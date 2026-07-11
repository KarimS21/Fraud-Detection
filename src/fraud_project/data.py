"""Carga, limpieza y creación de variables del dataset transaccional."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    FINAL_CLEAN_COLUMNS,
    INTERIM_DATA_DIR,
    RANDOM_STATE,
    RAW_COLUMN_MAP,
    RAW_DATA_DIR,
    TARGET_COLUMN,
)


def load_raw_datasets(
    raw_dir: str | Path = RAW_DATA_DIR,
    train_name: str = "fraudTrain.csv",
    test_name: str = "fraudTest.csv",
    *,
    index_col: int | str | None = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los CSV originales y valida que existan."""
    raw_dir = Path(raw_dir)
    train_path = raw_dir / train_name
    test_path = raw_dir / test_name

    missing = [str(path) for path in (train_path, test_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "No se encontraron los datasets originales:\n- " + "\n- ".join(missing)
        )

    return (
        pd.read_csv(train_path, index_col=index_col),
        pd.read_csv(test_path, index_col=index_col),
    )


def rename_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra las columnas originales al esquema utilizado en el proyecto."""
    return df.rename(columns=RAW_COLUMN_MAP).copy()


def cast_fraud_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Estandariza tipos sin modificar el DataFrame original."""
    result = df.copy()

    for col in ("numero_tarjeta", "codigo_postal", "id_transaccion"):
        if col in result:
            result[col] = result[col].astype("string")

    if "fecha_hora_transaccion" in result:
        result["fecha_hora_transaccion"] = pd.to_datetime(
            result["fecha_hora_transaccion"], errors="coerce"
        )

    if "fecha_nacimiento" in result:
        result["fecha_nacimiento"] = pd.to_datetime(
            result["fecha_nacimiento"], errors="coerce"
        )

    if TARGET_COLUMN in result:
        result[TARGET_COLUMN] = pd.to_numeric(
            result[TARGET_COLUMN], errors="coerce"
        ).astype("Int8")

    for col in ("genero", "estado", "categoria_comercio"):
        if col in result:
            result[col] = result[col].astype("category")

    return result


def haversine_distance_km(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:
    """Calcula distancia Haversine vectorizada entre cliente y comercio."""
    lat1_r = np.radians(pd.to_numeric(lat1, errors="coerce"))
    lon1_r = np.radians(pd.to_numeric(lon1, errors="coerce"))
    lat2_r = np.radians(pd.to_numeric(lat2, errors="coerce"))
    lon2_r = np.radians(pd.to_numeric(lon2, errors="coerce"))

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    return pd.Series(6371.0 * 2 * np.arcsin(np.sqrt(a)), index=lat1.index)


def build_clean_dataset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Une train/test, elimina PII directa y crea variables base."""
    train = cast_fraud_dtypes(rename_raw_columns(train_df))
    test = cast_fraud_dtypes(rename_raw_columns(test_df))
    train["type"] = "train"
    test["type"] = "test"

    data = pd.concat([train, test], ignore_index=True)

    numeric_cols = [
        "monto",
        "latitud_cliente",
        "longitud_cliente",
        "poblacion_ciudad",
        "timestamp_unix",
        "latitud_comercio",
        "longitud_comercio",
        TARGET_COLUMN,
        "codigo_postal",
    ]
    for col in numeric_cols:
        if col in data:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if "fecha_hora_transaccion" in data:
        data["fecha_hora_transaccion"] = pd.to_datetime(
            data["fecha_hora_transaccion"], errors="coerce"
        )
        data["hora_transaccion"] = data["fecha_hora_transaccion"].dt.hour.astype(
            "Int64"
        )
        data["dia_semana_transaccion"] = (
            data["fecha_hora_transaccion"].dt.day_name()
        )
        data["mes_transaccion"] = data["fecha_hora_transaccion"].dt.month.astype(
            "Int64"
        )

    if {"fecha_hora_transaccion", "fecha_nacimiento"}.issubset(data.columns):
        data["fecha_nacimiento"] = pd.to_datetime(
            data["fecha_nacimiento"], errors="coerce"
        )
        age_years = (
            data["fecha_hora_transaccion"] - data["fecha_nacimiento"]
        ).dt.days / 365.25
        data["edad_cliente"] = np.floor(age_years).astype("Int64")

    geo_cols = {
        "latitud_cliente",
        "longitud_cliente",
        "latitud_comercio",
        "longitud_comercio",
    }
    if geo_cols.issubset(data.columns):
        data["distancia_km"] = haversine_distance_km(
            data["latitud_cliente"],
            data["longitud_cliente"],
            data["latitud_comercio"],
            data["longitud_comercio"],
        )

    data = data.drop(
        columns=[
            col
            for col in ("nombre", "apellido", "direccion")
            if col in data.columns
        ]
    )

    if "numero_tarjeta" in data:
        data["tarjeta_ultimos4"] = data["numero_tarjeta"].astype(str).str[-4:]
        data = data.drop(columns=["numero_tarjeta"])

    ordered = [col for col in FINAL_CLEAN_COLUMNS if col in data.columns]
    remaining = [col for col in data.columns if col not in ordered]
    return data.loc[:, ordered + remaining]


def classify_day_period(hour: Any) -> str:
    """Clasifica una hora en madrugada, mañana, tarde o noche."""
    if pd.isna(hour):
        return "desconocido"
    hour = int(hour)
    if 0 <= hour <= 5:
        return "madrugada"
    if 6 <= hour <= 11:
        return "mañana"
    if 12 <= hour <= 17:
        return "tarde"
    return "noche"


def engineer_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Genera las variables utilizadas por el preprocesador final."""
    result = df.copy()

    if "fecha_hora_transaccion" in result:
        result["fecha_hora_transaccion"] = pd.to_datetime(
            result["fecha_hora_transaccion"], errors="coerce"
        )
        result["hora_transaccion"] = result["fecha_hora_transaccion"].dt.hour
        result["dia_semana_num"] = (
            result["fecha_hora_transaccion"].dt.dayofweek
        )
        result["dia_semana_transaccion"] = (
            result["fecha_hora_transaccion"].dt.day_name()
        )
        result["mes_transaccion"] = result["fecha_hora_transaccion"].dt.month

    if "dia_semana_num" in result:
        result["es_fin_de_semana"] = (
            result["dia_semana_num"].isin([5, 6]).astype(int)
        )

    if "hora_transaccion" in result:
        result["hora_sin"] = np.sin(
            2 * np.pi * result["hora_transaccion"] / 24
        )
        result["hora_cos"] = np.cos(
            2 * np.pi * result["hora_transaccion"] / 24
        )
        result["periodo_dia"] = result["hora_transaccion"].map(
            classify_day_period
        )

    if "mes_transaccion" in result:
        result["mes_sin"] = np.sin(
            2 * np.pi * result["mes_transaccion"] / 12
        )
        result["mes_cos"] = np.cos(
            2 * np.pi * result["mes_transaccion"] / 12
        )

    if "monto" in result:
        amount = pd.to_numeric(result["monto"], errors="coerce")
        if (amount.dropna() < -1).any():
            raise ValueError("La variable monto contiene valores menores que -1.")
        result["log_monto"] = np.log1p(amount)

    return result


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa el dataset integrado usando la columna ``type``."""
    if "type" not in df:
        raise ValueError("El dataset no contiene la columna 'type'.")
    return (
        df.loc[df["type"].eq("train")].copy(),
        df.loc[df["type"].eq("test")].copy(),
    )


def sample_binary_dataset(
    df: pd.DataFrame,
    *,
    max_rows: int,
    max_positive: int,
    target_col: str = TARGET_COLUMN,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Crea la muestra experimental usada por el notebook.

    Esta operación altera la prevalencia original. Las métricas sensibles a la
    prevalencia deben interpretarse y documentarse con esa limitación.
    """
    if target_col not in df:
        raise ValueError(f"No existe la variable objetivo '{target_col}'.")

    valid = df.dropna(subset=[target_col]).copy()
    valid[target_col] = pd.to_numeric(
        valid[target_col], errors="raise"
    ).astype(int)

    if len(valid) <= max_rows:
        return valid.reset_index(drop=True)

    positives = valid.loc[valid[target_col].eq(1)]
    negatives = valid.loc[valid[target_col].eq(0)]
    n_positive = min(len(positives), max_positive)
    n_negative = min(len(negatives), max_rows - n_positive)

    sampled = pd.concat(
        [
            positives.sample(n=n_positive, random_state=random_state),
            negatives.sample(n=n_negative, random_state=random_state),
        ],
        ignore_index=True,
    )
    return sampled.sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)


def save_interim_dataset(
    df: pd.DataFrame,
    path: str | Path = INTERIM_DATA_DIR / "transactions_clean_v2.csv",
) -> Path:
    """Guarda el dataset limpio y devuelve su ruta."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
