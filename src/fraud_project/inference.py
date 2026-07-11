"""Preparación y validación de datos para inferencia.

La capa de inferencia acepta dos entradas:

1. CSV con nombres originales del dataset de Kaggle.
2. Datos limpios en español, como los generados por el formulario visual.

El módulo no ajusta transformadores ni vuelve a entrenar el modelo. Únicamente
normaliza columnas y reconstruye las variables base necesarias antes de aplicar
el preprocesador serializado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from .config import RAW_COLUMN_MAP, TARGET_COLUMN
from .data import cast_fraud_dtypes, haversine_distance_km, rename_raw_columns

InputSchema = Literal["kaggle_raw", "clean_spanish", "unknown"]

# Columnas base que permiten reconstruir todas las variables usadas por el
# preprocesador final. ``edad_cliente`` y ``distancia_km`` también pueden ser
# calculadas a partir de fecha de nacimiento y coordenadas.
CORE_COLUMNS = [
    "fecha_hora_transaccion",
    "monto",
    "poblacion_ciudad",
    "genero",
    "categoria_comercio",
    "estado",
    "ciudad",
    "comercio",
    "ocupacion",
    "codigo_postal",
]

AGE_ALTERNATIVES = ("edad_cliente", "fecha_nacimiento")
DISTANCE_COLUMNS = (
    "latitud_cliente",
    "longitud_cliente",
    "latitud_comercio",
    "longitud_comercio",
)

RAW_KAGGLE_SIGNATURE = {
    "trans_date_trans_time",
    "merchant",
    "category",
    "amt",
    "gender",
    "city",
    "state",
    "zip",
    "city_pop",
    "job",
}

CLEAN_SIGNATURE = {
    "fecha_hora_transaccion",
    "comercio",
    "categoria_comercio",
    "monto",
    "genero",
    "ciudad",
    "estado",
    "codigo_postal",
    "poblacion_ciudad",
    "ocupacion",
}


@dataclass(slots=True)
class PreparedInferenceData:
    """Resultado de normalizar un lote de transacciones."""

    data: pd.DataFrame
    input_schema: InputSchema
    warnings: list[str] = field(default_factory=list)
    original_columns: list[str] = field(default_factory=list)


class InferenceValidationError(ValueError):
    """Error legible producido por una entrada que no puede puntuarse."""


def _drop_technical_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    technical = [
        column
        for column in result.columns
        if str(column).strip().lower().startswith("unnamed:")
    ]
    if technical:
        result = result.drop(columns=technical)
    return result


def detect_input_schema(columns: Iterable[str]) -> InputSchema:
    """Detecta si las columnas usan el esquema Kaggle o el esquema limpio."""
    available = {str(column) for column in columns}
    if RAW_KAGGLE_SIGNATURE.issubset(available):
        return "kaggle_raw"
    if CLEAN_SIGNATURE.issubset(available):
        return "clean_spanish"
    return "unknown"


def _missing_core_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in CORE_COLUMNS if column not in df.columns]


def _coerce_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    string_columns = [
        "genero",
        "categoria_comercio",
        "estado",
        "ciudad",
        "comercio",
        "ocupacion",
        "codigo_postal",
        "id_transaccion",
    ]
    for column in string_columns:
        if column in result.columns:
            result[column] = result[column].astype("string").str.strip()
    return result


def _calculate_age(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "edad_cliente" in result.columns:
        result["edad_cliente"] = pd.to_numeric(
            result["edad_cliente"], errors="coerce"
        )

    missing_age = (
        "edad_cliente" not in result.columns
        or result["edad_cliente"].isna().any()
    )
    if missing_age and {
        "fecha_hora_transaccion",
        "fecha_nacimiento",
    }.issubset(result.columns):
        transaction_date = pd.to_datetime(
            result["fecha_hora_transaccion"], errors="coerce"
        )
        birth_date = pd.to_datetime(result["fecha_nacimiento"], errors="coerce")
        calculated = np.floor((transaction_date - birth_date).dt.days / 365.25)
        if "edad_cliente" not in result.columns:
            result["edad_cliente"] = calculated
        else:
            result["edad_cliente"] = result["edad_cliente"].fillna(calculated)
    return result


def _calculate_distance(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "distancia_km" in result.columns:
        result["distancia_km"] = pd.to_numeric(
            result["distancia_km"], errors="coerce"
        )

    missing_distance = (
        "distancia_km" not in result.columns
        or result["distancia_km"].isna().any()
    )
    if missing_distance and set(DISTANCE_COLUMNS).issubset(result.columns):
        calculated = haversine_distance_km(
            result["latitud_cliente"],
            result["longitud_cliente"],
            result["latitud_comercio"],
            result["longitud_comercio"],
        )
        if "distancia_km" not in result.columns:
            result["distancia_km"] = calculated
        else:
            result["distancia_km"] = result["distancia_km"].fillna(calculated)
    return result


def _validate_values(df: pd.DataFrame) -> None:
    errors: list[str] = []

    missing = _missing_core_columns(df)
    if missing:
        errors.append("faltan columnas obligatorias: " + ", ".join(missing))

    if not any(column in df.columns for column in AGE_ALTERNATIVES):
        errors.append(
            "se requiere 'edad_cliente' o 'fecha_nacimiento' para calcular la edad"
        )

    has_distance = "distancia_km" in df.columns
    has_coordinates = set(DISTANCE_COLUMNS).issubset(df.columns)
    if not has_distance and not has_coordinates:
        errors.append(
            "se requiere 'distancia_km' o las cuatro coordenadas de cliente y comercio"
        )

    if errors:
        raise InferenceValidationError("; ".join(errors) + ".")

    invalid_datetime = df["fecha_hora_transaccion"].isna()
    if invalid_datetime.any():
        rows = (invalid_datetime[invalid_datetime].index[:5] + 1).tolist()
        errors.append(f"fecha de transacción inválida en filas {rows}")

    for column in ("monto", "poblacion_ciudad", "edad_cliente", "distancia_km"):
        if column not in df.columns:
            continue
        invalid = pd.to_numeric(df[column], errors="coerce").isna()
        if invalid.any():
            rows = (invalid[invalid].index[:5] + 1).tolist()
            errors.append(f"'{column}' contiene valores inválidos en filas {rows}")

    amount = pd.to_numeric(df["monto"], errors="coerce")
    if amount.lt(0).any():
        errors.append("'monto' no puede contener valores negativos")

    if "edad_cliente" in df.columns:
        age = pd.to_numeric(df["edad_cliente"], errors="coerce")
        if age.lt(0).any() or age.gt(120).any():
            errors.append("'edad_cliente' debe encontrarse entre 0 y 120 años")

    if "distancia_km" in df.columns:
        distance = pd.to_numeric(df["distancia_km"], errors="coerce")
        if distance.lt(0).any():
            errors.append("'distancia_km' no puede contener valores negativos")

    if errors:
        raise InferenceValidationError("; ".join(errors) + ".")


def prepare_inference_data(df: pd.DataFrame) -> PreparedInferenceData:
    """Normaliza un lote y reconstruye edad y distancia cuando sea necesario.

    Parameters
    ----------
    df:
        DataFrame con columnas originales de Kaggle o columnas limpias en
        español. Se permiten columnas adicionales; el preprocesador serializado
        seleccionará únicamente las que fueron usadas durante el entrenamiento.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("La entrada debe ser un pandas.DataFrame.")
    if df.empty:
        raise InferenceValidationError("El archivo no contiene transacciones.")

    original_columns = [str(column) for column in df.columns]
    data = _drop_technical_columns(df)
    schema = detect_input_schema(data.columns)

    if schema == "kaggle_raw":
        data = rename_raw_columns(data)
    elif schema == "unknown":
        # Se permite un esquema parcial siempre que, después del renombrado de
        # alias conocidos, satisfaga los campos obligatorios.
        known_raw = set(data.columns).intersection(RAW_COLUMN_MAP)
        if known_raw:
            data = rename_raw_columns(data)
        schema = (
            "clean_spanish"
            if CLEAN_SIGNATURE.issubset(set(data.columns))
            else "unknown"
        )

    _validate_values(data)

    data = cast_fraud_dtypes(data)
    data["fecha_hora_transaccion"] = pd.to_datetime(
        data["fecha_hora_transaccion"], errors="coerce"
    )
    if "fecha_nacimiento" in data.columns:
        data["fecha_nacimiento"] = pd.to_datetime(
            data["fecha_nacimiento"], errors="coerce"
        )

    for column in (
        "monto",
        "poblacion_ciudad",
        "edad_cliente",
        "distancia_km",
        *DISTANCE_COLUMNS,
    ):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data = _calculate_age(data)
    data = _calculate_distance(data)
    data = _coerce_string_columns(data)

    # Validación final una vez calculadas las variables derivadas.
    _validate_values(data)

    warnings: list[str] = []
    if TARGET_COLUMN in data.columns:
        labels = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
        invalid_labels = labels.dropna().loc[~labels.dropna().isin([0, 1])]
        if not invalid_labels.empty:
            warnings.append(
                "La columna de etiqueta contiene valores distintos de 0 y 1; "
                "se conservará para referencia, pero no se usará para puntuar."
            )
        data[TARGET_COLUMN] = labels

    if data["codigo_postal"].astype("string").str.len().lt(3).any():
        warnings.append(
            "Algunos códigos postales tienen menos de tres caracteres; revise "
            "que no se hayan perdido ceros a la izquierda al abrir el CSV."
        )

    data = data.reset_index(drop=True)
    data["fila_entrada"] = np.arange(1, len(data) + 1)
    return PreparedInferenceData(
        data=data,
        input_schema=schema,
        warnings=warnings,
        original_columns=original_columns,
    )
