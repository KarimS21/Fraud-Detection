"""Funciones de presentación independientes de Streamlit."""

from __future__ import annotations

from typing import Any

import pandas as pd


RISK_ORDER = ["Crítico", "Alto", "Medio", "Bajo"]


def top_transaction_columns(frame: pd.DataFrame) -> list[str]:
    """Selecciona columnas legibles para tablas de la demo."""
    preferred = [
        "rank_modelo_fuerte",
        "score_modelo_fuerte",
        "probabilidad_fraude_pct",
        "nivel_riesgo",
        "prediccion_fraude",
        "es_fraude",
        "fecha_hora_transaccion",
        "categoria_comercio",
        "monto",
        "comercio",
        "ciudad",
        "estado",
        "distancia_km",
        "edad_cliente",
        "fila_entrada",
    ]
    return [column for column in preferred if column in frame.columns]


def risk_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Cuenta transacciones por nivel de riesgo respetando orden semántico."""
    if "nivel_riesgo" not in frame.columns:
        return pd.DataFrame(columns=["nivel_riesgo", "transacciones"])
    counts = frame["nivel_riesgo"].value_counts()
    return pd.DataFrame(
        {
            "nivel_riesgo": RISK_ORDER,
            "transacciones": [int(counts.get(level, 0)) for level in RISK_ORDER],
        }
    )


def category_risk_summary(frame: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Resume score promedio y volumen por categoría."""
    required = {"categoria_comercio", "score_modelo_fuerte"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    summary = (
        frame.groupby("categoria_comercio", dropna=False)
        .agg(
            score_promedio=("score_modelo_fuerte", "mean"),
            score_maximo=("score_modelo_fuerte", "max"),
            transacciones=("score_modelo_fuerte", "size"),
        )
        .reset_index()
        .sort_values(["score_promedio", "transacciones"], ascending=[False, False])
        .head(top)
    )
    return summary


def compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Selecciona los campos más relevantes para la vista de operación."""
    keys = [
        "model_name",
        "model_version",
        "artifact_status",
        "exported_at_utc",
        "model_components",
        "classification_threshold",
        "training_rows",
        "test_rows",
        "experimental_train_fraud_rate",
        "experimental_test_fraud_rate",
        "notebook",
    ]
    return {key: metadata.get(key) for key in keys if key in metadata}
