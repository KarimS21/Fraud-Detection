"""Baseline histórico, candidate pool y priorización de transacciones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .config import TARGET_COLUMN


@dataclass(slots=True)
class HistoricalBaseline:
    group_column: str
    risk_table: pd.Series
    global_fraud_rate: float

    def score(self, data: pd.DataFrame) -> np.ndarray:
        if self.group_column not in data:
            return np.repeat(self.global_fraud_rate, len(data))
        return (
            data[self.group_column]
            .map(self.risk_table)
            .fillna(self.global_fraud_rate)
            .astype(float)
            .to_numpy()
        )


def fit_historical_baseline(
    training_data: pd.DataFrame,
    *,
    target_col: str = TARGET_COLUMN,
    preferred_columns: Sequence[str] = (
        "categoria_comercio",
        "estado",
        "genero",
        "periodo_dia",
    ),
) -> HistoricalBaseline:
    """Ajusta el baseline simple utilizado en Semana 10."""
    if target_col not in training_data:
        raise ValueError(f"No existe la variable objetivo '{target_col}'.")

    group_column = next(
        (col for col in preferred_columns if col in training_data),
        "tasa_global",
    )
    target = pd.to_numeric(training_data[target_col], errors="raise").astype(int)
    global_rate = float(target.mean())

    if group_column == "tasa_global":
        risk_table = pd.Series(dtype=float, name="tasa_fraude")
    else:
        risk_table = (
            training_data.assign(**{target_col: target})
            .groupby(group_column, observed=True)[target_col]
            .mean()
            .sort_index()
        )
        risk_table.name = "tasa_fraude"

    return HistoricalBaseline(group_column, risk_table, global_rate)


def build_candidate_pool(
    data: pd.DataFrame,
    *,
    y_true,
    strong_scores,
    baseline: HistoricalBaseline,
) -> pd.DataFrame:
    """Combina verdad, baseline y modelo fuerte en una tabla rankeada."""
    result = data.reset_index(drop=True).copy()
    truth = pd.Series(y_true).reset_index(drop=True).astype(int)
    strong = pd.Series(strong_scores).reset_index(drop=True).astype(float)

    if len(result) != len(truth) or len(result) != len(strong):
        raise ValueError("Las entradas del candidate pool no están alineadas.")

    result["fraude_real"] = truth
    result["score_baseline"] = baseline.score(result)
    result["score_modelo_fuerte"] = strong
    result["rank_baseline"] = result["score_baseline"].rank(
        ascending=False, method="first"
    ).astype(int)
    result["rank_modelo_fuerte"] = result["score_modelo_fuerte"].rank(
        ascending=False, method="first"
    ).astype(int)
    return result


def top_risk_transactions(
    candidate_pool: pd.DataFrame,
    *,
    n: int = 25,
) -> pd.DataFrame:
    """Devuelve las transacciones con mayor score del modelo."""
    required = {"score_modelo_fuerte", "rank_modelo_fuerte"}
    missing = required.difference(candidate_pool.columns)
    if missing:
        raise ValueError(f"Faltan columnas del ranking: {sorted(missing)}")
    return candidate_pool.nsmallest(n, "rank_modelo_fuerte").copy()
