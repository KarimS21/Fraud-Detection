"""Servicios de aplicación utilizados por la demo visual y la CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .artifact_manager import (
    LoadedArtifacts,
    inspect_artifacts,
    load_training_artifacts,
)
from .config import ARTIFACTS_DIR, TARGET_COLUMN
from .inference import PreparedInferenceData, prepare_inference_data


@dataclass(slots=True)
class ScoringResult:
    """Resultado completo de puntuar un lote."""

    scored: pd.DataFrame
    prepared: PreparedInferenceData
    summary: dict[str, Any]
    evaluation: dict[str, float]


def risk_level(score: float) -> str:
    """Convierte un score en una categoría descriptiva para la interfaz."""
    if score >= 0.80:
        return "Crítico"
    if score >= 0.50:
        return "Alto"
    if score >= 0.20:
        return "Medio"
    return "Bajo"


def risk_action(level: str) -> str:
    actions = {
        "Crítico": "Priorizar para revisión inmediata.",
        "Alto": "Enviar a revisión manual prioritaria.",
        "Medio": "Revisar si existen señales adicionales.",
        "Bajo": "Mantener seguimiento según las reglas operativas.",
    }
    return actions.get(level, "Revisar la transacción.")


class VisualDemoService:
    """Fachada que carga artefactos una vez y puntúa transacciones."""

    def __init__(
        self,
        artifacts_dir: str | Path = ARTIFACTS_DIR,
        *,
        loaded: LoadedArtifacts | None = None,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.loaded = loaded or load_training_artifacts(self.artifacts_dir)

    @property
    def metadata(self) -> dict[str, Any]:
        return self.loaded.metadata

    @property
    def threshold(self) -> float:
        return float(self.loaded.pipeline.threshold)

    def artifact_status(self) -> pd.DataFrame:
        return inspect_artifacts(self.artifacts_dir)

    def _add_baseline_score(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        baseline = self.loaded.baseline_risk_table
        if baseline is None or baseline.empty:
            return result

        group_column = str(
            self.metadata.get("baseline_group_column", "categoria_comercio")
        )
        if group_column not in result.columns or group_column not in baseline.columns:
            return result

        value_candidates = [
            column
            for column in baseline.columns
            if column != group_column
            and pd.api.types.is_numeric_dtype(baseline[column])
        ]
        if not value_candidates:
            return result

        value_column = (
            "tasa_fraude" if "tasa_fraude" in value_candidates else value_candidates[0]
        )
        table = baseline.set_index(group_column)[value_column]
        default = float(
            self.metadata.get(
                "experimental_train_fraud_rate",
                pd.to_numeric(table, errors="coerce").mean(),
            )
        )
        result["score_baseline"] = (
            result[group_column].astype("string").map(table).fillna(default).astype(float)
        )
        return result

    def score_transactions(self, data: pd.DataFrame) -> ScoringResult:
        prepared = prepare_inference_data(data)
        scored = self.loaded.pipeline.score_dataframe(prepared.data)
        scored = self._add_baseline_score(scored)
        scored["probabilidad_fraude_pct"] = scored["score_modelo_fuerte"] * 100
        scored["nivel_riesgo"] = scored["score_modelo_fuerte"].map(risk_level)
        scored["accion_sugerida"] = scored["nivel_riesgo"].map(risk_action)

        count = len(scored)
        positives = int(scored["prediccion_fraude"].sum())
        summary = {
            "rows": count,
            "predicted_positive": positives,
            "predicted_positive_rate": positives / count if count else 0.0,
            "mean_score": float(scored["score_modelo_fuerte"].mean()),
            "max_score": float(scored["score_modelo_fuerte"].max()),
            "threshold": self.threshold,
            "critical_count": int(scored["nivel_riesgo"].eq("Crítico").sum()),
        }

        evaluation: dict[str, float] = {}
        if TARGET_COLUMN in scored.columns:
            labels = pd.to_numeric(scored[TARGET_COLUMN], errors="coerce")
            valid = labels.isin([0, 1])
            y_true = labels.loc[valid].astype(int)
            y_score = scored.loc[valid, "score_modelo_fuerte"].astype(float)
            if len(y_true) and y_true.nunique() == 2:
                evaluation["roc_auc"] = float(roc_auc_score(y_true, y_score))
                evaluation["pr_auc"] = float(
                    average_precision_score(y_true, y_score)
                )
            if len(y_true):
                evaluation["observed_fraud_rate"] = float(y_true.mean())
                top_n = max(1, int(np.ceil(len(y_true) * 0.05)))
                top_indices = y_score.nlargest(top_n).index
                total_fraud = int(y_true.sum())
                evaluation["recall_top5"] = (
                    float(y_true.loc[top_indices].sum() / total_fraud)
                    if total_fraud
                    else 0.0
                )

        return ScoringResult(
            scored=scored,
            prepared=prepared,
            summary=summary,
            evaluation=evaluation,
        )


def score_bins(scores: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Construye una tabla de frecuencias para graficar scores."""
    clean = pd.to_numeric(scores, errors="coerce").dropna().clip(0, 1)
    edges = np.linspace(0, 1, bins + 1)
    categories = pd.cut(clean, bins=edges, include_lowest=True, right=True)
    counts = categories.value_counts(sort=False)
    result = counts.rename_axis("intervalo").reset_index(name="transacciones")
    result["rango_score"] = result["intervalo"].astype(str)
    return result[["rango_score", "transacciones"]]


def metadata_comparison(metadata: dict[str, Any]) -> pd.DataFrame:
    """Tabla compacta baseline versus modelo para la portada."""
    rows = []
    mappings = [
        ("ROC-AUC", "roc_auc_baseline", "roc_auc_model"),
        ("PR-AUC", "pr_auc_baseline", "pr_auc_model"),
        ("Recall@Top 5 %", "recall_top5_baseline", "recall_top5_model"),
    ]
    for metric, baseline_key, model_key in mappings:
        baseline = metadata.get(baseline_key)
        model = metadata.get(model_key)
        if baseline is not None and model is not None:
            rows.append(
                {
                    "métrica": metric,
                    "Baseline histórico": float(baseline),
                    "XGBoost optimizado": float(model),
                }
            )
    return pd.DataFrame(rows)
