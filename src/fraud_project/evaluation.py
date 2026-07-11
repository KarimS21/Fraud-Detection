"""Métricas offline y métricas operativas Top-K."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_metric(function, y_true, y_score) -> float:
    try:
        return float(function(y_true, y_score))
    except (TypeError, ValueError):
        return float("nan")


def evaluate_system(
    system_name: str,
    y_true,
    y_score,
    *,
    threshold: float = 0.5,
    system_type: str | None = None,
) -> dict[str, float | int | str]:
    """Evalúa scores continuos y predicciones al umbral indicado."""
    true = pd.Series(y_true).reset_index(drop=True).astype(int)
    score = pd.Series(y_score).reset_index(drop=True).astype(float)
    if len(true) != len(score):
        raise ValueError("y_true y y_score deben tener la misma longitud.")

    predicted = score.ge(threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        true, predicted, labels=[0, 1]
    ).ravel()

    return {
        "sistema": system_name,
        "tipo": system_type
        or ("baseline" if "baseline" in system_name.lower() else "modelo_fuerte"),
        "threshold": float(threshold),
        "roc_auc": _safe_metric(roc_auc_score, true, score),
        "pr_auc": _safe_metric(average_precision_score, true, score),
        "precision": float(
            precision_score(true, predicted, zero_division=0)
        ),
        "recall": float(recall_score(true, predicted, zero_division=0)),
        "f1": float(f1_score(true, predicted, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_comparison(
    y_true,
    systems: dict[str, np.ndarray | pd.Series],
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Evalúa varios sistemas sobre la misma población."""
    return pd.DataFrame(
        [
            evaluate_system(name, y_true, score, threshold=threshold)
            for name, score in systems.items()
        ]
    )


def topk_metrics(
    system_name: str,
    y_true,
    y_score,
    fraction: float,
) -> dict[str, float | int | str]:
    """Calcula Precision@K, Recall@K y Lift@K."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction debe estar en el intervalo (0, 1].")

    data = pd.DataFrame(
        {
            "y_true": pd.Series(y_true).reset_index(drop=True).astype(int),
            "score": pd.Series(y_score).reset_index(drop=True).astype(float),
        }
    )
    if data.empty:
        raise ValueError("No existen observaciones para evaluar.")

    k = max(1, int(len(data) * fraction))
    top = data.nlargest(k, "score")
    total_frauds = int(data["y_true"].sum())
    frauds_top = int(top["y_true"].sum())
    base_rate = float(data["y_true"].mean())
    precision_at_k = frauds_top / k
    recall_at_k = (
        frauds_top / total_frauds if total_frauds else float("nan")
    )
    lift_at_k = (
        precision_at_k / base_rate if base_rate else float("nan")
    )

    return {
        "sistema": system_name,
        "fraccion_pool": f"{fraction:.0%}",
        "k_transacciones_revisadas": k,
        "fraudes_en_top_k": frauds_top,
        "precision_at_k": float(precision_at_k),
        "recall_at_k": float(recall_at_k),
        "lift_at_k": float(lift_at_k),
    }


def build_topk_report(
    systems: dict[str, np.ndarray | pd.Series],
    y_true,
    *,
    fractions: tuple[float, ...] = (0.01, 0.05, 0.10),
) -> pd.DataFrame:
    """Genera una tabla Top-K comparable entre sistemas."""
    rows = [
        topk_metrics(name, y_true, score, fraction)
        for name, score in systems.items()
        for fraction in fractions
    ]
    return pd.DataFrame(rows)
