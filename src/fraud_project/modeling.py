"""Entrenamiento y consumo del modelo XGBoost oficial."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from .config import CLASSIFICATION_THRESHOLD, MODEL_COMPONENTS, RANDOM_STATE
from .data import engineer_model_features
from .preprocessing import transform_for_model


DEFAULT_PARAM_GRID: dict[str, list[Any]] = {
    "n_estimators": [50, 100, 200],
    "max_depth": [4, 6],
    "learning_rate": [0.001, 0.01, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}


def create_xgboost_estimator(
    *,
    random_state: int = RANDOM_STATE,
    **params: Any,
) -> XGBClassifier:
    """Crea un clasificador compatible con el notebook."""
    defaults: dict[str, Any] = {
        "random_state": random_state,
        "eval_metric": "logloss",
        "n_jobs": -1,
    }
    defaults.update(params)
    return XGBClassifier(**defaults)


def train_best_xgboost(
    x_train: np.ndarray,
    y_train: pd.Series | np.ndarray,
    *,
    param_grid: Mapping[str, list[Any]] | None = None,
    cv: int = 5,
    scoring: str = "roc_auc",
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
) -> tuple[XGBClassifier, GridSearchCV]:
    """Ejecuta GridSearchCV y devuelve ``best_xgb``."""
    search = GridSearchCV(
        estimator=create_xgboost_estimator(random_state=random_state),
        param_grid=dict(param_grid or DEFAULT_PARAM_GRID),
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=False,
    )
    search.fit(x_train, np.asarray(y_train).astype(int))
    return search.best_estimator_, search


def predict_fraud_scores(
    model: XGBClassifier,
    x: np.ndarray,
) -> np.ndarray:
    """Devuelve la probabilidad de la clase fraude."""
    probabilities = model.predict_proba(x)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("El modelo no devolvió probabilidades binarias válidas.")
    return probabilities[:, 1].astype(float)


@dataclass(slots=True)
class FraudScoringPipeline:
    """Pipeline serializado para puntuar nuevas transacciones."""

    model: XGBClassifier
    preprocessor: Any
    svd_model: Any
    scaler: Any
    model_components: int = MODEL_COMPONENTS
    threshold: float = CLASSIFICATION_THRESHOLD

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        engineered = engineer_model_features(data)
        return transform_for_model(
            engineered,
            preprocessor=self.preprocessor,
            svd_model=self.svd_model,
            scaler=self.scaler,
            model_components=self.model_components,
        )

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        return predict_fraud_scores(self.model, self.transform(data))

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(data) >= self.threshold).astype(int)

    def score_dataframe(
        self,
        data: pd.DataFrame,
        *,
        score_column: str = "score_modelo_fuerte",
        prediction_column: str = "prediccion_fraude",
    ) -> pd.DataFrame:
        result = data.copy()
        result[score_column] = self.predict_proba(data)
        result[prediction_column] = (
            result[score_column] >= self.threshold
        ).astype(int)
        result["rank_modelo_fuerte"] = result[score_column].rank(
            ascending=False, method="first"
        ).astype(int)
        return result.sort_values(score_column, ascending=False)
