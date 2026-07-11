"""Preprocesamiento, codificación y reducción dimensional."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    HIGH_CARDINALITY_FEATURES,
    LOW_CARDINALITY_FEATURES,
    MODEL_COMPONENTS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    existing_features,
)


@dataclass(slots=True)
class PreparedMatrices:
    """Matrices y objetos ajustados que alimentan al modelo final."""

    preprocessor: ColumnTransformer
    svd_model: TruncatedSVD
    scaler: StandardScaler
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    model_components: int


def create_preprocessor(
    columns: Sequence[str],
    *,
    numeric_features: Sequence[str] = NUMERIC_FEATURES,
    low_cardinality: Sequence[str] = LOW_CARDINALITY_FEATURES,
    high_cardinality: Sequence[str] = HIGH_CARDINALITY_FEATURES,
) -> ColumnTransformer:
    """Construye el ``ColumnTransformer`` usado por el notebook."""
    numeric = existing_features(columns, numeric_features)
    low_card = existing_features(columns, low_cardinality)
    high_card = existing_features(columns, high_cardinality)

    if not numeric and not low_card and not high_card:
        raise ValueError("No se encontró ninguna variable del esquema del modelo.")

    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if low_card:
        transformers.append(
            (
                "low_cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=True,
                            ),
                        ),
                    ]
                ),
                low_card,
            )
        )
    if high_card:
        transformers.append(
            (
                "high_cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "target",
                            TargetEncoder(
                                handle_missing="value",
                                handle_unknown="value",
                                return_df=False,
                            ),
                        ),
                    ]
                ),
                high_card,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:
        names: list[str] = []
        for name, _, columns in preprocessor.transformers_:
            if name == "remainder":
                continue
            names.extend([f"{name}__{col}" for col in columns])
        return names


def fit_model_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target_col: str = TARGET_COLUMN,
    svd_components: int = 50,
    model_components: int = MODEL_COMPONENTS,
    random_state: int = RANDOM_STATE,
) -> PreparedMatrices:
    """Ajusta preprocesador, SVD y escalador sin fuga desde test."""
    if target_col not in train_df or target_col not in test_df:
        raise ValueError(f"Ambos datasets deben contener '{target_col}'.")

    x_train_df = train_df.drop(columns=[target_col])
    x_test_df = test_df.drop(columns=[target_col])
    y_train = pd.to_numeric(train_df[target_col], errors="raise").astype(int)
    y_test = pd.to_numeric(test_df[target_col], errors="raise").astype(int)

    preprocessor = create_preprocessor(x_train_df.columns)
    transformed_train = preprocessor.fit_transform(x_train_df, y_train)
    transformed_test = preprocessor.transform(x_test_df)

    maximum = min(
        svd_components,
        transformed_train.shape[0] - 1,
        transformed_train.shape[1] - 1,
    )
    if maximum < 2:
        raise ValueError("La matriz transformada no admite reducción SVD.")

    svd_model = TruncatedSVD(
        n_components=maximum, random_state=random_state
    )
    train_svd = svd_model.fit_transform(transformed_train)
    test_svd = svd_model.transform(transformed_test)

    selected = min(model_components, train_svd.shape[1])
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_svd[:, :selected])
    x_test = scaler.transform(test_svd[:, :selected])

    return PreparedMatrices(
        preprocessor=preprocessor,
        svd_model=svd_model,
        scaler=scaler,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        feature_names=_feature_names(preprocessor),
        model_components=selected,
    )


def transform_for_model(
    df: pd.DataFrame,
    *,
    preprocessor: ColumnTransformer,
    svd_model: TruncatedSVD,
    scaler: StandardScaler,
    model_components: int = MODEL_COMPONENTS,
) -> np.ndarray:
    """Transforma nuevas filas con los objetos ya ajustados."""
    transformed = preprocessor.transform(df)
    reduced = svd_model.transform(transformed)
    selected = min(model_components, reduced.shape[1])
    return scaler.transform(reduced[:, :selected])


def matrix_shape(value: object) -> tuple[int, ...]:
    """Retorna una forma uniforme para matrices densas o sparse."""
    if sparse.issparse(value):
        return value.shape
    return np.asarray(value).shape
