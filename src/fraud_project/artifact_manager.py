"""Persistencia, carga e inspección de artefactos entrenados."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
import json

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .config import (
    ARTIFACTS_DIR,
    CLASSIFICATION_THRESHOLD,
    MODEL_COMPONENTS,
    MODEL_INPUT_FEATURES,
    RANDOM_STATE,
)
from .modeling import FraudScoringPipeline

MODEL_FILENAME = "fraud_xgboost_model.json"
PREPROCESSOR_FILENAME = "preprocessor.joblib"
SVD_FILENAME = "svd_model.joblib"
SCALER_FILENAME = "scaler_cluster.joblib"
BASELINE_FILENAME = "historical_risk_table.csv"
METADATA_FILENAME = "model_metadata.json"
SCHEMA_FILENAME = "input_schema.json"
MANIFEST_FILENAME = "artifact_manifest.json"


@dataclass(slots=True)
class LoadedArtifacts:
    pipeline: FraudScoringPipeline
    baseline_risk_table: pd.DataFrame | None
    metadata: dict[str, Any]
    manifest: dict[str, Any]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime,)):
        return value.isoformat()
    raise TypeError(f"Objeto no serializable: {type(value).__name__}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_manifest(root: str | Path = ARTIFACTS_DIR) -> dict[str, Any]:
    """Genera un inventario con hashes para todos los artefactos."""
    root = Path(root)
    manifest_path = root / "metadata" / MANIFEST_FILENAME
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_root": root.name,
        "files": files,
    }
    _write_json(manifest_path, payload)
    return payload


def save_training_artifacts(
    *,
    model: XGBClassifier,
    preprocessor: Any,
    svd_model: Any,
    scaler: Any,
    risk_table: pd.Series | pd.DataFrame | None = None,
    metadata: dict[str, Any] | None = None,
    output_dir: str | Path = ARTIFACTS_DIR,
) -> dict[str, Path]:
    """Guarda el modelo oficial, transformadores, baseline y metadatos."""
    output_dir = Path(output_dir)
    models_dir = output_dir / "models"
    metadata_dir = output_dir / "metadata"
    baselines_dir = output_dir / "baselines"
    for directory in (models_dir, metadata_dir, baselines_dir):
        directory.mkdir(parents=True, exist_ok=True)

    paths = {
        "model": models_dir / MODEL_FILENAME,
        "preprocessor": models_dir / PREPROCESSOR_FILENAME,
        "svd_model": models_dir / SVD_FILENAME,
        "scaler": models_dir / SCALER_FILENAME,
        "baseline": baselines_dir / BASELINE_FILENAME,
        "metadata": metadata_dir / METADATA_FILENAME,
        "schema": metadata_dir / SCHEMA_FILENAME,
        "manifest": metadata_dir / MANIFEST_FILENAME,
    }

    if not hasattr(model, "save_model"):
        raise TypeError("El modelo oficial debe ser un XGBClassifier entrenado.")
    model.save_model(paths["model"])
    joblib.dump(preprocessor, paths["preprocessor"])
    joblib.dump(svd_model, paths["svd_model"])
    joblib.dump(scaler, paths["scaler"])

    if risk_table is not None:
        if isinstance(risk_table, pd.Series):
            baseline_df = risk_table.rename("tasa_fraude").reset_index()
        else:
            baseline_df = risk_table.copy()
        baseline_df.to_csv(paths["baseline"], index=False)

    base_metadata: dict[str, Any] = {
        "model_name": "XGBoost optimizado",
        "model_version": "1.0.0",
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "model_components": MODEL_COMPONENTS,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "artifact_status": "ready",
    }
    base_metadata.update(metadata or {})
    _write_json(paths["metadata"], base_metadata)

    schema = {
        "schema_version": "1.0.0",
        "required_model_features": MODEL_INPUT_FEATURES,
        "target_column": "es_fraude",
        "raw_datetime_columns": [
            "fecha_hora_transaccion",
            "fecha_nacimiento",
        ],
        "notes": [
            "engineer_model_features crea variables temporales y log_monto.",
            "Las categorías desconocidas se gestionan mediante los encoders ajustados.",
        ],
    }
    _write_json(paths["schema"], schema)
    create_manifest(output_dir)
    return paths


def load_training_artifacts(
    artifacts_dir: str | Path = ARTIFACTS_DIR,
) -> LoadedArtifacts:
    """Carga el pipeline completo y verifica los archivos obligatorios."""
    artifacts_dir = Path(artifacts_dir)
    models_dir = artifacts_dir / "models"
    metadata_dir = artifacts_dir / "metadata"
    baselines_dir = artifacts_dir / "baselines"

    required = {
        "model": models_dir / MODEL_FILENAME,
        "preprocessor": models_dir / PREPROCESSOR_FILENAME,
        "svd_model": models_dir / SVD_FILENAME,
        "scaler": models_dir / SCALER_FILENAME,
        "metadata": metadata_dir / METADATA_FILENAME,
        "manifest": metadata_dir / MANIFEST_FILENAME,
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan artefactos entrenados:\n- " + "\n- ".join(missing)
        )

    metadata = json.loads(required["metadata"].read_text(encoding="utf-8"))
    manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
    model = XGBClassifier()
    model.load_model(required["model"])
    pipeline = FraudScoringPipeline(
        model=model,
        preprocessor=joblib.load(required["preprocessor"]),
        svd_model=joblib.load(required["svd_model"]),
        scaler=joblib.load(required["scaler"]),
        model_components=int(metadata.get("model_components", MODEL_COMPONENTS)),
        threshold=float(
            metadata.get("classification_threshold", CLASSIFICATION_THRESHOLD)
        ),
    )

    baseline_path = baselines_dir / BASELINE_FILENAME
    baseline = pd.read_csv(baseline_path) if baseline_path.exists() else None
    return LoadedArtifacts(pipeline, baseline, metadata, manifest)


def inspect_artifacts(
    artifacts_dir: str | Path = ARTIFACTS_DIR,
) -> pd.DataFrame:
    """Devuelve una tabla verificable con tamaño, hash y estado."""
    artifacts_dir = Path(artifacts_dir)
    manifest_path = artifacts_dir / "metadata" / MANIFEST_FILENAME
    if not manifest_path.exists():
        return pd.DataFrame(
            [
                {
                    "path": str(manifest_path),
                    "exists": False,
                    "size_bytes": 0,
                    "sha256_matches": False,
                }
            ]
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for item in manifest.get("files", []):
        path = artifacts_dir / item["path"]
        exists = path.exists()
        rows.append(
            {
                "path": item["path"],
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else 0,
                "sha256_matches": (
                    sha256_file(path) == item["sha256"] if exists else False
                ),
            }
        )
    return pd.DataFrame(rows)
