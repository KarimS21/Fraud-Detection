"""Validaciones de estructura, contenido y artefactos finales."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from .artifact_manager import inspect_artifacts
from .config import PROJECT_ROOT


@dataclass(slots=True)
class ValidationResult:
    check: str
    status: str
    detail: str
    path: str = ""


def _result(
    check: str,
    ok: bool,
    detail: str,
    path: Path | None = None,
    *,
    warning: bool = False,
) -> ValidationResult:
    status = "WARNING" if warning and not ok else ("OK" if ok else "ERROR")
    return ValidationResult(check, status, detail, str(path or ""))


def _validate_csv(
    path: Path,
    *,
    required_columns: set[str] | None = None,
    min_rows: int = 1,
    exact_rows: int | None = None,
) -> tuple[bool, str]:
    if not path.exists():
        return False, "El archivo no existe."
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return False, f"No se pudo leer: {exc}"
    if "mensaje" in frame.columns:
        return False, "El archivo contiene un placeholder en vez de resultados."
    if exact_rows is not None and len(frame) != exact_rows:
        return False, f"Se esperaban {exact_rows} filas y existen {len(frame)}."
    if len(frame) < min_rows:
        return False, f"Se esperaban al menos {min_rows} filas."
    missing = (required_columns or set()).difference(frame.columns)
    if missing:
        return False, f"Faltan columnas: {sorted(missing)}"
    return True, f"CSV válido con {len(frame)} filas y {len(frame.columns)} columnas."


def validate_repository(
    project_root: str | Path = PROJECT_ROOT,
) -> list[ValidationResult]:
    """Ejecuta controles que sustituyen el simple ``path.exists()``."""
    root = Path(project_root)
    results: list[ValidationResult] = []

    required_directories = [
        "src/fraud_project",
        "data/raw",
        "data/interim",
        "data/week5",
        "data/week7",
        "data/week10",
        "data/week12",
        "artifacts",
        "reports",
        "delivery/week14_final_integrated_delivery",
        ".streamlit",
        "data/demo",
    ]
    for relative in required_directories:
        path = root / relative
        results.append(
            _result(
                f"Directorio {relative}",
                path.is_dir(),
                "Directorio disponible." if path.is_dir() else "Falta el directorio.",
                path,
            )
        )

    for relative in (
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        "README.md",
        "main.py",
        "streamlit_app.py",
        ".streamlit/config.toml",
        "src/fraud_project/inference.py",
        "src/fraud_project/demo_service.py",
        "src/fraud_project/demo_visuals.py",
    ):
        path = root / relative
        results.append(
            _result(
                f"Archivo raíz {relative}",
                path.is_file(),
                "Archivo disponible." if path.is_file() else "Falta el archivo.",
                path,
            )
        )

    raw_files = [
        root / "data/raw/fraudTrain.csv",
        root / "data/raw/fraudTest.csv",
    ]
    for path in raw_files:
        results.append(
            _result(
                f"Dataset local {path.name}",
                path.exists(),
                (
                    "Dataset disponible localmente."
                    if path.exists()
                    else "No está localmente; puede descargarse según el README."
                ),
                path,
                warning=True,
            )
        )

    csv_checks = [
        (
            root / "data/week10/offline_evaluation_report_week10.csv",
            {"sistema", "roc_auc", "pr_auc"},
            2,
            None,
        ),
        (
            root / "data/week10/topk_ranking_report_week10.csv",
            {"sistema", "fraccion_pool", "recall_at_k", "lift_at_k"},
            6,
            None,
        ),
        (
            root
            / "delivery/week14_final_integrated_delivery/demo/"
            "final_demo_top25_risk_transactions_week14.csv",
            {
                "score_modelo_fuerte",
                "rank_modelo_fuerte",
                "fraude_real",
            },
            25,
            25,
        ),
    ]
    for path, columns, min_rows, exact_rows in csv_checks:
        ok, detail = _validate_csv(
            path,
            required_columns=columns,
            min_rows=min_rows,
            exact_rows=exact_rows,
        )
        results.append(_result(f"Contenido {path.name}", ok, detail, path))


    demo_sample = root / "data/demo/sample_transactions_kaggle_format.csv"
    demo_columns = {
        "trans_date_trans_time",
        "merchant",
        "category",
        "amt",
        "gender",
        "city",
        "state",
        "zip",
        "lat",
        "long",
        "city_pop",
        "job",
        "dob",
        "merch_lat",
        "merch_long",
    }
    ok, detail = _validate_csv(
        demo_sample, required_columns=demo_columns, min_rows=1
    )
    results.append(_result("CSV de ejemplo para demo visual", ok, detail, demo_sample))

    pyproject_path = root / "pyproject.toml"
    if pyproject_path.exists():
        pyproject_text = pyproject_path.read_text(encoding="utf-8")
        streamlit_declared = "streamlit" in pyproject_text.lower()
        results.append(
            _result(
                "Dependencia Streamlit",
                streamlit_declared,
                (
                    "Streamlit está declarado en pyproject.toml."
                    if streamlit_declared
                    else "Falta declarar Streamlit en pyproject.toml."
                ),
                pyproject_path,
            )
        )

    artifact_table = inspect_artifacts(root / "artifacts")
    if artifact_table.empty:
        results.append(
            _result(
                "Artefactos entrenados",
                False,
                "No se encontraron artefactos ni manifiesto.",
                root / "artifacts",
            )
        )
    else:
        all_exist = bool(artifact_table["exists"].all())
        hashes_ok = bool(artifact_table["sha256_matches"].all())
        results.append(
            _result(
                "Integridad de artefactos",
                all_exist and hashes_ok,
                (
                    "Todos los hashes coinciden."
                    if all_exist and hashes_ok
                    else "Faltan artefactos o algún hash no coincide."
                ),
                root / "artifacts",
            )
        )

    report_paths = [
        root / "reports/final/technical_report.md",
        root / "reports/final/model_card.md",
        root / "reports/final/data_card.md",
        root / "reports/final/monitoring_plan.md",
        root / "reports/final/limitations_future_work.md",
    ]
    for path in report_paths:
        valid = path.exists() and path.stat().st_size > 200
        results.append(
            _result(
                f"Reporte {path.name}",
                valid,
                (
                    "Documento disponible y no vacío."
                    if valid
                    else "Falta el documento o es demasiado pequeño."
                ),
                path,
            )
        )
    return results


def results_dataframe(results: list[ValidationResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check": item.check,
                "status": item.status,
                "detail": item.detail,
                "path": item.path,
            }
            for item in results
        ]
    )


def has_errors(results: list[ValidationResult]) -> bool:
    return any(item.status == "ERROR" for item in results)
