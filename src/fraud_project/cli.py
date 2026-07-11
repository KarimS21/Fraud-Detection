"""Interfaz de línea de comandos del proyecto."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd

from .artifact_manager import inspect_artifacts, load_training_artifacts
from .config import ARTIFACTS_DIR, PROJECT_ROOT
from .reporting import generate_report_bundle
from .validation import has_errors, results_dataframe, validate_repository


def _print_frame(frame: pd.DataFrame, max_rows: int = 30) -> None:
    if frame.empty:
        print("No hay registros para mostrar.")
        return
    with pd.option_context(
        "display.max_rows",
        max_rows,
        "display.max_columns",
        None,
        "display.width",
        180,
    ):
        print(frame.head(max_rows).to_string(index=False))


def command_validate(args: argparse.Namespace) -> int:
    results = validate_repository(args.project_root)
    frame = results_dataframe(results)
    _print_frame(frame, max_rows=100)
    errors = int(frame["status"].eq("ERROR").sum())
    warnings = int(frame["status"].eq("WARNING").sum())
    print(f"\nResumen: {errors} error(es), {warnings} advertencia(s).")
    return 1 if has_errors(results) else 0


def command_demo(args: argparse.Namespace) -> int:
    root = Path(args.project_root)
    preferred = (
        root
        / "delivery/week14_final_integrated_delivery/demo/"
        "final_demo_top25_risk_transactions_week14.csv"
    )
    fallback = root / "data/week10/candidate_pool_ranked_week10.csv"
    path = preferred if preferred.exists() else fallback
    if not path.exists():
        print("No existe un candidate pool o demo final para mostrar.")
        return 1

    frame = pd.read_csv(path)
    if "mensaje" in frame:
        print("El archivo encontrado es un placeholder y no un demo válido.")
        return 1
    if "rank_modelo_fuerte" in frame:
        frame = frame.nsmallest(args.top, "rank_modelo_fuerte")
    elif "score_modelo_fuerte" in frame:
        frame = frame.nlargest(args.top, "score_modelo_fuerte")
    else:
        print("El archivo no contiene columnas de ranking del modelo.")
        return 1

    columns = [
        col
        for col in (
            "rank_modelo_fuerte",
            "score_modelo_fuerte",
            "fraude_real",
            "fecha_hora_transaccion",
            "categoria_comercio",
            "monto",
            "comercio",
            "ciudad",
            "estado",
        )
        if col in frame
    ]
    print(f"Demo cargado desde: {path}")
    _print_frame(frame[columns], max_rows=args.top)
    return 0


def command_inspect_artifacts(args: argparse.Namespace) -> int:
    table = inspect_artifacts(args.artifacts_dir)
    _print_frame(table, max_rows=100)
    valid = (
        not table.empty
        and "exists" in table
        and bool(table["exists"].all())
        and bool(table["sha256_matches"].all())
    )
    if valid:
        loaded = load_training_artifacts(args.artifacts_dir)
        print("\nMetadatos:")
        for key, value in loaded.metadata.items():
            print(f"- {key}: {value}")
    return 0 if valid else 1


def command_generate_reports(args: argparse.Namespace) -> int:
    outputs = generate_report_bundle(
        project_root=args.project_root,
        reports_dir=Path(args.project_root) / "reports",
    )
    print("Reportes generados:")
    for group in ("tables", "figures", "documents"):
        for path in outputs[group]:
            print(f"- {path}")
    print(f"- {outputs['manifest']}")
    return 0


def command_export_artifacts(args: argparse.Namespace) -> int:
    """Ejecuta el notebook que contiene la celda de exportación."""
    root = Path(args.project_root)
    notebook = root / args.notebook
    if not notebook.exists():
        print(f"No existe el notebook: {notebook}")
        return 1

    output = root / args.output_notebook
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(notebook),
        "--output",
        output.name,
        "--output-dir",
        str(output.parent),
        f"--ExecutePreprocessor.timeout={args.timeout}",
    ]
    print("Ejecutando notebook para entrenar y exportar artefactos...")
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode:
        print("La ejecución del notebook falló.")
        return completed.returncode

    manifest = root / "artifacts/metadata/artifact_manifest.json"
    if not manifest.exists():
        print(
            "El notebook terminó, pero no generó artifact_manifest.json. "
            "Verifica que contenga la celda de integración proporcionada."
        )
        return 1
    print(f"Artefactos exportados. Notebook ejecutado: {output}")
    return 0



def command_visual_demo(args: argparse.Namespace) -> int:
    """Inicia la aplicación Streamlit usando el mismo entorno de Python."""
    root = Path(args.project_root)
    app_path = root / args.app
    if not app_path.exists():
        print(f"No existe la aplicación visual: {app_path}")
        return 1

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
    ]
    if args.headless:
        command.extend(["--server.headless", "true"])

    print(f"Iniciando demo visual: {app_path}")
    print(f"URL local esperada: http://localhost:{args.port}")
    completed = subprocess.run(command, cwd=root, check=False)
    return int(completed.returncode)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fraud-project",
        description="CLI reproducible del proyecto de detección de fraude.",
    )
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="Raíz del repositorio.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="Valida estructura, contenido y artefactos."
    )
    validate.set_defaults(func=command_validate)

    demo = subparsers.add_parser(
        "demo", help="Muestra las transacciones de mayor riesgo."
    )
    demo.add_argument("--top", type=int, default=25)
    demo.set_defaults(func=command_demo)

    export = subparsers.add_parser(
        "export-artifacts",
        help="Ejecuta el notebook completo y exporta objetos entrenados.",
    )
    export.add_argument(
        "--notebook",
        default="notebooks/fraudenotebook_with_exports.ipynb",
    )
    export.add_argument(
        "--output-notebook",
        default="notebooks/fraudenotebook_executed.ipynb",
    )
    export.add_argument(
        "--timeout",
        type=int,
        default=-1,
        help="Timeout por celda; -1 desactiva el límite.",
    )
    export.set_defaults(func=command_export_artifacts)

    reports = subparsers.add_parser(
        "generate-reports",
        help="Consolida tablas, figuras y documentos finales.",
    )
    reports.set_defaults(func=command_generate_reports)

    inspect = subparsers.add_parser(
        "inspect-artifacts",
        help="Verifica hashes y muestra metadatos del modelo.",
    )
    inspect.add_argument(
        "--artifacts-dir",
        default=str(ARTIFACTS_DIR),
    )
    inspect.set_defaults(func=command_inspect_artifacts)

    visual = subparsers.add_parser(
        "visual-demo",
        help="Inicia la aplicación visual de Streamlit.",
    )
    visual.add_argument(
        "--app",
        default="streamlit_app.py",
        help="Ruta relativa de la aplicación Streamlit.",
    )
    visual.add_argument("--port", type=int, default=8501)
    visual.add_argument(
        "--headless",
        action="store_true",
        help="Ejecuta Streamlit sin abrir automáticamente el navegador.",
    )
    visual.set_defaults(func=command_visual_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
