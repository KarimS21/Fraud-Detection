"""Consolidación de tablas, figuras y documentación final."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2
from typing import Any
import json

import pandas as pd

from .config import PROJECT_ROOT, REPORTS_DIR


TABLE_SOURCES = {
    "model_metrics.csv": "data/week10/offline_evaluation_report_week10.csv",
    "topk_metrics.csv": "data/week10/topk_ranking_report_week10.csv",
    "cluster_validation.csv": "data/week7/validation_table_week7.csv",
    "graph_metrics.csv": "data/week12/ranking_comparison_report_week12.csv",
    "graph_merchant_ranking.csv": "data/week12/graph_vs_model_ranking_week12.csv",
}

FIGURE_SOURCES = {
    "model_performance.png": "data/week10/analisis_rendimiento_completo.png",
    "score_distribution_and_topk.png": (
        "data/week10/distribucion_scores_y_metricas_topk.png"
    ),
    "graph_ranking_comparison.png": (
        "data/week12/comparacion_rankings_week12.png"
    ),
}


def _copy_available(
    project_root: Path,
    destination: Path,
    mapping: dict[str, str],
) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for target_name, source_relative in mapping.items():
        source = project_root / source_relative
        if source.exists():
            target = destination / target_name
            copy2(source, target)
            copied.append(target)
    return copied


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _metric(
    frame: pd.DataFrame,
    system_contains: str,
    column: str,
) -> float | None:
    if frame.empty or column not in frame:
        return None
    mask = frame["sistema"].astype(str).str.contains(
        system_contains, case=False, regex=False
    )
    if not mask.any():
        return None
    return float(frame.loc[mask, column].iloc[0])


def _format_metric(value: float | None) -> str:
    return "No disponible" if value is None else f"{value:.4f}"


def generate_report_bundle(
    *,
    project_root: str | Path = PROJECT_ROOT,
    reports_dir: str | Path = REPORTS_DIR,
) -> dict[str, list[Path] | Path]:
    """Copia evidencia y genera documentación coherente con los resultados."""
    project_root = Path(project_root)
    reports_dir = Path(reports_dir)
    tables_dir = reports_dir / "tables"
    figures_dir = reports_dir / "figures"
    final_dir = reports_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    copied_tables = _copy_available(
        project_root, tables_dir, TABLE_SOURCES
    )
    copied_figures = _copy_available(
        project_root, figures_dir, FIGURE_SOURCES
    )

    evaluation = _read_csv(
        project_root / "data/week10/offline_evaluation_report_week10.csv"
    )
    topk = _read_csv(
        project_root / "data/week10/topk_ranking_report_week10.csv"
    )
    graph_comparison = _read_csv(
        project_root / "data/week12/ranking_comparison_report_week12.csv"
    )

    roc_model = _metric(evaluation, "XGBoost", "roc_auc")
    pr_model = _metric(evaluation, "XGBoost", "pr_auc")
    roc_baseline = _metric(evaluation, "Baseline", "roc_auc")
    pr_baseline = _metric(evaluation, "Baseline", "pr_auc")

    recall5_model = None
    recall5_baseline = None
    if not topk.empty:
        top5 = topk.loc[topk["fraccion_pool"].astype(str).eq("5%")]
        if not top5.empty:
            model_rows = top5[
                top5["sistema"].astype(str).str.contains("XGBoost", case=False)
            ]
            baseline_rows = top5[
                top5["sistema"].astype(str).str.contains("Baseline", case=False)
            ]
            if not model_rows.empty:
                recall5_model = float(model_rows["recall_at_k"].iloc[0])
            if not baseline_rows.empty:
                recall5_baseline = float(
                    baseline_rows["recall_at_k"].iloc[0]
                )

    technical_report = f"""# Informe técnico final

## 1. Problema y objetivo

El proyecto implementa un sistema offline para detectar y priorizar transacciones
con riesgo de fraude. La salida principal no es únicamente una clase binaria:
también se construye un *candidate pool* ordenado para orientar la revisión
operativa hacia las transacciones de mayor riesgo.

## 2. Datos y preparación

Se emplean los archivos `fraudTrain.csv` y `fraudTest.csv` del dataset público de
Kaggle. El pipeline elimina identificadores personales directos, estandariza
fechas y crea variables de edad, distancia cliente-comercio, horario, fin de
semana, periodo del día y monto logarítmico.

La experimentación usa muestras de 75 000 registros de entrenamiento y 21 000
de evaluación, ambas con una prevalencia de fraude cercana al 10 %. Esta
prevalencia fue inducida para la experimentación y no representa directamente
el escenario productivo.

## 3. Pipeline de modelado

1. Imputación y escalado de variables numéricas.
2. One-hot encoding para variables categóricas de baja cardinalidad.
3. Target encoding para variables de alta cardinalidad.
4. TruncatedSVD con un máximo de 50 componentes.
5. Selección y escalado de las primeras 23 componentes.
6. XGBoost optimizado mediante GridSearchCV.
7. Comparación contra un baseline histórico por categoría de comercio.

## 4. Resultados

| Métrica | Baseline histórico | XGBoost optimizado |
|---|---:|---:|
| ROC-AUC | {_format_metric(roc_baseline)} | {_format_metric(roc_model)} |
| PR-AUC | {_format_metric(pr_baseline)} | {_format_metric(pr_model)} |
| Recall@Top 5 % | {_format_metric(recall5_baseline)} | {_format_metric(recall5_model)} |

Los resultados muestran que el modelo fuerte mejora la discriminación global y
la concentración de fraudes dentro de la fracción priorizada.

## 5. Análisis de grafos

Se construye un grafo bipartito dirigido cliente-comercio. Las aristas agregan
frecuencia, monto, fraude observado y score promedio del modelo. PageRank,
grado ponderado y PageRank ajustado por riesgo permiten identificar comercios
estructuralmente relevantes. El ranking del grafo complementa al modelo, pero
no debe interpretarse como evidencia causal de fraude.

## 6. Reproducibilidad

El entorno se fija mediante `.python-version`, `pyproject.toml` y `uv.lock`.
La lógica reutilizable se encuentra en `src/fraud_project`, los objetos
entrenados en `artifacts` y la evidencia consolidada en `reports`.

## 7. Conclusión

El sistema conecta preparación de datos, modelado supervisado, ranking operativo,
análisis de grafos y documentación reproducible. Su uso actual es académico y
offline; antes de una adopción real requiere validación temporal, calibración,
monitoreo y retroalimentación de fraude confirmado.
"""

    model_card = f"""# Model Card — XGBoost optimizado

## Identificación

- **Nombre:** XGBoost optimizado
- **Versión:** 1.0.0
- **Uso previsto:** priorización offline de transacciones para revisión.
- **Salida:** probabilidad estimada de fraude y ranking descendente.
- **Umbral descriptivo:** 0.50.

## Transformaciones requeridas

`preprocessor → TruncatedSVD → primeras 23 componentes → StandardScaler → XGBoost`

## Desempeño observado

- ROC-AUC: **{_format_metric(roc_model)}**
- PR-AUC: **{_format_metric(pr_model)}**
- Recall@Top 5 %: **{_format_metric(recall5_model)}**

## Uso no previsto

No debe emplearse como decisión automática para bloquear transacciones, acusar
a una persona o reemplazar la revisión especializada.

## Riesgos y limitaciones

- La muestra de evaluación tiene una prevalencia de fraude artificialmente alta.
- Las métricas son offline y no reflejan drift ni retraso de etiquetas.
- El target encoding y las categorías nuevas requieren monitoreo.
- Un score alto expresa prioridad de revisión, no certeza de fraude.
"""

    data_card = """# Data Card — Credit Card Transactions Fraud Detection

## Fuente

Dataset público de Kaggle: Credit Card Transactions Fraud Detection.

## Archivos originales

- `data/raw/fraudTrain.csv`
- `data/raw/fraudTest.csv`

Los archivos originales no se versionan por su tamaño. El repositorio incluye
instrucciones para descargarlos y ubicarlos en `data/raw`.

## Variables derivadas principales

- Edad del cliente al momento de la transacción.
- Distancia Haversine entre cliente y comercio.
- Hora, día de semana, mes y periodo del día.
- Indicador de fin de semana.
- Transformaciones cíclicas de hora y mes.
- Logaritmo del monto.

## Privacidad

Se eliminan nombre, apellido y dirección. El número completo de tarjeta se
descarta y solo se conserva un proxy parcial para análisis académico.

## Sesgos y representatividad

El dataset es simulado y no representa necesariamente la distribución,
tipologías de fraude o comportamiento de una institución financiera real.
Además, el muestreo experimental modifica la prevalencia de fraude.
"""

    monitoring_plan = """# Plan de monitoreo y operacionalización

## Supuestos de servicio

El sistema se ejecutaría por lotes y produciría un ranking de transacciones. La
revisión humana ocurre después del scoring y las etiquetas reales pueden llegar
con retraso.

## Controles por ejecución

- Validar esquema, tipos, valores nulos y volumen.
- Registrar versión del modelo y hashes de artefactos.
- Medir tiempo total, errores y cantidad de filas procesadas.
- Guardar distribución del score y proporción sobre umbrales operativos.
- Confirmar la creación del candidate pool y de todos los reportes.

## Drift

- Comparar PSI o KS de variables numéricas y del score frente al periodo base.
- Registrar categorías desconocidas y cambios en su frecuencia.
- Revisar drift semanalmente y rendimiento cuando las etiquetas estén disponibles.

## Alertas iniciales sugeridas

- Error si falta una variable obligatoria.
- Alerta si el volumen se desvía más de 30 % respecto a la media móvil.
- Alerta si PSI ≥ 0.20 en una variable crítica o en el score.
- Alerta si Recall@Top 5 % cae más de 10 puntos porcentuales frente al valor de referencia.
- Alerta si la tasa de categorías desconocidas supera 5 %.

Estos umbrales son supuestos iniciales y deben calibrarse con datos operativos.

## Reentrenamiento y rollback

Evaluar reentrenamiento trimestral o antes si existe drift sostenido o pérdida
de rendimiento. Conservar el modelo anterior y su manifiesto para realizar
rollback. Un nuevo modelo solo se promueve si supera al vigente en PR-AUC y
Recall@K bajo una evaluación temporal comparable.

## Responsables propuestos

- Data Engineer: calidad, esquema y ejecución del pipeline.
- Data Scientist: drift, evaluación y reentrenamiento.
- Analista de fraude: validación de casos y definición de capacidad Top-K.
- Responsable técnico: versionado, despliegue y rollback.
"""

    limitations = """# Limitaciones y trabajo futuro

## Limitaciones

- La evaluación es offline y no incluye comportamiento en producción.
- El dataset es simulado y puede no representar fraude real.
- La muestra de entrenamiento y evaluación fue construida con una prevalencia de
  fraude cercana al 10 %, superior a la distribución original. Por ello,
  precisión, PR-AUC y Lift no deben trasladarse directamente a producción.
- No se realizó validación temporal estricta ni calibración de probabilidades.
- El costo de falsos positivos y falsos negativos no está incorporado en una
  función económica.
- El proxy de cliente usado en grafos puede fusionar o separar entidades de forma
  imperfecta.
- PageRank representa relevancia estructural, no causalidad ni culpabilidad.
- El flujo actual depende todavía del notebook para el entrenamiento completo.

## Trabajo futuro

- Crear un pipeline de entrenamiento ejecutable enteramente desde CLI.
- Aplicar validación temporal y evaluación con prevalencia real.
- Calibrar probabilidades y optimizar umbrales por costo.
- Incorporar SHAP para explicación local y global.
- Automatizar PSI, categorías desconocidas y métricas con etiquetas retrasadas.
- Exponer scoring mediante API o procesamiento batch.
- Implementar dashboard de revisión y retroalimentación.
"""

    documents = {
        "technical_report.md": technical_report,
        "model_card.md": model_card,
        "data_card.md": data_card,
        "monitoring_plan.md": monitoring_plan,
        "limitations_future_work.md": limitations,
    }
    generated = []
    for name, content in documents.items():
        path = final_dir / name
        path.write_text(content.strip() + "\n", encoding="utf-8")
        generated.append(path)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tables_copied": [
            path.relative_to(reports_dir).as_posix() for path in copied_tables
        ],
        "figures_copied": [
            path.relative_to(reports_dir).as_posix() for path in copied_figures
        ],
        "documents_generated": [
            path.relative_to(reports_dir).as_posix() for path in generated
        ],
        "metrics": {
            "roc_auc_baseline": roc_baseline,
            "pr_auc_baseline": pr_baseline,
            "roc_auc_model": roc_model,
            "pr_auc_model": pr_model,
            "recall_top5_baseline": recall5_baseline,
            "recall_top5_model": recall5_model,
        },
        "graph_comparison_rows": len(graph_comparison),
    }
    summary_path = reports_dir / "report_manifest.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "tables": copied_tables,
        "figures": copied_figures,
        "documents": generated,
        "manifest": summary_path,
    }
