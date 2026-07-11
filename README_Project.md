# Credit Card Fraud Detection

Proyecto reproducible de detección y priorización de fraude transaccional mediante
feature engineering, reducción dimensional, XGBoost, ranking Top-K y análisis de
grafos cliente-comercio.

## Resultados finales

| Métrica | Baseline histórico | XGBoost optimizado |
|---|---:|---:|
| ROC-AUC | 0.726 | 0.992 |
| PR-AUC | 0.208 | 0.958 |
| Recall@Top 5 % | 0.130 | 0.500 |

La muestra experimental usa 75 000 filas de entrenamiento y 21 000 de evaluación,
con una prevalencia de fraude cercana al 10 %. Esta prevalencia fue inducida y
debe considerarse al interpretar métricas sensibles a la distribución.

## Estructura

```text
Fraud-Project/
├── artifacts/        # modelo, transformadores, baseline y manifiesto
├── data/
│   ├── raw/          # CSV originales, no versionados
│   ├── interim/      # datasets intermedios grandes, no versionados
│   ├── week5/
│   ├── week7/
│   ├── week10/
│   └── week12/
├── delivery/
├── notebooks/
├── reports/          # evidencia consolidada
├── src/fraud_project/
├── main.py
├── pyproject.toml
└── uv.lock
```

## Instalación

Requisitos: Python definido en `.python-version`, `uv` y Git LFS.

```bash
git lfs install
uv sync
```

Los datasets originales deben descargarse desde Kaggle y colocarse en:

```text
data/raw/fraudTrain.csv
data/raw/fraudTest.csv
```

Fuente: Credit Card Transactions Fraud Detection Dataset, de Kartik2112.

## Ejecución

```bash
uv run python main.py validate
uv run python main.py demo --top 25
uv run python main.py generate-reports
uv run python main.py inspect-artifacts
```

Para ejecutar el notebook completo y exportar el modelo:

```bash
uv run python main.py export-artifacts \
  --notebook notebooks/fraudenotebook_with_exports.ipynb
```

La ejecución puede tardar debido a GridSearchCV.

## Pipeline oficial

```text
variables limpias
→ imputación y encoding
→ TruncatedSVD
→ primeras 23 componentes
→ StandardScaler
→ best_xgb
→ score y ranking
```

El modelo oficial es `best_xgb` sin `cluster_label`, porque la prueba con clusters
no produjo una mejora suficientemente significativa.

## Artefactos

La celda final del notebook genera:

- `artifacts/models/fraud_xgboost_model.json`
- `artifacts/models/preprocessor.joblib`
- `artifacts/models/svd_model.joblib`
- `artifacts/models/scaler_cluster.joblib`
- `artifacts/baselines/historical_risk_table.csv`
- `artifacts/metadata/model_metadata.json`
- `artifacts/metadata/input_schema.json`
- `artifacts/metadata/artifact_manifest.json`

## Limitaciones

El sistema es académico y offline. No debe emplearse para bloquear operaciones
automáticamente. Requiere validación temporal, calibración, monitoreo de drift y
retroalimentación con fraude confirmado antes de un uso real.

## Demo visual

La aplicación visual carga el modelo y los transformadores serializados una sola
vez, acepta una transacción manual o un CSV en formato original de Kaggle y
genera scores y rankings reales.

```powershell
uv lock
uv sync
uv run streamlit run streamlit_app.py
```

También puede iniciarse desde la CLI:

```powershell
uv run python main.py visual-demo
```

La interfaz incluye:

- Comparación de métricas baseline versus XGBoost.
- Formulario para una transacción individual.
- Scoring masivo y ranking Top-K desde CSV.
- Descarga de resultados con probabilidades.
- Validación visual de artefactos y hashes SHA-256.

Archivo de ejemplo:

```text
data/demo/sample_transactions_kaggle_format.csv
```

La guía de exposición se encuentra en
`delivery/week14_final_integrated_delivery/docs/visual_demo_guide_week14.md`.
