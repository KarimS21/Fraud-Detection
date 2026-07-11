# Guía de la demo visual — Semana 14

## Objetivo

Demostrar que los artefactos serializados pueden recibir transacciones nuevas,
reconstruir variables, ejecutar el modelo XGBoost y generar un ranking de riesgo
sin depender de las variables vivas del notebook.

## Requisitos previos

Deben existir y superar la validación:

```text
artifacts/models/fraud_xgboost_model.json
artifacts/models/preprocessor.joblib
artifacts/models/svd_model.joblib
artifacts/models/scaler_cluster.joblib
artifacts/metadata/model_metadata.json
artifacts/metadata/artifact_manifest.json
```

## Instalación

```powershell
uv lock
uv sync
```

La configuración compatible con los artefactos utiliza Python 3.12,
`scikit-learn==1.5.2` y `category-encoders==2.7.0`.

## Ejecución

Opción directa:

```powershell
uv run streamlit run streamlit_app.py
```

Opción mediante la CLI:

```powershell
uv run python main.py visual-demo
```

## Flujo de demostración sugerido

1. Abrir **Resumen** y explicar la comparación baseline versus XGBoost.
2. Abrir **Transacción individual**, completar el formulario y calcular riesgo.
3. Abrir **Scoring masivo** y cargar
   `data/demo/sample_transactions_kaggle_format.csv` o una muestra de
   `data/raw/fraudTest.csv`.
4. Mostrar distribución de scores, niveles de riesgo y Top-K.
5. Descargar `fraud_scoring_results.csv`.
6. Abrir **Artefactos** y demostrar que los hashes SHA-256 coinciden.

## Entrada admitida

La carga masiva acepta el esquema original de Kaggle. Se requieren, como mínimo:

```text
trans_date_trans_time, merchant, category, amt, gender, city, state, zip,
lat, long, city_pop, job, dob, merch_lat, merch_long
```

Las columnas adicionales son permitidas. `is_fraud` es opcional; cuando está
presente, la interfaz calcula métricas offline del lote.

## Limitaciones

- Uso académico y offline.
- El score prioriza revisión; no constituye una decisión automática.
- La muestra de evaluación del proyecto tiene prevalencia de fraude inducida.
- No hay calibración productiva ni validación temporal en un entorno real.
