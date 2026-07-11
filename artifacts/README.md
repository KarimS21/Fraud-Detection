# Artefactos entrenados

Esta carpeta almacena los objetos necesarios para reproducir el scoring sin
volver a entrenar el modelo:

```text
artifacts/
├── models/
│   ├── fraud_xgboost_model.json
│   ├── preprocessor.joblib
│   ├── svd_model.joblib
│   └── scaler_cluster.joblib
├── baselines/
│   └── historical_risk_table.csv
└── metadata/
    ├── model_metadata.json
    ├── input_schema.json
    └── artifact_manifest.json
```

Los archivos reales se generan desde el notebook con
`save_training_artifacts(...)`. El manifiesto incluye tamaño y SHA-256 para
detectar archivos faltantes o modificados.

Los modelos y matrices se versionan con Git LFS:

```bash
git lfs install
git lfs track "*.npz" "*.npy" "artifacts/models/*.joblib" \
  "artifacts/models/*.json" "*.gexf"
git add .gitattributes
```
