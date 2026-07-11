# Integración del notebook con `src`, `artifacts` y `reports`

La copia `fraudenotebook_with_exports.ipynb` incluye dos celdas al final. No
reemplazan el entrenamiento existente; consumen las variables ya creadas.

## Variables requeridas

- `best_xgb`
- `preprocessor`
- `svd_model`
- `scaler_cluster`
- `risk_table`
- `evaluation_report`
- `topk_report`
- `df_model`
- `df_model_test`
- `candidate_pool`

## Exportación manual alternativa

```python
from fraud_project.artifact_manager import save_training_artifacts

best_row = evaluation_report[
    evaluation_report["sistema"] == "XGBoost optimizado"
].iloc[0]

top5_model = topk_report[
    (topk_report["sistema"] == "XGBoost optimizado")
    & (topk_report["fraccion_pool"] == "5%")
].iloc[0]

save_training_artifacts(
    model=best_xgb,
    preprocessor=preprocessor,
    svd_model=svd_model,
    scaler=scaler_cluster,
    risk_table=risk_table,
    metadata={
        "training_rows": len(df_model),
        "test_rows": len(df_model_test),
        "experimental_train_fraud_rate": float(df_model["es_fraude"].mean()),
        "experimental_test_fraud_rate": float(df_model_test["es_fraude"].mean()),
        "roc_auc_model": float(best_row["roc_auc"]),
        "pr_auc_model": float(best_row["pr_auc"]),
        "recall_top5_model": float(top5_model["recall_at_k"]),
        "candidate_pool_rows": len(candidate_pool),
        "baseline_group_column": baseline_group_col,
        "best_params": best_xgb.get_params(),
    },
)
```
