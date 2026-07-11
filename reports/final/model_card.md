# Model Card — XGBoost optimizado

## Identificación

- **Nombre:** XGBoost optimizado
- **Versión:** 1.0.0
- **Uso previsto:** priorización offline de transacciones para revisión.
- **Salida:** probabilidad estimada de fraude y ranking descendente.
- **Umbral descriptivo:** 0.50.

## Transformaciones requeridas

`preprocessor → TruncatedSVD → primeras 23 componentes → StandardScaler → XGBoost`

## Desempeño observado

- ROC-AUC: **0.9917**
- PR-AUC: **0.9578**
- Recall@Top 5 %: **0.4995**

## Uso no previsto

No debe emplearse como decisión automática para bloquear transacciones, acusar
a una persona o reemplazar la revisión especializada.

## Riesgos y limitaciones

- La muestra de evaluación tiene una prevalencia de fraude artificialmente alta.
- Las métricas son offline y no reflejan drift ni retraso de etiquetas.
- El target encoding y las categorías nuevas requieren monitoreo.
- Un score alto expresa prioridad de revisión, no certeza de fraude.
