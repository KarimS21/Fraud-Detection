# Limitaciones y trabajo futuro

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
