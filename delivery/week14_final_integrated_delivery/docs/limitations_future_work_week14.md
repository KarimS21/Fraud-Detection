# Limitaciones y Trabajo Futuro

## Limitaciones
- El análisis depende de la calidad y disponibilidad de los datos transaccionales.
- El modelo se evalúa de forma offline; no reemplaza una validación productiva con fraude confirmado posteriormente.
- La clase fraude suele estar desbalanceada, por lo que las métricas deben interpretarse con cuidado.
- El grafo cliente-comercio usa identificadores anonimizados o proxy, lo que puede limitar la trazabilidad fina.
- El baseline histórico puede verse afectado por categorías nuevas o con pocos registros.
- La notebook debe ejecutarse en orden para asegurar que todas las variables intermedias existan.

## Trabajo futuro
- Implementar un pipeline automatizado con scripts independientes del notebook.
- Guardar el modelo entrenado y versionarlo con fecha, métricas y parámetros.
- Incorporar monitoreo de drift de datos y drift de performance.
- Desplegar una API para consultar el score de riesgo de nuevas transacciones.
- Crear un dashboard operativo para revisar transacciones priorizadas.
- Mejorar la explicación del modelo con SHAP u otras técnicas de interpretabilidad.
- Validar el sistema con retroalimentación real de casos confirmados de fraude.
