# Plan de Monitoreo y Operacionalización

## 1. Objetivo
El monitoreo busca controlar que el sistema mantenga un comportamiento estable al priorizar transacciones con riesgo de fraude.

## 2. Variables a monitorear
- Volumen diario de transacciones procesadas.
- Porcentaje de transacciones con score alto de fraude.
- Distribución del score del modelo.
- Tasa real de fraude confirmada, si se cuenta con retroalimentación posterior.
- Métricas de ranking como Precision@K, Recall@K y Lift@K.
- Tiempo de ejecución del pipeline.
- Errores de carga, columnas faltantes o cambios de esquema.

## 3. Alertas sugeridas
- Alerta si el volumen de transacciones cambia de forma abrupta.
- Alerta si la distribución de scores se desplaza significativamente.
- Alerta si aparecen categorías nuevas no vistas durante entrenamiento.
- Alerta si la tasa de fraude detectada cae o sube de forma inesperada.
- Alerta si el pipeline no genera los archivos finales esperados.

## 4. Operacionalización
En un entorno productivo, el sistema podría ejecutarse de forma programada. Cada ejecución debe guardar logs, métricas, versión del modelo, fecha de procesamiento y archivos de salida. El modelo debe recalibrarse o reentrenarse cuando exista evidencia de drift, pérdida de rendimiento o cambio en el comportamiento transaccional.
