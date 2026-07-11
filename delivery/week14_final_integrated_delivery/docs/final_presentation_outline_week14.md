# Presentación Final - Guion de Diapositivas

## Diapositiva 1: Título
Sistema de detección y priorización de fraude mediante machine learning, ranking y análisis de grafos.

## Diapositiva 2: Problema
El fraude transaccional requiere priorizar operaciones sospechosas de forma eficiente, especialmente cuando no es posible revisar manualmente todo el volumen de transacciones.

## Diapositiva 3: Objetivo
Construir un sistema reproducible que procese datos, genere variables, entrene un modelo, evalúe su rendimiento y produzca rankings de riesgo.

## Diapositiva 4: Pipeline general
Carga de datos, limpieza, feature engineering, reducción de dimensionalidad, clustering, modelo fuerte, evaluación offline, grafos y entrega final.

## Diapositiva 5: Feature engineering
Variables temporales, monto transformado, distancia cliente-comercio, edad aproximada, variables categóricas y representación final para modelado.

## Diapositiva 6: Modelo y baseline
Comparación entre baseline histórico y modelo fuerte. Explicar por qué se usan PR-AUC, Recall@K y Lift@K en lugar de depender solo de accuracy.

## Diapositiva 7: Resultados de evaluación
Mostrar tabla de métricas y candidate pool rankeado. Explicar qué transacciones quedan arriba y cómo se usarían para revisión.

## Diapositiva 8: Análisis de grafos
Explicar nodos, aristas, pesos, centralidad y comparación entre ranking estructural, popularidad y modelo predictivo.

## Diapositiva 9: Demo final
Mostrar los archivos generados en Semana 14: inventario, resumen, demo top 25, runbook, reporte final y plan de monitoreo.

## Diapositiva 10: Limitaciones y futuro
Explicar limitaciones de datos, evaluación offline, desbalance de clases y mejoras futuras como API, dashboard, monitoreo y reentrenamiento.

## Diapositiva 11: Cierre
Defender que el sistema es coherente porque conecta problema, datos, modelo, evaluación, grafos, evidencia procesada y documentación reproducible.
