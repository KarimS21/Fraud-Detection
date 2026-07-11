# Informe técnico final

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
| ROC-AUC | 0.7260 | 0.9917 |
| PR-AUC | 0.2077 | 0.9578 |
| Recall@Top 5 % | 0.1300 | 0.4995 |

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
