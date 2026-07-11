# Informe Técnico Final - Sistema de Detección y Priorización de Fraude

## 1. Objetivo del proyecto
El objetivo del proyecto es construir un sistema analítico para detección y priorización de riesgo de fraude en transacciones financieras. La solución integra limpieza de datos, ingeniería de variables, reducción de dimensionalidad, clustering, modelado supervisado, evaluación offline, ranking de transacciones y análisis de grafos cliente-comercio.

## 2. Coherencia de la solución
El proyecto mantiene coherencia porque cada semana construye una capa del sistema final. Primero se preparan los datos y variables relevantes. Luego se reduce la dimensionalidad para trabajar con representaciones más manejables. Después se exploran patrones no supervisados mediante clustering. Posteriormente se entrena y evalúa un modelo fuerte para asignar scores de riesgo. Finalmente, se complementa el análisis predictivo con grafos para entender relaciones entre clientes y comercios.

## 3. Datos y artefactos procesados
La entrega final no se basa únicamente en diapositivas. Se generó un inventario de artefactos procesados en `processed_artifacts/final_processed_artifacts_inventory_week14.csv`. En esta ejecución se identificaron 21 artefactos disponibles de 21 artefactos esperados. Los principales resultados incluyen matrices de features, tablas de evaluación, candidate pool rankeado, reportes de clustering, reportes de centralidad y archivos exportables del grafo.

## 4. Modelo predictivo y evaluación offline
El sistema fuerte utilizado es: **XGBoost optimizado**. Este modelo se compara contra un baseline histórico. Las métricas principales disponibles son:

| Sistema | ROC-AUC | PR-AUC | Recall@Top 5% |
|---|---:|---:|---:|
| Baseline histórico | 0.725953514739229 | 0.2076501631928543 | 0.13 |
| Modelo fuerte | 0.9916765180146132 | 0.9577833515942488 | 0.49952380952380954 |

En un problema de fraude, la evaluación offline no debe depender únicamente de accuracy, porque la clase positiva suele ser minoritaria. Por ello se consideran métricas como PR-AUC, Recall@K y Lift@K, ya que permiten evaluar si el sistema prioriza correctamente los casos más relevantes para revisión.

## 5. Análisis de grafos
Además del score predictivo, se construyó un grafo dirigido entre clientes anonimizados y comercios. El grafo permite analizar la estructura de interacción, detectar comercios centrales y comparar popularidad, centralidad y riesgo estimado. En la ejecución actual se registran 1616 nodos y 20481 aristas, siempre que las celdas de Semana 12 hayan sido ejecutadas previamente.

## 6. Reproducibilidad
La reproducibilidad se garantiza mediante el notebook, las carpetas de salida por semana y el runbook final. El proyecto debe ejecutarse siguiendo el orden lógico de las semanas: preparación de datos, features, clustering, modelo, evaluación, grafos y entrega final.

## 7. Resultado final
El resultado final es un sistema integrado que permite transformar datos transaccionales en artefactos de análisis, métricas de evaluación, rankings de riesgo y evidencia de demo. Esto permite defender la solución desde una perspectiva técnica, funcional y operativa.
