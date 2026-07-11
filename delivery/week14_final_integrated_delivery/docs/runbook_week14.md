# Runbook - Ejecución Reproducible del Proyecto

## 1. Requisitos previos
- Python 3.10 o superior.
- Jupyter Notebook o JupyterLab.
- Librerías principales: pandas, numpy, scikit-learn, matplotlib, scipy, networkx, xgboost y sweetviz.
- Archivos base `fraudTrain.csv` y `fraudTest.csv` ubicados en la ruta esperada por el notebook.

## 2. Estructura esperada del proyecto
```text
project_root/
├── data/
│   ├── interim/
│   ├── week5/
│   ├── week7/
│   ├── week10/
│   └── week12/
├── delivery/
│   └── week14_final_integrated_delivery/
├── notebooks/
│   └── fraudenotebook.ipynb
└── README.md
```

## 3. Orden de ejecución
1. Ejecutar las celdas iniciales de carga de datos.
2. Ejecutar limpieza, tipado y feature engineering.
3. Ejecutar Semana 5 para generar matriz de features y reducción de dimensionalidad.
4. Ejecutar Semana 7 para clustering y validación.
5. Ejecutar Semana 10 para modelo fuerte, baseline, evaluación offline y ranking.
6. Ejecutar Semana 12 para grafo cliente-comercio y centralidad.
7. Ejecutar Semana 14 para consolidar la entrega final.

## 4. Validación de salida
Al finalizar, revisar la carpeta:

```text
delivery/week14_final_integrated_delivery/
```

Debe contener documentos finales, inventario de artefactos, demo de transacciones priorizadas, resumen de ejecución y checklist de defensa.

## 5. Problemas comunes
- Si falta `candidate_pool`, ejecutar primero Semana 10.
- Si falta `merchant_ranking`, ejecutar primero Semana 12.
- Si el inventario muestra artefactos pendientes, revisar qué celda anterior no fue ejecutada.
- Si faltan librerías, instalarlas en el entorno antes de volver a ejecutar.
