# Plan de monitoreo y operacionalización

## Supuestos de servicio

El sistema se ejecutaría por lotes y produciría un ranking de transacciones. La
revisión humana ocurre después del scoring y las etiquetas reales pueden llegar
con retraso.

## Controles por ejecución

- Validar esquema, tipos, valores nulos y volumen.
- Registrar versión del modelo y hashes de artefactos.
- Medir tiempo total, errores y cantidad de filas procesadas.
- Guardar distribución del score y proporción sobre umbrales operativos.
- Confirmar la creación del candidate pool y de todos los reportes.

## Drift

- Comparar PSI o KS de variables numéricas y del score frente al periodo base.
- Registrar categorías desconocidas y cambios en su frecuencia.
- Revisar drift semanalmente y rendimiento cuando las etiquetas estén disponibles.

## Alertas iniciales sugeridas

- Error si falta una variable obligatoria.
- Alerta si el volumen se desvía más de 30 % respecto a la media móvil.
- Alerta si PSI ≥ 0.20 en una variable crítica o en el score.
- Alerta si Recall@Top 5 % cae más de 10 puntos porcentuales frente al valor de referencia.
- Alerta si la tasa de categorías desconocidas supera 5 %.

Estos umbrales son supuestos iniciales y deben calibrarse con datos operativos.

## Reentrenamiento y rollback

Evaluar reentrenamiento trimestral o antes si existe drift sostenido o pérdida
de rendimiento. Conservar el modelo anterior y su manifiesto para realizar
rollback. Un nuevo modelo solo se promueve si supera al vigente en PR-AUC y
Recall@K bajo una evaluación temporal comparable.

## Responsables propuestos

- Data Engineer: calidad, esquema y ejecución del pipeline.
- Data Scientist: drift, evaluación y reentrenamiento.
- Analista de fraude: validación de casos y definición de capacidad Top-K.
- Responsable técnico: versionado, despliegue y rollback.
