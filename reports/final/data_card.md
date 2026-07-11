# Data Card — Credit Card Transactions Fraud Detection

## Fuente

Dataset público de Kaggle: Credit Card Transactions Fraud Detection.

## Archivos originales

- `data/raw/fraudTrain.csv`
- `data/raw/fraudTest.csv`

Los archivos originales no se versionan por su tamaño. El repositorio incluye
instrucciones para descargarlos y ubicarlos en `data/raw`.

## Variables derivadas principales

- Edad del cliente al momento de la transacción.
- Distancia Haversine entre cliente y comercio.
- Hora, día de semana, mes y periodo del día.
- Indicador de fin de semana.
- Transformaciones cíclicas de hora y mes.
- Logaritmo del monto.

## Privacidad

Se eliminan nombre, apellido y dirección. El número completo de tarjeta se
descarta y solo se conserva un proxy parcial para análisis académico.

## Sesgos y representatividad

El dataset es simulado y no representa necesariamente la distribución,
tipologías de fraude o comportamiento de una institución financiera real.
Además, el muestreo experimental modifica la prevalencia de fraude.
