# Reportes consolidados

`uv run python main.py generate-reports` copia las tablas y figuras finales
desde `data/week7`, `data/week10` y `data/week12`, y genera:

- `reports/final/technical_report.md`
- `reports/final/model_card.md`
- `reports/final/data_card.md`
- `reports/final/monitoring_plan.md`
- `reports/final/limitations_future_work.md`
- `reports/report_manifest.json`

Los archivos de `reports` son evidencia consolidada. Los resultados originales
por semana se conservan en `data/weekX`.
