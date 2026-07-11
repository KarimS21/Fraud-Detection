"""Demo visual del sistema de priorización de fraude.

Ejecución:
    uv run streamlit run streamlit_app.py

También puede iniciarse mediante:
    uv run python main.py visual-demo
"""

from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path
import json
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fraud_project.config import ARTIFACTS_DIR, TARGET_COLUMN  # noqa: E402
from fraud_project.demo_service import (  # noqa: E402
    VisualDemoService,
    metadata_comparison,
    score_bins,
)
from fraud_project.demo_visuals import (  # noqa: E402
    category_risk_summary,
    compact_metadata,
    risk_distribution,
    top_transaction_columns,
)

st.set_page_config(
    page_title="Fraud Detection Demo",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Cargando modelo y transformadores...")
def load_service(artifacts_path: str) -> VisualDemoService:
    """Carga una sola vez los artefactos compartidos por la aplicación."""
    return VisualDemoService(artifacts_path)


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


def show_disclaimer() -> None:
    st.info(
        "Esta herramienta es una demostración académica. El score prioriza "
        "transacciones para revisión y no debe utilizarse por sí solo para "
        "bloquear operaciones o tomar decisiones financieras definitivas."
    )


def show_risk_result(score: float, level: str, action: str, threshold: float) -> None:
    st.metric("Probabilidad estimada de fraude", f"{score:.2%}")
    st.progress(min(max(float(score), 0.0), 1.0))
    message = (
        f"**Nivel de riesgo: {level}.** {action} "
        f"El umbral operativo configurado es {threshold:.0%}."
    )
    if level == "Crítico":
        st.error(message)
    elif level == "Alto":
        st.warning(message)
    elif level == "Medio":
        st.info(message)
    else:
        st.success(message)


def render_overview(service: VisualDemoService) -> None:
    st.subheader("Resumen del sistema")
    metadata = service.metadata

    cols = st.columns(4)
    cols[0].metric("Modelo", metadata.get("model_name", "XGBoost"))
    cols[1].metric("ROC-AUC", f"{float(metadata.get('roc_auc_model', 0)):.3f}")
    cols[2].metric("PR-AUC", f"{float(metadata.get('pr_auc_model', 0)):.3f}")
    cols[3].metric(
        "Recall@Top 5 %",
        f"{float(metadata.get('recall_top5_model', 0)):.3f}",
    )

    comparison = metadata_comparison(metadata)
    if not comparison.empty:
        st.markdown("#### Baseline histórico frente al modelo final")
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        chart = comparison.set_index("métrica")
        st.bar_chart(chart)

    st.markdown("#### Pipeline reproducible")
    st.code(
        "Entrada → validación → feature engineering → preprocessor → "
        "TruncatedSVD → 23 componentes → StandardScaler → XGBoost → ranking",
        language=None,
    )

    train_rows = int(metadata.get("training_rows", 0) or 0)
    test_rows = int(metadata.get("test_rows", 0) or 0)
    fraud_rate = float(metadata.get("experimental_test_fraud_rate", 0) or 0)
    st.caption(
        f"Evidencia experimental: {train_rows:,} filas de entrenamiento, "
        f"{test_rows:,} filas de evaluación y prevalencia de fraude inducida "
        f"de {fraud_rate:.1%} en la muestra de prueba."
    )
    show_disclaimer()


def render_manual_prediction(service: VisualDemoService) -> None:
    st.subheader("Evaluar una transacción")
    st.write(
        "Complete los datos de una operación. La aplicación calculará edad, "
        "distancia y variables temporales antes de ejecutar el modelo serializado."
    )

    categories = [
        "grocery_pos",
        "gas_transport",
        "shopping_net",
        "shopping_pos",
        "misc_net",
        "misc_pos",
        "entertainment",
        "food_dining",
        "home",
        "kids_pets",
        "health_fitness",
        "personal_care",
        "travel",
    ]

    with st.form("manual_transaction_form"):
        col1, col2, col3 = st.columns(3)
        transaction_date = col1.date_input(
            "Fecha de transacción", value=date(2020, 10, 10)
        )
        transaction_time = col2.time_input(
            "Hora de transacción", value=time(22, 47)
        )
        amount = col3.number_input(
            "Monto", min_value=0.0, value=250.0, step=1.0, format="%.2f"
        )

        col1, col2, col3 = st.columns(3)
        category = col1.selectbox("Categoría", categories, index=0)
        merchant = col2.text_input("Comercio", value="fraud_Demo Merchant")
        gender = col3.selectbox("Género registrado", ["M", "F"])

        col1, col2, col3 = st.columns(3)
        city = col1.text_input("Ciudad", value="Lima")
        state = col2.text_input("Estado / región", value="LI", max_chars=3)
        postal_code = col3.text_input("Código postal", value="15001")

        col1, col2, col3 = st.columns(3)
        city_population = col1.number_input(
            "Población de la ciudad", min_value=1, value=100000, step=1000
        )
        occupation = col2.text_input("Ocupación", value="Data analyst")
        birth_date = col3.date_input(
            "Fecha de nacimiento", value=date(1985, 6, 15)
        )

        st.markdown("##### Ubicación aproximada")
        col1, col2, col3, col4 = st.columns(4)
        client_lat = col1.number_input(
            "Latitud cliente", value=-12.0464, format="%.6f"
        )
        client_lon = col2.number_input(
            "Longitud cliente", value=-77.0428, format="%.6f"
        )
        merchant_lat = col3.number_input(
            "Latitud comercio", value=-12.1000, format="%.6f"
        )
        merchant_lon = col4.number_input(
            "Longitud comercio", value=-77.0300, format="%.6f"
        )

        submitted = st.form_submit_button(
            "Calcular riesgo", type="primary", use_container_width=True
        )

    if not submitted:
        return

    transaction_datetime = datetime.combine(transaction_date, transaction_time)
    input_frame = pd.DataFrame(
        [
            {
                "fecha_hora_transaccion": transaction_datetime,
                "comercio": merchant,
                "categoria_comercio": category,
                "monto": amount,
                "genero": gender,
                "ciudad": city,
                "estado": state.upper(),
                "codigo_postal": postal_code,
                "latitud_cliente": client_lat,
                "longitud_cliente": client_lon,
                "poblacion_ciudad": city_population,
                "ocupacion": occupation,
                "fecha_nacimiento": birth_date,
                "latitud_comercio": merchant_lat,
                "longitud_comercio": merchant_lon,
                "id_transaccion": "manual-demo-001",
            }
        ]
    )

    try:
        with st.spinner("Aplicando el pipeline serializado..."):
            result = service.score_transactions(input_frame)
    except Exception as exc:  # Streamlit debe mostrar un error útil al usuario.
        st.error(f"No se pudo evaluar la transacción: {exc}")
        return

    row = result.scored.iloc[0]
    left, right = st.columns([1, 1.4])
    with left:
        show_risk_result(
            float(row["score_modelo_fuerte"]),
            str(row["nivel_riesgo"]),
            str(row["accion_sugerida"]),
            service.threshold,
        )
    with right:
        st.markdown("#### Datos derivados y salida")
        details = pd.DataFrame(
            {
                "Campo": [
                    "Edad calculada",
                    "Distancia cliente-comercio",
                    "Predicción binaria",
                    "Score baseline",
                    "Ranking en el lote",
                ],
                "Valor": [
                    f"{float(row['edad_cliente']):.0f} años",
                    f"{float(row['distancia_km']):.2f} km",
                    int(row["prediccion_fraude"]),
                    (
                        f"{float(row['score_baseline']):.4f}"
                        if "score_baseline" in row
                        else "No disponible"
                    ),
                    int(row["rank_modelo_fuerte"]),
                ],
            }
        )
        st.dataframe(details, use_container_width=True, hide_index=True)

    show_disclaimer()


def render_batch_scoring(service: VisualDemoService) -> None:
    st.subheader("Evaluación masiva desde CSV")
    st.write(
        "Suba un CSV con el esquema original de Kaggle. Se aceptan columnas "
        "adicionales; la aplicación selecciona únicamente las variables requeridas."
    )

    sample_path = PROJECT_ROOT / "data/demo/sample_transactions_kaggle_format.csv"
    if sample_path.exists():
        st.download_button(
            "Descargar CSV de ejemplo",
            data=sample_path.read_bytes(),
            file_name=sample_path.name,
            mime="text/csv",
        )

    uploaded = st.file_uploader("Archivo CSV", type=["csv"])
    if uploaded is None:
        st.caption(
            "También puede utilizar `data/raw/fraudTest.csv`. Para una demo rápida, "
            "se recomienda cargar una muestra en lugar de las 500 mil filas completas."
        )
        return

    try:
        frame = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"No se pudo leer el CSV: {exc}")
        return

    st.markdown("#### Vista previa")
    st.dataframe(frame.head(10), use_container_width=True, hide_index=True)
    st.caption(f"Filas detectadas: {len(frame):,} · Columnas: {len(frame.columns)}")

    max_rows = 50_000
    if len(frame) > max_rows:
        st.warning(
            f"La demo procesa como máximo {max_rows:,} filas por ejecución. "
            "Se utilizarán las primeras filas del archivo."
        )
        frame = frame.head(max_rows).copy()

    col1, col2 = st.columns([1, 1])
    top_n = col1.slider(
        "Cantidad de transacciones a mostrar", 5, min(100, max(5, len(frame))), 25
    )
    execute = col2.button(
        "Ejecutar scoring masivo", type="primary", use_container_width=True
    )
    if not execute:
        return

    try:
        with st.spinner("Validando, transformando y puntuando transacciones..."):
            result = service.score_transactions(frame)
    except Exception as exc:
        st.error(f"No se pudo procesar el archivo: {exc}")
        st.exception(exc)
        return

    for warning in result.prepared.warnings:
        st.warning(warning)

    summary = result.summary
    cols = st.columns(4)
    cols[0].metric("Filas procesadas", f"{summary['rows']:,}")
    cols[1].metric(
        "Sobre el umbral", f"{summary['predicted_positive_rate']:.1%}"
    )
    cols[2].metric("Score promedio", f"{summary['mean_score']:.3f}")
    cols[3].metric("Riesgo crítico", f"{summary['critical_count']:,}")

    if result.evaluation:
        st.markdown("#### Evaluación con etiquetas incluidas en el CSV")
        metric_cols = st.columns(len(result.evaluation))
        labels = {
            "roc_auc": "ROC-AUC",
            "pr_auc": "PR-AUC",
            "observed_fraud_rate": "Fraude observado",
            "recall_top5": "Recall@Top 5 %",
        }
        for container, (key, value) in zip(metric_cols, result.evaluation.items()):
            container.metric(labels.get(key, key), f"{value:.3f}")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Distribución de scores")
        distribution = score_bins(result.scored["score_modelo_fuerte"])
        st.bar_chart(distribution.set_index("rango_score"))
    with right:
        st.markdown("#### Distribución por nivel de riesgo")
        levels = risk_distribution(result.scored)
        st.bar_chart(levels.set_index("nivel_riesgo"))

    category_summary = category_risk_summary(result.scored)
    if not category_summary.empty:
        st.markdown("#### Categorías con mayor score promedio")
        st.dataframe(
            category_summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "score_promedio": st.column_config.NumberColumn(format="%.4f"),
                "score_maximo": st.column_config.NumberColumn(format="%.4f"),
            },
        )

    st.markdown(f"#### Top {top_n} transacciones priorizadas")
    top = result.scored.head(top_n)
    display_columns = top_transaction_columns(top)
    st.dataframe(
        top[display_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "score_modelo_fuerte": st.column_config.ProgressColumn(
                "Score", min_value=0.0, max_value=1.0, format="%.4f"
            ),
            "probabilidad_fraude_pct": st.column_config.NumberColumn(
                "Probabilidad (%)", format="%.2f"
            ),
            "monto": st.column_config.NumberColumn(format="%.2f"),
            "distancia_km": st.column_config.NumberColumn(format="%.2f km"),
        },
    )

    st.download_button(
        "Descargar resultados con scores",
        data=to_csv_bytes(result.scored),
        file_name="fraud_scoring_results.csv",
        mime="text/csv",
        type="primary",
    )
    show_disclaimer()


def render_artifacts(service: VisualDemoService) -> None:
    st.subheader("Artefactos y operación")
    status = service.artifact_status()
    all_valid = (
        not status.empty
        and status["exists"].all()
        and status["sha256_matches"].all()
    )
    if all_valid:
        st.success("Todos los artefactos existen y sus hashes SHA-256 coinciden.")
    else:
        st.error("Uno o más artefactos están ausentes o no superaron la validación.")

    st.dataframe(status, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Metadatos del modelo")
        st.json(compact_metadata(service.metadata), expanded=True)
    with right:
        st.markdown("#### Esquema de entrada")
        schema_path = ARTIFACTS_DIR / "metadata/input_schema.json"
        if schema_path.exists():
            st.json(json.loads(schema_path.read_text(encoding="utf-8")))
        else:
            st.warning("No se encontró input_schema.json.")

    st.markdown("#### Consideraciones operativas")
    st.markdown(
        """
- Conservar la versión anterior del modelo para rollback.
- Verificar drift de volumen, variables y distribución de scores.
- Evaluar Recall@Top-K cuando existan etiquetas confirmadas.
- No interpretar el score como causalidad ni como una decisión automática.
- Registrar versión, fecha, hash y cantidad de filas de cada ejecución.
"""
    )


def main() -> None:
    st.title("💳 Fraud Detection — Demo visual")
    st.caption(
        "Priorización de transacciones mediante XGBoost, reducción dimensional "
        "y artefactos reproducibles."
    )

    try:
        service = load_service(str(ARTIFACTS_DIR))
    except Exception as exc:
        st.error(
            "No fue posible cargar los artefactos. Ejecute primero la exportación "
            "del notebook y verifique las versiones del entorno."
        )
        st.code(str(exc))
        st.stop()

    with st.sidebar:
        st.header("Estado del modelo")
        st.write(f"**Versión:** {service.metadata.get('model_version', 'N/D')}")
        st.write(f"**Estado:** {service.metadata.get('artifact_status', 'N/D')}")
        st.write(f"**Umbral:** {service.threshold:.0%}")
        st.write(
            f"**Componentes:** {service.metadata.get('model_components', 'N/D')}"
        )
        st.divider()
        st.caption("Proyecto académico — UPC · Big Data")

    tabs = st.tabs(
        [
            "📊 Resumen",
            "🔎 Transacción individual",
            "📁 Scoring masivo",
            "🧩 Artefactos",
        ]
    )
    with tabs[0]:
        render_overview(service)
    with tabs[1]:
        render_manual_prediction(service)
    with tabs[2]:
        render_batch_scoring(service)
    with tabs[3]:
        render_artifacts(service)


if __name__ == "__main__":
    main()
