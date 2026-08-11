"""Dashboard de operación del pipeline de churn.

    streamlit run dashboard/app.py

Está pensado para una demo de 2 minutos: cuatro secciones que van del contexto
(qué dicen los datos) al estado actual (qué está pasando ahora mismo con el
modelo en producción).

Todo lo que muestra sale del pipeline real — el histórico de drift que generó
Evidently, el Model Registry de MLflow y la API en marcha. No hay ni un dato
inventado; si algo no está disponible, lo dice en vez de rellenarlo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    API_URL,
    DRIFT_HISTORY_PATH,
    FIGURES_DIR,
    REPORTS_DIR,
)

st.set_page_config(
    page_title="Churn · panel de operación",
    page_icon="📉",
    layout="wide",
)

# Paleta compartida con las figuras del EDA, para que el dashboard y el
# repositorio se vean como un mismo sistema.
BLUE = "#2a78d6"
INK_MUTED = "#898781"
SEMAFORO = {
    "OK": ("#0ca30c", "Estable"),
    "VIGILAR": ("#fab219", "Vigilar"),
    "CRITICO": ("#d03b3b", "Requiere reentrenamiento"),
}


# --------------------------------------------------------------------------
# Carga de datos (con caché: el dashboard no debe golpear MLflow en cada clic)
# --------------------------------------------------------------------------


@st.cache_data(ttl=60)
def load_drift_history() -> dict | None:
    if not DRIFT_HISTORY_PATH.exists():
        return None
    return json.loads(DRIFT_HISTORY_PATH.read_text(encoding="utf-8"))


@st.cache_data(ttl=30)
def fetch_api(path: str) -> tuple[int, dict | None]:
    try:
        response = requests.get(f"{API_URL}{path}", timeout=5)
        return response.status_code, response.json()
    except requests.RequestException:
        return 0, None


def post_api(path: str, payload: dict) -> tuple[int, dict | None]:
    try:
        response = requests.post(f"{API_URL}{path}", json=payload, timeout=10)
        return response.status_code, response.json()
    except requests.RequestException:
        return 0, None


def figure_path(name: str) -> Path | None:
    ruta = FIGURES_DIR / f"{name}.png"
    return ruta if ruta.exists() else None


# --------------------------------------------------------------------------
# Cabecera: el estado del sistema en una línea
# --------------------------------------------------------------------------

historico = load_drift_history()
codigo_salud, salud = fetch_api("/health")

st.title("Predicción de churn · panel de operación")

if historico is None:
    st.error(
        "No hay histórico de drift. Genera los informes con `python monitoring/generate_report.py`."
    )
    st.stop()

meses = pd.DataFrame(historico["history"])
ultimo = meses.iloc[-1]
color, etiqueta = SEMAFORO.get(ultimo.status, ("#898781", ultimo.status))

col_estado, col_modelo, col_auc, col_api = st.columns([1.4, 1, 1, 1])

with col_estado:
    st.markdown(
        f"""
        <div style="border-left:5px solid {color};padding:.35rem 0 .35rem .9rem">
          <div style="color:{INK_MUTED};font-size:.8rem;text-transform:uppercase;
                      letter-spacing:.05em">Estado del modelo</div>
          <div style="color:{color};font-size:1.45rem;font-weight:700">{etiqueta}</div>
          <div style="color:{INK_MUTED};font-size:.85rem">último periodo: {ultimo.month}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

col_modelo.metric(
    "Modelo en producción",
    f"v{historico['model']['version']}",
    historico["model"]["algorithm"],
    delta_color="off",
)
col_auc.metric(
    "AUC del último mes",
    f"{ultimo.roc_auc:.4f}",
    f"{ultimo.auc_delta:+.4f} vs. referencia",
    delta_color="normal",
)
col_api.metric(
    "API",
    "en línea" if codigo_salud == 200 else ("degradada" if codigo_salud else "sin respuesta"),
    f"HTTP {codigo_salud}" if codigo_salud else "no alcanzable",
    delta_color="off",
)

if ultimo.status == "CRITICO":
    st.error(
        f"**El modelo se ha degradado.** El AUC ha caído {abs(ultimo.auc_delta):.4f} puntos "
        f"respecto al periodo de referencia ({historico['reference_roc_auc']:.4f}). "
        "La relación entre las variables y el churn ha cambiado: hay que reentrenar. "
        "Un modelo con drift de datos puede seguir siendo válido; este ya no lo es."
    )
elif ultimo.status == "VIGILAR":
    st.warning(
        "**Drift de datos por encima del umbral, pero el modelo aguanta.** "
        "La población de clientes ha cambiado y el modelo sigue prediciendo bien. "
        "Reentrenar ahora sería gastar sin motivo: toca vigilar."
    )

st.divider()

tab_drift, tab_modelo, tab_predicciones, tab_eda = st.tabs(
    ["🚨 Drift", "🧪 Salud del modelo", "🔮 Predicciones", "📊 EDA"]
)


# --------------------------------------------------------------------------
# Pestaña 1 · Drift
# --------------------------------------------------------------------------
with tab_drift:
    st.subheader("Las dos señales, mes a mes")
    st.caption(
        "Arriba, la proporción de columnas cuya distribución ha cambiado respecto al periodo "
        "de entrenamiento: está disponible en tiempo real, sin necesidad de etiquetas. "
        "Abajo, el AUC real: es la verdad, pero en producción llegaría con semanas de retraso, "
        "porque saber si un cliente se ha ido lleva tiempo. Por eso hacen falta las dos."
    )

    umbral_drift = historico["thresholds"]["drift_share"]

    fig_drift = go.Figure()
    fig_drift.add_bar(
        x=meses.month,
        y=meses.drifted_columns_share,
        marker_color=[SEMAFORO.get(s, ("#898781",))[0] for s in meses.status],
        name="Columnas con drift",
        hovertemplate="<b>%{x}</b><br>%{y:.0%} de las columnas<extra></extra>",
    )
    fig_drift.add_hline(
        y=umbral_drift,
        line_dash="dash",
        line_color=INK_MUTED,
        annotation_text=f"umbral {umbral_drift:.0%}",
        annotation_position="top right",
    )
    fig_drift.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="Columnas con drift",
        yaxis_tickformat=".0%",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_drift, use_container_width=True)

    fig_auc = go.Figure()
    fig_auc.add_scatter(
        x=meses.month,
        y=meses.roc_auc,
        mode="lines+markers",
        line=dict(color=BLUE, width=2),
        marker=dict(size=9),
        name="AUC",
        hovertemplate="<b>%{x}</b><br>AUC %{y:.4f}<extra></extra>",
    )
    fig_auc.add_hline(
        y=historico["reference_roc_auc"],
        line_dash="dash",
        line_color=INK_MUTED,
        annotation_text="referencia",
        annotation_position="top right",
    )
    fig_auc.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="AUC-ROC",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_auc, use_container_width=True)

    primera_alerta = meses.loc[meses.status == "VIGILAR", "month"]
    primera_caida = meses.loc[meses.status == "CRITICO", "month"]
    if len(primera_alerta) and len(primera_caida):
        st.info(
            f"El drift de datos superó el umbral en **{primera_alerta.iloc[0]}** y el modelo "
            f"no se degradó hasta **{primera_caida.iloc[0]}**. Ese desfase es el motivo de "
            "que reentrenar automáticamente ante cualquier alerta de drift sea mala idea."
        )

    st.subheader("Detalle por mes")
    mes_elegido = st.selectbox("Periodo", meses.month.tolist(), index=len(meses) - 1)
    fila = meses[meses.month == mes_elegido].iloc[0]

    izq, der = st.columns([1, 2])
    with izq:
        st.metric("Columnas con drift", f"{fila.drifted_columns_count} de 14")
        st.metric("Drift de predicciones", "sí" if fila.prediction_drift_detected else "no")
        st.metric("AUC", f"{fila.roc_auc:.4f}", f"{fila.auc_delta:+.4f}")
    with der:
        st.write("**Columnas cuya distribución ha cambiado:**")
        if fila.drifted_columns:
            st.write(" · ".join(f"`{c}`" for c in fila.drifted_columns))
        else:
            st.write("_ninguna_")

    informe = REPORTS_DIR / f"drift_{mes_elegido}.html"
    if informe.exists():
        with st.expander("Ver el informe completo de Evidently"):
            st.components.v1.html(informe.read_text(encoding="utf-8"), height=650, scrolling=True)
    else:
        st.caption(f"El informe de {mes_elegido} no está generado.")


# --------------------------------------------------------------------------
# Pestaña 2 · Salud del modelo
# --------------------------------------------------------------------------
with tab_modelo:
    st.subheader("Modelo que está sirviendo ahora mismo")

    codigo_info, info = fetch_api("/model-info")
    if codigo_info == 200 and info:
        a, b, c = st.columns(3)
        a.metric("Nombre", info["model_name"])
        b.metric("Versión", f"v{info['model_version']}")
        c.metric("Algoritmo", info["algorithm"])

        st.caption(f"run de MLflow: `{info['run_id']}`  ·  umbral de decisión: {info['threshold']}")

        st.write("**Métricas de validación con las que se eligió este modelo**")
        metricas = {
            k: v for k, v in info["validation_metrics"].items() if not k.startswith("train_")
        }
        st.dataframe(
            pd.DataFrame([metricas]).T.rename(columns={0: "valor"}),
            use_container_width=True,
        )

        with st.expander("Features que espera el modelo"):
            st.write(" · ".join(f"`{f}`" for f in info["features"]))
    else:
        st.warning(
            f"No se puede consultar `/model-info` (HTTP {codigo_info or 'sin respuesta'}). "
            f"Comprueba que la API está levantada en {API_URL}."
        )

    st.divider()
    st.write("**Promoción de modelos**")
    st.caption(
        "La API resuelve `models:/churn-model@champion` contra el Model Registry, no un archivo "
        "dentro de la imagen. Promover una versión nueva es mover ese alias y llamar a "
        "`/reload`: no hace falta reconstruir la imagen ni redesplegar el servicio."
    )
    if st.button("Recargar el modelo desde el registry"):
        codigo, respuesta = post_api("/reload", {})
        if codigo == 200:
            st.success(
                f"Modelo recargado: v{respuesta['model_version']} ({respuesta['algorithm']})"
            )
            st.cache_data.clear()
        else:
            st.error(f"No se pudo recargar (HTTP {codigo or 'sin respuesta'}): {respuesta}")


# --------------------------------------------------------------------------
# Pestaña 3 · Predicciones en vivo
# --------------------------------------------------------------------------
with tab_predicciones:
    st.subheader("Probar el modelo")
    st.caption("Llama a la API real. Los valores por defecto describen a un cliente de riesgo.")

    with st.form("prediccion"):
        c1, c2, c3 = st.columns(3)
        with c1:
            contract_type = st.selectbox(
                "Tipo de contrato", ["Month-to-month", "One year", "Two year"]
            )
            tenure_months = st.slider("Antigüedad (meses)", 0, 72, 3)
            monthly_charges = st.slider("Cargo mensual", 18.0, 190.0, 94.5)
            age = st.slider("Edad", 18, 92, 42)
            gender = st.selectbox("Género", ["F", "M"])
        with c2:
            support_tickets_30d = st.slider("Tickets de soporte (30 días)", 0, 12, 3)
            late_payments_3m = st.slider("Impagos (3 meses)", 0, 8, 2)
            monthly_usage_gb = st.slider("Consumo mensual (GB)", 0.0, 80.0, 8.2)
            avg_session_minutes = st.slider("Sesión media (min)", 1.0, 120.0, 18.4)
            num_services = st.slider("Servicios contratados", 1, 8, 2)
        with c3:
            region = st.selectbox("Región", ["Norte", "Centro", "Sur", "Occidente"])
            payment_method = st.selectbox(
                "Método de pago",
                ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
            )
            senior_citizen = st.selectbox("Tercera edad", ["No", "Yes"])
            paperless_billing = st.selectbox("Factura electrónica", ["Yes", "No"])

        enviado = st.form_submit_button("Predecir", type="primary")

    if enviado:
        payload = {
            "age": age,
            "gender": gender,
            "region": region,
            "senior_citizen": senior_citizen,
            "contract_type": contract_type,
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "payment_method": payment_method,
            "paperless_billing": paperless_billing,
            "monthly_usage_gb": monthly_usage_gb,
            "support_tickets_30d": support_tickets_30d,
            "avg_session_minutes": avg_session_minutes,
            "num_services": num_services,
            "late_payments_3m": late_payments_3m,
        }
        codigo, respuesta = post_api("/predict", payload)

        if codigo == 200:
            probabilidad = respuesta["churn_probability"]
            tramo = respuesta["risk_band"]
            color_tramo = {"bajo": "#0ca30c", "medio": "#fab219", "alto": "#d03b3b"}[tramo]

            medidor = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probabilidad * 100,
                    number={"suffix": " %", "font": {"size": 42}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1},
                        "bar": {"color": color_tramo, "thickness": 0.7},
                        "steps": [
                            {"range": [0, 35], "color": "rgba(12,163,12,.12)"},
                            {"range": [35, 65], "color": "rgba(250,178,25,.12)"},
                            {"range": [65, 100], "color": "rgba(208,59,59,.12)"},
                        ],
                    },
                )
            )
            medidor.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=10))

            izq, der = st.columns([2, 1])
            izq.plotly_chart(medidor, use_container_width=True)
            der.markdown(
                f"### Riesgo {tramo}\n\n"
                f"Clase predicha: **{'abandona' if respuesta['churn_prediction'] else 'permanece'}**\n\n"
                f"Umbral: {respuesta['threshold']}\n\n"
                f"Modelo: v{respuesta['model_version']}"
            )
        elif codigo == 422:
            st.error(f"La API ha rechazado la entrada (422): {respuesta}")
        else:
            st.error(f"Error de la API (HTTP {codigo or 'sin respuesta'}): {respuesta}")


# --------------------------------------------------------------------------
# Pestaña 4 · EDA
# --------------------------------------------------------------------------
with tab_eda:
    st.subheader("Lo que el análisis exploratorio encontró")
    st.caption(
        "Las figuras salen de los notebooks de la Fase 2. Están aquí para que el panel cuente "
        "la historia completa sin obligar a abrir el repositorio."
    )

    figuras = [
        (
            "01_desbalance_churn",
            "Desbalance de clases",
            "1 de cada 5 clientes abandona. Por eso la métrica principal es el AUC y no la "
            "accuracy: predecir siempre «permanece» acierta el 80 % de las veces y no sirve "
            "para nada.",
        ),
        (
            "04_data_drift_uso",
            "Data drift desde 2024-09",
            "El consumo medio pasa de 12 a 20 GB en rampa progresiva. La distribución de "
            "entrada cambia, pero la relación con el churn no.",
        ),
        (
            "04_concept_drift_tickets",
            "Concept drift desde 2025-01",
            "La correlación entre tickets de soporte y churn se apaga de golpe. El mejor "
            "predictor del modelo deja de informar: aquí es donde el modelo se rompe.",
        ),
        (
            "08_historico_drift",
            "Las dos señales juntas",
            "Meses de drift de datos con el modelo intacto, y después la caída real.",
        ),
    ]

    for nombre, titulo, explicacion in figuras:
        ruta = figure_path(nombre)
        if ruta is None:
            continue
        st.markdown(f"**{titulo}**")
        st.image(str(ruta), use_container_width=True)
        st.caption(explicacion)
        st.write("")

    st.info(
        "El análisis completo, con los contrastes estadísticos y las decisiones de feature "
        "engineering, está publicado en "
        "[GitHub Pages](https://danielpantoja08.github.io/mlops-churn-pipeline/)."
    )
