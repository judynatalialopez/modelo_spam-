"""Aplicación Streamlit para clasificar SMS como spam o ham.

Ejecutar:  streamlit run app.py
"""
import json
from pathlib import Path

import streamlit as st

from predict import load_model, predict_texts

BASE_DIR = Path(__file__).resolve().parent
METRICS_PATH = BASE_DIR / "reports" / "metrics.json"
CONFIG_PATH = BASE_DIR / "modelo" / "config.json"

EJEMPLOS = {
    "Spam 1": "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! Call 09061701461 now.",
    "Spam 2": "Free entry in a weekly competition to win FA Cup tickets. Text FA to 87121 to receive entry question.",
    "Ham 1": "Hey, are we still meeting at 6 for dinner? Let me know if you want me to pick you up.",
    "Ham 2": "I'll call you later, I'm in a meeting right now. Can you send me the notes?",
}

st.set_page_config(page_title="Detector de spam SMS", page_icon="📩", layout="centered")


@st.cache_resource(show_spinner="Cargando modelo...")
def get_model():
    return load_model()


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


# ----------------------------------------------------------------------------
# Barra lateral: información del modelo
# ----------------------------------------------------------------------------
config, metrics = read_json(CONFIG_PATH), read_json(METRICS_PATH)
with st.sidebar:
    st.header("Sobre el modelo")
    st.write(
        "Red neuronal recurrente (RNN/LSTM/GRU) entrenada con el dataset "
        "**SMS Spam Collection** (UCI). Recibe texto y devuelve la probabilidad de spam."
    )
    if config:
        st.caption(f"Arquitectura: **{config['model'].upper()}** · max_len = {config['max_len']}")
    if metrics:
        st.subheader("Métricas en test")
        c1, c2 = st.columns(2)
        c1.metric("F1", f"{metrics['f1']:.3f}")
        c2.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        c1.metric("Precisión", f"{metrics['precision']:.3f}")
        c2.metric("Recall", f"{metrics['recall']:.3f}")
    st.info("El modelo fue entrenado con mensajes en **inglés**; escribe los mensajes en ese idioma.")

# ----------------------------------------------------------------------------
# Cuerpo principal
# ----------------------------------------------------------------------------
st.title("📩 Detector de spam en SMS")
st.write("Escribe un mensaje (en inglés) o elige un ejemplo y pulsa **Predecir**.")

if "texto" not in st.session_state:
    st.session_state["texto"] = ""

cols = st.columns(len(EJEMPLOS))
for col, (nombre, ejemplo) in zip(cols, EJEMPLOS.items()):
    if col.button(nombre, use_container_width=True):
        st.session_state["texto"] = ejemplo

texto = st.text_area("Mensaje", key="texto", height=150, placeholder="Type your SMS here...")

if st.button("Predecir", type="primary"):
    if not texto.strip():
        st.warning("Por favor ingresa un mensaje antes de predecir.")
    else:
        try:
            resultado = predict_texts(get_model(), [texto])[0]
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

        p = resultado["prob_spam"]
        if resultado["etiqueta"] == "spam":
            st.error(f"🚨 **SPAM** — probabilidad de spam: {p:.1%}")
        else:
            st.success(f"✅ **HAM (mensaje legítimo)** — probabilidad de spam: {p:.1%}")
        st.progress(min(max(p, 0.0), 1.0), text="Probabilidad de spam")
