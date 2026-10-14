"""Aplicación Streamlit: análisis de sentimiento en español.

Ejecutar:  streamlit run app.py
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from predict import load_model, predict_texts

BASE_DIR = Path(__file__).resolve().parent
METRICS_PATH = BASE_DIR / "reports" / "metrics.json"
CONFIG_PATH = BASE_DIR / "modelo" / "config.json"

EJEMPLOS = {
    "Positivo": "Me encantó, llegó antes de lo esperado y la calidad es excelente. Lo recomiendo totalmente.",
    "Neutral": "El producto es como en la descripción, ni mejor ni peor de lo que esperaba.",
    "Negativo": "Muy mala experiencia, el producto llegó roto y nadie responde mis reclamos.",
}
ICONOS = {"positivo": "😊", "neutral": "😐", "negativo": "😞"}

st.set_page_config(page_title="Análisis de sentimiento", page_icon="💬", layout="centered")


@st.cache_resource(show_spinner="Cargando modelo...")
def get_model():
    return load_model()


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


config, metrics = read_json(CONFIG_PATH), read_json(METRICS_PATH)
with st.sidebar:
    st.header("Sobre el modelo")
    st.write(
        "Red neuronal recurrente (RNN/LSTM/GRU) entrenada con reseñas en español. "
        "Clasifica el texto como **positivo**, **neutral** o **negativo**."
    )
    if config:
        st.caption(f"Arquitectura: **{config['model'].upper()}** · max_len = {config['max_len']}")
    if metrics:
        st.subheader("Métricas en test")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        c2.metric("F1 macro", f"{metrics['f1_macro']:.3f}")
        c1.metric("Precisión macro", f"{metrics['precision_macro']:.3f}")
        c2.metric("Recall macro", f"{metrics['recall_macro']:.3f}")

st.title("💬 Análisis de sentimiento en español")
st.write("Escribe una reseña o comentario (o elige un ejemplo) y pulsa **Predecir**.")

if "texto" not in st.session_state:
    st.session_state["texto"] = ""

cols = st.columns(len(EJEMPLOS))
for col, (nombre, ejemplo) in zip(cols, EJEMPLOS.items()):
    if col.button(f"Ejemplo {nombre.lower()}", use_container_width=True):
        st.session_state["texto"] = ejemplo

texto = st.text_area("Texto", key="texto", height=150, placeholder="Escribe aquí tu comentario...")

if st.button("Predecir", type="primary"):
    if not texto.strip():
        st.warning("Por favor ingresa un texto antes de predecir.")
    else:
        try:
            r = predict_texts(get_model(), [texto])[0]
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

        st.subheader(f"{ICONOS[r['etiqueta']]} {r['etiqueta'].capitalize()} — confianza {r['confianza']:.1%}")
        probs = pd.Series(r["probabilidades"], name="Probabilidad")
        st.bar_chart(probs)
