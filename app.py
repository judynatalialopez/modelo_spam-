"""Aplicación Streamlit - Detector de spam en SMS (Bi-GRU).
Ejecutar:  streamlit run app.py
"""
import json, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from spam_utils import clean_text, clean_text_es, MODEL_PATH, MODEL_PATH_ES

st.set_page_config(page_title="Detector de Spam SMS", page_icon="📩", layout="centered")


@st.cache_resource(show_spinner="Cargando modelo entrenado...")
def get_model(path):
    return keras.models.load_model(path)


def predict_proba(model, messages, cleaner=clean_text):
    cleaned = tf.constant([cleaner(m) for m in messages], dtype=tf.string)
    return model.predict(cleaned, verbose=0).ravel()


EXAMPLES_ES = {
    "Spam: premio": "¡Felicidades! Has ganado un premio de 1000 euros. Llama al 09061701461 ahora para reclamarlo.",
    "Spam: suscripción": "Entrada gratis a nuestro sorteo semanal. Envía GANA al 80086 para recibir tu tono de llamada.",
    "Ham: cotidiano": "Hola, ¿sigues queriendo almorzar mañana? Avísame a qué hora te viene bien.",
    "Ham: estudio": "¿Me pasas los apuntes de la clase de hoy? Me perdí la última parte, gracias.",
}

EXAMPLES = {
    "Spam: premio": "WINNER!! You have been selected to receive a £900 prize reward. Call 09061701461 now. Claim code KL341.",
    "Spam: suscripción": "Free entry in a weekly competition! Text WIN to 80086 now to receive your latest ringtone. T&Cs apply.",
    "Ham: cotidiano": "Hey, are we still meeting for lunch tomorrow? Let me know what time works for you.",
    "Ham: estudio": "Can you send me the notes from class today? I missed the last part, thanks!",
}

st.title("📩 Detector de Spam en SMS")
st.caption("Red neuronal recurrente **Bi-GRU** entrenada con SMS Spam Collection (mensajes en inglés).")

LANGS = {"English (modelo base)": (MODEL_PATH, clean_text, EXAMPLES)}
if os.path.exists(MODEL_PATH_ES):
    LANGS["Español"] = (MODEL_PATH_ES, clean_text_es, EXAMPLES_ES)

with st.sidebar:
    st.header("⚙️ Configuración")
    lang = st.selectbox("Idioma del mensaje", list(LANGS), index=len(LANGS) - 1)
    if len(LANGS) == 1:
        st.caption("Para español, entrena `notebook_spam_es.ipynb` y copia el modelo a `models/`.")
    model_path, cleaner, EXAMPLES = LANGS[lang]
    threshold = st.slider("Umbral de decisión (spam)", 0.05, 0.95, 0.50, 0.05,
                          help="Si P(spam) ≥ umbral, el mensaje se clasifica como spam.")
    metrics_file = "reports/metrics.json" if model_path == MODEL_PATH else "reports/metrics_es.json"
    if os.path.exists(metrics_file):
        st.header("📊 Métricas en test")
        for k, v in json.load(open(metrics_file)).items():
            st.metric(k.upper().replace("_", " "), f"{v:.3f}")

if not os.path.exists(model_path):
    st.error(f"No se encontró el modelo en `{model_path}`.")
    st.stop()
model = get_model(model_path)

tab1, tab2 = st.tabs(["Mensaje individual", "Varios mensajes (CSV)"])

with tab1:
    choice = st.selectbox("Cargar un ejemplo (opcional)", ["—"] + list(EXAMPLES))
    default = EXAMPLES.get(choice, "")
    text = st.text_area("Escribe o pega un SMS:", value=default, height=130, key=f"txt_{lang}_{choice}")
    if st.button("Analizar mensaje", type="primary"):
        if not text.strip():
            st.warning("Escribe un mensaje para analizar.")
        else:
            p = float(predict_proba(model, [text], cleaner)[0])
            is_spam = p >= threshold
            (st.error if is_spam else st.success)(f"{'🚨 SPAM' if is_spam else '✅ NO ES SPAM (ham)'}  —  P(spam) = {p:.1%}")
            st.progress(min(max(p, 0.0), 1.0), text="Probabilidad de spam")
            with st.expander("Ver texto preprocesado"):
                st.code(cleaner(text))

with tab2:
    st.write("Sube un CSV con una columna llamada `mensaje` (o `text`) para clasificar varios SMS.")
    up = st.file_uploader("Archivo CSV", type="csv")
    if up:
        df = pd.read_csv(up)
        col = next((c for c in ("mensaje", "text", "message") if c in df.columns), None)
        if col is None:
            st.error("El CSV debe tener una columna `mensaje`, `text` o `message`.")
        else:
            df["prob_spam"] = predict_proba(model, df[col].astype(str).tolist(), cleaner).round(4)
            df["prediccion"] = (df["prob_spam"] >= threshold).map({True: "spam", False: "ham"})
            st.dataframe(df, use_container_width=True)
            st.download_button("Descargar resultados", df.to_csv(index=False).encode(), "resultados.csv", "text/csv")
