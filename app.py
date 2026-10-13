"""Aplicación Streamlit - Detector de spam en SMS (Bi-GRU).
Ejecutar:  streamlit run app.py
"""
import json, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from spam_utils import clean_text, MODEL_PATH

st.set_page_config(page_title="Detector de Spam SMS", page_icon="📩", layout="centered")


@st.cache_resource(show_spinner="Cargando modelo entrenado...")
def get_model():
    return keras.models.load_model(MODEL_PATH)


def predict_proba(model, messages):
    cleaned = tf.constant([clean_text(m) for m in messages], dtype=tf.string)
    return model.predict(cleaned, verbose=0).ravel()


EXAMPLES = {
    "Spam: premio": "WINNER!! You have been selected to receive a £900 prize reward. Call 09061701461 now. Claim code KL341.",
    "Spam: suscripción": "Free entry in a weekly competition! Text WIN to 80086 now to receive your latest ringtone. T&Cs apply.",
    "Ham: cotidiano": "Hey, are we still meeting for lunch tomorrow? Let me know what time works for you.",
    "Ham: estudio": "Can you send me the notes from class today? I missed the last part, thanks!",
}

st.title("📩 Detector de Spam en SMS")
st.caption("Red neuronal recurrente **Bi-GRU** entrenada con SMS Spam Collection (mensajes en inglés).")

if not os.path.exists(MODEL_PATH):
    st.error(f"No se encontró el modelo en `{MODEL_PATH}`. Ejecuta primero `python train.py`.")
    st.stop()
model = get_model()

with st.sidebar:
    st.header("⚙️ Configuración")
    threshold = st.slider("Umbral de decisión (spam)", 0.05, 0.95, 0.50, 0.05,
                          help="Si P(spam) ≥ umbral, el mensaje se clasifica como spam.")
    if os.path.exists("reports/metrics.json"):
        st.header("📊 Métricas en test")
        for k, v in json.load(open("reports/metrics.json")).items():
            st.metric(k.upper().replace("_", " "), f"{v:.3f}")

tab1, tab2 = st.tabs(["Mensaje individual", "Varios mensajes (CSV)"])

with tab1:
    choice = st.selectbox("Cargar un ejemplo (opcional)", ["—"] + list(EXAMPLES))
    default = EXAMPLES.get(choice, "")
    text = st.text_area("Escribe o pega un SMS:", value=default, height=130, key=f"txt_{choice}")
    if st.button("Analizar mensaje", type="primary"):
        if not text.strip():
            st.warning("Escribe un mensaje para analizar.")
        else:
            p = float(predict_proba(model, [text])[0])
            is_spam = p >= threshold
            (st.error if is_spam else st.success)(f"{'🚨 SPAM' if is_spam else '✅ NO ES SPAM (ham)'}  —  P(spam) = {p:.1%}")
            st.progress(min(max(p, 0.0), 1.0), text="Probabilidad de spam")
            with st.expander("Ver texto preprocesado"):
                st.code(clean_text(text))

with tab2:
    st.write("Sube un CSV con una columna llamada `mensaje` (o `text`) para clasificar varios SMS.")
    up = st.file_uploader("Archivo CSV", type="csv")
    if up:
        df = pd.read_csv(up)
        col = next((c for c in ("mensaje", "text", "message") if c in df.columns), None)
        if col is None:
            st.error("El CSV debe tener una columna `mensaje`, `text` o `message`.")
        else:
            df["prob_spam"] = predict_proba(model, df[col].astype(str).tolist()).round(4)
            df["prediccion"] = (df["prob_spam"] >= threshold).map({True: "spam", False: "ham"})
            st.dataframe(df, use_container_width=True)
            st.download_button("Descargar resultados", df.to_csv(index=False).encode(), "resultados.csv", "text/csv")
