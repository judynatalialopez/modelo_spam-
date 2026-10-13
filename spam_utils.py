"""Utilidades compartidas entre entrenamiento, predicción y la app Streamlit."""
import re

DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
MODEL_PATH = "models/spam_gru_model.keras"
THRESHOLD = 0.5

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\S+@\S+")
PHONE_RE = re.compile(r"\b\d{5,}\b")      # secuencias largas de dígitos (teléfonos, shortcodes)
NUM_RE = re.compile(r"\b\d+\b")           # otros números
MONEY_RE = re.compile(r"[£$€]")


def clean_text(text: str) -> str:
    """Limpieza de un SMS: minúsculas, normaliza URLs, emails, números y símbolos."""
    text = str(text).lower()
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = MONEY_RE.sub(" moneytoken ", text)
    text = PHONE_RE.sub(" phonetoken ", text)
    text = NUM_RE.sub(" numtoken ", text)
    text = re.sub(r"[^a-z\s]", " ", text)   # elimina puntuación y símbolos restantes
    return re.sub(r"\s+", " ", text).strip()
