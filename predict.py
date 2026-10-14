"""Carga del modelo entrenado y predicción sobre texto nuevo.

Uso por consola:
    python predict.py "Congratulations! You won a free prize, call now"
    python predict.py "Hey, are we still meeting at 6?" "WINNER!! Claim your £1000 now"
"""
import json
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from utils import clean_text

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo" / "modelo_spam.keras"
CONFIG_PATH = BASE_DIR / "modelo" / "config.json"


def load_model(path: Path = MODEL_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {path}. Ejecuta primero `python train.py` "
            "o copia aquí el modelo generado en el notebook."
        )
    return keras.models.load_model(path)


def load_threshold(default: float = 0.5) -> float:
    if CONFIG_PATH.exists():
        return float(json.loads(CONFIG_PATH.read_text()).get("threshold", default))
    return default


def predict_texts(model, texts, threshold: float | None = None):
    """Devuelve una lista de dicts con probabilidad de spam y etiqueta."""
    threshold = load_threshold() if threshold is None else threshold
    cleaned = [clean_text(t) for t in texts]
    probs = model.predict(tf.constant(cleaned), verbose=0).ravel()
    return [
        {
            "texto": t,
            "prob_spam": float(p),
            "etiqueta": "spam" if p >= threshold else "ham",
        }
        for t, p in zip(texts, probs)
    ]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Uso: python predict.py "mensaje 1" ["mensaje 2" ...]')
    mdl = load_model()
    for r in predict_texts(mdl, sys.argv[1:]):
        print(f"[{r['etiqueta'].upper():4}] p(spam)={r['prob_spam']:.3f} | {r['texto']}")
