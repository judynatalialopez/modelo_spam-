"""Predicción por línea de comandos.

Uso:
  python predict.py "Congratulations! You won a free prize, call 0800123456 now"
  python predict.py            # ejecuta ejemplos de demostración
"""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from spam_utils import clean_text, MODEL_PATH, THRESHOLD


def load_model(path: str = MODEL_PATH):
    return keras.models.load_model(path)


def predict(model, messages, threshold: float = THRESHOLD):
    """Devuelve lista de (mensaje, probabilidad_spam, etiqueta)."""
    cleaned = tf.constant([clean_text(m) for m in messages], dtype=tf.string)
    probs = model.predict(cleaned, verbose=0).ravel()
    return [(m, float(p), "spam" if p >= threshold else "ham") for m, p in zip(messages, probs)]


if __name__ == "__main__":
    msgs = sys.argv[1:] or [
        "WINNER!! You have won a £1000 prize. Call 09061701461 now to claim.",
        "Hey, are we still meeting for lunch tomorrow?",
        "Free entry in a weekly competition! Text WIN to 80086 now",
        "Can you send me the notes from class? Thanks!",
    ]
    for m, p, lab in predict(load_model(), msgs):
        print(f"[{lab.upper():4}] P(spam)={p:.3f} | {m}")
