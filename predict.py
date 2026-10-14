"""Carga del modelo entrenado y predicción de sentimiento sobre texto en español.

Uso por consola:
    python predict.py "Me encantó el producto, llegó rápido y funciona perfecto"
    python predict.py "Muy mala calidad, no lo recomiendo" "Está bien, nada especial"
"""
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from utils import LABELS, clean_text

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo" / "modelo_sentimiento.keras"


def load_model(path: Path = MODEL_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {path}. Ejecuta primero `python train.py` "
            "o copia aquí el modelo generado en el notebook."
        )
    return keras.models.load_model(path)


def predict_texts(model, texts):
    """Devuelve una lista de dicts con etiqueta, confianza y probabilidades por clase."""
    cleaned = [clean_text(t) for t in texts]
    probs = model.predict(tf.constant(cleaned), verbose=0)
    results = []
    for t, p in zip(texts, probs):
        i = int(p.argmax())
        results.append({
            "texto": t,
            "etiqueta": LABELS[i],
            "confianza": float(p[i]),
            "probabilidades": {lab: float(v) for lab, v in zip(LABELS, p)},
        })
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Uso: python predict.py "mensaje 1" ["mensaje 2" ...]')
    mdl = load_model()
    for r in predict_texts(mdl, sys.argv[1:]):
        print(f"[{r['etiqueta'].upper():8}] confianza={r['confianza']:.2f} | {r['texto']}")
