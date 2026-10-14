"""Utilidades compartidas por train.py, predict.py y app.py.

Mantener la limpieza de texto en un único lugar garantiza que el
preprocesamiento en entrenamiento y en inferencia sea idéntico.
"""
import os
import random
import re

import numpy as np
import pandas as pd

LABELS = ["positivo", "neutral", "negativo"]  # índice = id de clase (0, 1, 2)
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

HF_BASE = ("https://huggingface.co/datasets/tyqiangz/multilingual-sentiments/"
           "resolve/main/spanish/multilingual-sentiments-{split}.parquet")

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_MENTION_RE = re.compile(r"@\w+")
_REPEAT_RE = re.compile(r"(.)\1{2,}")  # "buenooooo" -> "buenoo"
_SPACES_RE = re.compile(r"\s+")


def set_seed(seed: int = 42) -> None:
    """Fija las semillas para que los resultados sean reproducibles."""
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def clean_text(text: str) -> str:
    """Limpieza ligera para sentimiento en español.

    - Minúsculas (se conservan tildes y la ñ).
    - URLs y menciones @usuario -> tokens genéricos.
    - Letras repetidas ("buenooooo") se reducen a 2.
    - NO se eliminan stopwords: "no", "muy", "nada" cambian el sentimiento.
    La puntuación (incluyendo ¡ y ¿) la elimina la capa TextVectorization.
    """
    text = str(text).lower()
    text = _URL_RE.sub(" urltoken ", text)
    text = _MENTION_RE.sub(" usuario ", text)
    text = _REPEAT_RE.sub(r"\1\1", text)
    return _SPACES_RE.sub(" ", text).strip()


def load_dataset(data_dir: str = "data") -> pd.DataFrame:
    """Devuelve un DataFrame con columnas ['text', 'label'] (label = 0/1/2).

    Fuente: Hugging Face `tyqiangz/multilingual-sentiments`, config `spanish`
    (se unen train + validation + test y se re-divide después).
    Alternativa offline: coloca data/dataset.csv con columnas `text,label`
    (label puede ser 0/1/2 o positivo/neutral/negativo).
    """
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "dataset.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)[["text", "label"]]
    else:
        print("Descargando dataset desde Hugging Face...")
        parts = [pd.read_parquet(HF_BASE.format(split=s)) for s in ("train", "validation", "test")]
        df = pd.concat(parts, ignore_index=True)[["text", "label"]]
        df.to_csv(csv_path, index=False)  # caché local

    if df["label"].dtype == object:
        df["label"] = df["label"].str.lower().map(LABEL2ID)
    df["label"] = df["label"].astype("int32")
    return df
