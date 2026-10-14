"""Utilidades compartidas por train.py, predict.py y app.py.

Mantener la limpieza de texto en un único lugar garantiza que el
preprocesamiento en entrenamiento y en inferencia sea idéntico.
"""
import io
import os
import random
import re
import urllib.request
import zipfile

import numpy as np
import pandas as pd

DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_LONGNUM_RE = re.compile(r"\b\d{5,}\b")  # teléfonos, códigos cortos, etc.
_MONEY_RE = re.compile(r"[£$€]")
_SPACES_RE = re.compile(r"\s+")


def set_seed(seed: int = 42) -> None:
    """Fija las semillas para que los resultados sean reproducibles."""
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def clean_text(text: str) -> str:
    """Limpieza ligera: minúsculas y normalización de URLs, emails,
    números largos y símbolos de moneda (señales muy útiles para spam).
    La puntuación restante la elimina la capa TextVectorization del modelo.
    """
    text = str(text).lower()
    text = _URL_RE.sub(" urltoken ", text)
    text = _EMAIL_RE.sub(" emailtoken ", text)
    text = _LONGNUM_RE.sub(" numtoken ", text)
    text = _MONEY_RE.sub(" moneytoken ", text)
    return _SPACES_RE.sub(" ", text).strip()


def load_dataset(data_dir: str = "data") -> pd.DataFrame:
    """Carga SMS Spam Collection (UCI). Devuelve un DataFrame con columnas
    ['label', 'text']. Si no existe en disco, lo descarga.

    También acepta la versión de Kaggle si colocas 'spam.csv' en data/.
    """
    os.makedirs(data_dir, exist_ok=True)
    kaggle_path = os.path.join(data_dir, "spam.csv")
    uci_path = os.path.join(data_dir, "SMSSpamCollection")

    if os.path.exists(kaggle_path):
        df = pd.read_csv(kaggle_path, encoding="latin-1")[["v1", "v2"]]
        df.columns = ["label", "text"]
        return df

    if not os.path.exists(uci_path):
        print("Descargando dataset desde UCI...")
        with urllib.request.urlopen(DATASET_URL) as resp:
            zipfile.ZipFile(io.BytesIO(resp.read())).extract("SMSSpamCollection", data_dir)

    return pd.read_csv(uci_path, sep="\t", header=None, names=["label", "text"])
