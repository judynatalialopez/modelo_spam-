"""Entrenamiento de un clasificador de spam (SMS) con RNN / LSTM / GRU.

Uso:
    python train.py                       # BiLSTM por defecto
    python train.py --model gru --epochs 30
    python train.py --model rnn

Genera:
    modelo/modelo_spam.keras   -> modelo entrenado (incluye TextVectorization)
    modelo/config.json         -> hiperparámetros y umbral
    reports/metrics.json       -> métricas en test
    reports/*.png              -> curvas de entrenamiento, matriz de confusión, ROC
    reports/predicciones_test.csv -> 10+ predicciones de ejemplo
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

from utils import clean_text, load_dataset, set_seed


# ----------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------
def prepare_data(seed: int):
    df = load_dataset()
    df = df.dropna().drop_duplicates(subset="text").reset_index(drop=True)
    df["y"] = (df["label"] == "spam").astype("int32")
    df["clean"] = df["text"].apply(clean_text)

    # 70 / 15 / 15 estratificado. Se divide ANTES de adaptar el vocabulario.
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["y"], random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["y"], random_state=seed)
    return train_df, val_df, test_df


def make_ds(texts, labels, batch_size, shuffle=False, seed=42):
    ds = tf.data.Dataset.from_tensor_slices((np.array(texts, dtype=str), np.array(labels, dtype="float32")))
    if shuffle:
        ds = ds.shuffle(len(texts), seed=seed)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ----------------------------------------------------------------------------
# Modelo
# ----------------------------------------------------------------------------
def build_vectorizer(train_texts, vocab_size):
    lengths = [len(t.split()) for t in train_texts]
    max_len = int(np.percentile(lengths, 95))  # percentil 95, no el máximo
    vec = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=max_len,
        standardize="lower_and_strip_punctuation",
    )
    vec.adapt(np.array(train_texts, dtype=str))  # solo con entrenamiento
    return vec, max_len


def build_model(vectorizer, kind, embed_dim, units, dropout):
    vocab = vectorizer.vocabulary_size()
    inputs = keras.Input(shape=(), dtype=tf.string, name="texto")
    x = vectorizer(inputs)
    x = layers.Embedding(vocab, embed_dim, mask_zero=True)(x)

    if kind == "rnn":
        x = layers.SimpleRNN(units)(x)
    elif kind == "lstm":
        x = layers.LSTM(units)(x)
    elif kind == "gru":
        x = layers.GRU(units)(x)
    elif kind == "bilstm":
        x = layers.Bidirectional(layers.LSTM(units))(x)
    else:
        raise ValueError(f"Modelo no soportado: {kind}")

    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prob_spam")(x)

    model = keras.Model(inputs, outputs, name=f"spam_{kind}")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],
    )
    return model


# ----------------------------------------------------------------------------
# Gráficas y evaluación
# ----------------------------------------------------------------------------
def plot_history(history, path):
    h = history.history
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(h["loss"], label="entrenamiento")
    axes[0].plot(h["val_loss"], label="validación")
    axes[0].set(title="Pérdida", xlabel="Época", ylabel="Binary crossentropy")
    axes[0].legend()
    axes[1].plot(h["accuracy"], label="entrenamiento")
    axes[1].plot(h["val_accuracy"], label="validación")
    axes[1].set(title="Exactitud", xlabel="Época", ylabel="Accuracy")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def evaluate(model, test_df, batch_size, reports_dir, threshold=0.5):
    X, y = test_df["clean"].values, test_df["y"].values
    probs = model.predict(make_ds(X, y, batch_size), verbose=0).ravel()
    preds = (probs >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
        "f1": f1_score(y, preds),
        "roc_auc": roc_auc_score(y, probs),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
        "n_test": int(len(y)),
        "threshold": threshold,
    }
    print("\n=== Evaluación en TEST ===")
    print(classification_report(y, preds, target_names=["ham", "spam"], digits=4))
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    ConfusionMatrixDisplay.from_predictions(y, preds, display_labels=["ham", "spam"], cmap="Blues")
    plt.title("Matriz de confusión (test)")
    plt.savefig(os.path.join(reports_dir, "matriz_confusion.png"), dpi=130, bbox_inches="tight")
    plt.close()

    RocCurveDisplay.from_predictions(y, probs)
    plt.title("Curva ROC (test)")
    plt.savefig(os.path.join(reports_dir, "roc.png"), dpi=130, bbox_inches="tight")
    plt.close()

    return metrics, probs, preds


def sample_predictions(test_df, probs, preds, reports_dir, n=12, seed=42):
    out = test_df.copy()
    out["prob_spam"] = probs
    out["prediccion"] = np.where(preds == 1, "spam", "ham")
    out["correcta"] = out["prediccion"] == out["label"]
    # mitad spam, mitad ham
    half = n // 2
    parts = [g.sample(min(half, len(g)), random_state=seed) for _, g in out.groupby("label")]
    sel = pd.concat(parts)[["text", "label", "prediccion", "prob_spam", "correcta"]]
    sel["text"] = sel["text"].str.slice(0, 90)
    sel.to_csv(os.path.join(reports_dir, "predicciones_test.csv"), index=False)
    print("\n=== Predicciones de ejemplo (test) ===")
    print(sel.to_string(index=False))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="bilstm", choices=["rnn", "lstm", "gru", "bilstm"])
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--units", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="modelo")
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--no-class-weight", action="store_true", help="Desactiva el balanceo de clases")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)
    set_seed(args.seed)

    train_df, val_df, test_df = prepare_data(args.seed)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print("Proporción de spam (train/val/test):",
          [round(d['y'].mean(), 3) for d in (train_df, val_df, test_df)])

    vectorizer, max_len = build_vectorizer(train_df["clean"].values, args.vocab_size)
    print(f"max_len (p95) = {max_len} | vocabulario = {vectorizer.vocabulary_size()}")

    model = build_model(vectorizer, args.model, args.embed_dim, args.units, args.dropout)
    model.summary()

    class_weight = None
    if not args.no_class_weight:
        w = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_df["y"].values)
        class_weight = {0: float(w[0]), 1: float(w[1])}
        print("class_weight:", class_weight)

    ckpt_path = os.path.join(args.out_dir, "checkpoint.keras")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        make_ds(train_df["clean"].values, train_df["y"].values, args.batch_size, shuffle=True, seed=args.seed),
        validation_data=make_ds(val_df["clean"].values, val_df["y"].values, args.batch_size),
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )
    plot_history(history, os.path.join(args.reports_dir, "curvas_entrenamiento.png"))

    metrics, probs, preds = evaluate(model, test_df, args.batch_size, args.reports_dir)
    sample_predictions(test_df, probs, preds, args.reports_dir, seed=args.seed)

    # Guardado
    model_path = os.path.join(args.out_dir, "modelo_spam.keras")
    model.save(model_path)
    config = {**vars(args), "max_len": max_len, "vocab_real": int(vectorizer.vocabulary_size()),
              "threshold": 0.5, "epochs_entrenadas": len(history.history["loss"])}
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(args.reports_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nModelo guardado en: {model_path}")


if __name__ == "__main__":
    main()
