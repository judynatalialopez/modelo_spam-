"""Entrenamiento de un clasificador de sentimiento en español (positivo / neutral /
negativo) con RNN, LSTM, BiLSTM o GRU.

Uso:
    python train.py                       # BiLSTM por defecto
    python train.py --model gru --epochs 30

Genera:
    modelo/modelo_sentimiento.keras  -> modelo entrenado (incluye TextVectorization)
    modelo/config.json               -> hiperparámetros y etiquetas
    reports/metrics.json             -> métricas en test
    reports/*.png                    -> curvas, matriz de confusión, ROC
    reports/predicciones_test.csv    -> 12 predicciones de ejemplo
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
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

from utils import LABELS, clean_text, load_dataset, set_seed

NUM_CLASSES = len(LABELS)


# ----------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------
def prepare_data(seed: int):
    df = load_dataset()
    df = df.dropna().drop_duplicates(subset="text").reset_index(drop=True)
    df["etiqueta"] = df["label"].map(dict(enumerate(LABELS)))
    df["clean"] = df["text"].apply(clean_text)

    print("\nDistribución de clases:\n", df["etiqueta"].value_counts())
    print("\nVerifica que las etiquetas tengan sentido (2 ejemplos por clase):")
    for lab in LABELS:
        for t in df.loc[df["etiqueta"] == lab, "text"].head(2):
            print(f"  [{lab}] {t[:100]}")

    # 70 / 15 / 15 estratificado. Se divide ANTES de adaptar el vocabulario.
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=seed)
    return train_df, val_df, test_df


def make_ds(texts, labels, batch_size, shuffle=False, seed=42):
    ds = tf.data.Dataset.from_tensor_slices((np.array(texts, dtype=str), np.array(labels, dtype="int32")))
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
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="probabilidades")(x)

    model = keras.Model(inputs, outputs, name=f"sentimiento_{kind}")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
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
    axes[0].set(title="Pérdida", xlabel="Época", ylabel="Crossentropy")
    axes[0].legend()
    axes[1].plot(h["accuracy"], label="entrenamiento")
    axes[1].plot(h["val_accuracy"], label="validación")
    axes[1].set(title="Exactitud", xlabel="Época", ylabel="Accuracy")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def evaluate(model, test_df, batch_size, reports_dir):
    X, y = test_df["clean"].values, test_df["label"].values
    probs = model.predict(make_ds(X, y, batch_size), verbose=0)
    preds = probs.argmax(axis=1)

    metrics = {
        "accuracy": accuracy_score(y, preds),
        "precision_macro": precision_score(y, preds, average="macro"),
        "recall_macro": recall_score(y, preds, average="macro"),
        "f1_macro": f1_score(y, preds, average="macro"),
        "f1_weighted": f1_score(y, preds, average="weighted"),
        "roc_auc_ovr": roc_auc_score(y, probs, multi_class="ovr", average="macro"),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
        "n_test": int(len(y)),
    }
    print("\n=== Evaluación en TEST ===")
    print(classification_report(y, preds, target_names=LABELS, digits=4))
    print(f"ROC-AUC (OvR, macro): {metrics['roc_auc_ovr']:.4f}")

    ConfusionMatrixDisplay.from_predictions(y, preds, display_labels=LABELS, cmap="Blues")
    plt.title("Matriz de confusión (test)")
    plt.savefig(os.path.join(reports_dir, "matriz_confusion.png"), dpi=130, bbox_inches="tight")
    plt.close()

    y_bin = label_binarize(y, classes=list(range(NUM_CLASSES)))
    for i, lab in enumerate(LABELS):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        plt.plot(fpr, tpr, label=f"{lab} (AUC={roc_auc_score(y_bin[:, i], probs[:, i]):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos")
    plt.title("Curvas ROC uno-contra-resto (test)")
    plt.legend()
    plt.savefig(os.path.join(reports_dir, "roc.png"), dpi=130, bbox_inches="tight")
    plt.close()

    return metrics, probs, preds


def sample_predictions(test_df, probs, preds, reports_dir, n_per_class=4, seed=42):
    out = test_df.copy()
    out["prediccion"] = [LABELS[i] for i in preds]
    out["confianza"] = probs.max(axis=1)
    out["correcta"] = out["prediccion"] == out["etiqueta"]
    parts = [g.sample(min(n_per_class, len(g)), random_state=seed) for _, g in out.groupby("etiqueta")]
    sel = pd.concat(parts)[["text", "etiqueta", "prediccion", "confianza", "correcta"]]
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
    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    vectorizer, max_len = build_vectorizer(train_df["clean"].values, args.vocab_size)
    print(f"max_len (p95) = {max_len} | vocabulario = {vectorizer.vocabulary_size()}")

    model = build_model(vectorizer, args.model, args.embed_dim, args.units, args.dropout)
    model.summary()

    class_weight = None
    if not args.no_class_weight:
        w = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=train_df["label"].values)
        class_weight = {i: float(v) for i, v in enumerate(w)}
        print("class_weight:", class_weight)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(os.path.join(args.out_dir, "checkpoint.keras"),
                                        monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        make_ds(train_df["clean"].values, train_df["label"].values, args.batch_size, shuffle=True, seed=args.seed),
        validation_data=make_ds(val_df["clean"].values, val_df["label"].values, args.batch_size),
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )
    plot_history(history, os.path.join(args.reports_dir, "curvas_entrenamiento.png"))

    metrics, probs, preds = evaluate(model, test_df, args.batch_size, args.reports_dir)
    sample_predictions(test_df, probs, preds, args.reports_dir, seed=args.seed)

    model_path = os.path.join(args.out_dir, "modelo_sentimiento.keras")
    model.save(model_path)
    config = {**vars(args), "max_len": max_len, "vocab_real": int(vectorizer.vocabulary_size()),
              "labels": LABELS, "epochs_entrenadas": len(history.history["loss"])}
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(args.reports_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nModelo guardado en: {model_path}")


if __name__ == "__main__":
    main()
