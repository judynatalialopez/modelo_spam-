"""Entrenamiento de un detector de spam SMS con Bi-GRU (TensorFlow/Keras).

Uso:  python train.py
Genera: models/spam_gru_model.keras, reports/*.png, reports/metrics.json, reports/predicciones_test.csv
"""
import json, os, random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report, roc_curve)
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from spam_utils import clean_text, MODEL_PATH, THRESHOLD

SEED = 42
MAX_TOKENS, SEQ_LEN, EMB_DIM = 8000, 50, 64
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
os.makedirs("models", exist_ok=True); os.makedirs("reports", exist_ok=True)

# 1. Carga ---------------------------------------------------------------
df = pd.read_csv("data/sms.tsv", sep="\t", header=None, names=["label", "text"])
print("Registros:", len(df), "| duplicados:", df.duplicated().sum())

# 2. EDA -----------------------------------------------------------------
df["length"] = df["text"].str.len()
df["n_words"] = df["text"].str.split().str.len()
print(df.groupby("label")[["length", "n_words"]].describe().T)

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
df["label"].value_counts().plot.bar(ax=ax[0], color=["#4C72B0", "#C44E52"]); ax[0].set_title("Distribución de clases")
sns.histplot(data=df, x="length", hue="label", bins=40, ax=ax[1], palette=["#4C72B0", "#C44E52"]); ax[1].set_title("Longitud (caracteres)")
sns.boxplot(data=df, x="label", y="n_words", hue="label", legend=False, ax=ax[2], palette=["#4C72B0", "#C44E52"]); ax[2].set_title("Nº de palabras")
plt.tight_layout(); plt.savefig("reports/eda.png", dpi=120); plt.close()

# 3. Limpieza ------------------------------------------------------------
df = df.drop_duplicates(subset="text").reset_index(drop=True)
df["clean"] = df["text"].apply(clean_text)
df = df[df["clean"].str.len() > 0].reset_index(drop=True)
df["y"] = (df["label"] == "spam").astype(int)
print("Tras limpieza:", len(df), "| spam:", df["y"].sum(), f"({df['y'].mean():.1%})")

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for a, lab, col in zip(ax, ["ham", "spam"], ["#4C72B0", "#C44E52"]):
    cnt = Counter(" ".join(df[df.label == lab]["clean"]).split())
    w, c = zip(*cnt.most_common(15)); a.barh(w[::-1], c[::-1], color=col); a.set_title(f"Top palabras - {lab}")
plt.tight_layout(); plt.savefig("reports/top_words.png", dpi=120); plt.close()

# 4. Split 70/15/15 estratificado ---------------------------------------
X_train, X_tmp, y_train, y_tmp = train_test_split(df["clean"], df["y"], test_size=0.30, stratify=df["y"], random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
raw_test = df.loc[X_test.index, "text"]
print(f"Train={len(X_train)} Val={len(X_val)} Test={len(X_test)}")
X_train, X_val, X_test = X_train.values, X_val.values, X_test.values
y_train, y_val, y_test = y_train.values, y_val.values, y_test.values

# 5. Modelo (el vectorizador va DENTRO del modelo: recibe texto crudo) ---
vectorizer = layers.TextVectorization(max_tokens=MAX_TOKENS, output_sequence_length=SEQ_LEN)
vectorizer.adapt(X_train)  # solo con train, evita fuga de datos

inputs = keras.Input(shape=(), dtype="string")
x = vectorizer(inputs)
x = layers.Embedding(MAX_TOKENS, EMB_DIM, mask_zero=True)(x)
x = layers.Bidirectional(layers.GRU(32))(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(32, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs, name="spam_bigru")
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")])
model.summary()

cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
class_weight = {0: cw[0], 1: cw[1]}
cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)]
def make_ds(X, y=None, shuffle=False):
    t = tf.constant(X, dtype=tf.string)
    ds = tf.data.Dataset.from_tensor_slices((t, y.astype("float32")) if y is not None else t)
    if shuffle: ds = ds.shuffle(len(X), seed=SEED)
    return ds.batch(32).prefetch(1)

history = model.fit(make_ds(X_train, y_train, True), validation_data=make_ds(X_val, y_val), epochs=30,
                    class_weight=class_weight, callbacks=cb, verbose=2)

# 6. Gráficas de entrenamiento -----------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(17, 4))
for a, m, t in zip(ax, ["loss", "accuracy", "recall"], ["Pérdida", "Exactitud", "Recall"]):
    a.plot(history.history[m], label="train"); a.plot(history.history[f"val_{m}"], label="val")
    a.set_title(t); a.set_xlabel("Época"); a.legend()
plt.tight_layout(); plt.savefig("reports/training_curves.png", dpi=120); plt.close()

# 7. Evaluación en TEST --------------------------------------------------
proba = model.predict(make_ds(X_test), verbose=0).ravel()
pred = (proba >= THRESHOLD).astype(int)
metrics = {"accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred),
           "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred), "roc_auc": roc_auc_score(y_test, proba)}
print(json.dumps(metrics, indent=2)); print(classification_report(y_test, pred, target_names=["ham", "spam"]))
json.dump({k: round(float(v), 4) for k, v in metrics.items()}, open("reports/metrics.json", "w"), indent=2)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues", ax=ax[0],
            xticklabels=["ham", "spam"], yticklabels=["ham", "spam"]); ax[0].set_title("Matriz de confusión (test)")
ax[0].set_xlabel("Predicho"); ax[0].set_ylabel("Real")
fpr, tpr, _ = roc_curve(y_test, proba); ax[1].plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.3f}"); ax[1].plot([0, 1], [0, 1], "--")
ax[1].set_title("Curva ROC"); ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR"); ax[1].legend()
plt.tight_layout(); plt.savefig("reports/evaluation.png", dpi=120); plt.close()

# 8. Predicciones de prueba (>=10) --------------------------------------
res = pd.DataFrame({"mensaje": raw_test.values, "real": np.where(y_test == 1, "spam", "ham"),
                    "prob_spam": proba.round(4), "predicho": np.where(pred == 1, "spam", "ham")})
res["correcto"] = res["real"] == res["predicho"]
res.to_csv("reports/predicciones_test.csv", index=False)
sample = pd.concat([res[res.real == "spam"].sample(7, random_state=SEED), res[res.real == "ham"].sample(8, random_state=SEED)])
pd.set_option("display.max_colwidth", 70, "display.width", 200)
print(sample.to_string(index=False))

# 9. Guardado ------------------------------------------------------------
model.save(MODEL_PATH); print("Modelo guardado en", MODEL_PATH)
