# 💬 Análisis de sentimiento en español con RNN / LSTM / GRU

Proyecto de Deep Learning que clasifica textos en español como **positivo**, **neutral** o **negativo**
usando redes recurrentes (SimpleRNN, LSTM, BiLSTM y GRU) en TensorFlow/Keras, con una app en Streamlit.

## 1. Dataset

- **multilingual-sentiments, config `spanish`** (Hugging Face: `tyqiangz/multilingual-sentiments`, licencia Apache 2.0).
- Textos en español con etiqueta de sentimiento: 0 = positivo, 1 = neutral, 2 = negativo.
- Es un problema **secuencial**: el sentimiento depende del orden y contexto ("no está *mal*" ≠ "está *mal*").
- Se descarga automáticamente (y se guarda en `data/dataset.csv`). Alternativa offline: coloca tu propio
  `data/dataset.csv` con columnas `text,label` (label = 0/1/2 o positivo/neutral/negativo).
- Se unen los splits originales y se re-dividen 70/15/15 estratificados.

## 2. Estructura

```
proyecto-dl/
├── notebook_experimentacion.ipynb   # Google Colab: EDA, experimentos, evaluación
├── train.py                         # Entrenamiento por línea de comandos
├── predict.py                       # Carga del modelo y predicción
├── app.py                           # Aplicación Streamlit
├── utils.py                         # Limpieza de texto, carga de datos, semillas
├── modelo/                          # modelo_sentimiento.keras + config.json
├── reports/                         # gráficas, métricas y predicciones de test
├── data/                            # dataset (se descarga solo)
├── requirements.txt
└── README.md
```

## 3. Metodología

1. **EDA**: balance de clases, longitud de textos, palabras frecuentes por clase, duplicados y nulos.
2. **Limpieza**: minúsculas (conserva tildes y ñ), URLs y @menciones a tokens, reducción de letras repetidas.
   No se eliminan stopwords porque palabras como "no" o "muy" cambian el sentimiento.
3. **División**: 70 % / 15 % / 15 % estratificada; el vocabulario se ajusta solo con entrenamiento.
4. **Modelo**: `TextVectorization → Embedding(mask_zero) → RNN/LSTM/GRU → Dropout → Dense → Softmax(3)`.
   La vectorización va **dentro** del modelo guardado.
5. **Entrenamiento**: Adam (1e-3), `sparse_categorical_crossentropy`, `class_weight` balanceado,
   `EarlyStopping` (patience=3) y `ModelCheckpoint`.
6. **Evaluación**: accuracy, precisión/recall/F1 macro, ROC-AUC uno-contra-resto y matriz de confusión.

## 4. Instalación

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Entrenar el modelo

```bash
python train.py                    # BiLSTM (por defecto)
python train.py --model gru        # opciones: rnn | lstm | gru | bilstm
python train.py --model lstm --epochs 30 --units 128 --dropout 0.5
```

Salida: `modelo/modelo_sentimiento.keras`, `modelo/config.json`, y en `reports/` las curvas de
entrenamiento, matriz de confusión, curva ROC, `metrics.json` y `predicciones_test.csv`.

> Alternativa: ejecuta `notebook_experimentacion.ipynb` en Google Colab, descarga los dos `.zip` finales
> y descomprímelos dentro de `proyecto-dl/` para obtener `modelo/` y `reports/`.

## 6. Predecir por consola

```bash
python predict.py "Me encantó el producto, llegó rápido y funciona perfecto"
```

## 7. Ejecutar la aplicación Streamlit

```bash
streamlit run app.py
```

Abre http://localhost:8501, escribe un texto **en español** o elige un ejemplo, y pulsa **Predecir**.

## 8. Resultados

Pega aquí tus métricas tras entrenar (están en `reports/metrics.json`):

| Modelo | Accuracy | F1 macro | ROC-AUC |
|--------|----------|----------|---------|
| (completa con tus resultados) | | | |

## 9. Limitaciones y trabajo futuro

- Entrenado con reseñas/comentarios; puede rendir peor en otros dominios.
- No detecta bien ironía ni sarcasmo; la clase *neutral* suele ser la más difícil.
- Mejoras: embeddings preentrenados en español, más datos, o un Transformer (BETO).
