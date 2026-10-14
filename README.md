# 📩 Detección de spam en SMS con RNN / LSTM / GRU

Proyecto de Deep Learning para clasificar mensajes SMS como **spam** o **ham** (legítimo)
usando redes recurrentes (SimpleRNN, LSTM, BiLSTM y GRU) en TensorFlow/Keras, con una
aplicación web en Streamlit.

## 1. Dataset

- **SMS Spam Collection** (UCI Machine Learning Repository): ~5.574 mensajes en inglés,
  etiquetados `ham` / `spam`. ~13 % son spam (clases desbalanceadas).
- Fuente: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- Es un problema **secuencial**: el significado depende del orden y contexto de las palabras.
- Se descarga automáticamente al entrenar. Alternativa: coloca `spam.csv` (versión Kaggle) en `data/`.

## 2. Estructura

```
proyecto-dl/
├── notebook_experimentacion.ipynb   # Google Colab: EDA, experimentos, evaluación
├── train.py                         # Entrenamiento por línea de comandos
├── predict.py                       # Carga del modelo y predicción
├── app.py                           # Aplicación Streamlit
├── utils.py                         # Limpieza de texto, carga de datos, semillas
├── modelo/                          # modelo_spam.keras + config.json
├── reports/                         # gráficas, métricas y predicciones de test
├── data/                            # dataset (se descarga solo)
├── requirements.txt
└── README.md
```

## 3. Metodología

1. **EDA**: balance de clases, longitud de mensajes, palabras frecuentes por clase, duplicados.
2. **Limpieza**: minúsculas, normalización de URLs, emails, números largos y símbolos de
   moneda a tokens especiales; eliminación de duplicados. La puntuación se elimina dentro del modelo.
3. **División**: 70 % entrenamiento / 15 % validación / 15 % prueba, **estratificada**.
   El vocabulario se ajusta solo con entrenamiento (sin fuga de datos).
4. **Modelo**: `TextVectorization → Embedding(mask_zero) → RNN/LSTM/GRU → Dropout → Dense → Sigmoid`.
   La vectorización va **dentro** del modelo guardado, por lo que el modelo recibe texto.
5. **Entrenamiento**: Adam (1e-3), `binary_crossentropy`, `class_weight` balanceado,
   `EarlyStopping` (patience=3) y `ModelCheckpoint`.
6. **Evaluación**: precision, recall, F1, ROC-AUC y matriz de confusión
   (la exactitud sola es engañosa con clases desbalanceadas).

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

Salida: `modelo/modelo_spam.keras`, `modelo/config.json`, y en `reports/` las curvas de
entrenamiento, matriz de confusión, curva ROC, `metrics.json` y `predicciones_test.csv`.

> Alternativa: ejecuta `notebook_experimentacion.ipynb` en Google Colab (GPU opcional),
> descarga el `.zip` final y descomprímelo dentro de `proyecto-dl/` para obtener `modelo/` y `reports/`.

## 6. Predecir por consola

```bash
python predict.py "WINNER!! You won a £1000 prize, call 09061701461 now"
```

## 7. Ejecutar la aplicación Streamlit

```bash
streamlit run app.py
```

Abre http://localhost:8501, escribe un mensaje **en inglés** o usa los ejemplos, y pulsa **Predecir**.

## 8. Resultados

Pega aquí tus métricas tras entrenar (están en `reports/metrics.json`):

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|----------|-----------|--------|----|---------|
| (completa con tus resultados) | | | | | |

## 9. Limitaciones y trabajo futuro

- Entrenado solo con SMS en inglés; no generaliza a otros idiomas.
- Dataset pequeño y de una época concreta: el spam moderno puede diferir.
- Mejoras: embeddings preentrenados (GloVe), ajuste del umbral, comparar con un Transformer.
