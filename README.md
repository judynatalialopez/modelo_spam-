# 📩 Detector de Spam en SMS con Deep Learning (Bi-GRU)

Proyecto de Deep Learning con **redes recurrentes (GRU bidireccional)** para clasificar mensajes SMS como **spam** o **ham** (legítimo), con aplicación web en **Streamlit**.

## Resultados (conjunto de test, 776 mensajes)

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|:-:|:-:|:-:|:-:|:-:|
| 0.983 | 0.921 | 0.949 | 0.935 | 0.994 |

## Dataset
**SMS Spam Collection** (Almeida & Hidalgo, UCI): 5 572 SMS en inglés (4 825 ham / 747 spam). Tras eliminar 403 duplicados quedan 5 167 mensajes (12.6 % spam → clases desbalanceadas, por lo que se evalúa con precisión, recall, F1 y AUC).
Fuente usada: `https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv` (copia local en `data/sms.tsv`).

## Estructura
```
spam_project/
├── notebook_spam_gru.ipynb   # Notebook de experimentación (Google Colab)
├── train.py                  # Entrenamiento + EDA + evaluación + gráficas
├── predict.py                # Predicción por línea de comandos
├── spam_utils.py             # Limpieza de texto compartida (train/predict/app)
├── app.py                    # Aplicación Streamlit
├── models/spam_gru_model.keras   # Modelo entrenado
├── data/sms.tsv              # Dataset
├── reports/                  # Gráficas (EDA, curvas, matriz de confusión), métricas y predicciones de test
├── requirements.txt
└── README.md
```

## Pipeline
1. **EDA:** distribución de clases, longitud de mensajes, palabras más frecuentes.
2. **Limpieza:** minúsculas, eliminación de duplicados, normalización de URLs, emails, monedas, teléfonos y números a *tokens*, eliminación de puntuación.
3. **Split estratificado:** 70 % train / 15 % validación / 15 % test.
4. **Modelo:** `TextVectorization → Embedding(64) → Bidirectional(GRU 32) → Dropout → Dense(32) → Dropout → Dense(1, sigmoid)` (≈ 533 k parámetros). El vectorizador está **dentro** del modelo, que recibe texto.
5. **Entrenamiento:** Adam, `binary_crossentropy`, *class weights*, `EarlyStopping` sobre `val_loss`.
6. **Evaluación:** accuracy, precision, recall, F1, ROC-AUC, matriz de confusión, 15 predicciones de prueba (`reports/predicciones_test.csv` contiene todas).

## Ejecución local

```bash
# 1. (Opcional) entorno virtual — Python 3.10–3.12
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. (Opcional) reentrenar el modelo (~1 min en CPU). Ya se incluye uno entrenado.
python train.py

# 4. Predicción por consola
python predict.py "WINNER!! You won a £1000 prize. Call 09061701461 now"

# 5. Lanzar la app web
streamlit run app.py
```
La app se abre en `http://localhost:8501`.

### Funciones de la app
- Escribir un SMS (o cargar un ejemplo) y ver la probabilidad de spam y el veredicto.
- Umbral de decisión ajustable en la barra lateral.
- Métricas del modelo en test.
- Clasificación por lotes subiendo un CSV con columna `mensaje`, `text` o `message`.

> ⚠️ El modelo fue entrenado con mensajes **en inglés**; en otros idiomas las predicciones no son fiables.

## Google Colab
1. Sube `notebook_spam_gru.ipynb` a Colab (Archivo → Subir notebook).
2. Ejecuta *Entorno de ejecución → Ejecutar todo*. El notebook descarga el dataset por sí mismo.
3. Al final, descarga el modelo (`models/spam_gru_model.keras`) descomentando la línea de `files.download`.

## Solución de problemas
- **Error al cargar el modelo** (versión de Keras/TensorFlow distinta): ejecuta `python train.py` para regenerarlo con tu versión.
- **`ModuleNotFoundError: spam_utils`**: ejecuta los comandos desde la carpeta raíz del proyecto.
