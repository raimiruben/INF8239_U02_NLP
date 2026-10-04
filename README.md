# INF-8239 · Unidad 02 · Proyecto NLP

Autor académico: Edwin Ramón José Nolasco

Proyecto base para LAB04–LAB06. No sustituya la comprensión por ejecución mecánica.

## Inicio rápido

```bash
uv python install 3.12
uv sync
uv run pytest -q
uv run python scripts/prepare_sentiment_data.py
uv run python scripts/audit_data.py
```

Copie `.env.example` como `.env` y configure el dataset aprobado.

## Dataset

Complete `docs/DATASET_CARD.md`, el diccionario, la licencia y el procedimiento de obtención.
El archivo `data/sample/demo_text.csv` solamente comprueba la arquitectura.

## Entrenamiento

```bash
uv run python scripts/train_text.py
uv run streamlit run app/streamlit_app.py
```

## Interpretación

Toda conclusión debe separar observación, evidencia, interpretación y decisión.

## Correcciones aplicadas

**Selección de modelo en `train_text.py` (LAB05).** El script original tenía
`selected = models["logistic"]` fijo, por lo que la matriz de confusión, el
análisis de errores y el modelo guardado (`models/text_model.joblib`)
siempre correspondían a la regresión logística, sin importar cuál de los
tres modelos obtuviera mejor F1 macro. En esta corrida, `naive_bayes`
superó a `logistic` (F1 macro 0.822 vs 0.779), por lo que se corrigió la
selección para que sea dinámica:

```python
best_name = max(predictions, key=lambda name: f1_score(y_test, predictions[name], average="macro"))
selected = models[best_name]
selected_pred = predictions[best_name]
```

Con este cambio, los artefactos generados (`reports/confusion_text.png`,
`reports/error_analysis.csv`, `models/text_model.joblib`) corresponden
siempre al modelo con mejor desempeño real, no a uno fijo de antemano.
