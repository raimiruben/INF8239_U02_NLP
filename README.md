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

## Cierre interpretativo — Ejercicio 03

Se entrenaron y compararon tres clasificadores de texto sobre el corpus
Sentiment Labelled Sentences (aprobado y auditado en LAB04): un baseline
(DummyClassifier, F1 macro=0.333), Complement Naive Bayes (F1 macro=0.822)
y una regresión logística (F1 macro=0.779). Ambos modelos reales superan
ampliamente al baseline, confirmando que el corpus contiene señal real de
sentimiento recuperable mediante TF-IDF.

El modelo seleccionado fue Complement Naive Bayes, con F1 macro=0.822 y una
matriz de confusión razonablemente equilibrada entre clases: 304 verdaderos
negativos, 69 falsos positivos, 64 falsos negativos y 309 verdaderos
positivos, para una exactitud global de 82.2%. Esta selección se hizo de
forma automática comparando el F1 macro de los tres modelos, después de
detectar y corregir un error en el script original que fijaba la regresión
logística como modelo elegido sin importar cuál tuviera mejor desempeño
real — un hallazgo importante documentado en el repositorio, ya que sin esa
corrección se habría evaluado y publicado un modelo que no era el mejor
disponible.

La clase 0 (sentimiento negativo) presentó la mayor dificultad, con un
recall de 0.82 frente a 0.83 de la clase positiva. El error más frecuente
fueron 69 reseñas negativas reales clasificadas como positivas, ligeramente
por encima de los 64 errores en la dirección opuesta.

Al categorizar manualmente 25 de los 133 errores del conjunto de prueba, la
causa más frecuente (13 de 25) fue la ambigüedad por sentimiento implícito:
oraciones negativas o positivas que no usan palabras de sentimiento
explícitas y requieren conocimiento de dominio para interpretarse ("the
battery runs down quickly", "it only worked once"). El segundo patrón más
común (8 de 25) fue la negación: construcciones como "not disappointed" o
"wouldn't see this movie again" confunden al modelo porque TF-IDF con
bag-of-words trata "not" y la palabra siguiente como tokens independientes,
perdiendo el efecto de inversión del sentimiento — una limitación conocida
y bien documentada de este tipo de representación.

En cuanto al impacto en contexto: si este modelo se usara para monitorear
reseñas negativas de clientes, los 69 falsos positivos representan el
riesgo más costoso, ya que implican dejar pasar quejas reales sin que
lleguen a revisión humana. Además, al probar el modelo con una frase
positiva en español ("Excelente servicio"), el sistema la clasificó
incorrectamente como negativa, confirmando que el modelo no tiene ninguna
capacidad de generalización fuera del idioma inglés en el que fue
entrenado — algo que la aplicación local de Streamlit sí advierte
explícitamente a los usuarios.

La principal limitación del dataset es su alcance: 3000 oraciones en
inglés de tres dominios específicos (cine, productos, restaurantes),
datadas alrededor de 2015, con un balance de clases perfectamente 50/50
que es artificial y no representa la proporción real de opiniones
positivas y negativas en plataformas reales. Cualquier conclusión debe
limitarse a este alcance.

Antes de considerar un despliegue más allá de esta demostración académica,
seríamos prudentes en: (1) incorporar manejo explícito de negación, por
ejemplo mediante n-gramas o reglas de preprocesamiento, dado que es la
segunda fuente de error más frecuente y tiene una solución técnica conocida;
y (2) restringir o advertir explícitamente el uso del modelo a texto en
inglés, dado que no existe ningún mecanismo de detección de idioma en el
pipeline actual y el modelo falla silenciosamente —sin ninguna señal de baja
confianza— ante texto en otros idiomas.
