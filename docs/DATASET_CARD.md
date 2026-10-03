# Dataset Card

## Identificación

- Nombre: Sentiment Labelled Sentences
- Fuente original: UCI Machine Learning Repository 
        https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences
- Responsable: Compilado por D. Kotzias et al. (2015); documentado para este laboratorio por Raimi De Jesús
- Versión o fecha: Descargado el 2026-10-01 (hash SHA-256: b4b252cb51b2e24cf3ad87c69297bc3cbc8cf30668c85162dd5b7b22e26deef3)
- Licencia: CC BY 4.0
- Idioma: Inglés

## Propósito y variable objetivo

Clasificación binaria de sentimiento en reseñas cortas de texto. La variable
objetivo es `label` (0 = sentimiento negativo, 1 = sentimiento positivo).
El dataset combina tres dominios distintos —reseñas de películas (IMDb),
productos (Amazon) y restaurantes (Yelp)— con 1000 oraciones por dominio
(500 positivas + 500 negativas cada uno), para un total de 3000 oraciones
perfectamente balanceadas.

## Diccionario de datos

| Columna | Tipo | Descripción | Valores o unidad |
|---|---|---|---|
| text | string | Oración de reseña en inglés | Texto libre, 7 a 479 caracteres |
| label | int64 | Sentimiento de la oración | 0 = negativo, 1 = positivo |
| source | string | Dominio de origen de la reseña | imdb / amazon / yelp |

## Ejemplos (muestra aleatoria, semilla=42)
| text | label | source |
|---|---|---|
| Avoid at ALL costs! | 0 (negativo) | imdb |
| Garbo, who showed right off the bat that her talents could carry over from the silent era (I wanted to see some of her silent work, but Netflix doesn't seem to be stocking them. | 1 (positivo) | imdb |
| You will leave the theater wanting to go out and dance under the stars. | 1 (positivo) | imdb |
| O my gosh the best phone I have ever had. | 1 (positivo) | amazon |
| I would not recommend this place. | 0 (negativo) | yelp |

## Procedimiento de obtención

1. Se descargó el archivo `sentiment+labelled+sentences.zip` directamente
   desde UCI mediante `scripts/prepare_sentiment_data.py`.
2. El script extrae los tres archivos de texto (`imdb_labelled.txt`,
   `amazon_cells_labelled.txt`, `yelp_labelled.txt`), cada uno separado por
   tabulador, y los combina en un único `data/raw/dataset.csv`.
3. Se usó `quoting=csv.QUOTE_NONE` al leer los archivos para evitar que las
   comillas dentro del texto (parte normal del contenido de las reseñas)
   fueran mal interpretadas como delimitadores CSV — un primer intento sin
   este ajuste perdió 252 de las 3000 filas esperadas.
4. `scripts/download_data.py` (modo `local`) y `scripts/audit_data.py`
   confirman la integridad del archivo final (ver hash SHA-256 arriba).

## Calidad observada

- 3000 filas, 3 columnas. Sin valores nulos en `text` ni en `label`.
- 17 duplicados de texto (0.57% del total) — se conservan, documentados como
  limitación, dado que no afectan significativamente el balance de clases.
- Distribución de clases perfectamente balanceada: 50% negativo / 50% positivo.
- Longitud de texto: mínimo 7 caracteres, mediana 55.5, percentil 90 en
  121.1, percentil 99 en 215.0, máximo 479. No se observan textos vacíos ni
  truncados de forma sospechosa.

## Población cubierta y excluida

Cubre opiniones cortas en inglés de tres dominios de consumo (cine,
productos electrónicos/varios de Amazon, restaurantes), escritas por
usuarios anónimos de plataformas públicas de reseñas alrededor de 2015.
Excluye: idiomas distintos al inglés, reseñas largas o con lenguaje técnico
especializado, sentimiento neutro o mixto (el dataset solo etiqueta
positivo/negativo, sin clase neutral), y cualquier metadato del usuario
(no incluye edad, ubicación, ni identidad del autor).

## Riesgos, sesgos y usos prohibidos

- Riesgo de privacidad: bajo — las oraciones son reseñas públicas ya
  publicadas, sin identificadores personales.
- Sesgo de dominio: el balance perfecto es artificial (curado por los
  autores originales), no refleja la proporción real de sentimiento positivo
  vs. negativo en reseñas "naturales" de estas plataformas.
- Sesgo temporal: el lenguaje y las referencias (productos, restaurantes)
  corresponden a ~2015 y anteriores; puede no generalizar a lenguaje o
  jerga actual.
- Uso prohibido: no debe usarse para perfilar o tomar decisiones sobre
  autores individuales (el dataset no fue recolectado con consentimiento
  explícito para ese fin), ni presentarse como representativo de opinión
  pública general más allá de estos tres dominios específicos.
- Pérdida de contexto: al ser oraciones individuales extraídas de reseñas
  más largas, algunas pierden el contexto necesario para interpretar su
  sentimiento de forma aislada (ej. una oración neutral en apariencia puede
  estar etiquetada como positiva por el tono general de la reseña completa
  de la que proviene) — esto es una limitación real del dataset, no un
  error de etiquetado, pero debe tenerse en cuenta al interpretar errores
  del modelo en labs posteriores.


## Cierre interpretativo

**Resultado principal:** Se localizó, aprobó y auditó el corpus Sentiment
Labelled Sentences (UCI) para clasificación de sentimiento: 3000 oraciones
en inglés, etiqueta binaria positivo/negativo, perfectamente balanceada
(50/50), combinando tres dominios (IMDb, Amazon, Yelp; 1000 oraciones cada
uno).

**Evidencia de calidad y procedencia:** Fuente oficial UCI con licencia
CC BY 4.0, procedencia documentada (Kotzias et al., 2015). Hash SHA-256
registrado para trazabilidad de la descarga. Auditoría: 0 valores nulos,
17 duplicados (0.57%, documentados), longitud de texto razonable (mediana
55.5 caracteres, sin truncamientos sospechosos).

**Riesgo o sesgo identificado:** El balance perfecto 50/50 es artificial
(curado por los autores originales), no refleja la proporción real de
opinión positiva/negativa en estas plataformas. Además, al tratarse de
oraciones individuales extraídas de reseñas más largas, algunas pierden el
contexto necesario para interpretar su sentimiento de forma aislada.

**Decisión de aprobación o rechazo:** Aprobado. Cumple los seis criterios
del manual (pertinencia, licencia, representatividad, calidad,
reproducibilidad, riesgo) y el contrato de datos automatizado
(`test_data_contract.py`) pasa 3/3 pruebas. Queda listo para los
laboratorios siguientes de embeddings y entrenamiento de texto.

**Limitación que debe comunicarse:** El lenguaje y las referencias del
corpus corresponden a ~2015 o antes, por lo que puede no generalizar bien a
texto o jerga actual. No incluye metadatos de usuario ni consentimiento
explícito de los autores originales para fines de perfilado individual.

**Siguiente verificación:** Antes de entrenar el primer modelo de texto,
revisar si alguno de los 17 duplicados detectados cae simultáneamente en
el conjunto de entrenamiento y en el de prueba (fuga de datos), y evaluar
si el balance artificial 50/50 sigue siendo representativo una vez
definida la partición real de train/test.

