# Mercado Laboral Colombiano: pronóstico del desempleo con machine learning

Proyecto final del curso de Machine Learning (Universidad del Norte) de **Andrés Parejo, Santiago Hurtado y Juan Marín**.
Pronostica la tasa de desempleo nacional de Colombia del mes siguiente. Compara las 140 combinaciones de modelos,
técnicas de balanceo y optimizadores del segundo entregable y propone un modelo original, **ALEE** (agregación local
entrópica de expertos), que se evalúa frente a los ocho modelos de regresión del curso y LightGBM en Colombia y en
ocho países más.

## Qué hay en la carpeta

| Ruta | Contenido |
|---|---|
| `paper/` | Paper en LaTeX (`EDA_Mercado_Laboral_Colombiano_ALEE.tex`), con la estructura de los artículos del curso, sus figuras y tablas, y `EDA_ALEE_Overleaf.zip` para compilarlo en Overleaf. El PDF `_version_anterior` es la versión previa a la reestructuración. |
| `docs/revision_sistematica/` | Registro de la revisión sistemática en Scopus. |
| `iniciar_dashboard.bat` | Abre el dashboard en el navegador (Windows). |
| `app.py`, `pages/`, `assets/`, `labor/`, `vendor/` | Dashboard **LABOR INSIGHTS** en Dash, con seis secciones: Overview, Exploratory Analytics, Model Intelligence, Unemployment Forecast, Global Intelligence e Insights & Conclusions. Incluye el globo de la portada, la introducción animada, el pronóstico a un mes con intervalo conforme del 90 % en los nueve países y el informe PDF por país (`/reporte.pdf?pais=<id>`). `labor/engine.py` es la única puerta al modelo. Three.js, GSAP, Bootstrap y el mapa se sirven desde `vendor/`, así que funciona sin internet. |
| `artifacts/paises/`, `data/paises/` | ALEE ajustado con la serie de cada país, series y predicciones fuera de muestra del dashboard (`scripts/construir_paises.py` los regenera). `data/paises/colombia_posterior.csv` son las cifras del DANE de enero a septiembre de 2026, solo como contraste. |
| `propuesta_diseno/` | Versión del dashboard con diseño de artículo académico. Usa las mismas seis páginas de `pages/` sin modificarlas, con otra hoja de estilos (`estilo.css`), otra paleta de gráficos (`tema.py`), navegación en español y tres paneles nuevos (`extras.py`): anatomía del pronóstico, bitácora de pronósticos y mapa de estados. Se abre con `.venv\Scripts\python propuesta_diseno\app_propuesta.py` en <http://127.0.0.1:8051>. |
| `legacy/` | Interfaz anterior del dashboard (Contexto, EDA, Modelos, Resultados y Conclusiones) y sus pruebas, solo como referencia. |
| `src/` | Código del proyecto: datos, modelos, validación anidada, búsqueda de hiperparámetros, pruebas estadísticas, modelo original (`original_model.py`), artefactos y dashboard. |
| `artifacts/` | Modelos finales serializados con `joblib` (`model.joblib`, `metadata.json` y `roundtrip.npz` por modelo), resultados del paper (`resultados/`) y de la validación confirmatoria (`confirmatorio/`, `adicionales/`). |
| `data/` | `MLC.xlsx` (DANE, vía Banco de la República), series de los países adicionales (`adicionales/`) y datos precalculados del dashboard (`dashboard/`). |
| `experiments/` | Protocolo pre-registrado (`protocolo.md`), scripts de los experimentos, tabla maestra de las 140 combinaciones y registros de ejecución (`logs/`). |
| `book/` | Jupyter Book con los 11 capítulos de la guía. |
| `docs/` | Registro de las iteraciones de diseño, diseño del modelo original, guion de la sustentación (`ponencia/`) y código del dashboard del primer entregable (`dash_antiguo/`, solo como referencia). |
| `tests/` | Pruebas de los artefactos, del modelo original y del dashboard (`test_labor.py`). |
| `MODEL_CARD.md`, `DATA_STATEMENT.md` | Ficha del modelo y declaración de los datos. |
| `historial_claude/` | Historial de las sesiones de trabajo con Claude. |

## Cómo ejecutarlo

Requiere Python 3.9 (con el que se crearon los artefactos).

```bash
python -m venv .venv
.venv\Scripts\activate          # en Linux o macOS: source .venv/bin/activate
pip install -r requirements.txt
python app.py                   # dashboard en http://127.0.0.1:8050
python -m pytest                # pruebas
```

En Windows también sirve `iniciar_dashboard.bat`. El asistente está oculto por defecto: se activa con `LI_CHATBOT=1`
y usa la API de Groq, que necesita la variable de entorno `GROQ_API_KEY`.

### Regenerar los resultados

```bash
python -m src.runner full                       # las 140 combinaciones del segundo entregable (cerca de una hora)
python -m src.artifacts                         # mejores combinaciones, reentrenamiento y serialización
python -m src.artifacts --original --curso      # artefactos de ALEE, regresión bayesiana, SVR (RBF) y LightGBM
python experiments/reproducir_todo.py           # validación confirmatoria, conjuntos adicionales y tablas del paper
python scripts/build_dashboard_data.py          # predicciones, intervalos y valores de Shapley del dashboard
jupyter-book build book/                        # el libro
```

`experiments/reproducir_todo.py` y el paper se generaron con el entorno de `requirements-paper.txt` (Python 3.13).

## Despliegue en Render

`render.yaml` crea el servicio con `requirements-dash.txt` y `gunicorn app:server`. La clave `GROQ_API_KEY` se escribe
a mano en el panel de Render. Las pruebas corren en GitHub Actions (`.github/workflows/tests.yml`).

## Resultados principales

- ALEE tiene el menor RMSE en los nueve conjuntos (Colombia, Estados Unidos, España, México, Canadá, Australia, Japón,
  Alemania y Corea del Sur), con entre 21 % y 33 % menos error que el mejor modelo de regresión del curso.
- Diebold-Mariano-HLN y Giacomini-White con Holm rechazan la igualdad en las 81 comparaciones; entre conjuntos, el
  Wilcoxon con Holm da p = 0.035 frente a cada modelo.
- La ventaja viene sobre todo de los expertos (persistencia, núcleo del cambio y cambio estacional); la localización
  por estado solo aporta de forma significativa en México y Japón.

El detalle, las limitaciones y la separación entre lo pre-registrado y lo agregado después están en el paper y en
`experiments/protocolo.md`.
