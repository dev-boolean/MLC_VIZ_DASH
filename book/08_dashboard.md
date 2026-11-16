# Dashboard y despliegue

El dashboard lleva los resultados del proyecto a una aplicación web que cualquier persona puede usar sin instalar nada. Se construyó con Dash y `dash-bootstrap-components`, con cinco páginas (`use_pages=True`): Contexto, EDA, Modelos, Resultados y Conclusiones. Conserva el diseño del dashboard del primer entregable (MLC_VIZ_DASH): el video de entrada con el logo, el encabezado con los autores, las pestañas y un asistente conversacional.

**URL pública:** pendiente del despliegue en Render. Para verlo localmente: `python app.py` (o `iniciar_dashboard.bat` en Windows) y abrir <http://127.0.0.1:8050>.

## Arquitectura

| Ruta | Contenido |
|---|---|
| `app.py` | Crea la aplicación, el video de entrada, el encabezado y las pestañas, registra el asistente y expone `server` para `gunicorn` |
| `pages/` | Una página por sección (`contexto.py`, `eda.py`, `modelos.py`, `resultados.py` y `conclusiones.py`), con sus propios *callbacks* |
| `src/dash_data.py` | Carga única de datos, artefactos y resultados (`functools.lru_cache`) |
| `src/dash_plots.py` | Funciones que construyen las figuras, con una plantilla y una paleta comunes |
| `src/explain.py` | Valores de Shapley por permutaciones, intervalos conformes, ACF y PACF, sin dependencias pesadas |
| `src/chatbot.py`, `assets/chatbot.js` | Asistente: el servidor arma el contexto con los datos y los resultados del proyecto y consulta la API de Groq |
| `assets/` | Estilos (`base_viz_dash.css` del primer entregable y `style.css`), video y animación de entrada |
| `scripts/build_dashboard_data.py` | Precalcula las predicciones fuera de muestra, el MCS, Friedman, los intervalos y los valores de Shapley globales |
| `artifacts/` | Modelos serializados con `metadata.json` (capítulo 3) y tablas de resultados del paper (capítulo 7) |

El dashboard **nunca entrena modelos ni corre búsquedas de hiperparámetros**: solo carga artefactos y archivos precalculados.

## Secciones

```{figure} images/dash_contexto.jpg
:width: 90%
:alt: Encabezado del dashboard con el logo, los autores y las pestañas, y la portada de Contexto con la cifra de desempleo más reciente

Contexto: el problema, la cifra más reciente con su fuente, la pregunta de investigación, las definiciones de los indicadores, los datos con su diccionario, la metodología y las limitaciones.
```

```{figure} images/dash_eda.jpg
:width: 90%
:alt: Página de EDA con los filtros de fecha, años y cobertura y la serie de las tres tasas del mercado laboral

EDA: los filtros de fechas, años y cobertura actualizan todos los gráficos, y cada uno lleva una frase de interpretación calculada con los datos filtrados.
```

```{figure} images/dash_modelos.jpg
:width: 90%
:alt: Página de Modelos con el Model Confidence Set y la sección del modelo original ALEE con su pseudocódigo

Modelos: la tabla de líderes, el diagnóstico y la interpretabilidad de cada modelo, la comparación estadística y la sección del modelo original.
```

```{figure} images/dash_resultados.jpg
:width: 90%
:alt: Página de Resultados con tres indicadores y la tabla del RMSE de los diez modelos en los nueve conjuntos

Resultados: el desempeño de ALEE frente a los modelos del curso y LightGBM en los nueve conjuntos, con las pruebas de significancia.
```

Lo que contiene cada página:

- **Modelos.**
  - La tabla de líderes con la mejor combinación de cada modelo del segundo entregable, ALEE y los modelos que agregó el paper.
  - El diagnóstico del modelo elegido. En regresión: observado frente a pronosticado, residuos, histograma y ACF. En clasificación: matriz de confusión, curvas ROC y de precisión y exhaustividad, y diagrama de calibración.
  - La importancia global con valores de Shapley, el Model Confidence Set y el diagrama de diferencia crítica.
  - La sección de ALEE: idea, pseudocódigo, curva de convergencia, ablación en Colombia, costo computacional y ficha del modelo.
- **Resultados.**
  - La tabla de las cinco métricas en los nueve conjuntos y un mapa del error relativo a ALEE.
  - Las pruebas de Diebold-Mariano-HLN, Giacomini-White y la prueba bayesiana con ROPE por conjunto; el diagrama de diferencia crítica y las pruebas de Wilcoxon y Conover entre conjuntos.
  - La ablación y el error por régimen de volatilidad de cada conjunto.
- **Conclusiones.** Los hallazgos sobre los datos y los modelos, las limitaciones y la bibliografía del paper.

## Decisiones de diseño

- **Valores de Shapley propios.** `shap` y sus dependencias (`numba`, `llvmlite`) son pesados para los 512 MB del plan gratuito de Render. `src/explain.py` estima los valores por muestreo de permutaciones {cite}`strumbelj2014`. Sirve para cualquier modelo, incluido ALEE, y cumple de forma exacta la propiedad de eficiencia. Una prueba (`tests/test_dashboard.py`) comprueba que en un modelo lineal da el valor exacto.
- **Asistente.** El contexto que recibe el modelo de lenguaje se arma con los datos y las tablas del proyecto, de modo que sus cifras coinciden con las del dashboard. Las instrucciones le prohíben inventar cifras y responder temas ajenos al proyecto. Necesita la variable de entorno `GROQ_API_KEY`; sin ella responde con un aviso y el resto de la aplicación funciona igual.
- **Paleta y accesibilidad.** Cinco colores, y el color nunca es la única señal: las series también se distinguen por tipo de línea o de marcador, y las barras con diferencias significativas llevan rayas. Todos los gráficos tienen título, ejes con unidades y texto en el *hover*, y las imágenes tienen texto alternativo.
- **Rendimiento.** Los datos y los modelos se cargan una sola vez al iniciar el servidor. El video de entrada solo se reproduce al abrir la aplicación: cambiar de pestaña no recarga la página.
- **Diseño adaptable.** Las páginas usan la rejilla de Bootstrap (`dbc.Row` y `dbc.Col`) y se revisaron en escritorio y en tableta.

```{figure} images/dash_tableta.jpg
:width: 45%
:alt: Página de EDA en una pantalla de tableta, con los filtros y la serie apilados

La página de EDA en una tableta de 768 píxeles de ancho.
```

## Ficha del modelo

```{include} ../MODEL_CARD.md
:start-line: 1
```

## Guía de despliegue en Render

1. Verificar localmente: `python app.py`, y que `app.server` está expuesto.
2. Subir el repositorio a GitHub. `requirements-dash.txt` fija versiones exactas e incluye `gunicorn`. Para ahorrar memoria usa `xgboost-cpu`, que evita las librerías de NVIDIA.
3. En Render, crear un *Blueprint* o un *Web Service* desde el repositorio. `render.yaml` define la construcción (`pip install -r requirements-dash.txt`), el arranque (`gunicorn app:server --workers 1 --threads 4`), la ruta de salud (`/health`) y Python 3.11.9.
4. Escribir la clave `GROQ_API_KEY` en las variables de entorno del servicio, en el panel de Render.
5. Revisar los registros de construcción y comprobar las cinco páginas y sus *callbacks* en la URL pública.
6. GitHub Actions (`.github/workflows/tests.yml`) corre todas las pruebas con la misma versión de Python que Render en cada *push*. Así se verifica que los artefactos, creados en Windows con Python 3.9, cargan en Linux.

El plan gratuito suspende el servicio cuando no tiene actividad: conviene abrir la URL unos minutos antes de la sustentación.
