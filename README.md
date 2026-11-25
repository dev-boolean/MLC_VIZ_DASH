# LABOR INSIGHTS

**Global Unemployment Intelligence.** Plataforma que pronostica la tasa de desempleo del mes siguiente en nueve países con
**ALEE** (agregación local entrópica de expertos), el modelo original de Andrés Parejo, Santiago Hurtado y Juan Marín
(Universidad del Norte, proyecto de Machine Learning).

## Ejecutar

Requiere Python 3.9.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Abra <http://127.0.0.1:8050>. En Windows también sirve `iniciar_dashboard.bat`. La portada 3D, las animaciones y las fuentes se
cargan de CDN (cdnjs, jsDelivr y Google Fonts); sin internet, los datos, los pronósticos y los gráficos siguen funcionando, pero se pierde parte del estilo (Bootstrap también viene de CDN).
Las animaciones están siempre activas, aunque el sistema tenga «reducir movimiento».
Al abrir la portada se muestra una introducción de unos 12 s (`assets/li_intro.js`): la tasa de desempleo observada de Colombia
se dibuja mes a mes de enero de 2001 a diciembre de 2025 (la misma serie del EDA, incrustada en la página por `app.py`) y el último
punto viaja hasta Colombia en el globo. Se ve una vez por pestaña; se salta con el botón o con Escape y se repite desde la barra lateral.
El asistente está oculto por defecto (no se ha probado): se activa con `LI_CHATBOT=1` y necesita la variable de entorno `GROQ_API_KEY`. Three.js, GSAP, Bootstrap y el mapa del globo se sirven desde `vendor/`, así que la plataforma funciona sin internet (salvo las fuentes de Google, que tienen respaldo).

Pruebas: `pip install pytest` y `python -m pytest tests -q`.

## Estructura

| Ruta | Contenido |
|---|---|
| `app.py` | Aplicación, navegación de seis secciones, selector de país, `/api/globo` y `/reporte.pdf?pais=<id>` |
| `pages/` | Las seis secciones: Overview, Exploratory Analytics (Colombia), Model Intelligence, Unemployment Forecast, Global Intelligence e Insights & Conclusions |
| `labor/catalog.py` | Países, fuentes y formato en español |
| `labor/engine.py` | Única puerta al modelo: series, variables, pronóstico, intervalo conforme, validación y métricas |
| `labor/stats.py` | Estadística descriptiva |
| `labor/figs.py`, `labor/theme.py` | Figuras y plantilla de Plotly, con la semántica de color de la plataforma |
| `labor/ui.py` | Componentes visuales |
| `labor/report.py` | Informe académico en PDF (fpdf2 y matplotlib) con los datos de la plataforma |
| `assets/` | `labor.css` (sistema de diseño académico), `alee_arquitectura.svg`, `li_globe.js` (globo de la portada, Three.js), `li_motion.js` (GSAP), `chatbot.js` |
| `src/` | Código del proyecto de investigación. `original_model.py` (ALEE) no se modificó |
| `artifacts/reg_ALEE` | ALEE de Colombia, congelado |
| `artifacts/paises/`, `data/paises/` | ALEE ajustado con la serie de cada país, series y predicciones fuera de muestra |
| `scripts/construir_paises.py` | Regenera esas dos carpetas desde el proyecto completo y comprueba que reproduce lo guardado |
| `legacy/` | Interfaz anterior, solo como referencia |

## Reglas científicas

- El pronóstico es de un mes. La interfaz no lo extiende.
- Colombia usa 17 variables; los demás países, 14 (no hay indicadores de las 13 áreas).
- Cada país tiene su propio ALEE, ajustado con su serie: generaliza el método, no un modelo entrenado en Colombia.
- Los intervalos del 90 % son de predicción conforme con los errores fuera de muestra; su cobertura se mide hacia adelante.
  En el historial, el intervalo de cada mes se calcula solo con los errores anteriores (`engine.forward_halfwidths`).
- Cada cifra se etiqueta como observada, pronosticada o calculada.
- `data/paises/colombia_posterior.csv` son datos de 2026 recopilados por el equipo. Solo se muestran como contraste.
- Octubre de 2025 en Estados Unidos es una interpolación lineal (no se publicó).

## Despliegue

`gunicorn app:server` (ver `Procfile`). Ruta de salud: `/health`.
