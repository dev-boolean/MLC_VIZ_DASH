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
Las animaciones respetan «reducir movimiento» del sistema; se activan en la barra lateral o con `?movimiento=1`.
El asistente necesita la variable de entorno `GROQ_API_KEY`.

Pruebas: `pip install pytest` y `python -m pytest tests -q`.

## Estructura

| Ruta | Contenido |
|---|---|
| `app.py` | Aplicación, navegación, selector de país y `/api/globo` |
| `pages/` | Las siete secciones, cada una con su layout y sus callbacks |
| `labor/catalog.py` | Países, fuentes y formato en español |
| `labor/engine.py` | Única puerta al modelo: series, variables, pronóstico, intervalo, escenarios y métricas |
| `labor/stats.py` | Estadística descriptiva |
| `labor/figs.py`, `labor/theme.py` | Figuras y plantilla de Plotly |
| `labor/ui.py` | Componentes visuales |
| `assets/` | `labor.css` (design system), `li_globe.js` (Three.js), `li_motion.js` (GSAP), `chatbot.js` |
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
- Los escenarios se ejecutan con el modelo y no se interpretan como efectos causales.
- `data/paises/colombia_posterior.csv` son datos de 2026 recopilados por el equipo. Solo se muestran como contraste.
- Octubre de 2025 en Estados Unidos es una interpolación lineal (no se publicó).

## Despliegue

`gunicorn app:server` (ver `Procfile`). Ruta de salud: `/health`.
