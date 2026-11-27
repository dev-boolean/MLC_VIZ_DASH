"""Sección 1: Contexto del problema, los datos y la metodología."""
import dash
import dash_bootstrap_components as dbc
from dash import dash_table, html

from src import config
from src import dash_data as D

dash.register_page(__name__, path="/", name="Contexto", title="Contexto · Mercado Laboral Colombiano")

raw = D.raw()
last = raw.iloc[-1]
peak = raw.loc[raw["tasa_desempleo_nacional"].idxmax()]
first_2025 = raw[raw["fecha"].dt.year == 2025]["tasa_desempleo_nacional"].mean()


def card(title, children, cls=""):
    return dbc.Card([dbc.CardHeader(title), dbc.CardBody(children)], className=f"mlc-card {cls}")


dictionary = [
    {"Variable": label, "Grupo": group, "Descripción": desc, "Nombre en el código": name}
    for name, (label, group, desc) in D.FEATURE_INFO.items()
]

steps = ["ETL", "EDA", "14 modelos base", "4 balanceos × 4 optimizadores", "Validación anidada (ventana expansiva)",
         "Comparación estadística", "Modelo original (ALEE)", "Evaluación en 9 conjuntos", "Dashboard"]

# Indicadores del mercado laboral (definiciones del dashboard del primer entregable).
INDICATORS = [
    ("Tasa de desempleo — Nacional", "TD NAL", "Proporción de personas desocupadas sobre la fuerza laboral a nivel "
     "nacional. Es la variable objetivo del pronóstico."),
    ("Tasa de desempleo — 13 áreas", "TD ÁREA", "Proporción de desocupados sobre la fuerza laboral en las 13 principales "
     "áreas metropolitanas; aproxima la presión laboral urbana."),
    ("Tasa de ocupación — Nacional", "TO NAL", "Relación entre la población ocupada y la población en edad de trabajar; "
     "mide la capacidad de la economía para generar empleo."),
    ("Tasa de ocupación — 13 áreas", "TO ÁREA", "Ocupados sobre población en edad de trabajar en las 13 áreas; mide la "
     "absorción de empleo en el entorno urbano."),
    ("Tasa global de participación — Nacional", "TGP NAL", "Relación entre la fuerza laboral y la población en edad de "
     "trabajar; refleja la disposición a participar en el mercado laboral."),
    ("Tasa global de participación — 13 áreas", "TGP ÁREA", "Fuerza laboral sobre población en edad de trabajar en las "
     "13 áreas metropolitanas."),
]


def indicator(title, badge, text):
    return html.Details([html.Summary([html.Span(title, className="acc-title"), html.Span(badge, className="acc-badge")]),
                         html.Ul([html.Li(text), html.Li("Cuantitativa continua, escala de razón, en porcentaje, mensual.")],
                                 className="acc-list")], className="acc-item")

layout = html.Div([
    html.Div([
        dbc.Row([
            dbc.Col([
                html.H1("¿Cuál será la tasa de desempleo de Colombia el próximo mes?"),
                html.P("Pronóstico a un mes de la tasa de desempleo nacional con 140 combinaciones de modelos de "
                       "machine learning, validación anidada que respeta el tiempo y un modelo original de "
                       "combinación de expertos, ALEE, evaluado también en ocho países.", className="lead"),
            ], md=8),
            dbc.Col([
                html.Div(f"{last['tasa_desempleo_nacional']:.1f} %", className="mlc-kpi"),
                html.Div(f"Desempleo nacional, {D.month_es(last['fecha'])}", className="mlc-kpi-label"),
                html.Div(f"Máximo de la serie: {peak['tasa_desempleo_nacional']:.1f} % en {D.month_es(peak['fecha'])}",
                         className="mlc-kpi-label mt-2"),
                html.Div("Fuente: DANE, Gran Encuesta Integrada de Hogares, series del Banco de la República.",
                         className="mlc-kpi-label mt-2", style={"fontSize": "0.78rem"}),
            ], md=4, className="text-md-end mt-3 mt-md-0"),
        ]),
    ], className="mlc-hero"),

    dbc.Row([
        dbc.Col(card("Por qué importa", [
            html.P("La tasa de desempleo es el indicador laboral que más se sigue en Colombia: orienta la política "
                   "monetaria y fiscal, los programas de empleo y las decisiones de contratación. Un pronóstico "
                   "confiable del mes siguiente permite anticipar cambios antes de que el DANE publique la cifra."),
            html.P(f"La serie muestra por qué es difícil: pasó de cifras cercanas al 15 % a comienzos de los 2000 a "
                   f"{peak['tasa_desempleo_nacional']:.1f} % en {D.month_es(peak['fecha'])} con la pandemia, y a un "
                   f"promedio de {first_2025:.1f} % en 2025 (DANE, vía Banco de la República)."),
        ]), md=6),
        dbc.Col(card("Pregunta de investigación y objetivos", [
            html.P(html.B("¿Puede un modelo de combinación de expertos con pesos que dependen del régimen del mercado "
                          "laboral pronosticar la tasa de desempleo del mes siguiente mejor que los modelos de "
                          "regresión estudiados en clase?")),
            html.Ul([
                html.Li("Comparar 140 combinaciones de modelos, técnicas de balanceo y optimizadores con validación anidada."),
                html.Li("Llevar la mejor combinación de cada modelo a un producto que cualquier persona pueda usar."),
                html.Li("Proponer, implementar y validar con rigor un modelo original (ALEE)."),
                html.Li("Comprobar si su ventaja se mantiene en ocho países con mercados laborales distintos."),
            ]),
        ]), md=6),
    ]),

    card("Indicadores del mercado laboral", [
        html.P("El DANE publica cada mes tres tasas para el total nacional y para las 13 principales áreas "
               "metropolitanas. Toque cada una para ver su definición.", className="mb-2"),
        html.Div([indicator(*i) for i in INDICATORS]),
    ]),

    card("Datos", [
        dbc.Row([
            dbc.Col([
                html.Ul([
                    html.Li([html.B("Fuente: "), "DANE, Gran Encuesta Integrada de Hogares (GEIH); series publicadas por el Banco de la República."]),
                    html.Li([html.B("Periodo: "), f"{D.month_es(raw['fecha'].min())} a {D.month_es(raw['fecha'].max())}, mensual ({len(raw)} meses)."]),
                    html.Li([html.B("Variables originales: "), "tasas de desempleo, ocupación y participación, nacionales y de las 13 áreas metropolitanas (porcentaje)."]),
                    html.Li([html.B("Licencia: "), "datos públicos; los términos de uso del Banco de la República están por verificar (ver DATA_STATEMENT.md)."]),
                    html.Li([html.B("Variable objetivo: "), "tasa de desempleo nacional del mes t+1."]),
                    html.Li([html.B("Tareas: "), "regresión (la tasa) y clasificación (Bajo < 10 %, Medio 10–14 %, Alto ≥ 14 %)."]),
                    html.Li([html.B("Observaciones para modelar: "), f"{len(D.dataset().X)} meses con 17 variables (se pierden los primeros 11 por los rezagos y el último por no tener mes siguiente)."]),
                    html.Li([html.B("Exclusión: "), "la ocupación y la participación nacionales del mismo mes no se usan, porque TD = 1 − TO/TGP (correlación 0.9999993) y el modelo solo despejaría una identidad contable."]),
                ]),
            ], md=5),
            dbc.Col([
                html.Div("Diccionario de variables de los modelos", className="fw-semibold mb-2"),
                dash_table.DataTable(
                    data=dictionary, columns=[{"name": c, "id": c} for c in ["Variable", "Grupo", "Descripción", "Nombre en el código"]],
                    page_size=9, style_table={"overflowX": "auto"},
                    style_cell={"fontFamily": "Inter, Segoe UI, Arial", "fontSize": 13, "textAlign": "left", "padding": "6px"},
                    style_header={"backgroundColor": "#0f172a", "color": "white", "fontWeight": "600"},
                ),
            ], md=7),
        ]),
    ]),

    card("Metodología", [
        html.Div(sum([[html.Div(s, className="mlc-step" + (" original" if "ALEE" in s else "")),
                       html.Span("→", className="mlc-arrow")] for s in steps], [])[:-1], className="mlc-pipeline"),
        html.P(["Cada modelo se evalúa con validación anidada: el bucle externo (5 pliegues de ventana expansiva) "
                "estima el desempeño y el interno (3 pliegues) elige los hiperparámetros, siempre entrenando con meses "
                "anteriores a los que se pronostican. Semilla global ", html.Code(str(config.SEED)), "."], className="mt-3 mb-0"),
    ]),

    dbc.Row([
        dbc.Col(card("Limitaciones, ética y sesgos", [
            html.Ul([
                html.Li("Solo 288 meses y un choque extremo (2020) que domina los errores: las conclusiones tienen mucha incertidumbre."),
                html.Li("Las categorías Bajo, Medio y Alto son una decisión del equipo, no una definición oficial."),
                html.Li("El pronóstico es una herramienta académica: no debe usarse para decisiones de política sin el criterio de expertos."),
                html.Li("En Colombia se evaluaron diez iteraciones de diseño de ALEE sobre los mismos 240 meses; por eso "
                        "los ocho países adicionales, evaluados una sola vez con el modelo congelado, son la evidencia "
                        "más confiable."),
            ], className="mb-0"),
        ]), md=7),
        dbc.Col(card("Autores y versión", [
            html.P("Andrés Parejo, Santiago Hurtado y Juan Marín", className="fw-semibold"),
            html.P("Pregrado en Ciencia de Datos · Machine Learning, tercer entregable"),
            html.P("Versión 1.0 · noviembre de 2026", className="text-muted mb-0"),
        ]), md=5),
    ]),
])
