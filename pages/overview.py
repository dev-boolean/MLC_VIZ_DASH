"""01. Overview: observatorio del mercado laboral del país seleccionado."""
import dash
import pandas as pd
from dash import Input, Output, callback, dcc, html

from labor import catalog as C
from labor import engine as E
from labor import figs as F
from labor import ui as U

dash.register_page(__name__, path="/", name="Overview", title="Overview · LABOR INSIGHTS")
N = C.num

hero = html.Section([
    html.Div([
        html.Div([html.Span("01", className="n"), html.Span("Overview · Observatorio académico del desempleo")], className="li-kicker"),
        html.H1([html.Span("Comprender el mercado laboral."), html.Span("Evaluar la evidencia predictiva.", className="li-accent")], className="hero-title split"),
        html.P("Una plataforma de investigación que pronostica la tasa de desempleo del mes siguiente con ALEE, un modelo original de combinación "
               "de expertos, y que muestra con la misma claridad su incertidumbre y la evidencia de su evaluación en nueve países.", className="hero-sub"),
        html.Div([dcc.Link("Ver el pronóstico", href="/forecast", className="btn-li primary"), dcc.Link("Cobertura internacional", href="/global", className="btn-li ghost")], className="hero-cta"),
        html.Div([html.Span(x, className="pill") for x in ("9 países evaluados", "Frecuencia mensual", "Horizonte: un mes", "Intervalo conforme del 90 %")], className="pills"),
    ], className="hero-copy"),
    html.Div([html.Div(className="li-globe hero-globe", **{"data-mode": "hero"}),
              html.Div(id="ov-hero-card", className="hero-card")], className="hero-visual"),
], className="li-hero")

# Indicadores que publica el DANE cada mes (definiciones del primer entregable) y su papel en el modelo de Colombia.
INDICATORS = [
    ("Tasa de desempleo, nacional", "TD NAL", "Variable objetivo",
     "Proporción de personas desocupadas sobre la fuerza laboral a nivel nacional. El modelo pronostica su valor del mes siguiente."),
    ("Tasa de desempleo, 13 áreas", "TD ÁREA", "Variable del modelo",
     "Proporción de desocupados sobre la fuerza laboral en las 13 principales áreas metropolitanas; aproxima la presión laboral urbana."),
    ("Tasa de ocupación, nacional", "TO NAL", "Excluida",
     "Relación entre la población ocupada y la población en edad de trabajar; mide la capacidad de la economía para generar empleo."),
    ("Tasa de ocupación, 13 áreas", "TO ÁREA", "Variable del modelo",
     "Ocupados sobre población en edad de trabajar en las 13 áreas; mide la absorción de empleo en el entorno urbano."),
    ("Tasa global de participación, nacional", "TGP NAL", "Excluida",
     "Relación entre la fuerza laboral y la población en edad de trabajar; refleja la disposición a participar en el mercado laboral."),
    ("Tasa global de participación, 13 áreas", "TGP ÁREA", "Variable del modelo",
     "Fuerza laboral sobre población en edad de trabajar en las 13 áreas metropolitanas."),
]

# Recorrido metodológico del proyecto, con la sección del dashboard donde se ve cada paso.
STEPS = [
    ("Datos", "Serie mensual del DANE (GEIH) y de ocho países más; variables con rezagos, medias móviles y estacionalidad.", None),
    ("Análisis exploratorio", "Tendencia, estacionalidad, autocorrelación, estacionariedad y anomalías de la serie.", ("/analytics", "02")),
    ("Modelos del curso", "140 combinaciones de modelos, balanceo y optimizadores, con validación anidada.", ("/modelo", "03")),
    ("ALEE", "Modelo original: tres expertos combinados con pesos que dependen del estado del mercado laboral.", ("/modelo", "03")),
    ("Pronóstico", "Tasa del mes siguiente con un intervalo conforme del 90 % calculado con errores fuera de muestra.", ("/forecast", "04")),
    ("Evaluación internacional", "El mismo método, con un ALEE propio por país, en nueve países.", ("/global", "05")),
    ("Conclusiones", "Respuestas a la pregunta de investigación, limitaciones y trabajo futuro.", ("/insights", "06")),
]


def _indicator(title, code, role, text):
    kind = {"Variable objetivo": "est", "Variable del modelo": "obs", "Excluida": "calc"}[role]
    return html.Details([
        html.Summary([html.Span(title, className="ind-t"), html.Span(code, className="ind-code"), html.Span(role, className=f"chip {kind}")]),
        html.P(text + " Cuantitativa continua, en porcentaje, mensual."),
    ], className="ctx-ind")


def context():
    """Contexto del estudio: problema, pregunta, datos, indicadores y metodología. Cifras calculadas con la serie guardada."""
    s, m = E.series("colombia"), E.meta("colombia")
    peak, first, last = s.idxmax(), s.index[0], s.index[-1]
    early = s[s.index.year <= first.year + 2]
    final = s[s.index.year == last.year]
    others = sorted({C.COUNTRIES[k]["source"].split(" (")[0].replace("OCDE, Infra-annual labour statistics", "OCDE").replace(", vía", " vía")
                     for k in C.COUNTRIES if k != "colombia"})
    others = ", ".join(others[:-1]) + " y " + others[-1]
    n_other = len(E.meta("eeuu")["features"])
    data = [
        ("Fuente", "DANE, Gran Encuesta Integrada de Hogares (GEIH); series publicadas por el Banco de la República."),
        ("Periodo", f"{C.month_long(first)} a {C.month_long(last)}, mensual ({len(s)} meses)."),
        ("Variable objetivo", "Tasa de desempleo nacional del mes t+1: el horizonte es siempre de un mes."),
        ("Variables del modelo", f"{len(m['features'])}: rezagos de 1, 2, 3, 6 y 12 meses; medias y desviaciones móviles de 3, 6 y 12 meses; "
                                 "el mes del año (seno y coseno); una tendencia, y las tres tasas de las 13 áreas metropolitanas."),
        ("Observaciones", f"{m['n']} meses para entrenar y evaluar: se pierden los primeros por los rezagos y el último por no tener mes siguiente."),
        ("Exclusión", "La ocupación y la participación nacionales del mismo mes no se usan: TD = 1 − TO/TGP, así que el modelo solo despejaría una identidad contable."),
        ("Otros países", f"Ocho series mensuales de {others}. Cada país tiene su propio ALEE con {n_other} variables: "
                         "las mismas, sin las de las 13 áreas."),
    ]
    return html.Section([
        U.block("Contexto del estudio", "Qué se pregunta, con qué datos y cómo se responde",
                "El problema, la pregunta de investigación, los datos y el recorrido metodológico. El análisis exploratorio detallado "
                "es de Colombia; la evaluación internacional aplica el mismo método en ocho países más."),
        U.grid(
            U.panel("Por qué importa", html.Div([
                html.P("La tasa de desempleo es el indicador laboral que más se sigue en Colombia: orienta la política monetaria y fiscal, los programas "
                       "de empleo y las decisiones de contratación. Un pronóstico confiable del mes siguiente permite anticipar cambios antes de que el "
                       "DANE publique la cifra."),
                html.P(f"La serie muestra por qué es difícil: promedió {N(early.mean(), 1)} % entre {first.year} y {first.year + 2}, llegó a "
                       f"{N(s.max(), 1)} % en {C.month_long(peak)}, durante la pandemia, y promedió {N(final.mean(), 1)} % en {last.year}."),
            ], className="li-prose"), sub="Observado · DANE"),
            U.panel("Pregunta de investigación", html.Div([
                html.P(html.B("¿Puede un modelo de combinación de expertos, con pesos que dependen del estado del mercado laboral, pronosticar la tasa "
                              "de desempleo del mes siguiente mejor que los modelos de regresión estudiados en clase?"), className="ctx-q"),
                html.Ul([
                    html.Li("Comparar 140 combinaciones de modelos, técnicas de balanceo y optimizadores con validación anidada."),
                    html.Li("Proponer, implementar y validar con rigor un modelo original: ALEE."),
                    html.Li("Comprobar si su ventaja se mantiene en ocho países con mercados laborales distintos."),
                    html.Li("Llevar el resultado a un producto que cualquier persona pueda consultar."),
                ], className="ctx-list"),
            ], className="li-prose"), sub="Objetivos del proyecto"),
            cols="2",
        ),
        U.grid(
            U.panel("Datos", html.Dl([x for t, v in data for x in (html.Dt(t), html.Dd(v))], className="li-interp ctx-data"), sub="Colombia y comparación internacional"),
            U.panel("Indicadores del mercado laboral", html.Div([
                html.P("El DANE publica cada mes tres tasas para el total nacional y para las 13 principales áreas metropolitanas. "
                       "Despliegue cada una para ver su definición y su papel en el modelo.", className="li-cap ctx-lead"),
                html.Div([_indicator(*i) for i in INDICATORS]),
            ]), sub="Definiciones"),
            cols="2",
        ),
        U.panel("Metodología", html.Div([
            html.Ol([html.Li([
                html.Span(f"{i:02d}", className="st-n"), html.B(t), html.P(d),
                dcc.Link(f"Sección {link[1]} →", href=link[0], className="li-link") if link else html.Span("Base de todo el análisis", className="st-none"),
            ]) for i, (t, d, link) in enumerate(STEPS, 1)], className="ctx-steps"),
            html.P("Todos los modelos se evalúan con validación anidada: el bucle externo (cinco pliegues de ventana expansiva) estima el desempeño y el "
                   "interno (tres pliegues) elige los hiperparámetros, siempre entrenando con meses anteriores a los que se pronostican.", className="li-cap"),
        ]), sub="Del dato al pronóstico"),
    ], className="ctx")


layout = html.Div([
    hero,
    html.Div([
        context(),
        U.block("Pronóstico vigente", "El país seleccionado, hoy", "Todo lo que sigue se actualiza con el país elegido en la barra superior, en el mapa de la sección 05 o en las tarjetas de abajo."),
        html.Div(id="ov-kpis"), U.loading(html.Div(id="ov-main")), html.Div(id="ov-more"),
    ], className="li-wrap"),
])


def _move(d):
    return "por encima" if d > 0 else "por debajo"


def brief(c):
    """Informe breve calculado con los datos vigentes. Cada frase dice de dónde sale; ninguna atribuye causas."""
    f, conf, s = E.forecast(c), E.conformal(c), E.series(c)
    items = []
    d = f["observed"] - f["previous"]
    items.append(("obs", f"En {C.month_long(f['last_date'])} la tasa fue {N(f['observed'])} %, "
                         f"{N(abs(d))} pp {_move(d)} del mes anterior." if abs(d) >= 0.005 else f"En {C.month_long(f['last_date'])} la tasa fue {N(f['observed'])} %, igual que el mes anterior."))
    if len(s) >= 24:
        ly = s.iloc[-12:].mean() - s.iloc[-24:-12].mean()
        items.append(("calc", f"La media de los últimos 12 meses ({N(s.iloc[-12:].mean())} %) está {N(abs(ly))} pp {_move(ly)} de la de los 12 anteriores."))
    if len(s) >= 13:
        items.append(("obs", f"En el mismo mes del pronóstico, un año antes ({C.month_long(f['target_date'] - pd.DateOffset(years=1))}), la tasa fue {N(f['same_month'])} %."))
    inside = f["lo"] <= f["observed"] <= f["hi"]
    items.append(("est", f"ALEE pronostica {N(f['pred'])} % para {C.month_long(f['target_date'])} (intervalo del 90 %: {N(f['lo'])} a {N(f['hi'])} %). "
                         + ("El último dato observado cae dentro de ese intervalo: el cambio pronosticado es menor que el error habitual del modelo."
                            if inside else "El último dato observado queda fuera de ese intervalo: el cambio pronosticado supera el error habitual del modelo.")))
    z = (f["observed"] - s.mean()) / s.std()
    items.append(("calc", f"El nivel actual está a {N(abs(z), 1)} desviaciones estándar {_move(z)} de la media histórica del país ({N(s.mean())} %, {s.index[0].year}–{s.index[-1].year})."))
    best, b = E.best_baseline(c)
    r = E.metric_table().loc[c, "ALEE"]
    items.append(("calc", f"En {conf['n']} meses de prueba fuera de muestra, ALEE tuvo un RMSE medio por pliegue de {N(r, 3)} pp frente a {N(b, 3)} del mejor modelo de comparación "
                          f"({C.MODEL_LABELS[best]}). Su intervalo del 90 % cubrió el {N(100 * conf['coverage'], 1)} % de los casos medidos hacia adelante."))
    if E.frame(c)["interpolado"].iloc[-12:].any():
        items.append(("obs", "Uno de los últimos 12 meses no se publicó y está interpolado; entra así en las variables del modelo."))
    return items


@callback(Output("ov-hero-card", "children"), Output("ov-kpis", "children"), Output("ov-more", "children"), Input("pais", "value"))
@U.safe
def body(c):
    f = E.forecast(c)
    d = f["observed"] - f["previous"]
    card = [html.Div([html.Span("País seleccionado", className="hc-k"), html.Span(C.name(c), className="hc-name")]),
            html.Div([html.Div([html.Span(f"{N(f['observed'])} %", className="hc-v obs"), html.Small(f"observado, {C.month_short(f['last_date'])}")]),
                      html.Span("→", className="hc-arrow", **{"aria-hidden": "true"}),
                      html.Div([html.Span(f"{N(f['pred'])} %", className="hc-v est"), html.Small(f"pronosticado, {C.month_short(f['target_date'])}")])], className="hc-row")]
    kp = U.kpis(
        U.kpi("Tasa del mes siguiente", f["pred"], "est", "%", f"{C.month_long(f['target_date']).capitalize()}. Intervalo del 90 %: {N(f['lo'])} a {N(f['hi'])} %", big=True),
        U.kpi("Última tasa observada", f["observed"], "obs", "%", f"{C.month_long(f['last_date']).capitalize()}. Mes anterior: {N(f['previous'])} %"),
        U.kpi("Variación pronosticada", f["change"], "calc", "pp", "Pronóstico menos último dato"),
        U.kpi("Última observación", C.month_short(f["last_date"]), "obs", foot=f"Variación del mes: {N(d, 2, True)} pp"),
        U.kpi("Periodo pronosticado", C.month_short(f["target_date"]), "est", foot="Un mes después del último dato"),
        cls="ov",
    )
    cards = []
    for k, m in C.COUNTRIES.items():
        g = E.forecast(k)
        cards.append(html.Button([
            html.Div([html.Span(m["name"], className="mc-name"), html.Span(m["short"], className="mc-code")], className="mc-top"),
            U.graph(F.sparkline(k), cls="spark"),
            html.Div([html.Span([html.Small("Observado"), f"{N(g['observed'])} %"], className="obs"), html.Span([html.Small("Pronóstico"), f"{N(g['pred'])} %"], className="est"),
                      html.Span([html.Small("Cambio"), f"{N(g['change'], 2, True)} pp"], className="calc")], className="mc-row"),
            html.Small(f"{C.month_short(g['last_date'])} → {C.month_short(g['target_date'])}", className="mc-when"),
        ], className="mini-country" + (" on" if k == c else ""), id={"type": "pick", "c": k}, n_clicks=0, title=f"Seleccionar {m['name']}"))
    labels = {"obs": "Observado", "est": "Pronosticado", "calc": "Calculado"}
    more = html.Div([
        U.panel("Informe de investigación", html.Div([
            html.Ul([html.Li([html.Span(labels[k], className=f"chip {k}"), html.Span(t)]) for k, t in brief(c)], className="li-brief"),
            html.P("Generado con la serie vigente y los resultados guardados de la evaluación. Describe lo observado y lo estimado; no atribuye causas.", className="li-cap"),
            dcc.Link("Ver la evaluación completa →", href="/modelo", className="li-link"),
        ]), sub=f"{C.name(c)} · se actualiza con el país seleccionado", cls="brief"),
        U.block("Comparación internacional", "Nueve países, el mismo método",
                "Seleccione una tarjeta para cambiar de país en toda la plataforma. Cada línea usa su propia escala: las encuestas y las edades cubiertas no son idénticas, así que aquí no se ordenan los niveles."),
        html.Div(cards, className="mini-countries reveal"),
        html.Div(dcc.Link("Explorar la evaluación internacional →", href="/global", className="li-link"), className="li-more"),
    ])
    return card, kp, more


@callback(Output("ov-main", "children"), Input("pais", "value"))
@U.safe
def main(c):
    return U.panel(f"Evolución y pronóstico: {C.name(c)}", html.Div([
        U.seg("ov-span", [("5 años", 60), ("10 años", 120), ("20 años", 240), ("Toda la serie", 0)], 120, "Ventana histórica"),
        U.graph(F.forecast_main(c, 120, 460), id="ov-fig", cls="forecast-anim"),
        U.legend(("Historia observada", "obs"), ("Pronóstico de ALEE, un mes", "est"), ("Intervalo del 90 %", "unc")),
        html.P("La ventana solo cambia cuánta historia se muestra. El pronóstico es siempre el del mes siguiente al último dato.", className="li-cap"),
    ]), sub=U.source(c), cls="hero-chart")


@callback(Output("ov-fig", "figure"), Input("ov-span", "value"), dash.State("pais", "value"), prevent_initial_call=True)
def span(months, c):
    return F.forecast_main(c, months, 460)


@callback(Output("pais", "value"), Input({"type": "pick", "c": dash.ALL}, "n_clicks"), prevent_initial_call=True)
def pick(n):
    t = dash.ctx.triggered_id
    if not t or not any(n):
        return dash.no_update
    return t["c"]
