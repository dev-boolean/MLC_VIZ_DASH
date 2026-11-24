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
    html.Div([html.Div(className="li-globe hero-globe", **{"data-mode": "hero"}), html.Span("Globo ilustrativo: ubica los países, no mide nada", className="illus"),
              html.Div(id="ov-hero-card", className="hero-card")], className="hero-visual"),
], className="li-hero")

layout = html.Div([
    hero,
    html.Div([html.Div(id="ov-kpis"), U.loading(html.Div(id="ov-main")), html.Div(id="ov-more")], className="li-wrap"),
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
