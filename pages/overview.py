"""01. Overview: visión ejecutiva del país seleccionado."""
import dash
from dash import Input, Output, callback, dcc, html

from labor import catalog as C
from labor import engine as E
from labor import figs as F
from labor import stats as S
from labor import ui as U

dash.register_page(__name__, path="/", name="Overview", title="Overview · LABOR INSIGHTS")
N = C.num

hero = html.Section([
    html.Div(className="li-globe hero-globe", **{"data-mode": "hero"}),
    html.Div([
        html.Div([html.Span(className="li-dash"), "GLOBAL UNEMPLOYMENT INTELLIGENCE"], className="li-kicker"),
        html.H1([html.Span("Comprende el presente."), html.Span("Anticipa el futuro", className="li-accent"), html.Span("del mercado laboral.")], className="hero-title split"),
        html.P("Una plataforma de inteligencia económica impulsada por un modelo original de predicción del desempleo.", className="hero-sub"),
        html.Div([dcc.Link("Ver el pronóstico", href="/forecast", className="btn-li primary"), dcc.Link("Explorar el mundo", href="/global", className="btn-li ghost")], className="hero-cta"),
        html.Div([html.Span(x, className="pill") for x in ("9 países evaluados", "Frecuencia mensual", "Horizonte: 1 mes", "Modelo original ALEE")], className="pills"),
    ], className="hero-copy"),
    html.Div(id="ov-hero-card", className="hero-card"),
    html.Span("Globo ilustrativo", className="illus"),
], className="li-hero")

layout = html.Div([
    hero,
    html.Div(id="ov-kpis"),
    html.Div(U.loading(html.Div(id="ov-main")), className="li-wrap"),
    html.Div(id="ov-more"),
])


def _move(d):
    return "por encima" if d > 0 else "por debajo"


@callback(Output("ov-hero-card", "children"), Output("ov-kpis", "children"), Output("ov-more", "children"), Input("pais", "value"))
@U.safe
def body(c):
    f, conf = E.forecast(c), E.conformal(c)
    d = f["observed"] - f["previous"]
    card = [html.Div([html.Span("PAÍS SELECCIONADO", className="hc-k"), html.Span(C.name(c), className="hc-name")]),
            html.Div([html.Div([html.Span(f"{N(f['observed'])} %", className="hc-v obs"), html.Small(f"observado, {C.month_short(f['last_date'])}")]),
                      html.Span("→", className="hc-arrow"),
                      html.Div([html.Span(f"{N(f['pred'])} %", className="hc-v est"), html.Small(f"pronosticado, {C.month_short(f['target_date'])}")])], className="hc-row")]
    kp = U.kpis(
        U.kpi("Última tasa observada", f["observed"], "obs", "%", f"{C.month_long(f['last_date']).capitalize()}. Mes anterior: {N(f['previous'])} %"),
        U.kpi("Tasa del mes siguiente", f["pred"], "est", "%", f"Intervalo del 90 %: {N(f['lo'])} a {N(f['hi'])} %"),
        U.kpi("Variación pronosticada", f["change"], "calc", "pp", "Pronóstico menos último dato"),
        U.kpi("Última observación", C.month_short(f["last_date"]), "obs", foot=f"Variación del mes: {N(d, 2, True)} pp"),
        U.kpi("Periodo estimado", C.month_short(f["target_date"]), "est", foot=f"Hace un año, en el mismo mes: {N(f['same_month'])} %"),
    )
    # ¿Cómo se compara? Cada país frente a su propia historia, sin ordenar niveles de encuestas distintas.
    cards = []
    for k, m in C.COUNTRIES.items():
        g = E.forecast(k)
        cards.append(html.Button([
            html.Div([html.Span(m["name"], className="mc-name"), html.Span(m["short"], className="mc-code")], className="mc-top"),
            U.graph(F.sparkline(k), cls="spark"),
            html.Div([html.Span([html.Small("Observado"), f"{N(g['observed'])} %"], className="obs"), html.Span([html.Small("Siguiente"), f"{N(g['pred'])} %"], className="est"),
                      html.Span([html.Small("Cambio"), f"{N(g['change'], 2, True)} pp"], className="calc")], className="mc-row"),
        ], className="mini-country" + (" on" if k == c else ""), id={"type": "pick", "c": k}, n_clicks=0, title=f"Seleccionar {m['name']}"))
    s = E.series(c)
    last_year = s.iloc[-12:].mean() - s.iloc[-24:-12].mean()
    z = (f["observed"] - s.mean()) / s.std()
    up, dn = S.runs(s)
    facts = []
    if up:
        facts.append(html.Li([html.B("La mayor subida de la tendencia: "), f"{N(up[0], 1)} pp, de {C.month_long(up[1])} a {C.month_long(up[2])}."]))
        facts.append(html.Li([html.B("La mayor bajada: "), f"{N(dn[0], 1)} pp, de {C.month_long(dn[1])} a {C.month_long(dn[2])}."]))
    facts.append(html.Li([html.B("El último año: "), f"la media de los últimos 12 meses está {N(abs(last_year))} pp {_move(last_year)} de la de los 12 anteriores."]))
    facts.append(html.Li([html.B("El nivel actual: "), f"está a {N(abs(z), 1)} desviaciones estándar {_move(z)} de la media histórica del país ({N(s.mean())} %)."]))
    more = html.Div([
        U.grid(
            U.panel("¿Cómo ha evolucionado?", html.Ul(facts, className="li-facts"), sub=f"Calculado con la serie completa, {s.index[0].year} a {s.index[-1].year}"),
            U.panel("¿Qué tan confiable es el pronóstico?", html.Div([
                html.P([f"En {conf['n']} meses de prueba, con pronósticos hechos siempre con información anterior, ALEE tuvo un error medio por pliegue de ",
                        html.B(f"{N(E.metric_table().loc[c, 'ALEE'], 3)} pp"), f" de RMSE. El mejor modelo de comparación, {C.MODEL_LABELS[E.best_baseline(c)[0]]}, tuvo {N(E.best_baseline(c)[1], 3)}."]),
                html.P([f"El intervalo del 90 % tiene un semiancho de {N(conf['hw'])} pp. Medido hacia adelante en {conf['n_forward']} meses, cubrió el {N(100 * conf['coverage'], 1)} % de los casos."]),
                dcc.Link("Ver la evaluación completa →", href="/modelo", className="li-link"),
            ], className="li-prose"), sub="Validación anidada de origen móvil, cinco pliegues"),
        ),
        U.block("COMPARE", "Nueve países, el mismo modelo", "Seleccione una tarjeta para cambiar de país en toda la plataforma. Cada línea usa su propia escala: las encuestas y las edades cubiertas no son idénticas, así que aquí no se ordenan los niveles."),
        html.Div(cards, className="mini-countries reveal"),
    ], className="li-wrap")
    return card, kp, more


@callback(Output("ov-main", "children"), Input("pais", "value"))
@U.safe
def main(c):
    return U.panel(f"Evolución y pronóstico: {C.name(c)}", html.Div([
        U.seg("ov-span", [("5 años", 60), ("10 años", 120), ("20 años", 240), ("Todo", 0)], 120, "Ver"),
        U.graph(F.forecast_main(c, 120), id="ov-fig", cls="forecast-anim"),
        html.Div([html.Span("Historia observada", className="lg obs"), html.Span("Pronóstico de ALEE, un mes", className="lg est"), html.Span("Intervalo del 90 %", className="lg unc")], className="lgs"),
    ]), sub=U.source(c))


@callback(Output("ov-fig", "figure"), Input("ov-span", "value"), dash.State("pais", "value"), prevent_initial_call=True)
def span(months, c):
    return F.forecast_main(c, months)


@callback(Output("pais", "value"), Input({"type": "pick", "c": dash.ALL}, "n_clicks"), prevent_initial_call=True)
def pick(n):
    t = dash.ctx.triggered_id
    if not t or not any(n):
        return dash.no_update
    return t["c"]
