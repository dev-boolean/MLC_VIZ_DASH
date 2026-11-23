"""06. Global Intelligence: el mismo modelo en nueve países."""
import dash
from dash import Input, Output, callback, dcc, html

from labor import catalog as C
from labor import engine as E
from labor import figs as F
from labor import ui as U

dash.register_page(__name__, path="/global", name="Global Intelligence", title="Global Intelligence · LABOR INSIGHTS")
N = C.num
Y0 = min(E.series(c).index[0].year for c in C.COUNTRIES)
Y1 = max(E.series(c).index[-1].year for c in C.COUNTRIES)


def status_rows():
    rows = []
    for c, m in C.COUNTRIES.items():
        f, d = E.forecast(c), E.oof(c)
        rows.append({"a": m["name"], "b": f"{E.series(c).index[0].year}–{C.month_short(f['last_date'])}", "c": f"{E.meta(c)['n']}", "d": f"{len(E.meta(c)['features'])}", "e": f"{len(d)} meses",
                     "f": "Sí, con su propia serie", "g": "Desarrollo y prueba" if c == "colombia" else "Prueba con el diseño congelado"})
    return rows


def forecast_rows():
    rows = []
    for c, m in C.COUNTRIES.items():
        f, k = E.forecast(c), E.conformal(c)
        rows.append({"a": m["name"], "b": C.month_short(f["last_date"]), "c": f"{N(f['observed'])} %", "d": C.month_short(f["target_date"]), "e": f"{N(f['pred'])} %", "f": f"{N(f['change'], 2, True)} pp", "g": f"{N(f['lo'])} – {N(f['hi'])} %", "h": f"{N(100 * k['coverage'], 1)} %"})
    return rows


layout = html.Div([
    html.Section([
        html.Div(className="li-globe big", **{"data-mode": "full"}),
        html.Div([html.Div([html.Span(className="li-dash"), "06  ·  GLOBAL INTELLIGENCE"], className="li-kicker"),
                  html.H1([html.Span("Nueve países, "), html.Span("un mismo modelo.", className="li-accent")], className="li-title split"),
                  html.P("Arrastre para girar. Toque un país resaltado para seleccionarlo en toda la plataforma.", className="li-lead")], className="gl-head"),
        html.Div(id="gl-card", className="gl-card"),
        html.Span("Globo y conexiones ilustrativos", className="illus"),
    ], className="gl-hero"),
    html.Div([
        U.block("6.1 · COMPARAR", "La evolución, lado a lado"),
        U.panel("Comparación internacional", html.Div([
            html.Div([html.Label("Países", htmlFor="gl-countries"), dcc.Dropdown(id="gl-countries", options=[{"label": m["name"], "value": c} for c, m in C.COUNTRIES.items()], value=["colombia", "eeuu", "espana", "mexico"], multi=True, searchable=False, className="li-dd wide", labels={"selected_count": "{num_selected} elegidos", "select_all": "Elegir todos", "deselect_all": "Quitar todos", "clear_selection": "Quitar la selección"})], className="li-ctl"),
            html.Div([U.seg("gl-mode", [("Frente a su propia historia", "z"), ("Cambio interanual", "yoy"), ("Tasa en %", "level")], "z"),
                      html.Div(dcc.RangeSlider(id="gl-years", min=Y0, max=Y1, step=1, value=[2001, Y1], marks={y: {"label": str(y)} for y in (1950, 1970, 1990, 2010)} , pushable=3, tooltip={"placement": "bottom"}), className="range grow")], className="row-tools"),
            U.loading(html.Div(id="gl-compare")),
        ])),
        U.block("6.2 · PRONÓSTICOS", "Lo que ALEE espera para el mes siguiente en cada país"),
        U.grid(
            U.panel("Cambio pronosticado", U.graph(F.forecast_changes(380)), sub="Rombo: pronóstico menos último dato. Línea: intervalo del 90 %"),
            U.panel("Detalle", html.Div([U.table(forecast_rows(), [("a", "País"), ("b", "Último"), ("c", "Observado"), ("d", "Siguiente"), ("e", "Pronóstico"), ("f", "Cambio"), ("g", "Intervalo 90 %"), ("h", "Cobertura")], numeric=("c", "e", "f", "g", "h")),
                                         html.P("Orden fijo, no es un ranking. Los meses difieren: la serie de Colombia llega a diciembre de 2025.", className="li-cap")])), cols="1"),
        U.block("6.3 · RENDIMIENTO", "¿Se sostiene la ventaja fuera de Colombia?", "El diseño se desarrolló con Colombia. Los otros ocho países se evaluaron una sola vez, con el diseño ya congelado: son la evidencia más confiable de generalización."),
        U.grid(
            U.panel("Reducción del error por país", U.graph(F.reductions()), sub="Frente al mejor de los ocho modelos del curso y LightGBM en cada país"),
            U.panel("Rango medio de cada modelo", U.graph(F.cd_fig(360)), sub="Friedman y diferencia crítica de Nemenyi. Lo que cae en la franja no se distingue de ALEE con esta prueba"),
        ),
        U.panel("Todos los modelos, todos los países", html.Div([U.seg("gl-metric", [("RMSE", "RMSE"), ("MAE", "MAE"), ("MAPE", "MAPE"), ("MASE", "MASE")], "RMSE"), U.graph(F.relative_heat(), id="gl-heat")]), sub="Violeta intenso: igual que ALEE. Más oscuro: error mayor. Escala logarítmica, hasta ocho veces"),
        html.Div(id="gl-tests"),
        U.block("6.4 · ESTADO", "Entrenado, evaluado y disponible"),
        U.panel("Estado de cada país", html.Div([
            U.table(status_rows(), [("a", "País"), ("b", "Serie"), ("c", "Meses de ajuste"), ("d", "Variables"), ("e", "Prueba"), ("f", "Entrenado"), ("g", "Papel")], numeric=("c", "d", "e")),
            U.note(["Cada país tiene su propio ALEE, ajustado con su serie. Lo que está comprobado es que el ", html.B("método"), " generaliza, no que un modelo entrenado en Colombia sirva para otro país. No hay países con datos y sin evaluación: ninguno fuera de esta lista es compatible."], "info", "Qué significa generalizar."),
        ])),
    ], className="li-wrap"),
])


@callback(Output("gl-card", "children"), Input("pais", "value"))
@U.safe
def card(c):
    f, (best, b) = E.forecast(c), E.best_baseline(c)
    r = E.metric_table().loc[c, "ALEE"]
    return [html.Small("PAÍS SELECCIONADO"), html.H2(C.name(c)), U.graph(F.sparkline(c, 120), cls="spark"),
            html.Div([html.Div([html.Small(f"Observado, {C.month_short(f['last_date'])}"), html.B(f"{N(f['observed'])} %", className="obs")]),
                      html.Div([html.Small(f"Pronóstico, {C.month_short(f['target_date'])}"), html.B(f"{N(f['pred'])} %", className="est")]),
                      html.Div([html.Small("RMSE de ALEE"), html.B(N(r, 3), className="model")]),
                      html.Div([html.Small(f"Frente a {C.MODEL_LABELS[best]}"), html.B(f"−{N(100 * (1 - r / b), 0)} %", className="model")])], className="gl-stats"),
            html.P(U.source(c), className="li-cap")]


@callback(Output("gl-compare", "children"), Input("gl-countries", "value"), Input("gl-mode", "value"), Input("gl-years", "value"))
@U.safe
def compare(cs, mode, years):
    if not cs:
        return U.note("Elija al menos un país.", "warn")
    cs = [c for c in C.COUNTRIES if c in cs]
    msg = {"z": "Cada serie se mide en desviaciones estándar frente a su propia media histórica, así que se comparan movimientos y no niveles.",
           "yoy": "Cambio frente al mismo mes del año anterior: quita el calendario y permite comparar el ritmo.",
           "level": "Cuidado con los niveles: las encuestas, las edades cubiertas (15 o 16 años en adelante) y las definiciones de búsqueda de empleo no son idénticas. Una tasa más alta no es por sí sola un mercado peor."}[mode]
    return [U.graph(F.compare(cs, mode, years)), U.note(msg, "warn" if mode == "level" else "info")]


@callback(Output("gl-heat", "figure"), Input("gl-metric", "value"))
def heat(m):
    return F.relative_heat(m)


@callback(Output("gl-tests", "children"), Input("gl-metric", "id"))
@U.safe
def tests(_):
    b = E.result("pruebas_entre_conjuntos")
    t = E.result("pruebas_por_baseline")
    pf = lambda p: "< 0,001" if p < 0.001 else N(p, 3)  # noqa: E731
    rows = [{"m": C.MODEL_LABELS[r["baseline"]], "g": f"{int(r['conjuntos_ALEE_gana'])} de 9", "w": pf(r["wilcoxon_p_holm"]), "c": pf(r["conover_p_holm"])} for _, r in b.iterrows()]
    ns = b[b["conover_p_holm"] >= 0.05]["baseline"].map(C.MODEL_LABELS).tolist()
    return U.panel("Pruebas entre países", html.Div([
        U.table(rows, [("m", "Modelo"), ("g", "ALEE gana en"), ("w", "Wilcoxon (Holm), p"), ("c", "Conover (Holm), p")], numeric=("g", "w", "c")),
        U.reading(f"ALEE tiene el menor RMSE en los nueve países y Diebold-Mariano rechaza la igualdad en {int((t['dm_p_holm'] < 0.05).sum())} de {len(t)} comparaciones.",
                  f"Wilcoxon da p = {N(b['wilcoxon_p_holm'].max(), 3)} frente a cada modelo, el mínimo posible con nueve países." + (f" Conover no distingue a ALEE de {', '.join(x.lower() for x in ns)}." if ns else ""),
                  "Nueve países dan una potencia moderada. La comparación incluye los modelos del curso y LightGBM, no todos los métodos de series de tiempo que existen."),
    ]))
