"""05. Global Intelligence: evaluación internacional de ALEE."""
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
        f, d, s = E.forecast(c), E.oof(c), E.series(c)
        notes = []
        if c == "colombia":
            notes.append("Diez iteraciones de diseño sobre los mismos 240 meses; resultado posiblemente optimista")
        else:
            notes.append("Evaluado una sola vez, con el diseño ya congelado")
        if E.frame(c)["interpolado"].any():
            notes.append("oct 2025 interpolado")
        rows.append({"a": m["name"], "b": f"{C.month_short(s.index[0])} – {C.month_short(f['last_date'])}", "c": f"{len(s)}", "d": f"{len(E.meta(c)['features'])}",
                     "e": f"Sí · {E.meta(c)['n']} meses", "f": f"Sí · {len(d)} meses ({d['fecha'].min().year}–{d['fecha'].max().year})", "g": "; ".join(notes) + "."})
    return rows


def forecast_rows():
    rows = []
    for c, m in C.COUNTRIES.items():
        f, k = E.forecast(c), E.conformal(c)
        rows.append({"a": m["name"], "b": C.month_short(f["last_date"]), "c": f"{N(f['observed'])} %", "d": C.month_short(f["target_date"]), "e": f"{N(f['pred'])} %",
                     "f": f"{N(f['change'], 2, True)} pp", "g": f"{N(f['lo'])} – {N(f['hi'])} %", "h": f"{N(100 * k['coverage'], 1)} %"})
    return rows


EXPLAIN = [
    ("Métrica de evaluación", "RMSE medio de los cinco pliegues externos de la validación anidada (en puntos porcentuales). Es la métrica principal, fijada antes de evaluar; MAE, MAPE y MASE se muestran como secundarias."),
    ("Modelos de comparación", "Los ocho modelos de regresión del curso (KNN, Ridge, Lasso, regresión bayesiana, árbol de decisión, Random Forest, XGBoost y SVR con kernel RBF) y LightGBM, con las mismas particiones, variables y presupuesto de ajuste. La reducción del error se mide frente al mejor de ellos en cada país."),
    ("Periodo de evaluación", "En cada país, los meses de prueba de los cinco pliegues externos (TimeSeriesSplit con un mes de separación). Las fechas exactas están en la tabla de estado experimental."),
    ("Pruebas dentro de cada país", "Diebold-Mariano-HLN con corrección de Holm. H0: ALEE y el modelo de comparación tienen la misma pérdida cuadrática esperada."),
    ("Pruebas entre países", "Friedman con diferencia crítica de Nemenyi sobre los rangos medios; Wilcoxon y Conover con Holm frente a cada modelo. H0: los modelos tienen el mismo desempeño a través de los nueve países. Con nueve países la potencia es moderada."),
    ("Cómo interpretarlo", "La significancia estadística indica que la diferencia no es atribuible al azar con estos datos; no mide por sí sola cuánto importa en la práctica. La reducción del error en porcentaje sí lo hace."),
]

layout = html.Div([
    U.head("05", "Global Intelligence", "Nueve países,", "un mismo método.",
           "Cada país tiene su propio ALEE, ajustado con su serie. Lo que se pone a prueba aquí es si el método generaliza fuera de Colombia, no si un modelo entrenado en Colombia sirve en otro país."),
    html.Div([
        html.Section([
            html.Div([
                U.panel("Cobertura del estudio", html.Div([
                    U.graph(F.world_map(C.DEFAULT), id="gl-map", cls="map"),
                    html.Div([html.Span("Desarrollo del diseño y evaluación", className="lg dev"), html.Span("Evaluación con el diseño congelado", className="lg evalc"),
                              html.Span("Sin datos en el proyecto", className="lg none")], className="lgs"),
                    html.P("Toque un país coloreado para seleccionarlo. Los nueve tienen los tres estados a la vez: datos disponibles, un ALEE propio entrenado con su serie y evaluación fuera de muestra. "
                           "Ningún país del proyecto tiene datos sin evaluación.", className="li-cap"),
                ]), cls="map-panel"),
                html.Aside(id="gl-profile", className="li-panel gl-profile reveal"),
            ], className="gl-top"),
        ], id="s51"),
        html.Section([
            U.block("5.1 · Comparación histórica", "La evolución, lado a lado"),
            U.panel("Comparación internacional", html.Div([
                html.Div([html.Label("Países", htmlFor="gl-countries"), dcc.Dropdown(id="gl-countries", options=[{"label": m["name"], "value": c} for c, m in C.COUNTRIES.items()], value=["colombia", "eeuu", "espana", "mexico"], multi=True, searchable=False, className="li-dd wide")], className="li-ctl"),
                html.Div([U.seg("gl-mode", [("Frente a su propia historia", "z"), ("Cambio interanual", "yoy"), ("Tasa en %", "level")], "z"),
                          html.Div(dcc.RangeSlider(id="gl-years", min=Y0, max=Y1, step=1, value=[2001, Y1], marks={y: {"label": str(y)} for y in (1950, 1970, 1990, 2010, Y1)}, pushable=3, tooltip={"placement": "bottom"}), className="range grow")], className="row-tools"),
                U.loading(html.Div(id="gl-compare")),
                U.note("Las estadísticas laborales de cada país usan encuestas, edades cubiertas (15 o 16 años en adelante) y definiciones de búsqueda de empleo distintas. Las series no son perfectamente comparables: por eso la vista por defecto compara movimientos frente a la propia historia y no niveles.", "warn", "Comparabilidad."),
            ])),
        ], id="s52"),
        html.Section([
            U.block("5.2 · Pronósticos internacionales", "Lo que ALEE espera para el mes siguiente en cada país"),
            U.grid(
                U.panel("Cambio pronosticado e intervalo", U.graph(F.forecast_changes(400)), sub="Rombo: pronóstico menos último dato. Barra: intervalo del 90 %"),
                U.panel("Detalle por país", html.Div([U.table(forecast_rows(), [("a", "País"), ("b", "Último dato"), ("c", "Observado"), ("d", "Mes pronosticado"), ("e", "Pronóstico"), ("f", "Cambio"), ("g", "Intervalo 90 %"), ("h", "Cobertura empírica")],
                                                              numeric=("c", "e", "f", "g", "h")),
                                                      html.P("Orden fijo, no es un ranking. Los meses difieren entre países: la serie de Colombia llega a diciembre de 2025.", className="li-cap")])), cols="1"),
        ], id="s53"),
        html.Section([
            U.block("5.3 · Generalización y comparación", "¿Se sostiene la ventaja fuera de Colombia?",
                    "El diseño se desarrolló con Colombia. Los otros ocho países se evaluaron una sola vez, con el diseño ya congelado: son la evidencia más confiable de generalización."),
            U.grid(
                U.panel("Reducción del error por país", U.graph(F.reductions(380)), sub="RMSE de ALEE frente al mejor de los ocho modelos del curso y LightGBM en cada país"),
                U.panel("Rango medio de cada modelo", U.graph(F.cd_fig(380)), sub="Friedman y diferencia crítica de Nemenyi. Lo que cae en la franja ocre no se distingue de ALEE con esta prueba"),
            ),
            U.panel("Todos los modelos, todos los países", html.Div([U.seg("gl-metric", [("RMSE", "RMSE"), ("MAE", "MAE"), ("MAPE", "MAPE"), ("MASE", "MASE")], "RMSE"), U.graph(F.relative_heat(height=440), id="gl-heat")]),
                    sub="Blanco: igual que ALEE. Más oscuro: error mayor, en escala logarítmica hasta ocho veces el de ALEE"),
            html.Div(id="gl-tests"),
            U.panel("Cómo se evaluó", html.Dl([x for t, d in EXPLAIN for x in (html.Dt(t), html.Dd(d))], className="eval-guide")),
        ], id="s54"),
        html.Section([
            U.block("5.4 · Estado experimental", "Datos, entrenamiento y evaluación de cada país"),
            U.panel("Estado de cada país", html.Div([
                U.table(status_rows(), [("a", "País"), ("b", "Periodo disponible"), ("c", "Observaciones"), ("d", "Variables"), ("e", "Entrenamiento (ALEE propio)"), ("f", "Evaluación fuera de muestra"), ("g", "Notas")],
                        numeric=("c", "d")),
                U.note(["Cada país tiene su propio ALEE, ajustado con su serie. Lo que está comprobado es que el ", html.B("método"), " generaliza, no que un modelo entrenado en Colombia sirva para otro país."], "info", "Qué significa generalizar."),
            ])),
        ], id="s55"),
    ], className="li-wrap"),
])


@callback(Output("gl-profile", "children"), Output("gl-map", "figure"), Input("pais", "value"))
@U.safe
def profile(c):
    f, (best, b), k = E.forecast(c), E.best_baseline(c), E.conformal(c)
    r = E.metric_table().loc[c, "ALEE"]
    s, d = E.series(c), E.oof(c)
    t = E.result("pruebas_por_baseline")
    t = t[t["conjunto"] == c]
    check = [("Datos disponibles", f"{C.month_short(s.index[0])} – {C.month_short(s.index[-1])} · {len(s)} meses"),
             ("ALEE propio entrenado", f"{E.meta(c)['n']} meses · {len(E.meta(c)['features'])} variables"),
             ("Evaluado fuera de muestra", f"{len(d)} meses · {d['fecha'].min().year}–{d['fecha'].max().year}")]
    return [
        html.Small("País seleccionado", className="gp-k"), html.H2(C.name(c)),
        html.Span("Desarrollo del diseño y evaluación" if c == "colombia" else "Evaluación con el diseño congelado", className=f"gp-role {'dev' if c == 'colombia' else 'evalc'}"),
        html.Ul([html.Li([html.Span("✓", className="ok", **{"aria-hidden": "true"}), html.Div([html.B(a), html.Small(x)])]) for a, x in check], className="gp-status"),
        U.graph(F.sparkline(c, 120), cls="spark"),
        html.Div([
            html.Div([html.Small(f"Observado, {C.month_short(f['last_date'])}"), html.B(f"{N(f['observed'])} %", className="obs")]),
            html.Div([html.Small(f"Pronóstico, {C.month_short(f['target_date'])}"), html.B(f"{N(f['pred'])} %", className="est")]),
            html.Div([html.Small("Intervalo del 90 %"), html.B(f"{N(f['lo'])} – {N(f['hi'])} %", className="est")]),
            html.Div([html.Small("Cobertura empírica"), html.B(f"{N(100 * k['coverage'], 1)} %")]),
            html.Div([html.Small("RMSE de ALEE"), html.B(f"{N(r, 3)} pp", className="model")]),
            html.Div([html.Small(f"Frente a {C.MODEL_LABELS[best]}"), html.B(f"−{N(100 * (1 - r / b), 0)} %", className="model")]),
        ], className="gl-stats"),
        html.P(f"Diebold-Mariano (Holm) rechaza la igualdad frente a {int((t['dm_p_holm'] < 0.05).sum())} de {len(t)} modelos en este país.", className="li-cap"),
        html.P(U.source(c), className="li-cap"),
    ], F.world_map(c)


@callback(Output("pais", "value", allow_duplicate=True), Input("gl-map", "clickData"), prevent_initial_call=True)
def pick(cd):
    try:
        c = cd["points"][0]["customdata"]
    except (TypeError, KeyError, IndexError):
        return dash.no_update
    return c if c in C.COUNTRIES else dash.no_update


@callback(Output("gl-compare", "children"), Input("gl-countries", "value"), Input("gl-mode", "value"), Input("gl-years", "value"))
@U.safe
def compare(cs, mode, years):
    if not cs:
        return U.note("Elija al menos un país.", "warn")
    cs = [c for c in C.COUNTRIES if c in cs]
    msg = {"z": "Cada serie se mide en desviaciones estándar frente a su propia media histórica, así que se comparan movimientos y no niveles.",
           "yoy": "Cambio frente al mismo mes del año anterior: quita el calendario y permite comparar el ritmo.",
           "level": "Cuidado con los niveles: una tasa más alta no es por sí sola un mercado peor, porque las encuestas y las definiciones no son idénticas."}[mode]
    return [U.graph(F.compare(cs, mode, years, 460)), html.P(msg, className="li-cap")]


@callback(Output("gl-heat", "figure"), Input("gl-metric", "value"))
def heat(m):
    return F.relative_heat(m, 440)


@callback(Output("gl-tests", "children"), Input("gl-metric", "id"))
@U.safe
def tests(_):
    b = E.result("pruebas_entre_conjuntos")
    t = E.result("pruebas_por_baseline")
    pf = lambda p: "< 0,001" if p < 0.001 else N(p, 3)  # noqa: E731
    sig = t.assign(s=t["dm_p_holm"] < 0.05).groupby("baseline")["s"].sum()
    rows = [{"m": C.MODEL_LABELS[r["baseline"]], "g": f"{int(r['conjuntos_ALEE_gana'])} de 9", "dm": f"{int(sig.get(r['baseline'], 0))} de 9", "w": pf(r["wilcoxon_p_holm"]), "c": pf(r["conover_p_holm"])} for _, r in b.iterrows()]
    ns = b[b["conover_p_holm"] >= 0.05]["baseline"].map(C.MODEL_LABELS).tolist()
    return U.panel("Pruebas estadísticas por pares", html.Div([
        U.table(rows, [("m", "Modelo de comparación"), ("g", "ALEE con menor RMSE en"), ("dm", "DM-HLN significativo en"), ("w", "Wilcoxon (Holm), p"), ("c", "Conover (Holm), p")], numeric=("g", "dm", "w", "c")),
        U.reading(f"ALEE tiene el menor RMSE en los nueve países y Diebold-Mariano rechaza la igualdad en {int((t['dm_p_holm'] < 0.05).sum())} de {len(t)} comparaciones país-modelo.",
                  f"Entre países, Wilcoxon da p = {N(b['wilcoxon_p_holm'].max(), 3)} frente a cada modelo, el mínimo posible con nueve países." + (f" Conover no distingue a ALEE de {', '.join(x.lower() for x in ns)}." if ns else ""),
                  "Nueve países dan una potencia moderada. La comparación incluye los modelos del curso y LightGBM, no todos los métodos de series de tiempo que existen."),
    ]), sub="Valores p con corrección de Holm")
