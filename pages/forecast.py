"""04. Unemployment Forecast: el pronóstico del mes siguiente, su incertidumbre y su validación."""
import dash
import numpy as np
from dash import Input, Output, State, callback, html

from labor import catalog as C
from labor import engine as E
from labor import figs as F
from labor import ui as U

dash.register_page(__name__, path="/forecast", name="Unemployment Forecast", title="Unemployment Forecast · LABOR INSIGHTS")
N = C.num

layout = html.Div([
    U.head("04", "Unemployment Forecast", "El mes siguiente,", "con su margen de error.",
           "ALEE solo se entrenó y se evaluó a un paso. Por eso aquí hay un único pronóstico, el del mes que sigue al último dato observado, con su intervalo del 90 % y la evidencia de cómo le ha ido."),
    html.Div(U.loading(html.Div(id="fc-body")), className="li-wrap"),
])


def _yes(v):
    return "Sí" if v else "No"


@callback(Output("fc-body", "children"), Input("pais", "value"))
@U.safe
def body(c):
    f, conf = E.forecast(c), E.conformal(c)
    later = E.later_observations(c)
    val = E.later_validation(c)
    arrow = "▲" if f["change"] > 0.005 else "▼" if f["change"] < -0.005 else "■"
    tr = E.recent_track(c, 24)
    has = tr["hw"].notna()
    inside = int(tr.loc[has, "dentro"].sum())
    sign = int(((tr["ALEE"] - tr["f_persistencia"]) * (tr["y_true"] - tr["f_persistencia"]) > 0).sum())

    hero = html.Div([
        html.Div([
            html.Div([html.Span("Pronóstico de ALEE · un mes adelante", className="fk"), html.Span("Pronosticado", className="chip est")], className="k-top"),
            html.Div([html.Span(N(f["pred"]), className="num", **{"data-count": f"{f['pred']:.6f}", "data-dec": "2", "data-sign": "0"}), html.Span("%", className="unit")], className="fc-big"),
            html.Div(f"{C.name(c)} · {C.month_long(f['target_date'])}", className="fc-when"),
            html.Div([html.Span(arrow, className="fc-arrow", **{"aria-hidden": "true"}), html.Span(f"{N(f['change'], 2, True)} pp frente al último dato"), html.Span("Calculado", className="chip calc")], className="fc-delta"),
            html.Div([html.Div([html.Small("Límite inferior"), html.B(f"{N(f['lo'])} %")]), html.Div([html.Small("Intervalo del 90 %"), html.Span(className="fc-bar")]),
                      html.Div([html.Small("Límite superior"), html.B(f"{N(f['hi'])} %")])], className="fc-int"),
            html.Dl([
                html.Dt("Último dato observado"), html.Dd(f"{N(f['observed'])} %", className="obs"),
                html.Dt("Periodo de observación"), html.Dd(C.month_long(f["last_date"]), className="obs"),
                html.Dt("Periodo pronosticado"), html.Dd(C.month_long(f["target_date"]), className="est"),
                html.Dt("Mismo mes, un año antes"), html.Dd(f"{N(f['same_month'])} %", className="obs"),
            ], className="fc-dl"),
        ], className="fc-side"),
        html.Div([
            html.Div(U.seg("fc-span", [("2 años", 24), ("5 años", 60), ("10 años", 120), ("Toda la serie", 0)], 60, "Ventana histórica"), className="row-tools"),
            U.graph(F.forecast_main(c, 60, 470, posterior=True), id="fc-fig", cls="forecast-anim"),
            U.legend(("Observado", "obs"), ("Pronóstico, un mes", "est"), ("Intervalo del 90 %", "unc"), *([("Publicado después del corte (no usado)", "later")] if later is not None else [])),
            html.P("La ventana solo cambia cuánta historia se muestra; no es un horizonte de pronóstico.", className="li-cap"),
        ], className="fc-chart"),
    ], className="li-panel fc-hero reveal")

    # Validación con datos oficiales publicados después del corte
    if len(val):
        ok = val[val["pred"].notna()]
        rows = [{"a": C.month_long(r["origen"]).capitalize(), "b": C.month_long(r["objetivo"]).capitalize(), "c": f"{N(r['pred'])} %", "d": f"{N(r['obs'], 1)} %",
                 "e": f"{N(r['error'], 2, True)} pp", "f": f"{N(r['lo'])} – {N(r['hi'])} %", "g": _yes(r["dentro"]), "h": html.A("Fuente", href=r["fuente"], target="_blank", rel="noopener")} for _, r in ok.iterrows()]
        ref = val[val["pred"].isna()]
        validation = U.panel("Validación con datos publicados después del corte", html.Div([
            U.table(rows, [("a", "Origen (último dato usado)"), ("b", "Mes pronosticado"), ("c", "Pronóstico"), ("d", "Observado publicado"), ("e", "Error"), ("f", "Intervalo del 90 %"), ("g", "¿Dentro?"), ("h", "Publicación")],
                    numeric=("c", "d", "e", "f"), cls="hl-first"),
            html.Div([
                html.P([html.B("Cuándo se hizo y cuándo se supo. "), f"El pronóstico usa la serie hasta {C.month_long(f['last_date'])}. La cifra oficial de {C.month_long(f['target_date'])} se publicó después y no entró en el modelo ni en el intervalo."]),
                html.P([html.B("Qué no se compara. "), f"El equipo recopiló también {len(ref)} meses posteriores ({C.month_short(ref['objetivo'].min())} a {C.month_short(ref['objetivo'].max())}). "
                        "No se pronostican: ALEE de Colombia necesita los indicadores de las 13 áreas de esos meses, que no están en el proyecto. Se muestran en la gráfica solo como referencia."]) if len(ref) else None,
                html.P("Un solo mes no valida ni invalida el modelo: la evidencia de desempeño es la evaluación fuera de muestra de más abajo.", className="li-cap"),
            ], className="li-prose"),
        ]), sub="Pronóstico hecho antes de conocer el dato oficial", cls="validation")
    else:
        validation = U.panel("Validación con datos publicados después del corte", html.Div([
            U.note(f"La serie de {C.name(c)} llega hasta {C.month_long(f['last_date'])}. El dato oficial de {C.month_long(f['target_date'])} no está todavía en el proyecto, así que este pronóstico aún no se puede contrastar. "
                   "La evidencia de desempeño es la evaluación fuera de muestra de más abajo.", "info"),
        ]), sub="Pronóstico hecho antes de conocer el dato oficial", cls="validation")

    uncertainty = U.panel("De dónde sale la incertidumbre", html.Div([
        html.Div([
            html.Div([html.Small("Cobertura nominal"), html.B("90 %"), html.Span("objetivo del método")]),
            html.Div([html.Small("Cobertura empírica"), html.B(f"{N(100 * conf['coverage'], 1)} %"), html.Span(f"medida hacia adelante en {conf['n_forward']} meses")]),
            html.Div([html.Small("Ancho medio"), html.B(f"{N(conf['width'])} pp"), html.Span("en esa misma medición")]),
            html.Div([html.Small("Errores usados"), html.B(str(conf["n"])), html.Span("fuera de muestra, para el intervalo vigente")]),
            html.Div([html.Small("Semiancho vigente"), html.B(f"± {N(conf['hw'])} pp"), html.Span("a cada lado del pronóstico")]),
        ], className="cov"),
        html.P(["El intervalo es ", html.B("conforme partido"), f" (split conformal): se toman los {conf['n']} errores que ALEE cometió fuera de muestra en la validación anidada y su cuantil ajustado del 90 % en valor absoluto, "
                f"{N(conf['hw'])} pp, se suma y se resta al pronóstico. Es simétrico y tiene el mismo ancho en cualquier mes."]),
        U.note("Los errores de una serie de tiempo dependen unos de otros, así que la cobertura del 90 % no está garantizada: por eso se mide hacia adelante, recalculando el intervalo de cada mes solo con los errores anteriores. "
               "El ancho tampoco crece en meses agitados como los de 2020.", "warn", "Limitación."),
        U.disclose("Cálculo exacto", U.math(r"""
Con los errores fuera de muestra $r_i = y_i - \hat y_i$, $i = 1,\dots,n$, el semiancho es el cuantil empírico de orden $\lceil (n+1)\cdot 0{,}9\rceil / n$ de $|r_i|$:

$$q = |r|_{(\lceil (n+1)\,0{,}9\rceil)},\qquad \text{intervalo} = [\,\hat y_{t+1} - q,\ \hat y_{t+1} + q\,].$$

La cobertura empírica se mide hacia adelante: para cada mes $t > 48$ se calcula $q_t$ solo con $r_1,\dots,r_{t-1}$ y se cuenta si $|r_t| \le q_t$.
""")),
    ], className="li-prose"))

    contribution = U.panel("Qué empuja el pronóstico", html.Div([U.graph(F.contribution(f, 320)), html.P(
        f"El experto estacional propone {N(f['experts'][2])} %, el núcleo del cambio {N(f['experts'][1])} % y la persistencia {N(f['experts'][0])} %. "
        f"Cada barra es el peso del experto por el cambio que propone; sumadas al último dato dan el pronóstico.", className="li-cap")]), sub="Descomposición exacta por expertos")

    rows = [{"a": C.month_short(r["origen"]), "b": C.month_short(r["fecha"]), "c": f"{N(r['ALEE'])} %", "d": f"{N(r['y_true'])} %", "e": f"{N(r['error'], 2, True)} pp",
             "f": (f"{N(r['ALEE'] - r['hw'])} – {N(r['ALEE'] + r['hw'])} %" if not np.isnan(r["hw"]) else "—"), "g": _yes(r["dentro"]) if not np.isnan(r["hw"]) else "—",
             "_cls": "" if (np.isnan(r["hw"]) or r["dentro"]) else "out"} for _, r in tr.iloc[::-1].iterrows()]
    history = U.panel("Historial reciente fuera de muestra", html.Div([
        U.graph(F.track(c, 36)),
        U.reading(f"En los últimos {len(tr)} meses de prueba, el error absoluto medio fue de {N(tr['error'].abs().mean())} pp.",
                  f"{inside} de {int(has.sum())} observaciones cayeron dentro del intervalo disponible en su momento, y en {sign} de {len(tr)} ALEE acertó la dirección del cambio frente a repetir el último dato.",
                  "Cada pronóstico se hizo solo con meses anteriores y cada intervalo, solo con errores anteriores: así se habría comportado en la práctica."),
        U.disclose(f"Tabla detallada de los últimos {len(tr)} pronósticos", U.table(rows, [("a", "Origen"), ("b", "Mes pronosticado"), ("c", "Pronóstico"), ("d", "Observado"), ("e", "Error"), ("f", "Intervalo del 90 %"), ("g", "¿Dentro?")],
                                                                           numeric=("c", "d", "e", "f")), open=True),
    ]), sub="Validación anidada: pronósticos e intervalos calculados con información anterior a cada mes")

    kids = [hero, validation, U.grid(uncertainty, contribution), history]
    if c == "eeuu":
        kids.append(U.note("Octubre de 2025 no se publicó por el cierre del gobierno federal. Se interpoló entre septiembre y noviembre y así entra en las variables.", "warn", "Dato interpolado."))
    return kids


@callback(Output("fc-fig", "figure"), Input("fc-span", "value"), State("pais", "value"), prevent_initial_call=True)
def span(months, c):
    return F.forecast_main(c, months, 470, posterior=months in (24, 60))
