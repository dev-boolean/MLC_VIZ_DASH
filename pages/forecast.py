"""04. Unemployment Forecast: la predicción del mes siguiente."""
import dash
from dash import Input, Output, State, callback, html

from labor import catalog as C
from labor import engine as E
from labor import figs as F
from labor import ui as U

dash.register_page(__name__, path="/forecast", name="Unemployment Forecast", title="Unemployment Forecast · LABOR INSIGHTS")
N = C.num

layout = html.Div([
    U.head("04", "UNEMPLOYMENT FORECAST", "Un mes adelante,", "con su margen de error.", "ALEE solo se entrenó y se evaluó a un paso. Por eso aquí hay un pronóstico, el del mes que sigue al último dato, y nada más allá."),
    html.Div(U.loading(html.Div(id="fc-body")), className="li-wrap"),
])


@callback(Output("fc-body", "children"), Input("pais", "value"))
@U.safe
def body(c):
    f, conf = E.forecast(c), E.conformal(c)
    later = E.later_observations(c)
    arrow = "▲" if f["change"] > 0.005 else "▼" if f["change"] < -0.005 else "■"
    tr = E.recent_track(c, 24)
    inside = int((tr["error"].abs() <= conf["hw"]).sum())
    sign = int(((tr["ALEE"] - (tr["f_persistencia"])) * (tr["y_true"] - tr["f_persistencia"]) > 0).sum())
    kids = [
        html.Div([
            html.Div([
                html.Div([html.Span("PRONÓSTICO DE ALEE", className="fk"), html.Span("Dato estimado", className="chip est")], className="k-top"),
                html.Div([html.Span(N(f["pred"]), className="num", **{"data-count": f"{f['pred']:.6f}", "data-dec": "2", "data-sign": "0"}), html.Span("%", className="unit")], className="fc-big"),
                html.Div(f"{C.name(c)}, {C.month_long(f['target_date'])}", className="fc-when"),
                html.Div([html.Span(arrow, className="fc-arrow"), html.Span(f"{N(f['change'], 2, True)} pp frente al último dato")], className="fc-delta"),
                html.Dl([
                    html.Dt("Último dato observado"), html.Dd(f"{N(f['observed'])} %", className="obs"),
                    html.Dt("Fecha de observación"), html.Dd(C.month_long(f["last_date"]), className="obs"),
                    html.Dt("Fecha estimada"), html.Dd(C.month_long(f["target_date"]), className="est"),
                    html.Dt("Intervalo del 90 %"), html.Dd(f"{N(f['lo'])} a {N(f['hi'])} %", className="unc"),
                    html.Dt("Mismo mes, un año antes"), html.Dd(f"{N(f['same_month'])} %"),
                ], className="fc-dl"),
            ], className="fc-side"),
            html.Div([
                html.Div([U.seg("fc-span", [("2 años", 24), ("5 años", 60), ("10 años", 120), ("Todo", 0)], 60, "Ver"),
                          html.Button("↻ Repetir animación", id="fc-replay", className="btn-li ghost sm", **{"data-replay": "1"})], className="row-tools"),
                U.graph(F.forecast_main(c, 60, 470, posterior=True), id="fc-fig", cls="forecast-anim"),
                html.Div([html.Span("Observado", className="lg obs"), html.Span("Pronóstico, un mes", className="lg est"), html.Span("Intervalo del 90 %", className="lg unc")] + ([html.Span("Publicado después del corte", className="lg later")] if later is not None else []), className="lgs"),
            ], className="fc-chart"),
        ], className="li-panel fc-hero reveal"),
    ]
    if later is not None:
        r = later.iloc[0]
        e = r["tasa_desempleo_nacional"] - f["pred"]
        kids.append(U.panel("Contraste con datos que el modelo no vio", html.Div([
            html.Div([
                html.Div([html.Small(f"ALEE pronosticó para {C.month_long(f['target_date'])}"), html.B(f"{N(f['pred'])} %", className="est")]),
                html.Div([html.Small("El DANE publicó después"), html.B(f"{N(r['tasa_desempleo_nacional'], 1)} %", className="obs")]),
                html.Div([html.Small("Error"), html.B(f"{N(e, 2, True)} pp")]),
                html.Div([html.Small("¿Dentro del intervalo?"), html.B("Sí" if abs(e) <= conf["hw"] else "No")]),
            ], className="contrast"),
            html.P(["La serie de la plataforma llega hasta diciembre de 2025. El equipo recopiló la tasa nacional de enero a septiembre de 2026, pero los indicadores de las 13 áreas de esos meses no están en el proyecto y ALEE los necesita en Colombia. Por eso los demás meses de 2026 se muestran como referencia y no se pronostican. ",
                    "Un solo mes no valida ni invalida el modelo."], className="li-cap"),
            html.Div([html.A(C.month_short(x["fecha"]) + f": {N(x['tasa_desempleo_nacional'], 1)} %", href=x["fuente_principal"], target="_blank", rel="noopener") for _, x in later.iterrows()], className="src-chips"),
        ]), sub="Cada cifra enlaza a la fuente que registró el equipo", cls="warnline"))
    kids += [
        U.grid(
            U.panel("De dónde sale la incertidumbre", html.Div([
                html.P(["El intervalo se obtiene por ", html.B("predicción conforme"), f": se toman los {conf['n']} errores que ALEE cometió fuera de muestra y se usa el cuantil del 90 % de su tamaño, {N(conf['hw'])} pp, a cada lado del pronóstico."]),
                html.Div([html.Div([html.B(f"{N(100 * conf['coverage'], 1)} %"), html.Small(f"cobertura real, medida hacia adelante en {conf['n_forward']} meses")]), html.Div([html.B("90 %"), html.Small("cobertura nominal")]), html.Div([html.B(f"{N(conf['width'])} pp"), html.Small("ancho medio en esa medición")])], className="cov"),
                html.P("Los errores de una serie de tiempo dependen unos de otros, así que la cobertura no está garantizada: por eso se mide. El ancho es el mismo en meses tranquilos y en meses agitados.", className="li-cap"),
            ], className="li-prose")),
            U.panel("Qué empuja el pronóstico", html.Div([U.graph(F.contribution(f, 300)), html.P(f"El experto estacional propone {N(f['experts'][2])} %, el núcleo {N(f['experts'][1])} % y la persistencia {N(f['experts'][0])} %.", className="li-cap")]), sub="Último dato más el aporte de cada experto"),
        ),
        U.panel("Su historial reciente", html.Div([U.graph(F.track(c)), U.reading(
            f"En los últimos {len(tr)} meses de prueba, el error absoluto medio fue de {N(tr['error'].abs().mean())} pp.",
            f"{inside} de {len(tr)} observaciones quedaron a menos de {N(conf['hw'])} pp del pronóstico y en {sign} acertó la dirección del cambio.",
            "Muestra cómo se habría comportado en la práctica: cada pronóstico se hizo solo con meses anteriores.")]), sub="Pronósticos fuera de muestra, los últimos 36 meses evaluados. La banda usa el semiancho actual"),
    ]
    if c == "eeuu":
        kids.append(U.note("Octubre de 2025 no se publicó por el cierre del gobierno federal. Se interpoló entre septiembre y noviembre y así entra en las variables.", "warn", "Dato interpolado."))
    return kids


@callback(Output("fc-fig", "figure"), Input("fc-span", "value"), State("pais", "value"), prevent_initial_call=True)
def span(months, c):
    return F.forecast_main(c, months, 470, posterior=months in (24, 60))
