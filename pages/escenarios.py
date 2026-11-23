"""05. Scenario Lab: escenarios hipotéticos ejecutados con el modelo original."""
import dash
from dash import ALL, Input, Output, callback, dcc, html

from labor import catalog as C
from labor import engine as E
from labor import figs as F
from labor import ui as U
from src.dash_data import FEATURE_INFO

dash.register_page(__name__, path="/escenarios", name="Scenario Lab", title="Scenario Lab · LABOR INSIGHTS")
N = C.num

layout = html.Div([
    U.head("05", "SCENARIO LAB", "¿Y si el dato", "hubiera sido otro?", "Cambie los datos recientes y ALEE vuelve a pronosticar. No hay fórmulas de atajo: cada resultado sale del modelo ya entrenado."),
    html.Div([
        html.Div([
            html.Aside([html.Div([html.H3("Panel de variables"), html.Button("Restablecer", id="sc-reset", className="btn-li ghost sm", n_clicks=0)], className="sc-head"),
                        html.Div(id="sc-controls")], className="li-panel sc-ctl reveal"),
            html.Div(U.loading(html.Div(id="sc-out")), className="sc-res"),
        ], className="sc-grid"),
        U.note(["Los rezagos, las medias y las volatilidades son funciones de la misma serie. Moverlos por separado crearía un mes imposible, así que los controles cambian los últimos tres datos y la plataforma recalcula juntas todas las variables. El mes, la tendencia y el modelo entrenado no cambian. ",
                "Un escenario describe cómo responde el modelo; no dice qué pasaría en la economía ni por qué."], "info", "Cómo leerlo."),
        html.Div(id="sc-sens"),
    ], className="li-wrap"),
])


def _slider(idx, label, base, lim, unit, orig):
    return html.Div([html.Div([html.Span(label), html.Span(orig, className="o")], className="sl-top"),
                     dcc.Slider(id={"type": "sc", "k": idx}, min=-lim if unit == "pp" else base[0], max=lim if unit == "pp" else base[1], step=0.1, value=0 if unit == "pp" else base[2], marks=None,
                                included=False, updatemode="drag", tooltip={"placement": "bottom", "always_visible": False})], className="sl")


@callback(Output("sc-controls", "children"), Output("sc-sens", "children"), Input("pais", "value"), Input("sc-reset", "n_clicks"))
@U.safe
def controls(c, _):
    s, lim = E.series(c), E.step_limit(c)
    ctl = [html.P(f"Datos recientes de {C.name(c)}. Cada control mueve el valor observado hasta ±{N(lim, 1)} pp, el percentil 99 de los cambios mensuales históricos.", className="li-cap")]
    for k in (1, 2, 3):
        ctl.append(_slider(f"d{k}", C.month_long(s.index[-k]).capitalize(), None, lim, "pp", f"{N(s.iloc[-k])} %"))
    if c == "colombia":
        ctl += [html.H4("13 áreas metropolitanas"), html.P("Se mueven dentro de su rango de entrenamiento.", className="li-cap")]
        rng, last = E.training_range(c), E.features(c).iloc[-1]
        for col, lab in E.AREA_LABELS.items():
            ctl.append(_slider(col, lab, (round(rng.loc[col, "min"], 1), round(rng.loc[col, "max"], 1), round(float(last[col]), 1)), None, "%", f"{N(last[col])} %"))
    else:
        ctl.append(U.note("Este país no tiene indicadores de área: el modelo solo usa su propia serie.", "info"))
    sens = U.panel("Sensibilidad al último dato", html.Div([U.graph(F.sensitivity_fig(c)), U.reading(*_sens_text(c))]), sub="Solo cambia el último mes; los demás quedan como se observaron")
    return ctl, sens


def _sens_text(c):
    d, f = E.sensitivity(c), E.forecast(c)
    slope = (d["pred"].iloc[-1] - d["pred"].iloc[0]) / (d["ultimo"].iloc[-1] - d["ultimo"].iloc[0])
    line = d["pred"].iloc[0] + slope * (d["ultimo"] - d["ultimo"].iloc[0])
    bend = float((d["pred"] - line).abs().max() / (d["ultimo"].max() - d["ultimo"].min()))
    rngw = d["w_estacional"].max() - d["w_estacional"].min()
    return (f"Con el último dato entre {N(d['ultimo'].min(), 1)} y {N(d['ultimo'].max(), 1)} %, el pronóstico va de {N(d['pred'].min())} a {N(d['pred'].max())} %.",
            f"En promedio, un punto más en el último dato se traduce en {N(slope)} puntos más en el pronóstico. El peso del experto estacional cambia hasta {N(100 * rngw, 0)} puntos a lo largo de la curva.",
            ("Los tramos donde la curva se dobla son cambios de estado: el mes deja de parecerse a los mismos antecedentes y ALEE reparte los pesos de otra manera." if bend > 0.03
             else "La curva es casi recta: en este país y en este mes, la respuesta de ALEE apenas depende del estado y traslada el último dato más el cambio del calendario."))


@callback(Output("sc-out", "children"), Input({"type": "sc", "k": ALL}, "value"), Input("pais", "value"))
@U.safe
def run(values, c):
    ids = [i["id"]["k"] for i in (dash.ctx.inputs_list[0] or [])]
    v = dict(zip(ids, values))
    if not v or v.get("d1") is None:
        return None
    if (c == "colombia") != ("tasa_desempleo_area" in v):  # los controles aún son los del país anterior
        raise dash.exceptions.PreventUpdate
    f = E.forecast(c)
    last = E.features(c).iloc[-1]
    area = {k: v[k] for k in E.AREA_LABELS if k in v and abs(v[k] - round(float(last[k]), 1)) > 1e-9}
    deltas = [v.get(f"d{k}") or 0 for k in (1, 2, 3)]
    sc = E.scenario(c, deltas, area)
    changed = any(deltas) or bool(area)
    diff = sc["pred"] - f["pred"]
    last_alt = sc["recent"][-1]
    rows = [{"a": C.EXPERT_LABELS[n], "b": f"{N(f['experts'][i])} %", "c": f"{N(100 * f['weights'][i], 1)} %", "d": f"{N(sc['experts'][i])} %", "e": f"{N(100 * sc['weights'][i], 1)} %"} for i, n in enumerate(C.EXPERT_LABELS)]
    x0 = E.features(c).iloc[-1]
    moved = [k for k in x0.index if abs(sc["x"][k] - x0[k]) > 1e-9]
    frow = [{"a": FEATURE_INFO.get(k, (k,))[0], "b": N(x0[k]), "c": N(sc["x"][k]), "d": N(sc["x"][k] - x0[k], 2, True)} for k in moved]
    return [
        html.Div([
            html.Div([html.Small("Escenario base"), html.B(f"{N(f['pred'])} %", className="est"), html.Span(f"último dato {N(f['observed'])} %")]),
            html.Div([html.Small("Escenario alternativo"), html.B(f"{N(sc['pred'])} %", className="model"), html.Span(f"último dato {N(last_alt)} %")]),
            html.Div([html.Small("Diferencia"), html.B(f"{N(diff, 2, True)} pp"), html.Span("alternativo menos base")]),
        ], className="sc-kpis"),
        html.Div("Escenario hipotético generado por el modelo." if changed else "Mueva un control para crear un escenario.", className="sc-tag"),
        html.Div([U.graph(F.scenario_fig(c, sc))], className="li-panel pad"),
        U.note("El escenario saca del rango de entrenamiento a: " + ", ".join(FEATURE_INFO.get(k, (k,))[0].lower() for k in sc["outside"]) + ". El modelo extrapola y su respuesta es menos confiable.", "warn", "Extrapolación.") if sc["outside"] else None,
        html.Div([html.H4("Qué cambió por dentro"), U.table(rows, [("a", "Experto"), ("b", "Base"), ("c", "Peso base"), ("d", "Escenario"), ("e", "Peso escenario")], numeric=("b", "c", "d", "e")),
                  html.Details([html.Summary(f"Ver las {len(frow)} variables que cambiaron"), U.table(frow, [("a", "Variable"), ("b", "Original"), ("c", "Modificada"), ("d", "Diferencia")], numeric=("b", "c", "d"))]) if frow else None], className="li-panel pad"),
    ]
