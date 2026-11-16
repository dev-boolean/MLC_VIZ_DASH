"""Resultados del paper: ALEE frente a los ocho modelos de regresión del curso y LightGBM, en nueve conjuntos.

Solo lee las tablas que genera ``experiments/resultados_curso.py`` en
``artifacts/resultados/solo_curso/``.
"""
import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, callback, dash_table, dcc, html

from src import dash_data as D
from src import dash_plots as P

dash.register_page(__name__, path="/resultados", name="Resultados", title="Resultados · Mercado Laboral Colombiano")

METRICS = {"RMSE": "RMSE (pp, menor es mejor)", "MAE": "MAE (pp, menor es mejor)", "MAPE": "MAPE (%, menor es mejor)",
           "R2": "R² (mayor es mejor)", "MASE": "MASE (menor es mejor)"}
TABLE_STYLE = dict(
    style_table={"overflowX": "auto"},
    style_cell={"fontFamily": "Inter, Segoe UI, Arial", "fontSize": 13, "padding": "6px", "textAlign": "right"},
    style_cell_conditional=[{"if": {"column_id": "Conjunto"}, "textAlign": "left", "fontWeight": "600"}],
    style_header={"backgroundColor": "#0f172a", "color": "white", "fontWeight": "600"},
)


def card(title, children, cls=""):
    return dbc.Card([dbc.CardHeader(title), dbc.CardBody(children)], className=f"mlc-card {cls}")


def p_fmt(p):
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


def summary():
    rmse = D.paper_table()
    others = rmse.drop(columns="ALEE")
    wins = int((rmse["ALEE"] < others.min(axis=1)).sum())
    red = 1 - rmse["ALEE"] / others.min(axis=1)
    tests = D.paper_tests()
    fr = D.paper_friedman()
    return {"wins": wins, "n": len(rmse), "red_min": red.min(), "red_min_at": D.DATASETS[red.idxmin()],
            "red_max": red.max(), "red_max_at": D.DATASETS[red.idxmax()], "n_tests": len(tests),
            "dm_sig": int((tests["dm_p_holm"] < 0.05).sum()), "gw_sig": int((tests["gw_p_holm"] < 0.05).sum()),
            "dm_max": tests["dm_p_holm"].max(), "gw_max": tests["gw_p_holm"].max(), "fr": fr}


S = summary()
DATASET_OPTS = [{"label": v, "value": k} for k, v in D.DATASETS.items()]

layout = html.Div([
    html.H2("Resultados en nueve conjuntos de datos", className="mlc-section-title"),
    html.P("ALEE se comparó con los ocho modelos de regresión del curso (KNN, Ridge, Lasso, regresión bayesiana, árbol de "
           "decisión, Random Forest, XGBoost y SVR con kernel RBF) y con LightGBM, con la misma validación anidada en la "
           "tasa de desempleo de Colombia y en ocho países más. Los ocho conjuntos adicionales se evaluaron una sola vez, "
           "con el modelo ya congelado."),
    dbc.Row([
        dbc.Col(html.Div([html.Div(f"{S['wins']} de {S['n']}", className="mlc-kpi-value"),
                          html.Div("conjuntos en los que ALEE tiene el menor RMSE", className="mlc-kpi-label")],
                         className="mlc-kpi-card"), md=4),
        dbc.Col(html.Div([html.Div(f"{S['red_min']:.0%} a {S['red_max']:.0%}", className="mlc-kpi-value"),
                          html.Div("menos error que el mejor modelo de comparación", className="mlc-kpi-label")],
                         className="mlc-kpi-card"), md=4),
        dbc.Col(html.Div([html.Div(f"{S['dm_sig']} de {S['n_tests']}", className="mlc-kpi-value"),
                          html.Div("comparaciones significativas (Diebold-Mariano-HLN, Holm)", className="mlc-kpi-label")],
                         className="mlc-kpi-card"), md=4),
    ], className="g-3 mb-3"),

    card("Desempeño por conjunto", [
        dbc.RadioItems(id="res-metric", value="RMSE", inline=True, className="mb-2",
                       options=[{"label": v, "value": k} for k, v in METRICS.items()]),
        dash_table.DataTable(id="res-table", **TABLE_STYLE),
        html.Div(f"Media de los cinco pliegues externos. La reducción del RMSE frente al mejor modelo de comparación va de "
                 f"{S['red_min']:.0%} ({S['red_min_at']}) a {S['red_max']:.0%} ({S['red_max_at']}).", className="mlc-insight"),
    ]),
    card("Mapa de errores", dcc.Graph(figure=P.relative_heatmap(D.paper_table(), D.DATASETS, D.MODEL_LABELS),
                                      config={"displaylogo": False})),

    html.H2("Pruebas de significancia", className="mlc-section-title"),
    dbc.Row([
        dbc.Col(card("Por conjunto", [
            dbc.Label("Conjunto", html_for="res-dataset"),
            dcc.Dropdown(id="res-dataset", options=DATASET_OPTS, value="colombia", clearable=False, className="mb-2"),
            dash_table.DataTable(id="res-tests", **TABLE_STYLE),
            html.Div(f"Diebold-Mariano con la corrección de Harvey-Leybourne-Newbold y Giacomini-White condicional, ambos con "
                     f"Holm sobre los nueve contrastes de cada conjunto. En los {S['n_tests']} contrastes los dos rechazan "
                     f"(p máximos {p_fmt(S['dm_max'])} y {p_fmt(S['gw_max'])}). La diferencia de pérdida es la de ALEE "
                     "menos la del modelo (negativa: ALEE se equivoca menos), con su intervalo bootstrap del 95 %.",
                     className="mlc-insight"),
        ]), lg=12),
        dbc.Col(card("Entre conjuntos", [
            dcc.Graph(figure=P.cd_diagram(S["fr"]["avg_ranks"], S["fr"]["cd"], D.MODEL_LABELS), config={"displaylogo": False}),
            html.Div(f"Friedman con Iman-Davenport rechaza la igualdad de los diez modelos (χ² = {S['fr']['chi2']:.1f}, "
                     f"p = {p_fmt(S['fr']['p_value'])}). ALEE tiene rango medio "
                     f"{S['fr']['avg_ranks']['ALEE']:.2f}; la diferencia crítica de Nemenyi es {S['fr']['cd']:.2f}.",
                     className="mlc-insight"),
            html.Div(id="res-between"),
        ]), lg=12),
    ]),

    html.H2("Ablación y regímenes de volatilidad", className="mlc-section-title"),
    dbc.Label("Conjunto", html_for="res-dataset-2"),
    dcc.Dropdown(id="res-dataset-2", options=DATASET_OPTS, value="colombia", clearable=False, className="mb-2"),
    dbc.Row([
        dbc.Col(card("Ablación", [dcc.Graph(id="res-ablation", config={"displaylogo": False}),
                                  html.Div(id="res-ablation-txt", className="mlc-insight")]), lg=6),
        dbc.Col(card("Error por régimen", [dcc.Graph(id="res-regime", config={"displaylogo": False}),
                                           html.Div("La volatilidad es la desviación de los últimos tres meses, dividida en "
                                                    "terciles. ALEE aparece en violeta.", className="mlc-insight")]), lg=6),
    ]),
])


@callback(Output("res-table", "data"), Output("res-table", "columns"), Output("res-table", "style_data_conditional"),
          Input("res-metric", "value"))
def update_table(metric):
    t = D.paper_table(metric)
    higher_better = metric == "R2"
    best = t.idxmax(axis=1) if higher_better else t.idxmin(axis=1)
    df = t.round(3).rename(index=D.DATASETS).rename(columns=D.MODEL_LABELS).reset_index().rename(columns={"conjunto": "Conjunto"})
    cols = [{"name": c, "id": c} for c in df.columns]
    style = [{"if": {"row_index": i, "column_id": D.MODEL_LABELS[m]}, "backgroundColor": "#f5f3ff", "fontWeight": "700"}
             for i, m in enumerate(best.values)]
    return df.to_dict("records"), cols, style


@callback(Output("res-tests", "data"), Output("res-tests", "columns"), Output("res-between", "children"),
          Input("res-dataset", "value"))
def update_tests(dataset):
    t = D.paper_tests()
    t = t[t["conjunto"] == dataset]
    df = pd.DataFrame({
        "Modelo": t["baseline"].map(D.MODEL_LABELS),
        "Dif. de pérdida": t["dif_perdida_media"].round(3).astype(str) + " [" + t["ic95_inf"].round(3).astype(str)
                           + ", " + t["ic95_sup"].round(3).astype(str) + "]",
        "DM-HLN p (Holm)": t["dm_p_holm"].map(p_fmt), "GW p (Holm)": t["gw_p_holm"].map(p_fmt),
        "P(ALEE mejor), ROPE ±1 %": t["p_bayes_mejor"].round(3), "d de Cohen": t["d_cohen"].round(2),
        "A12": t["a12_vargha"].round(2),
    })
    b = D.paper_between()
    between = html.Div([
        html.Div("Wilcoxon (Holm) y Conover (Holm) de ALEE frente a cada modelo en los nueve conjuntos:",
                 className="small fw-semibold mt-2"),
        dbc.Table.from_dataframe(pd.DataFrame({
            "Modelo": b["baseline"].map(D.MODEL_LABELS), "Gana ALEE": b["conjuntos_ALEE_gana"].astype(str) + " / 9",
            "Wilcoxon p": b["wilcoxon_p_holm"].map(p_fmt), "Conover p": b["conover_p_holm"].map(p_fmt)}),
            striped=True, size="sm", className="small"),
    ])
    return df.to_dict("records"), [{"name": c, "id": c} for c in df.columns], between


@callback(Output("res-ablation", "figure"), Output("res-ablation-txt", "children"), Output("res-regime", "figure"),
          Input("res-dataset-2", "value"))
def update_ablation(dataset):
    label = D.DATASETS[dataset]
    abl = D.paper_ablation()
    abl = abl[abl["conjunto"] == dataset]
    sig = abl[abl["dm_p_holm"] < 0.05]
    # dif_perdida = pérdida de ALEE completo menos la de la variante: negativa si la variante es peor.
    worse = sig[sig["dif_perdida"] < 0]["variante"].tolist()
    better = sig[sig["dif_perdida"] > 0]["variante"].tolist()
    txt = "Ninguna variante difiere de ALEE completo con significancia." if sig.empty else (
        (f"Con significancia, son peores que ALEE completo: {', '.join(worse).lower()}. " if worse else "")
        + (f"Con significancia, son mejores: {', '.join(better).lower()}." if better else ""))
    reg = D.paper_regimes()
    return (P.ablation_figure(abl, D.ABLATION_ORDER, label), txt,
            P.regime_figure(reg[reg["conjunto"] == dataset], D.MODEL_LABELS, label))
