"""Sección 3: Modelos.

Tabla de líderes, diagnóstico e interpretabilidad del modelo seleccionado,
comparación estadística y la subsección del modelo original con su ficha.
"""
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, dash_table, dcc, html
from sklearn.metrics import auc, average_precision_score, confusion_matrix, precision_recall_curve, roc_curve

from src import config
from src import dash_data as D
from src import dash_plots as P
from src.original_model import EXPERTS, solve_weights

dash.register_page(__name__, path="/modelos", name="Modelos", title="Modelos · Mercado Laboral Colombiano")

DS = D.dataset()
LABELS = {k: v[0] for k, v in D.FEATURE_INFO.items()}
ARTIFACTS = D.artifact_names()
DEFAULT_MODEL = "reg_ALEE"


def model_options():
    opts = []
    for name in ARTIFACTS:
        opts.append({"label": D.artifact_label(name), "value": name})
    return sorted(opts, key=lambda o: (o["value"] != "reg_ALEE", o["label"]))


def card(title, children, cls=""):
    return dbc.Card([dbc.CardHeader(title), dbc.CardBody(children)], className=f"mlc-card {cls}")


# --------------------------------------------------------- modelo original
def _convergence_history():
    pipe = D.model("reg_ALEE").named_steps["model"]
    a = pipe.local_weights(DS.X.values[-1:])[0]
    Q = (pipe.errors_ * a[:, None]).T @ pipe.errors_
    _, hist = solve_weights(Q, pipe.effective_lam(a), w0=np.full(len(EXPERTS), 1 / len(EXPERTS)), prior=pipe.prior_,
                            max_iter=200, tol=0.0, return_history=True)
    return hist


PSEUDOCODE = """Entrada: X (en orden cronológico), y, h, T½, λ
1. Para s = m0..n−1: ajustar los 3 expertos con 0..s−1 y guardar e_s = y_s − f^(s)(x_s)
2. π ← combinación global en el símplex (todos los meses, decaimiento γ = 0.5^(1/T½))
Para predecir en x:
3. a_s ∝ exp(−‖z(x) − z_s‖² / 2h²) · γ^(n−1−s)        z: volatilidad y brecha con la media de 12 meses
4. Q ← Σ a_s e_s e_sᵀ ;  λ̃ ← λ · (n_ref / n_eff) · tr(Q)/K ;  η ← 1 / (2 max|Q_ij|)
5. w ← π; repetir: w ← normalizar[(w ⊙ e^(−2ηQw))^(1/(1+ηλ̃)) ⊙ π^(ηλ̃/(1+ηλ̃))]
6. ŷ(x) = Σ_k w_k f_k(x)"""

RELATED = pd.DataFrame([
    {"Trabajo": "Li y Tkacz (2001)", "Qué hace": "Pesos de combinación por regresión de núcleo",
     "Diferencia de ALEE": "Localiza por un estado de régimen y resuelve un problema convexo en el símplex con previo global"},
    {"Trabajo": "Cerqueira y otros (2017, 2019), ADE", "Qué hace": "Meta-aprendices que predicen el error de cada modelo; softmax",
     "Diferencia de ALEE": "Sin meta-modelo: pesos con solución única y garantías demostradas"},
    {"Trabajo": "Combinación por regímenes (arXiv 2209.01697)", "Qué hace": "Pesos distintos por régimen discreto",
     "Diferencia de ALEE": "Localización continua por núcleo, sin estimar regímenes"},
    {"Trabajo": "Gradiente exponenciado (Kivinen y Warmuth)", "Qué hace": "Actualización multiplicativa en el símplex",
     "Diferencia de ALEE": "Se usa como solucionador; lo nuevo es la localización y la regularización adaptativa"},
])


def _alee_summary():
    """Cifras de ALEE en Colombia frente a los modelos del paper, leídas de los resultados."""
    rmse = D.paper_table().loc["colombia"]
    others = rmse.drop("ALEE")
    tests = D.paper_tests()
    col = tests[tests["conjunto"] == "colombia"]
    return {"alee": rmse["ALEE"], "best": others.idxmin(), "best_rmse": others.min(),
            "reduction": 1 - rmse["ALEE"] / others.min(), "n_sig": int((col["dm_p_holm"] < 0.05).sum()),
            "n_tests": len(col), "max_p": col["dm_p_holm"].max()}


def model_card_text():
    """Ficha del modelo con las cifras del paper."""
    s = _alee_summary()
    between = D.paper_between()
    hp = D.metadata("reg_ALEE")["hyperparameters"]
    return f"""
**Modelo:** ALEE (agregación local entrópica de expertos), tres expertos programados en NumPy:
persistencia, regresión de Nadaraya-Watson del cambio mensual y cambio estacional medio.

**Uso previsto:** pronóstico a un mes de la tasa de desempleo nacional de Colombia con fines académicos.
No sustituye las cifras oficiales del DANE.

**Datos:** 288 meses (2002-2025) con 17 variables construidas solo con información pasada. Se evaluó además
en ocho países con 14 variables nacionales (EE. UU., España, México, Canadá, Australia, Japón, Alemania y Corea).

**Validación:** anidada con `TimeSeriesSplit` (5 pliegues externos y 3 internos, un mes de separación),
optimización bayesiana de 40 evaluaciones y semilla 42.

**Hiperparámetros del artefacto (búsqueda con los 288 meses):** ancho de banda {hp['bandwidth']:.3f},
vida media {hp['half_life']:.1f} meses y λ = {hp['lam']:.4f}.

**Desempeño en Colombia:** RMSE medio por pliegue {s['alee']:.3f}, frente a {s['best_rmse']:.3f} del mejor modelo
de comparación ({D.MODEL_LABELS[s['best']]}): {s['reduction']:.0%} menos error. Diebold-Mariano-HLN con Holm es
significativo frente a los {s['n_sig']} de {s['n_tests']} modelos (p máximo {s['max_p']:.2g}).

**Entre conjuntos:** ALEE tiene el menor RMSE en los nueve conjuntos; el Wilcoxon con Holm da
p = {between['wilcoxon_p_holm'].max():.3f} frente a cada modelo.

**Limitaciones:** nueve conjuntos dan potencia moderada a las pruebas entre conjuntos; en Colombia se evaluaron
diez iteraciones de diseño sobre los mismos 240 meses, por lo que los ocho conjuntos adicionales son la evidencia
más confiable; la localización por estado solo aporta de forma significativa en dos conjuntos; los datos son la
versión vigente de las series y no la publicada en tiempo real.
"""


def original_section():
    s = _alee_summary()
    abl = D.paper_ablation()
    abl_col = abl[abl["conjunto"] == "colombia"]
    cost = pd.read_csv(D.RESULTS_DIR / "costo_computacional.csv")
    cost = cost[cost["modelo"].isin(D.PAPER_MODELS)].assign(
        Modelo=lambda d: d["modelo"].map(D.MODEL_LABELS),
        **{"Entrenamiento (ms)": lambda d: (d["entrenamiento_s"] * 1000).round(1),
           "Inferencia (ms)": lambda d: (d["inferencia_s"] * 1000).round(1),
           "Memoria pico (MB)": lambda d: d["memoria_pico_MB"].round(1), "Parámetros": lambda d: d["parametros"]})
    seeds = pd.read_csv(D.RESULTS_DIR / "sensibilidad_semillas.csv")
    seeds = seeds[seeds["semilla"].astype(str).str.isdigit()]
    seed_range = seeds.groupby("conjunto")["rmse_medio"].agg(lambda v: v.max() - v.min())
    return html.Div([
        html.H2("Modelo original: agregación local entrópica de expertos (ALEE)", className="mlc-section-title original"),
        dbc.Row([
            dbc.Col(card("Idea y limitación que resuelve", [
                html.P("Entre los modelos de aprendizaje automático del segundo entregable, el de menor error cambiaba "
                       "según la volatilidad reciente del mercado laboral. ALEE combina tres expertos programados desde "
                       "cero (persistencia, regresión de Nadaraya-Watson del cambio mensual y cambio estacional medio) "
                       "con pesos que dependen del estado del mercado laboral."),
                html.P("Los pesos resuelven, para cada mes, el problema convexo min_{w ∈ símplex} wᵀQ(x)w + λ̃·KL(w‖π): "
                       "el error cuadrático local de la combinación más una penalización que la acerca a los pesos globales "
                       "cuando hay pocos meses parecidos en la historia."),
                html.Ul([html.Li("P1: solución única (convexidad estricta)."),
                         html.Li("P2: el algoritmo nunca aumenta el objetivo (suavidad relativa y desigualdad de Pinsker)."),
                         html.Li("P3: el error nunca supera el del peor experto."),
                         html.Li("P4: recupera el promedio y la combinación global como casos límite.")]),
            ]), lg=6),
            dbc.Col(card("Pseudocódigo", html.Pre(PSEUDOCODE, className="mlc-code")), lg=6),
        ]),
        dbc.Row([
            dbc.Col(card("Curva de convergencia", dcc.Graph(figure=P.convergence_figure(_convergence_history()),
                                                            config={"displaylogo": False})), lg=6),
            dbc.Col(card("Ablación en Colombia", [
                dcc.Graph(figure=P.ablation_figure(abl_col, D.ABLATION_ORDER, "Colombia"), config={"displaylogo": False}),
                html.Div("Cada barra quita una pieza de ALEE. La mayor parte de la ventaja viene de los expertos: el "
                         "experto estacional incorpora la estructura que los modelos genéricos tienen que aprender. La "
                         "localización por estado solo aporta de forma significativa en México y Japón (página Resultados).",
                         className="mlc-insight"),
            ]), lg=6),
        ]),
        card("Diferencias frente a la literatura", dbc.Table.from_dataframe(RELATED, striped=True, size="sm")),
        dbc.Row([
            dbc.Col(card("Costo computacional (último pliegue de Colombia)", [
                dbc.Table.from_dataframe(cost[["Modelo", "Entrenamiento (ms)", "Inferencia (ms)", "Memoria pico (MB)",
                                               "Parámetros"]], striped=True, size="sm"),
                html.Div("ALEE es el más lento, pero entrena en menos de un segundo: para un pronóstico mensual ese "
                         "costo es irrelevante.", className="mlc-insight"),
            ]), lg=7),
            dbc.Col(card("Robustez y veredicto", [
                html.P(f"Con las semillas 42 a 46 del optimizador, el RMSE de ALEE varía como máximo "
                       f"{seed_range.max():.3f} pp entre semillas en los conjuntos evaluados."),
                dbc.Alert([html.B("Resultado: "),
                           f"en Colombia ALEE tiene un RMSE medio por pliegue de {s['alee']:.3f}, frente a "
                           f"{s['best_rmse']:.3f} del mejor modelo de comparación ({D.MODEL_LABELS[s['best']]}), "
                           f"{s['reduction']:.0%} menos error, y las diferencias son significativas frente a los "
                           f"{s['n_sig']} modelos (Diebold-Mariano-HLN con Holm)."], color="success", className="mb-2"),
                dbc.Button("Ver la ficha del modelo", id="btn-card", color="primary", outline=True, size="sm"),
            ]), lg=5),
        ]),
        dbc.Modal([dbc.ModalHeader(dbc.ModalTitle("Ficha del modelo (model card)")),
                   dbc.ModalBody(dcc.Markdown(model_card_text(), mathjax=False))], id="modal-card", size="xl", scrollable=True),
    ])


layout = html.Div([
    html.H2("Mejores combinaciones", className="mlc-section-title"),
    card("Tabla de líderes", [
        dbc.RadioItems(id="lb-task", value="reg", inline=True, className="mb-2",
                       options=[{"label": "Regresión (RMSE, menor es mejor)", "value": "reg"},
                                {"label": "Clasificación (F1-macro, mayor es mejor)", "value": "clf"}]),
        dash_table.DataTable(id="lb-table", sort_action="native", filter_action="native", page_size=10,
                             style_table={"overflowX": "auto"},
                             style_cell={"fontFamily": "Inter, Segoe UI, Arial", "fontSize": 13, "padding": "6px"},
                             style_header={"backgroundColor": "#0f172a", "color": "white", "fontWeight": "600"},
                             style_data_conditional=[{"if": {"filter_query": "{Estado} contains 'Mejor combinación'"},
                                                      "backgroundColor": "#eff6ff", "fontWeight": "600"},
                                                     {"if": {"filter_query": "{Estado} = 'Modelo original'"},
                                                      "backgroundColor": "#f5f3ff", "fontWeight": "600"}]),
        html.Div(id="lb-txt", className="mlc-insight"),
    ]),

    html.H2("Diagnóstico e interpretabilidad", className="mlc-section-title"),
    card("Modelo", [
        dcc.Dropdown(id="diag-model", options=model_options(), value=DEFAULT_MODEL, clearable=False),
        html.P("Los diagnósticos son fuera de muestra: las 240 predicciones de los cinco pliegues externos de la "
               "validación anidada.", className="small text-muted mt-2 mb-0"),
    ]),
    dcc.Loading(html.Div(id="diag-out")),

    html.H2("Comparación estadística", className="mlc-section-title"),
    dbc.Tabs([
        dbc.Tab(label="Regresión: Model Confidence Set", tab_id="tab-mcs", children=html.Div(id="cmp-mcs", className="pt-3")),
        dbc.Tab(label="Clasificación: diferencia crítica", tab_id="tab-cd", children=html.Div(id="cmp-cd", className="pt-3")),
    ], active_tab="tab-mcs"),

    original_section(),
])


@callback(Output("lb-table", "data"), Output("lb-table", "columns"), Output("lb-txt", "children"), Input("lb-task", "value"))
def update_leaderboard(task):
    df = D.leaderboard(task)
    cols = [{"name": c, "id": c, "type": "numeric" if c in ("Media", "Desviación", "Inferencia (ms)") else "text"}
            for c in ["Modelo", "Balanceo", "Optimizador", "Métrica", "Media ± desv.", "Media", "Desviación",
                      "Inferencia (ms)", "Finalista estadístico", "Estado"]]
    if task == "reg":
        txt = ("Entre las siete mejores combinaciones del segundo entregable, Ridge con Random Search tiene el menor RMSE "
               "medio. ALEE, el modelo original, tiene el menor error de todos. La regresión bayesiana, el SVR con kernel "
               "RBF y LightGBM son los modelos que el paper agregó, ajustados con optimización bayesiana.")
    else:
        fr = D.friedman()
        txt = (f"KNN con SMOTE y algoritmo genético tiene el mayor F1-macro medio. La prueba de Friedman no rechaza "
               f"la igualdad entre los siete (p = {fr['p_value']:.2f}), así que la ventaja no es estadísticamente clara.")
    return df.to_dict("records"), cols, txt


def _reliability(y_bin, p, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    mp, fo, ece = [], [], 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            mp.append(p[m].mean())
            fo.append(y_bin[m].mean())
            ece += m.mean() * abs(p[m].mean() - y_bin[m].mean())
    return mp, fo, ece


@callback(Output("diag-out", "children"), Input("diag-model", "value"))
def update_diagnostics(name):
    task, m = name.split("_", 1)
    sg = D.shap_global(name)
    X_rows = DS.X.iloc[sg["rows"]]
    explained = "del pronóstico" if task == "reg" else "de la probabilidad de la clase Alto"
    shap_fig = P.shap_summary(sg["phi"], X_rows, LABELS, f"Importancia global: valores de Shapley {explained}")
    top = pd.Series(np.abs(sg["phi"]).mean(axis=0), index=sg["columns"]).sort_values(ascending=False)
    shap_txt = (f"Las variables con mayor impacto medio son {LABELS[top.index[0]].lower()}, {LABELS[top.index[1]].lower()} "
                f"y {LABELS[top.index[2]].lower()}. Cada punto es un mes; el color indica si el valor de la variable era alto o bajo.")
    df = D.oof(task)
    if task == "reg":
        col = f"pred_{m}"
        resid = (df["y_true"] - df[col]).values
        rmse = np.sqrt(np.mean(resid ** 2))
        from src.explain import acf

        a1 = acf(resid, 1)[1]
        diag = [
            card("Observado frente a pronosticado", [dcc.Graph(figure=P.real_vs_pred(df, col, D.MODEL_LABELS.get(m, m)), config={"displaylogo": False}),
                                                     html.Div(f"RMSE agrupado de las 240 predicciones: {rmse:.3f} pp. "
                                                              "Los mayores errores se concentran en el choque de 2020.", className="mlc-insight")]),
            card("Residuos", [dcc.Graph(figure=P.residual_panels(resid, df[col].values), config={"displaylogo": False}),
                              html.Div(f"La autocorrelación de los residuos en el rezago 1 es {a1:.2f}"
                                       + (": queda dependencia temporal que el modelo no captura." if abs(a1) > 1.96 / np.sqrt(len(resid))
                                          else ": no hay dependencia de primer orden apreciable."), className="mlc-insight")]),
        ]
    else:
        y = df["y_true"].values.astype(int)
        pred = df[f"pred_{m}"].values.astype(int)
        proba = df[[f"proba_{m}_{c}" for c in config.CLASS_LABELS]].values
        cm = confusion_matrix(y, pred, labels=[0, 1, 2])
        roc, pr, cal = {}, {}, {}
        for k, c in enumerate(config.CLASS_LABELS):
            yb = (y == k).astype(int)
            fpr, tpr, _ = roc_curve(yb, proba[:, k])
            prec, rec, _ = precision_recall_curve(yb, proba[:, k])
            roc[c] = (fpr, tpr, auc(fpr, tpr))
            pr[c] = (rec, prec, average_precision_score(yb, proba[:, k]))
            cal[c] = _reliability(yb, proba[:, k])
        alto_recall = cm[2, 2] / max(cm[2].sum(), 1)
        diag = [
            dbc.Row([
                dbc.Col(card("Matriz de confusión", [dcc.Graph(figure=P.confusion_figure(cm, config.CLASS_LABELS), config={"displaylogo": False}),
                                                      html.Div(f"El modelo detecta {alto_recall:.0%} de los meses de desempleo Alto "
                                                               f"({cm[2, 2]} de {cm[2].sum()}).", className="mlc-insight")]), lg=6),
                dbc.Col(card("Calibración", [dcc.Graph(figure=P.calibration_figure(cal), config={"displaylogo": False}),
                                              html.Div("Cuanto más cerca de la diagonal, más confiables son las probabilidades. "
                                                       "ECE: error de calibración esperado.", className="mlc-insight")]), lg=6),
            ]),
            dbc.Row([
                dbc.Col(card("Curvas ROC", [dcc.Graph(figure=P.curves_figure(roc, "roc"), config={"displaylogo": False}),
                                             html.Div("Una curva por clase frente a las otras dos.", className="mlc-insight")]), lg=6),
                dbc.Col(card("Precisión y exhaustividad", [dcc.Graph(figure=P.curves_figure(pr, "pr"), config={"displaylogo": False}),
                                                            html.Div("Con clases desbalanceadas, esta curva es más exigente que la ROC.",
                                                                     className="mlc-insight")]), lg=6),
            ]),
        ]
    return html.Div(diag + [card("Interpretabilidad global", [dcc.Graph(figure=shap_fig, config={"displaylogo": False}),
                                                                   html.Div(shap_txt, className="mlc-insight")])])


def mcs_text():
    mcs = D.mcs()
    out = mcs.loc[~mcs["en_conjunto"].astype(bool), "modelo"].map(lambda m: D.MODEL_LABELS.get(m, m)).tolist()
    best = D.MODEL_LABELS.get(mcs.sort_values("rango")["modelo"].iloc[0])
    txt = (f"Las siete mejores combinaciones, ALEE y los tres modelos que agregó el paper, con la pérdida cuadrática de "
           f"las 240 predicciones de Colombia, α = 0.10 y bootstrap por bloques de 12 meses. El de menor pérdida es {best}. ")
    if out:
        return txt + f"El MCS elimina a {len(out)} modelo(s): {', '.join(out)}."
    return txt + "El MCS no elimina a ninguno: con una sola serie de 240 meses no hay evidencia suficiente para separarlos."


@callback(Output("cmp-mcs", "children"), Output("cmp-cd", "children"), Input("lb-task", "value"))
def update_comparison(_):
    mcs = D.mcs().copy()
    mcs["modelo"] = mcs["modelo"].map(lambda m: D.MODEL_LABELS.get(m, m))
    mcs = mcs.rename(columns={"modelo": "Modelo", "perdida_media": "Pérdida media (MSE)", "rango": "Rango",
                              "p_valor_mcs": "p-valor MCS", "en_conjunto": "En el conjunto de confianza"})
    mcs["En el conjunto de confianza"] = mcs["En el conjunto de confianza"].map({True: "Sí", False: "No"})
    mcs = mcs.round(4)
    mcs_block = [dbc.Table.from_dataframe(mcs, striped=True, hover=True, size="sm"),
                 html.Div(mcs_text(), className="mlc-insight")]
    fr = D.friedman()
    cd_block = [dcc.Graph(figure=P.cd_diagram(fr["avg_ranks"], fr["cd"], D.MODEL_LABELS), config={"displaylogo": False}),
                html.Div(f"Friedman con la mejor variante de cada clasificador en los {fr['n_blocks']} pliegues: χ² = {fr['statistic']:.2f}, "
                         f"p = {fr['p_value']:.3f}. Todos los rangos caben dentro de una diferencia crítica de {fr['cd']:.2f}: "
                         "Nemenyi no distingue a ningún par. El modelo original es de regresión y no aparece en este diagrama.",
                         className="mlc-insight")]
    return mcs_block, cd_block


@callback(Output("modal-card", "is_open"), Input("btn-card", "n_clicks"), State("modal-card", "is_open"), prevent_initial_call=True)
def toggle_card(n, is_open):
    return not is_open
