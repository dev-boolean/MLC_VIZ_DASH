"""03. Model Intelligence: cómo funciona ALEE y cómo se evaluó."""
import dash
import numpy as np
import pandas as pd
from dash import Input, Output, callback, dcc, html

from labor import catalog as C
from labor import engine as E
from labor import figs as F
from labor import ui as U
from src import config
from src import dash_data as D
from src import dash_plots as P
from src.dash_data import FEATURE_INFO

dash.register_page(__name__, path="/modelo", name="Model Intelligence", title="Model Intelligence · LABOR INSIGHTS")
N = C.num
FEAT = {k: v[0] for k, v in FEATURE_INFO.items()}
STAGES = ["datos", "variables", "expertos", "estado", "pesos", "pronostico"]
METRICS = {"RMSE": ("Raíz del error cuadrático medio", "pp", "Penaliza más los errores grandes. Es la métrica principal, fijada antes de evaluar."),
           "MAE": ("Error absoluto medio", "pp", "Cuánto se equivoca, en promedio, en puntos porcentuales."),
           "MAPE": ("Error porcentual absoluto medio", "%", "El error como porcentaje de la tasa observada."),
           "R2": ("Coeficiente de determinación", "", "Parte de la variación de la prueba que el modelo explica. 1 es perfecto; puede ser negativo."),
           "MASE": ("Error absoluto escalado", "", "Error relativo al de repetir el mismo mes del año anterior, medido en el entrenamiento. Menor que 1 es mejor.")}

layout = html.Div([
    U.head("03", "MODEL INTELLIGENCE", "Tres expertos,", "una decisión convexa.",
           "ALEE (agregación local entrópica de expertos) combina tres pronosticadores sencillos. Cada mes decide cuánto confiar en cada uno según qué tan bien les fue en meses con un estado parecido."),
    html.Div([html.Div(className="li-flow", **{"aria-hidden": "true"}), html.Span("Representación conceptual", className="illus")], className="flow-band"),
    html.Div([
        U.block("3.1 · ARQUITECTURA", "Del dato al pronóstico", "Toque cada etapa: los valores son los del pronóstico vigente del país seleccionado."),
        html.Div([U.seg("mi-stage", [(f"{i + 1}", s) for i, s in enumerate(STAGES)], "datos"), html.Div(id="mi-arch")], className="li-panel arch reveal"),
        U.loading(html.Div(id="mi-explain")),
        U.loading(html.Div(id="mi-eval")),
        U.block("3.5 · LABORATORIO", "Las 140 combinaciones de Colombia", "El segundo entregable comparó 14 modelos base, cuatro técnicas de balanceo y cuatro optimizadores. Aquí están las mejores de cada modelo, en regresión y en clasificación (Bajo, menos de 10 %; Medio, 10 a 14 %; Alto, 14 % o más)."),
        U.panel("Mejores combinaciones", html.Div([
            U.seg("lab-task", [("Regresión", "reg"), ("Clasificación", "clf")], "reg"), html.Div(id="lab-board"),
        ]), sub="Media de los cinco pliegues externos. Estos resultados son solo de Colombia"),
        U.panel("Diagnóstico de un modelo", html.Div([
            html.Div([html.Label("Modelo", htmlFor="lab-model"), dcc.Dropdown(id="lab-model", clearable=False, searchable=False, className="li-dd")], className="li-ctl"),
            U.loading(html.Div(id="lab-diag")),
        ]), sub="240 predicciones fuera de muestra y valores de Shapley"),
    ], className="li-wrap"),
])


@callback(Output("mi-arch", "children"), Input("mi-stage", "value"), Input("pais", "value"))
@U.safe
def arch(stage, c):
    f, m, info = E.forecast(c), E.model(c), E.meta(c)
    x = E.features(c).iloc[-1]
    labels = {"datos": ("Datos", f"{len(E.series(c))} meses"), "variables": ("Variables", f"{len(info['features'])} por mes"),
              "expertos": ("3 expertos", "NumPy, sin librerías de modelos"), "estado": ("Estado", "volatilidad y brecha"),
              "pesos": ("Pesos", "problema convexo"), "pronostico": ("Pronóstico", f"{N(f['pred'])} %")}
    nodes = []
    for i, s in enumerate(STAGES):
        if i:
            nodes.append(html.Span(className="a-link" + (" hot" if STAGES.index(stage) >= i else "")))
        nodes.append(html.Div([html.Span(f"{i + 1}", className="a-n"), html.B(labels[s][0]), html.Small(labels[s][1])],
                              role="button", tabIndex=0, className=f"a-node {s}" + (" on" if s == stage else " done" if STAGES.index(stage) > i else "")))
    area = " y los tres indicadores de las 13 áreas" if c == "colombia" else ""
    txt = {
        "datos": ("Entran solo meses ya observados", [html.P(f"La serie mensual de {C.name(c)} va de {C.month_long(E.series(c).index[0])} a {C.month_long(f['last_date'])}. {U.source(c)}"),
                                                      html.P("La ocupación y la participación nacionales del mismo mes no entran: determinan el desempleo por identidad contable." if c == "colombia" else "No existen indicadores de área comparables, así que el modelo de este país usa 14 variables y no 17.")]),
        "variables": ("Todo se construye con información pasada", [html.P(f"Rezagos (1, 2, 3, 6 y 12), medias y desviaciones móviles (3, 6 y 12), mes codificado con seno y coseno, tendencia{area}."),
                                                                  html.Div([html.Span([html.Small(FEAT.get(k, k)), html.B(N(v, 0 if k == "trend_idx" else 2))]) for k, v in x.items()], className="a-feats")]),
        "expertos": ("Tres maneras simples de ver el mes siguiente", [html.Div([
            html.Div([html.B(C.EXPERT_LABELS[n]), html.P(t), html.Span(f"{N(f['experts'][i])} %")], className=f"a-exp {n}") for i, (n, t) in enumerate([
                ("persistencia", "Repite el último dato."), ("nucleo", "Suma al último dato el cambio de los meses con variables parecidas (Nadaraya-Watson)."),
                ("estacional", "Suma el cambio medio histórico de ese mes del año.")])], className="a-exps")]),
        "estado": ("¿A qué meses se parece el actual?", [html.P(f"El estado tiene dos números: la volatilidad de los últimos tres meses ({N(f['state'][0])} pp) y la brecha frente a la media de 12 meses ({N(f['state'][1], 2, True)} pp)."),
                                                         html.P(f"Los meses parecidos y recientes pesan más (ancho de banda {N(info['hp']['bandwidth'])}, vida media {N(info['hp']['half_life'], 1)} meses). El tamaño efectivo de esa muestra es de {N(f['n_eff'], 1)} meses, frente a {N(f['n_ref'], 1)} si solo contara la cercanía en el tiempo.")]),
        "pesos": ("Minimizar el error local sin alejarse demasiado de lo global", [
            html.Div("min wᵀ Q(x) w + λ(x) · KL(w ‖ π),  con w en el símplex", className="a-eq"),
            html.P("Q(x) son los errores pasados de los expertos en meses parecidos. El término KL acerca los pesos a la combinación global π, con más fuerza cuando hay pocos meses parecidos. La solución es única y se obtiene por descenso de espejo."),
            html.Div([html.Div([html.Small(C.EXPERT_LABELS[n]), html.Div(html.Div(className=f"fill {n}", style={"width": f"{100 * f['weights'][i]:.1f}%"}), className="bar"), html.B(f"{N(100 * f['weights'][i], 1)} %"), html.Small(f"global {N(100 * f['prior'][i], 1)} %")], className="a-w") for i, n in enumerate(C.EXPERT_LABELS)])]),
        "pronostico": (f"{N(f['pred'])} % en {C.month_long(f['target_date'])}", [html.P("El pronóstico es el promedio de los tres expertos con esos pesos. Por construcción, su error nunca supera el del peor experto."), html.Div(F_graph(f), className="a-fig")]),
    }[stage]
    return [html.Div(nodes, className="a-row"), html.Div([html.H3(txt[0]), *txt[1]], className="a-detail")]


def F_graph(f):
    return U.graph(F.contribution(f, 260))


@callback(Output("mi-explain", "children"), Input("pais", "value"))
@U.safe
def explain(c):
    f = E.forecast(c)
    top = f["neighbours"].index[0]
    k = int(np.argmax(f["weights"]))
    blocks = [
        U.block("3.2 · EXPLICABILIDAD", "Por qué da ese número"),
        U.grid(
            U.panel("Lo que propone cada experto", html.Div([U.graph(F.votes(c)), html.P(f"La línea azul es el último dato; la cian, el pronóstico. Hoy pesa más «{C.EXPERT_LABELS[list(C.EXPERT_LABELS)[k]].lower()}» ({N(100 * f['weights'][k], 0)} %).", className="li-cap")]), sub="El tamaño del punto es su peso"),
            U.panel("Descomposición exacta", html.Div([U.graph(F.contribution(f, 250)), html.P("Último dato más el aporte de cada experto (peso por cambio propuesto). La suma es el pronóstico, sin aproximaciones.", className="li-cap")])),
        ),
        U.grid(
            U.panel("Los meses a los que hoy se parece", html.Div([U.graph(F.state_map(c)), html.P(f"El mes con más peso es {C.month_long(top)} ({N(100 * f['neighbours'].iloc[0], 1)} %).", className="li-cap")]), sub="Espacio de estados con el que ALEE localiza"),
            U.panel("Pesos a lo largo de la validación", html.Div([U.graph(F.weights_time(c, 380)), html.P("Fuera de muestra: cada peso se calculó solo con meses anteriores. Las líneas separan los pliegues.", className="li-cap")])),
        ),
    ]
    if c == "colombia":
        blocks.append(U.grid(
            U.panel("Importancia por permutación", U.graph(F.permutation_fig()), sub="Último pliegue de Colombia, 20 permutaciones"),
            U.panel("Valores de Shapley", U.graph(_shap("reg_ALEE", 300)), sub="96 meses de Colombia. El color es el valor de la variable"),
        ))
    else:
        blocks.append(U.note("La importancia por permutación y los valores de Shapley solo se calcularon para Colombia. En los demás países la explicación es la descomposición exacta por expertos.", "info"))
    return blocks


def _shap(name, height=420):
    sg = D.shap_global(name)
    fig = P.shap_summary(sg["phi"], D.dataset().X.iloc[sg["rows"]], FEAT, "", top=8)
    fig.update_layout(height=height, title=None, margin=dict(l=190, r=10, t=8, b=44))
    return fig


@callback(Output("mi-eval", "children"), Input("pais", "value"))
@U.safe
def evaluate(c):
    met = E.metrics()
    mine = met[(met["conjunto"] == c) & (met["modelo"] == "ALEE")]
    best, b = E.best_baseline(c)
    d = E.oof(c)
    err = (d["y_true"] - d["ALEE"])
    worst = d.loc[err.abs().nlargest(6).index]
    cards = [html.Div([html.Div([html.Span(k if k != "R2" else "R²", className="k-label"), html.Span(v[0], className="chip ref")], className="k-top"),
                       html.Div([html.Span(N(mine[k].mean(), 3 if k != "MAPE" else 2), className="num", **{"data-count": f"{mine[k].mean():.6f}", "data-dec": "3" if k != "MAPE" else "2", "data-sign": "0"}), html.Span(v[1], className="unit")], className="k-val model"),
                       html.Div(f"± {N(mine[k].std(), 3 if k != 'MAPE' else 2)} entre pliegues", className="k-foot"), html.Div(v[2], className="k-help")], className="li-kpi reveal") for k, v in METRICS.items()]
    t = E.result("pruebas_por_baseline")
    t = t[t["conjunto"] == c]
    pf = lambda p: "< 0,001" if p < 0.001 else N(p, 3)  # noqa: E731
    rows = [{"m": C.MODEL_LABELS[r["baseline"]], "r": N(E.metric_table().loc[c, r["baseline"]], 3), "d": f"{N(r['dif_perdida_media'], 3)}  [{N(r['ic95_inf'], 3)}; {N(r['ic95_sup'], 3)}]",
             "dm": pf(r["dm_p_holm"]), "gw": pf(r["gw_p_holm"]), "pb": N(r["p_bayes_mejor"], 2), "cd": N(r["d_cohen"], 2), "a": N(r["a12_vargha"], 2)} for _, r in t.iterrows()]
    abl = E.result("ablacion")
    a = abl[(abl["conjunto"] == c) & (abl["dm_p_holm"] < 0.05)]
    worse = [v.lower() for v in a[a["dif_perdida"] < 0]["variante"]]
    mcs = E.result("mcs")
    mk = mcs[(mcs["conjunto"] == c) & mcs["en_conjunto"].astype(bool)]["modelo"]
    mk = [m for m in mk if m in C.MODEL_LABELS]
    return html.Div([
        U.block("3.3 · EVALUACIÓN", f"Cuánto se equivoca en {C.name(c)}", f"Validación anidada de origen móvil: cinco pliegues, {len(d)} meses de prueba ({d['fecha'].min().year} a {d['fecha'].max().year}), un mes de separación y búsqueda bayesiana de 40 evaluaciones en el bucle interno."),
        html.Div(cards, className="li-kpis five"),
        U.panel("Pronosticado frente a observado, fuera de muestra", html.Div([U.graph(F.pred_vs_obs(c, best)), html.P(f"En gris, el mejor modelo de comparación de este país: {C.MODEL_LABELS[best]}.", className="li-cap")])),
        U.grid(
            U.panel("Error en cada pliegue", U.graph(F.fold_errors(c)), sub="Un solo promedio esconde que el error cambia con la época"),
            U.panel("Los meses más difíciles", U.table([{"a": C.month_long(r["fecha"]).capitalize(), "b": f"{N(r['y_true'])} %", "c": f"{N(r['ALEE'])} %", "d": f"{N(r['y_true'] - r['ALEE'], 2, True)} pp"} for _, r in worst.iterrows()],
                                                           [("a", "Mes"), ("b", "Observado"), ("c", "Pronóstico"), ("d", "Error")], numeric=("b", "c", "d")), sub="Seis mayores errores absolutos. Ningún estado observado anticipa un choque sin precedentes"),
        ),
        U.block("3.4 · COMPARACIÓN", "Frente a los modelos de referencia", "Ocho modelos de regresión del curso y LightGBM, con las mismas particiones, variables y presupuesto de ajuste."),
        U.grid(
            U.panel("Métrica por modelo", html.Div([U.seg("mi-metric", [(("R²" if k == "R2" else k), k) for k in METRICS], "RMSE"), html.Div(U.graph(F.model_bars(c), id="mi-bars"))])),
            U.panel("Error por régimen de volatilidad", U.graph(F.regimes(c, 440)), sub=f"En azul, {C.MODEL_LABELS[best]}. En gris, los demás"),
        ),
        U.panel("Pruebas de significancia", html.Div([
            U.table(rows, [("m", "Modelo"), ("r", "RMSE"), ("d", "Diferencia de pérdida [IC 95 %]"), ("dm", "DM-HLN, p"), ("gw", "GW, p"), ("pb", "P(ALEE mejor)"), ("cd", "d de Cohen"), ("a", "A₁₂")], numeric=("r", "d", "dm", "gw", "pb", "cd", "a")),
            html.P(f"Valores p con corrección de Holm. Diferencia negativa: ALEE se equivoca menos. La probabilidad bayesiana (ROPE de ±1 %) usa solo cinco pliegues dependientes y es orientativa. El conjunto de confianza de modelos conserva {len(mk)} de 10 modelos en este país"
                   + (", solo ALEE." if len(mk) == 1 else "; es decir, esta prueba no descarta a " + ("todos los competidores." if len(mk) > 1 else "")), className="li-cap"),
        ])),
        U.panel("Ablación: qué pasa si se quita una pieza", html.Div([
            U.graph(F.ablation(c)),
            html.P((f"Con significancia son peores que ALEE completo: {', '.join(worse)}. " if worse else "Ninguna variante es significativamente peor. ")
                   + "La ventaja viene sobre todo de los expertos; la localización por estado, la idea central, solo aporta con significancia en México y Japón.", className="li-cap"),
        ]), sub="Naranja rayado: diferencia significativa (Diebold-Mariano con Holm)"),
    ])


@callback(Output("mi-bars", "figure"), Input("mi-metric", "value"), Input("pais", "value"))
def bars(metric, c):
    return F.model_bars(c, metric)


# ------------------------------------------------------- laboratorio (Colombia)
@callback(Output("lab-board", "children"), Output("lab-model", "options"), Output("lab-model", "value"), Input("lab-task", "value"))
@U.safe
def board(task):
    df = D.leaderboard(task)
    rows = [{"m": r["Modelo"], "b": r["Balanceo"], "o": r["Optimizador"], "v": r["Media ± desv."].replace(".", ","), "f": r["Finalista estadístico"], "e": r["Estado"],
             "_cls": "hl" if r["Estado"] == "Modelo original" else ""} for _, r in df.iterrows()]
    metric = "RMSE (menor es mejor)" if task == "reg" else "F1-macro (mayor es mejor)"
    fr = D.friedman()
    txt = ("ALEE es el de menor RMSE medio. El conjunto de confianza de modelos, con 240 meses muy heterogéneos, no descarta a ninguno." if task == "reg"
           else f"KNN con SMOTE y algoritmo genético tiene el mayor F1-macro, pero Friedman no distingue a los siete (p = {N(fr['p_value'])}). ALEE es un modelo de regresión y no participa aquí.")
    opts = [{"label": D.artifact_label(a).split(" · ")[1], "value": a} for a in D.artifact_names() if a.startswith(task + "_")]
    return [U.table(rows, [("m", "Modelo"), ("b", "Balanceo"), ("o", "Optimizador"), ("v", metric), ("f", "Finalista"), ("e", "Estado")], numeric=("v",)), html.P(txt, className="li-cap")], opts, ("reg_ALEE" if task == "reg" else "clf_KNN")


def _fig(fig, h=None):
    fig.update_layout(title=None, margin=dict(t=34, l=fig.layout.margin.l or 56, r=14, b=48))
    if h:
        fig.update_layout(height=h)
    return U.graph(fig)


@callback(Output("lab-diag", "children"), Input("lab-model", "value"))
@U.safe
def diag(name):
    if not name:
        return None
    from sklearn.metrics import auc, average_precision_score, confusion_matrix, precision_recall_curve, roc_curve

    from src.explain import acf

    task, m = name.split("_", 1)
    df = D.oof(task)
    if task == "reg":
        r = (df["y_true"] - df[f"pred_{m}"]).values
        a1 = acf(r, 1)[1]
        out = [html.H4("Observado frente a pronosticado"), _fig(P.real_vs_pred(df, f"pred_{m}", D.MODEL_LABELS.get(m, m)), 360), html.H4("Residuos"), _fig(P.residual_panels(r, df[f"pred_{m}"].values)),
               html.P(f"RMSE agrupado de {N(np.sqrt(np.mean(r ** 2)), 3)} pp. Autocorrelación de los residuos en el rezago 1: {N(a1)}.", className="li-cap")]
    else:
        y, pred = df["y_true"].values.astype(int), df[f"pred_{m}"].values.astype(int)
        pr = df[[f"proba_{m}_{k}" for k in config.CLASS_LABELS]].values
        cm = confusion_matrix(y, pred, labels=[0, 1, 2])
        roc, prc = {}, {}
        for k, lab in enumerate(config.CLASS_LABELS):
            yb = (y == k).astype(int)
            fp, tp, _ = roc_curve(yb, pr[:, k])
            p, rc, _ = precision_recall_curve(yb, pr[:, k])
            roc[lab], prc[lab] = (fp, tp, auc(fp, tp)), (rc, p, average_precision_score(yb, pr[:, k]))
        out = [html.Div([html.Div([html.H4("Matriz de confusión"), _fig(P.confusion_figure(cm, config.CLASS_LABELS))]), html.Div([html.H4("Curvas ROC"), _fig(P.curves_figure(roc, "roc"))]),
                         html.Div([html.H4("Precisión y exhaustividad"), _fig(P.curves_figure(prc, "pr"))])], className="li-grid g3"),
               html.P(f"Detecta {cm[2, 2]} de los {cm[2].sum()} meses de desempleo Alto.", className="li-cap")]
    return out + [html.H4("Importancia global (valores de Shapley)"), U.graph(_shap(name, 400))]
