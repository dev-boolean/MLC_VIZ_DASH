"""03. Model Intelligence: arquitectura, fundamentos matemáticos y diagnósticos de ALEE."""
import dash
import numpy as np
from dash import Input, Output, callback, dcc, html

from labor import catalog as C
from labor import engine as E
from labor import figs as F
from labor import ui as U
from src import dash_data as D
from src import dash_plots as P
from src.dash_data import FEATURE_INFO

dash.register_page(__name__, path="/modelo", name="Model Intelligence", title="Model Intelligence · LABOR INSIGHTS")
N = C.num
FEAT = {k: v[0] for k, v in FEATURE_INFO.items()}
METRICS = {"RMSE": ("Raíz del error cuadrático medio", "pp", "Penaliza más los errores grandes. Es la métrica principal, fijada antes de evaluar."),
           "MAE": ("Error absoluto medio", "pp", "Cuánto se equivoca, en promedio, en puntos porcentuales."),
           "MAPE": ("Error porcentual absoluto medio", "%", "El error como porcentaje de la tasa observada."),
           "R2": ("Coeficiente de determinación", "", "Parte de la variación de la prueba que el modelo explica. 1 es perfecto; puede ser negativo."),
           "MASE": ("Error absoluto escalado", "", "Error relativo al de repetir el mismo mes del año anterior, medido en el entrenamiento. Menor que 1 es mejor.")}

# Fundamentos: ecuaciones y notación del paper (sección «Modelo propuesto: ALEE») y de src/original_model.py.
MATH = [
    ("Formulación del problema", r"""
Sea $y_t$ la tasa de desempleo en el mes $t$ y $\mathbf{x}_t$ un vector construido solo con información disponible al cierre del mes $t$. ALEE aprende $f$ tal que

$$\hat y_{t+1} = f(\mathbf{x}_t),$$

sin usar nunca información del mes $t+1$. La pérdida de referencia es el error cuadrático medio, $\mathcal{L}_{\mathrm{MSE}} = \frac{1}{n}\sum_t (y_{t+1}-\hat y_{t+1})^2$.
"""),
    ("Variables y estado del mercado laboral", r"""
$$\ell_{t,k} = y_{t-k+1},\ k\in\{1,2,3,6,12\};\qquad \bar y_{t,m} = \frac{1}{m}\sum_{j=0}^{m-1} y_{t-j},\qquad s_{t,m} = \Big(\frac{1}{m-1}\sum_{j=0}^{m-1}(y_{t-j}-\bar y_{t,m})^2\Big)^{1/2},\ m\in\{3,6,12\}$$

más el mes codificado como $(\sin(2\pi\,\mathrm{mes}_t/12), \cos(2\pi\,\mathrm{mes}_t/12))$, un índice de tendencia y, en Colombia, los tres indicadores de las 13 áreas metropolitanas: 17 variables en Colombia y 14 en los demás países.

El **estado** con el que ALEE localiza es $z(\mathbf{x}_t) = (s_{t,3},\ y_t - \bar y_{t,12})$, estandarizado: la volatilidad reciente y la brecha del último dato frente a su media anual.
"""),
    ("Los tres expertos", r"""
- **Persistencia:** $f_1 = y_t$.
- **Núcleo del cambio:** $f_2 = y_t + \widehat{\Delta y}(\mathbf{x})$, con $\widehat{\Delta y}$ una regresión de Nadaraya-Watson del cambio mensual sobre las variables estandarizadas (núcleo gaussiano, ancho por la heurística de la mediana).
- **Cambio estacional:** $f_3 = y_t$ más el cambio medio histórico del mismo mes del año, encogido hacia cero con una observación ficticia.

Cada experto se ajusta solo con información anterior al punto que pronostica. Sus errores fuera de muestra $e_{s,k} = y_s - f_k^{(s)}(\mathbf{x}_s)$ se registran en ventana expansiva.
"""),
    ("Pesos locales y error local", r"""
Los meses pasados reciben pesos de localización

$$a_s(\mathbf{x}) \propto \exp\!\left(-\frac{\lVert z(\mathbf{x})-z_s\rVert^2}{2h^2}\right)\gamma^{\,n-1-s},\qquad \gamma = 0{,}5^{1/\tau},$$

con ancho de banda $h$ y vida media $\tau$. Con ellos se forma la matriz de errores locales

$$Q(\mathbf{x}) = \sum_s a_s(\mathbf{x})\, e_s e_s^{\top},$$

semidefinida positiva, de modo que $w^{\top}Q(\mathbf{x})\,w$ es el error cuadrático local de la combinación $w$.
"""),
    ("Combinación convexa regularizada", r"""
$$w^*(\mathbf{x}) = \arg\min_{w\in\Delta_K}\; w^{\top}Q(\mathbf{x})\,w + \tilde\lambda(\mathbf{x})\,\mathrm{KL}(w\,\Vert\,\pi),\qquad \tilde\lambda(\mathbf{x}) = \lambda\,\frac{\operatorname{tr} Q(\mathbf{x})}{K}\,\frac{n_{\mathrm{ref}}}{n_{\mathrm{eff}}(a)}$$

$\pi$ son los pesos globales (la misma combinación sin localización), $n_{\mathrm{eff}}(a) = 1/\sum_s a_s^2$ es el tamaño efectivo de la muestra local y $n_{\mathrm{ref}}$ el de los pesos con solo decaimiento temporal. Cuando hay pocos meses parecidos, $\tilde\lambda$ crece y la combinación se acerca a $\pi$. El pronóstico es

$$\hat y = \sum_k w_k^*\, f_k(\mathbf{x}).$$
"""),
    ("Algoritmo: descenso de espejo proximal", r"""
Con paso $\eta = 1/L$, $L = 2\max_{ij}|Q_{ij}|$, y partiendo de $w = \pi$:

$$w \leftarrow \left(w \odot e^{-2\eta Q w}\right)^{1/(1+\eta\tilde\lambda)} \odot \pi^{\eta\tilde\lambda/(1+\eta\tilde\lambda)},\quad \text{normalizado a } \textstyle\sum_k w_k = 1,$$

hasta que $\lVert w_{\text{nuevo}} - w\rVert_1 < 10^{-10}$ o 500 iteraciones. El modelo no tiene parámetros entrenables por gradiente: lo que aprende es el historial de errores.
"""),
    ("Propiedades y complejidad", r"""
No son resultados nuevos de optimización: se derivan de resultados conocidos y se verifican numéricamente en las pruebas del código.

1. **Existencia y unicidad.** Si $\tilde\lambda>0$, el problema tiene una única solución, en el interior de $\Delta_K$.
2. **Descenso monótono.** Con $\eta = 1/L$, cada iteración no aumenta el objetivo (suavidad relativa a la entropía y desigualdad de Pinsker).
3. **Error acotado.** $|y-\hat y| \le \max_k |y-f_k(\mathbf{x})|$ y $(y-\hat y)^2 \le \sum_k w_k^*(y-f_k(\mathbf{x}))^2$: el error nunca supera el del peor experto.
4. **Casos límite.** Con $\lambda\to\infty$ se recuperan los pesos globales $\pi$; sin localización, los pesos no dependen de $\mathbf{x}$.

**Complejidad:** entrenar cuesta $O(n^3p)$ en tiempo y $O(n^2)$ en memoria; pronosticar un mes cuesta $O(np)$.
"""),
]

# Diagrama de arquitectura: cajas y flechas en assets/alee_arquitectura.svg (lienzo de 1160 x 640) y, encima,
# el contenido de cada caja compuesto en LaTeX con MathJax. (x, y, ancho, alto, clase, LaTeX)
VB_W, VB_H = 1160, 640
_ACC = {"á": r"\acute{a}", "é": r"\acute{e}", "ó": r"\acute{o}", "ú": r"\acute{u}", "ñ": r"\tilde{n}", "Á": r"\acute{A}", "Ó": r"\acute{O}", "Ú": r"\acute{U}"}


def _t(s, font="mathrm"):
    """Texto en la fuente de LaTeX: tildes como acentos de TeX (no caen en una fuente del sistema)."""
    for k, v in _ACC.items():
        s = s.replace(k, v)
    s = s.replace(" ", "\\ ").replace(":", "{:}")  # {:} = dos puntos de texto, sin espacio de relación
    return "\\" + font + "{" + s + "}"


def _box(title, *lines):
    return r"\begin{array}{l}" + _t(title, "mathbf") + r"\\[3pt]" + r"\\[2pt]".join(lines) + r"\end{array}"


ARCH = [
    (20, 10, 600, 26, "lbl", _t("RUTA DEL PRONÓSTICO", "mathsf")),
    (20, 344, 760, 26, "lbl", _t("RUTA DE LOS PESOS: CUÁNTO CONFIAR EN CADA EXPERTO", "mathsf")),
    (20, 80, 235, 200, "", _box("1. Entrada", r"\mathbf{x}_t:\ " + _t("solo meses ") + r"s\le t", r"\ell_{t,k},\ k\in\{1,2,3,6,12\}",
                                r"\bar y_{t,m},\ s_{t,m},\ m\in\{3,6,12\}", _t("mes ") + r"(\sin,\cos)" + _t(" y tendencia"),
                                _t("Colombia: ") + r"+3" + _t(" de 13 áreas"))),
    (300, 46, 270, 78, "", _box("2a. Persistencia", r"f_1(\mathbf{x}) = y_t")),
    (300, 138, 270, 78, "", _box("2b. Núcleo del cambio", r"f_2(\mathbf{x}) = y_t + \widehat{\Delta y}_{\mathrm{NW}}(\mathbf{x})")),
    (300, 230, 270, 78, "", _box("2c. Cambio estacional", r"f_3(\mathbf{x}) = y_t + \overline{\Delta y}_{\,\mathrm{mes}}")),
    (574, 146, 60, 26, "lbl", r"f_k(\mathbf{x})"),
    (640, 100, 275, 150, "", _box("4. Combinación convexa", r"\hat y = \displaystyle\sum_{k=1}^{3} w_k^*(\mathbf{x})\, f_k(\mathbf{x})",
                                  r"w^*\in\Delta_K:\ w_k\ge 0,\ \textstyle\sum_k w_k = 1")),
    (960, 115, 180, 120, "", _box("5. Pronóstico", r"\hat y_{t+1}", _t("un mes adelante"))),
    (20, 375, 235, 120, "", _box("3a. Estado", r"z(\mathbf{x}) = \big(s_{t,3},\ y_t-\bar y_{t,12}\big)", _t("volatilidad y brecha anual"))),
    (300, 370, 310, 130, "", _box("3b. Localización", r"a_s(\mathbf{x}) \propto e^{-\lVert z(\mathbf{x})-z_s\rVert^2/2h^2}\,\gamma^{\,n-1-s}",
                                  r"\gamma = 0{,}5^{1/\tau}", _t("estado parecido y meses recientes"))),
    (300, 525, 310, 95, "", _box("Errores fuera de muestra", r"e_{s,k} = y_s - f_k^{(s)}(\mathbf{x}_s)", _t("ventana expansiva"))),
    (660, 375, 240, 95, "", _box("3c. Error local", r"Q(\mathbf{x}) = \displaystyle\sum_s a_s(\mathbf{x})\, e_s e_s^{\top}")),
    (945, 375, 195, 95, "", _box("Combinación global", r"\pi:\ " + _t("sin localizar"))),
    (660, 495, 480, 125, "", _box("3d. Pesos: problema convexo con solución única",
                                  r"w^*(\mathbf{x}) = \underset{w\in\Delta_K}{\arg\min}\ w^{\top}Q(\mathbf{x})\,w + \tilde\lambda(\mathbf{x})\,\mathrm{KL}(w\,\Vert\,\pi)",
                                  r"\tilde\lambda(\mathbf{x}) = \lambda\,\dfrac{\operatorname{tr}Q(\mathbf{x})}{K}\,\dfrac{n_{\mathrm{ref}}}{n_{\mathrm{eff}}(a)}")),
    (930, 284, 80, 26, "lbl", r"w^*(\mathbf{x})"),
]


def arch_diagram():
    """Diagrama de ALEE: el SVG dibuja cajas y flechas; cada texto es LaTeX compuesto con MathJax."""
    pct = lambda v, t: f"{100 * v / t:.3f}%"  # noqa: E731
    boxes = [html.Div(U.math(f"${tex}$"), className=f"arch-box {cls}",
                      style={"left": pct(x, VB_W), "top": pct(y, VB_H), "width": pct(w, VB_W), "height": pct(h, VB_H)})
             for x, y, w, h, cls, tex in ARCH]
    return html.Div([html.Img(src=dash.get_asset_url("alee_arquitectura.svg"), alt="", className="arch-svg"), *boxes],
                    className="arch-diagram", role="img",
                    **{"aria-label": "Arquitectura de ALEE: la entrada alimenta tres expertos que se combinan con pesos w*; los pesos salen del estado, "
                                     "la localización de los meses pasados, sus errores fuera de muestra y un problema convexo regularizado hacia la combinación global."})


TABLE_GUIDE = [
    ("Qué se compara", "El error al cuadrado de ALEE y el de cada modelo, mes a mes, en los mismos meses fuera de muestra."),
    ("Diferencia de pérdida", "Media de (error² de ALEE − error² del modelo). Negativa: ALEE se equivoca menos. Entre corchetes, su IC del 95 %."),
    ("DM-HLN y GW", "H0: igual capacidad predictiva (GW: condicional). Valores p corregidos por Holm; menores que 0,05 rechazan la igualdad."),
    ("P(ALEE mejor)", "Prueba bayesiana con ROPE de ±1 % sobre cinco pliegues dependientes: orientativa."),
    ("d de Cohen y A₁₂", "Tamaños del efecto: cuánto importa la diferencia, no solo si existe."),
]

GROUPS = [("g-exp", "Expertos"), ("g-con", "Contribuciones"), ("g-int", "Interpretabilidad"), ("g-err", "Errores y residuos"),
          ("g-eval", "Evaluación estadística"), ("g-cmp", "Modelos de referencia")]

layout = html.Div([
    U.head("03", "Model Intelligence", "ALEE: tres expertos,", "una combinación convexa.",
           "La agregación local entrópica de expertos combina tres pronosticadores sencillos y decide cada mes cuánto confiar en cada uno según cómo les fue en los meses pasados con un estado parecido."),
    html.Div([
        html.Section([
            U.block("3.1 · Arquitectura", "Del dato al pronóstico", "Diagrama estático construido a partir de la implementación (src/original_model.py) y del paper."),
            html.Figure([arch_diagram(),
                         html.Figcaption("Arriba, la ruta del pronóstico. Abajo, cómo se obtienen los pesos. La flecha discontinua indica que los errores pasados se registran fuera de muestra, en ventana expansiva.")],
                        className="li-panel pad arch reveal"),
            U.loading(html.Div(id="mi-now")),
        ], id="s31"),
        html.Section([
            U.block("3.2 · Formulación matemática", "El modelo completo, en la notación del paper",
                    "Problema, variables, expertos, pesos, optimización y propiedades."),
            html.Div([html.Article([html.H3([html.Span(f"{i + 1}", className="math-n"), t]), U.math(m)], className=f"math-card{' wide' if i not in (2, 3) else ''}")
                      for i, (t, m) in enumerate(MATH)], className="math-grid reveal"),
        ], id="s32"),
        html.Section([
            U.block("3.3 · Diagnósticos", "Cómo se comporta y cuánto se equivoca"),
            html.Nav([dcc.Link(t, href=f"#{i}") for i, t in GROUPS], className="li-subnav", **{"aria-label": "Grupos de diagnóstico"}),
            U.loading(html.Div(id="mi-explain")),
            U.loading(html.Div(id="mi-eval")),
        ], id="s33"),
    ], className="li-wrap"),
])


@callback(Output("mi-now", "children"), Input("pais", "value"))
@U.safe
def now(c):
    """Los valores del diagrama para el pronóstico vigente del país seleccionado."""
    f, info, m = E.forecast(c), E.meta(c), E.model(c)
    rows = [{"a": C.EXPERT_LABELS[n], "b": f"{N(f['experts'][i])} %", "c": f"{N(100 * f['weights'][i], 1)} %", "d": f"{N(100 * f['prior'][i], 1)} %",
             "e": f"{N(f['contrib'][i], 2, True)} pp"} for i, n in enumerate(C.EXPERT_LABELS)]
    hp = info["hp"]
    return U.panel(f"El diagrama con los valores de {C.name(c)}", html.Div([
        html.Div([
            html.Div([html.Small("1 · Entrada"), html.B(f"{len(info['features'])} variables"), html.Span(f"{info['n']} meses de ajuste")]),
            html.Div([html.Small("3a · Estado z(x)"), html.B(f"{N(f['state'][0])} pp · {N(f['state'][1], 2, True)} pp"), html.Span("volatilidad · brecha")]),
            html.Div([html.Small("3b · Muestra local efectiva"), html.B(f"{N(f['n_eff'], 1)} meses"), html.Span(f"frente a {N(f['n_ref'], 1)} solo por cercanía en el tiempo")]),
            html.Div([html.Small("Hiperparámetros"), html.B(f"h = {N(hp['bandwidth'])} · τ = {N(hp['half_life'], 1)}"), html.Span(f"λ = {N(hp['lam'], 4)} · {info['evals']} evaluaciones bayesianas")]),
            html.Div([html.Small("5 · Pronóstico"), html.B(f"{N(f['pred'])} %", className="est"), html.Span(C.month_long(f["target_date"]))]),
        ], className="now-strip"),
        U.table(rows, [("a", "Experto"), ("b", "Propone (f_k)"), ("c", "Peso local w*"), ("d", "Peso global π"), ("e", "Aporte w*·(f_k − y_t)")], numeric=("b", "c", "d", "e")),
        html.P(f"Último dato {N(f['last'])} % más la suma de los aportes = {N(f['pred'])} %, sin aproximaciones. Mínimo de meses para empezar el historial de errores: {m.min_train}.", className="li-cap"),
    ]), sub="Pronóstico vigente; se actualiza con el país seleccionado")


@callback(Output("mi-explain", "children"), Input("pais", "value"))
@U.safe
def explain(c):
    f = E.forecast(c)
    top = f["neighbours"].index[0]
    k = int(np.argmax(f["weights"]))
    out = [
        html.Section([
            html.H3("Comportamiento de los expertos", className="li-group"),
            U.grid(
                U.panel("Lo que propone cada experto", html.Div([U.graph(F.votes(c, height=270)), html.P(f"La línea punteada es el último dato y la continua, el pronóstico. Hoy pesa más «{C.EXPERT_LABELS[list(C.EXPERT_LABELS)[k]].lower()}» ({N(100 * f['weights'][k], 0)} %).", className="li-cap")]), sub="El tamaño del punto es su peso"),
                U.panel("Los meses a los que hoy se parece", html.Div([U.graph(F.state_map(c, 360)), html.P(f"El mes con más peso es {C.month_long(top)} ({N(100 * f['neighbours'].iloc[0], 1)} %).", className="li-cap")]), sub="Espacio de estados con el que ALEE localiza"),
            ),
            U.panel("Pesos a lo largo de la validación", html.Div([U.graph(F.weights_time(c, 360)), html.P("Fuera de muestra: cada peso se calculó solo con meses anteriores. Las líneas punteadas separan los cinco pliegues.", className="li-cap")])),
        ], id="g-exp"),
        html.Section([
            html.H3("Contribuciones al pronóstico", className="li-group"),
            U.panel("Descomposición exacta del pronóstico vigente", html.Div([U.graph(F.contribution(f, 320)),
                                                                          html.P("Último dato más el aporte de cada experto (peso por cambio propuesto). La suma es el pronóstico, sin aproximaciones.", className="li-cap")])),
        ], id="g-con"),
    ]
    interp = [html.H3("Interpretabilidad", className="li-group")]
    if c == "colombia":
        interp.append(U.grid(
            U.panel("Importancia por permutación", U.graph(F.permutation_fig(320)), sub="Último pliegue de Colombia, 20 permutaciones"),
            U.panel("Valores de Shapley", U.graph(_shap("reg_ALEE", 320)), sub="96 meses de Colombia. El color es el valor de la variable"),
        ))
    else:
        interp.append(U.note("La importancia por permutación y los valores de Shapley solo se calcularon para Colombia. En los demás países la explicación es la descomposición exacta por expertos.", "info"))
    out.append(html.Section(interp, id="g-int"))
    return out


def _shap(name, height=420):
    sg = D.shap_global(name)
    fig = P.shap_summary(sg["phi"], D.dataset().X.iloc[sg["rows"]], FEAT, "", top=8)
    fig.update_layout(height=height, title=None, margin=dict(l=190, r=10, t=8, b=44))
    return fig


@callback(Output("mi-eval", "children"), Input("pais", "value"))
@U.safe
def evaluate(c):
    from src.explain import acf

    met = E.metrics()
    mine = met[(met["conjunto"] == c) & (met["modelo"] == "ALEE")]
    best, b = E.best_baseline(c)
    d = E.oof(c)
    err = d["y_true"] - d["ALEE"]
    worst = d.loc[err.abs().nlargest(6).index]
    cards = [html.Div([html.Div([html.Span(k if k != "R2" else "R²", className="k-label"), html.Span("Calculado", className="chip calc")], className="k-top"),
                       html.Div([html.Span(N(mine[k].mean(), 3 if k != "MAPE" else 2), className="num"), html.Span(v[1], className="unit")], className="k-val model"),
                       html.Div(f"± {N(mine[k].std(), 3 if k != 'MAPE' else 2)} entre pliegues · {v[0].lower()}", className="k-foot"), html.Div(v[2], className="k-help")], className="li-kpi reveal") for k, v in METRICS.items()]
    t = E.result("pruebas_por_baseline")
    t = t[t["conjunto"] == c]
    pf = lambda p: "< 0,001" if p < 0.001 else N(p, 3)  # noqa: E731
    rows = [{"m": C.MODEL_LABELS[r["baseline"]], "r": N(E.metric_table().loc[c, r["baseline"]], 3), "d": f"{N(r['dif_perdida_media'], 3)}  [{N(r['ic95_inf'], 3)}; {N(r['ic95_sup'], 3)}]",
             "dm": pf(r["dm_p_holm"]), "gw": pf(r["gw_p_holm"]), "pb": N(r["p_bayes_mejor"], 2), "cd": N(r["d_cohen"], 2), "a": N(r["a12_vargha"], 2)} for _, r in t.iterrows()]
    # ALEE es la referencia de todas las pruebas: va primero, resaltado, con su RMSE y sin pruebas contra sí mismo.
    rows.insert(0, {"m": "ALEE (referencia)", "r": N(E.metric_table().loc[c, "ALEE"], 3), "d": "—", "dm": "—", "gw": "—", "pb": "—", "cd": "—", "a": "—", "_cls": "hl"})
    n_sig = int((t["dm_p_holm"] < 0.05).sum())
    abl = E.result("ablacion")
    a = abl[(abl["conjunto"] == c) & (abl["dm_p_holm"] < 0.05)]
    worse = [v.lower() for v in a[a["dif_perdida"] < 0]["variante"]]
    loc = abl[(abl["variante"] == "Sin localización") & (abl["dm_p_holm"] < 0.05)]["conjunto"].map(C.name).tolist()
    jury = _seasonal_answer(abl)
    mcs = E.result("mcs")
    mk = [m for m in mcs[(mcs["conjunto"] == c) & mcs["en_conjunto"].astype(bool)]["modelo"] if m in C.MODEL_LABELS]
    r = err.values
    pr = d["ALEE"].values
    return html.Div([
        html.Section([
            html.H3("Errores y residuos", className="li-group"),
            html.P(f"Validación anidada de origen móvil: cinco pliegues, {len(d)} meses de prueba ({d['fecha'].min().year} a {d['fecha'].max().year}), un mes de separación y búsqueda bayesiana de 40 evaluaciones en el bucle interno.", className="li-lead sm"),
            U.panel("Pronosticado frente a observado, fuera de muestra", html.Div([U.graph(F.pred_vs_obs(c, best, 420)), html.P(f"En ocre, el mejor modelo de comparación en {C.name(c)}: {C.MODEL_LABELS[best]}. Las líneas verticales separan los pliegues.", className="li-cap")])),
            U.grid(
                U.panel("Error en cada pliegue", U.graph(F.fold_errors(c, height=340)), sub="Un solo promedio esconde que el error cambia con la época"),
                U.panel("Los meses más difíciles", U.table([{"a": C.month_long(x["fecha"]).capitalize(), "b": f"{N(x['y_true'])} %", "c": f"{N(x['ALEE'])} %", "d": f"{N(x['y_true'] - x['ALEE'], 2, True)} pp"} for _, x in worst.iterrows()],
                                                           [("a", "Mes"), ("b", "Observado"), ("c", "Pronóstico"), ("d", "Error")], numeric=("b", "c", "d")), sub="Seis mayores errores absolutos"),
            ),
            U.panel("Diagnóstico de los residuos de ALEE", html.Div([_fig(P.residual_panels(r, pr)),
                                                                   html.P(f"RMSE agrupado de {N(np.sqrt(np.mean(r ** 2)), 3)} pp. Autocorrelación de los residuos en el rezago 1: {N(acf(r, 1)[1])}. Los residuos no son independientes: por eso la cobertura del intervalo se mide y no se supone.", className="li-cap")]),
                    sub=f"{len(r)} errores fuera de muestra de {C.name(c)}"),
        ], id="g-err"),
        html.Section([
            html.H3("Evaluación estadística", className="li-group"),
            html.Div(cards, className="li-kpis five"),
            U.panel("Pruebas de significancia frente a cada modelo", html.Div([
                U.table(rows, [("m", "Modelo"), ("r", "RMSE"), ("d", "Diferencia de pérdida [IC 95 %]"), ("dm", "DM-HLN, p"), ("gw", "GW, p"), ("pb", "P(ALEE mejor)"), ("cd", "d de Cohen"), ("a", "A₁₂")], numeric=("r", "d", "dm", "gw", "pb", "cd", "a")),
                html.P(f"Diebold-Mariano-HLN rechaza la igualdad de la pérdida en {n_sig} de {len(t)} comparaciones en {C.name(c)}. El conjunto de confianza de modelos conserva {len(mk)} de 10 modelos"
                       + (": solo ALEE." if len(mk) == 1 else "; con estos meses, esa prueba no descarta a los demás."), className="li-cap"),
                html.Div([html.H4("Cómo leer la tabla"), html.Dl([x for t, d in TABLE_GUIDE for x in (html.Dt(t), html.Dd(d))])], className="table-guide"),
            ])),
            U.panel("Ablación: qué pasa si se quita una pieza", html.Div([
                U.graph(F.ablation(c, 380)),
                html.P((f"Con significancia son peores que ALEE completo: {', '.join(worse)}. " if worse else "Ninguna variante es significativamente peor que ALEE completo en este país. ")
                       + f"En los nueve países, la localización por estado solo aporta con significancia en {' y '.join(loc) if loc else 'ninguno'}; la mayor parte de la ventaja viene de los expertos.", className="li-cap"),
                jury,
            ]), sub="Rayado: diferencia significativa frente a ALEE completo (Diebold-Mariano con Holm)"),
        ], id="g-eval"),
        html.Section([
            html.H3("Comparación con modelos de referencia", className="li-group"),
            html.P("Los ocho modelos de regresión del curso y LightGBM, con las mismas particiones, variables y presupuesto de ajuste. "
                   "Antecedente: en Colombia, ALEE también tuvo el menor RMSE medio entre las 140 combinaciones de modelos, balanceo y optimizadores del segundo entregable.", className="li-lead sm"),
            U.grid(
                U.panel("Métrica por modelo", html.Div([U.seg("mi-metric", [(("R²" if k == "R2" else k), k) for k in METRICS], "RMSE"), html.Div(U.graph(F.model_bars(c, height=420), id="mi-bars"))]),
                        sub="Media de los cinco pliegues externos"),
                U.panel("Error por régimen de volatilidad", U.graph(F.regimes(c, 460)), sub=f"Azul marino: ALEE. Ocre: {C.MODEL_LABELS[best]}. Gris: los demás"),
            ),
        ], id="g-cmp"),
    ])


def _seasonal_answer(abl):
    """Respuesta a «¿por qué no usar solo el experto estacional?», calculada con la ablación de los nueve países."""
    s = abl[abl["variante"] == "Solo estacional"].set_index("conjunto")
    alee = E.metric_table()["ALEE"]
    sig = s["dm_p_holm"] < 0.05
    alee_better = s[sig & (s["dif_perdida"] < 0)].index.map(C.name).tolist()
    seas_better = s[sig & (s["dif_perdida"] > 0)].index.map(C.name).tolist()
    lower = int((alee.reindex(s.index) < s["RMSE"]).sum())
    co = s.loc["colombia"] if "colombia" in s.index else None
    txt = [html.B("¿Por qué no usar solo el experto estacional? "),
           (f"En Colombia tiene un RMSE de {N(co['RMSE'], 3)} frente a {N(alee['colombia'], 3)} de ALEE, " + ("y la diferencia no es significativa. " if co["dm_p_holm"] >= 0.05 else "y la diferencia es significativa. ") if co is not None else ""),
           ("En ningún país el experto estacional solo es significativamente mejor que ALEE. " if not seas_better else f"Es significativamente mejor que ALEE en {', '.join(seas_better)}. "),
           (f"ALEE sí es significativamente mejor en {_and(alee_better)}. " if alee_better else ""),
           f"ALEE tiene menor RMSE en {lower} de {len(s)} países. Lo que aporta ALEE es que no hay que saber de antemano qué experto funciona en cada país: "
           "lo decide con los datos, y su error nunca supera al del peor experto."]
    return U.note(txt, "info")


def _and(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " y " + xs[-1]


@callback(Output("mi-bars", "figure"), Input("mi-metric", "value"), Input("pais", "value"))
def bars(metric, c):
    return F.model_bars(c, metric, 420)


def _fig(fig, h=None):
    fig.update_layout(title=None, margin=dict(t=34, l=fig.layout.margin.l or 56, r=14, b=48))
    if h:
        fig.update_layout(height=h)
    return U.graph(fig)
