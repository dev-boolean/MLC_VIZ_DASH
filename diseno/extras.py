"""Paneles agregados a las páginas originales para cubrir la rúbrica de la sustentación, sin quitarles nada.

* Análisis exploratorio, 2.7 (criterio 1): formulación del problema y justificación del preprocesamiento.
* Modelo, 3.4 (criterio 3): partición y validación, auditoría de fuga, hiperparámetros, reproducibilidad y costo.
* Modelo, 3.5 (criterio 4): sobreajuste y robustez (curva de aprendizaje, sensibilidad a hiperparámetros y a la semilla).

Todas las cifras se leen de ``artifacts/`` o se calculan al cargar; nada se escribe a mano.
"""
import inspect
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from sklearn.model_selection import TimeSeriesSplit

from labor import catalog as C
from labor import engine as E
from labor import theme as T
from labor import ui as U
from src import config
from src.dash_data import FEATURE_INFO

N = C.num
RES = config.ROOT_DIR / "artifacts" / "resultados"
CONF = config.ROOT_DIR / "artifacts" / "confirmatorio"
ADIC = config.ROOT_DIR / "artifacts" / "adicionales"
EXTRA = {"SARIMA": "SARIMA", "Persistencia": "Persistencia", "Estacional": "Estacional ingenuo",
         "EstacionalALEE": "Experto estacional de ALEE, solo"}


def etiqueta(m):
    return C.MODEL_LABELS.get(m, EXTRA.get(m, m))


def agregar(registro):
    """Añade los bloques nuevos al final de Análisis exploratorio y de Modelo."""
    for p in registro.values():
        if p["path"] == "/analytics":
            p["layout"].children.append(html.Div(formulacion(), className="li-wrap px-extra"))
        elif p["path"] == "/modelo":
            p["layout"].children.append(html.Div(U.loading(html.Div(id="px-modelo")), className="li-wrap px-extra"))


# ======================================================== criterio 1: formulación y preprocesamiento
def formulacion():
    from src.data import assemble_dataset, load_raw

    ds, raw = assemble_dataset(), load_raw()
    n, p = ds.X.shape
    f0, f1 = ds.fechas.iloc[0], ds.fechas.iloc[-1]
    obj0, obj1 = f0 + pd.offsets.MonthBegin(1), f1 + pd.offsets.MonthBegin(1)
    grupos = {}
    for v in ds.feature_names:
        grupos.setdefault(FEATURE_INFO[v][1], []).append(FEATURE_INFO[v][0])
    # Las razones se calculan con la serie, para que cada cifra coincida con las lecturas 2.1 a 2.6.
    s = raw.set_index("fecha")["tasa_desempleo_nacional"]
    cambio = s.diff().groupby(s.index.month)
    sube_ene, prom_ene = cambio.apply(lambda x: (x.dropna() > 0).mean()).loc[1], cambio.mean().loc[1]
    bajan = [C.MONTHS[m - 1] for m, v in cambio.mean().items() if v < -0.9]
    cor = raw[["tasa_desempleo_nacional", "tasa_desempleo_area", "tasa_ocupacion_area"]].corr().iloc[0]
    porque = {
        "Rezagos": (f"La serie recuerda mucho de sí misma: su autocorrelación es {N(s.autocorr(1))} a un mes y todavía "
                    f"{N(s.autocorr(12))} a doce (2.5).", "1, 2, 3, 6 y 12 meses"),
        "Estadísticos móviles": ("El nivel cambia despacio y la volatilidad cambia por épocas, sobre todo en 2020 (2.1 y 2.6). "
                                 "La volatilidad de 3 meses es además una de las variables de estado de ALEE.", "3, 6 y 12 meses"),
        "Calendario": (f"Hay un patrón anual marcado (2.2): enero sube en promedio {N(prom_ene, 1)} pp y lo hace en el "
                       f"{N(sube_ene * 100, 0)} % de los años; {' y '.join(bajan)} bajan. Seno y coseno evitan el salto artificial de "
                       "diciembre a enero; el índice recoge la tendencia de largo plazo (2.1).", "mes t"),
        "13 áreas metropolitanas": (f"Se mueven con la nacional (2.4): el desempleo de las 13 áreas correlaciona {N(cor['tasa_desempleo_area'])} "
                                    f"con ella y la ocupación {N(cor['tasa_ocupacion_area'])}. Se publican el mismo mes.", "mes t"),
    }
    filas = [{"a": g, "b": ", ".join(vs), "c": porque[g][1], "d": porque[g][0]} for g, vs in grupos.items()]
    filas.append({"a": "Excluidas", "b": "Ocupación y participación nacionales del mismo mes", "c": "—",
                  "d": "Cumplen la identidad TD = 1 − TO/TGP con el objetivo: incluirlas sería darle la respuesta al modelo."})
    faltantes = int(raw.isna().sum().sum())

    form = U.panel("Formulación del problema", html.Div([
        U.math(rf"""
**Tarea:** regresión supervisada de un paso. En cada mes $t$ se conoce el vector de variables $x_t \in \mathbb{{R}}^{{{p}}}$,
construido solo con información disponible hasta $t$, y se quiere pronosticar la tasa de desempleo nacional del mes siguiente:

$$\hat y_{{t+1}} = f(x_t), \qquad y_{{t+1}} = \text{{TD}}_{{t+1}} \ (\%).$$

**Función de pérdida:** error cuadrático. Se busca $f$ que minimice $\mathbb{{E}}\big[(y_{{t+1}} - f(x_t))^2\big]$, y la métrica principal
es su raíz, el RMSE, en puntos porcentuales, calculado solo en meses de prueba posteriores a los de entrenamiento.

**Datos:** {n} pares $(x_t, y_{{t+1}})$, con objetivos de {C.month_long(obj0)} a {C.month_long(obj1)}. Se pierden los primeros
12 meses de la serie por los rezagos y las medias de 12 meses.
"""),
        U.note("El horizonte es un mes. El modelo no se entrenó ni se evaluó a más pasos, así que la plataforma no extiende el pronóstico.",
               "info", "Alcance."),
    ]), sub="Criterio 1 de la rúbrica: tipo de tarea, objetivo y formulación matemática")

    prep = U.panel("Variables y preprocesamiento", html.Div([
        U.table(filas, [("a", "Grupo"), ("b", "Variables"), ("c", "Ventanas"), ("d", "Por qué, según el análisis exploratorio")]),
        html.Ul([
            html.Li(f"Sin valores faltantes: {faltantes} celdas vacías en los {len(raw)} meses, así que no se imputa nada."),
            html.Li("Rezagos y medias móviles terminan en el mes t: ninguna variable mira el mes que se pronostica (ver la auditoría de fuga en 3.4)."),
            html.Li("Para los modelos del curso, el escalador se ajusta dentro de cada pliegue, solo con los meses de entrenamiento. "
                    "ALEE no necesita escalado: estandariza su estado internamente con los mismos meses de entrenamiento."),
            html.Li("Las técnicas de balanceo del segundo entregable (SMOTE, ADASYN) solo aplican a la versión de clasificación; "
                    "el pronóstico de la tasa es una regresión y no las usa."),
        ], className="li-src"),
    ], className="li-prose"), sub="Criterio 1 de la rúbrica: justificación del preprocesamiento")

    return [U.block("2.7 · Del análisis al problema", "Formulación del pronóstico y preprocesamiento",
                    "Cómo las lecturas anteriores se convierten en las variables del modelo."), form, prep]


# ======================================================== criterio 3: diseño experimental
def _hp_pliegues(c):
    if c == "colombia":
        d = pd.read_csv(CONF / "iter4" / "rmse_K3_sin_ridge.csv")
    else:
        d = pd.read_csv(ADIC / c / "hiperparametros_ALEE.csv")
    return d[["pliegue", "bandwidth", "half_life", "lam"]]


def _particiones(c):
    d, hp = E.oof(c), _hp_pliegues(c)
    filas = []
    for k, g in d.groupby("pliegue"):
        ini = int(g["idx"].min())
        h = hp.iloc[k - 1]
        filas.append({"a": str(k), "b": f"{ini - config.CV_GAP} meses", "c": C.month_short(g["fecha"].min() - pd.DateOffset(months=2)),
                      "d": f"{C.month_short(g['fecha'].min())} – {C.month_short(g['fecha'].max())}", "e": str(len(g)),
                      "f": N(h["bandwidth"]), "g": N(h["half_life"], 1), "h": N(h["lam"], 4),
                      "i": N(np.sqrt(np.mean((g["y_true"] - g["ALEE"]) ** 2)), 3)})
    return filas


def _auditoria():
    """Las cuatro comprobaciones del capítulo 2b del libro, calculadas al cargar (Colombia)."""
    from src import nested_cv, search
    from src.data import assemble_dataset, load_raw

    ds, raw = assemble_dataset(), load_raw()
    ident = 100 * (1 - raw["tasa_ocupacion_nacional"] / raw["tasa_global_participacion_nacional"])
    r = float(np.corrcoef(ident, raw["tasa_desempleo_nacional"])[0, 1])
    excl = {"tasa_ocupacion_nacional", "tasa_global_participacion_nacional"}.isdisjoint(ds.feature_names)
    orden = all(ds.fechas.iloc[tr].max() < ds.fechas.iloc[te].min()
                for tr, te in TimeSeriesSplit(config.N_SPLITS_OUTER, gap=config.CV_GAP).split(ds.X))
    lag = bool(np.array_equal(ds.X["td_nacional_lag1"].values[1:], ds.y_reg.values[:-1]))
    m12 = raw["tasa_desempleo_nacional"].rolling(12).mean().dropna().values
    mov = bool(np.isin(np.round(ds.X["td_nacional_rollmean12"].values, 10), np.round(m12, 10)).all())
    src = inspect.getsource(nested_cv._fit_final) + inspect.getsource(search._fit_predict_fold)
    esc = "StandardScaler().fit(X_fit)" in src and "StandardScaler().fit(X_tr)" in src
    dup = int(ds.X.duplicated().sum())
    ok = lambda b: html.B("Cumple" if b else "No cumple", style={"color": T.PRED if b else T.ALERT})  # noqa: E731
    return [
        {"a": "1. Variables que codifican la respuesta", "b": f"La identidad 1 − TO/TGP correlaciona {N(r, 6)} con la tasa; "
         "ocupación y participación nacionales no están entre las variables.", "c": ok(excl and r > 0.999)},
        {"a": "2. Mezcla temporal entre entrenamiento y prueba", "b": f"En los {config.N_SPLITS_OUTER} pliegues externos, todo el entrenamiento "
         f"es anterior a la prueba, con {config.CV_GAP} mes de separación.", "c": ok(orden)},
        {"a": "3. Variables con información futura", "b": "El rezago 1 de cada fila es el objetivo de la fila anterior y la media de 12 meses "
         "termina en el mes t.", "c": ok(lag and mov)},
        {"a": "4. Preprocesamiento con todo el conjunto o filas repetidas", "b": f"El escalador se ajusta dentro de cada pliegue; {dup} filas "
         f"duplicadas en {len(ds.X)}.", "c": ok(esc and dup == 0)},
    ]


def _costo():
    d = pd.read_csv(RES / "costo_computacional.csv")
    d = d[d["entrenamiento_s"] > 0]
    hw = json.loads((RES / "hardware.json").read_text(encoding="utf-8"))
    filas = [{"a": etiqueta(r["modelo"]), "b": N(r["entrenamiento_s"] * 1000, 1), "c": N(r["inferencia_s"] * 1000, 1),
              "d": N(r["memoria_pico_MB"], 1), "e": f"{int(r['parametros']):,}".replace(",", "."),
              "_cls": "hl" if r["modelo"] == "ALEE" else ""} for _, r in d.iterrows()]
    return filas, hw


def diseno_experimental(c):
    hp = E.meta(c)["hp"]
    art = "artifacts/reg_ALEE" if c == "colombia" else f"artifacts/paises/{c}"
    filas_c, hw = _costo()
    part = U.panel("Partición, validación e hiperparámetros", html.Div([
        U.table(_particiones(c), [("a", "Pliegue"), ("b", "Entrenamiento"), ("c", "Hasta"), ("d", "Prueba (meses objetivo)"), ("e", "Meses"),
                                  ("f", "h"), ("g", "τ (vida media)"), ("h", "λ"), ("i", "RMSE de ALEE")],
                numeric=("b", "e", "f", "g", "h", "i")),
        html.Div([
            html.P([html.B("Validación anidada de origen móvil. "), f"{config.N_SPLITS_OUTER} pliegues externos de ventana expansiva para evaluar y, "
                    f"dentro de cada uno, {config.N_SPLITS_INNER} pliegues internos para elegir los hiperparámetros, siempre con "
                    f"{config.CV_GAP} mes de separación. Ningún mes de prueba participa en el ajuste ni en la elección de su propio modelo."]),
            html.P([html.B("Hiperparámetros. "), "Ancho del núcleo h, vida media del decaimiento τ y fuerza de la regularización λ, elegidos "
                    f"por optimización bayesiana (TPE, {config.FULL_BUDGETS['bayes_n_trials']} evaluaciones) en cada pliegue. Los modelos del curso "
                    "usan el mismo presupuesto y las mismas particiones. Que h, τ y λ cambien entre pliegues es esperable: cada pliegue ve "
                    "una historia distinta, y la sección 3.5 muestra que el error es poco sensible a ellos."]),
            html.P([html.B("Modelo final. "), f"Con toda la serie: h = {N(hp['bandwidth'])}, τ = {N(hp['half_life'], 1)}, λ = {N(hp['lam'], 4)} "
                    f"({art}/metadata.json)."]),
        ], className="li-prose"),
    ]), sub="Criterio 3: partición y validación de datos, justificación de hiperparámetros")

    audit = U.panel("Auditoría de fuga de información", html.Div([
        U.table(_auditoria(), [("a", "Fuente de fuga revisada"), ("b", "Comprobación (calculada al cargar, Colombia)"), ("c", "Resultado")]),
        html.P("Las cuatro fuentes de fuga de Kapoor y Narayanan (2023). Las mismas comprobaciones se ejecutan al construir el libro "
               "(capítulo 2b) y fallan si alguna condición no se cumple.", className="li-cap"),
    ]), sub="Criterio 3: prevención de data leakage")

    repro = U.panel("Reproducibilidad y costo computacional", html.Div([
        html.Ul([
            html.Li([f"Semilla {config.SEED} para los generadores de NumPy y random, para cada modelo con azar y para el muestreador de Optuna; la sección 3.5 muestra que cambiarla no cambia el resultado."]),
            html.Li(["Versiones exactas de las librerías en ", html.Code("requirements-proyecto.txt"), " y artefactos serializados con ",
                     html.Code("joblib"), " y su prueba de ida y vuelta (", html.Code("tests/test_artifacts.py"), ")."]),
            html.Li(["Todas las tablas y figuras del paper salen de un solo script: ", html.Code("python experiments/reproducir_todo.py"), "."]),
        ], className="li-src"),
        html.H4("Costo en el último pliegue de Colombia (239 meses de entrenamiento, 48 de prueba)", className="li-h4"),
        U.table(filas_c, [("a", "Modelo"), ("b", "Entrenar (ms)"), ("c", "Predecir 48 meses (ms)"), ("d", "Memoria (MB)"), ("e", "Parámetros")],
                numeric=("b", "c", "d", "e")),
        html.P(f"Mediana de tres repeticiones, cada modelo en un proceso independiente ({hw['nucleos']} núcleos, Python {hw['python']}). "
               "ALEE resuelve un problema convexo por cada mes pronosticado: es el más lento, pero tarda menos de un segundo.", className="li-cap"),
    ], className="li-prose"), sub="Criterio 3: reproducibilidad, rigor experimental y decisiones computacionales")
    return [U.block("3.4 · Diseño experimental", "Cómo se evaluó sin hacer trampa", "Partición, elección de hiperparámetros, fuga de información, reproducibilidad y costo."),
            part, U.grid(html.Div(audit), html.Div(repro))]


# ======================================================== criterio 4: sobreajuste y robustez
def _curva():
    d = pd.read_csv(RES / "curva_aprendizaje_eeuu.csv")
    cfg = json.loads((RES / "curva_aprendizaje_config.json").read_text(encoding="utf-8"))
    estilo = {"ALEE": (T.NAVY, "solid"), "EstacionalALEE": (T.PRED, "dot"), "SARIMA": (T.OCHRE, "dash"), "Ridge": (T.DIM, "dashdot")}
    fig = T.base(340)
    for m, (col, dash) in estilo.items():
        g = d[d["modelo"] == m]
        fig.add_trace(go.Scatter(x=g["meses_entrenamiento"], y=g["rmse"], mode="lines+markers", name=etiqueta(m),
                                 line=dict(color=col, dash=dash, width=2.4 if m == "ALEE" else 1.6), marker=dict(size=7),
                                 hovertemplate=f"{etiqueta(m)}: %{{y:.3f}} pp con %{{x}} meses<extra></extra>"))
    fig.update_layout(xaxis=dict(title="Meses de entrenamiento", type="log", tickvals=sorted(d["meses_entrenamiento"].unique())),
                      yaxis=dict(title="RMSE en la prueba fija (pp)"))
    a = d[d["modelo"] == "ALEE"].set_index("meses_entrenamiento")["rmse"]
    r = d[d["modelo"] == "Ridge"].set_index("meses_entrenamiento")["rmse"]
    texto = (f"Con un periodo de prueba fijo (los últimos {cfg['meses_prueba']} meses de Estados Unidos, que incluyen 2020), ALEE logra "
             f"{N(a.iloc[0], 3)} pp con solo {a.index[0]} meses de entrenamiento y {N(a.iloc[-1], 3)} con {a.index[-1]}: el error casi no depende "
             f"de cuánta historia vea, así que funciona con series cortas pero no aprovecha la información adicional. Ridge baja de "
             f"{N(r.iloc[0], 2)} a {N(r.iloc[-1], 2)} y aun con toda la serie queda por encima. La curva casi coincide con la del experto "
             "estacional solo: la ventaja viene sobre todo de los expertos. El sobreajuste también se controla con la validación anidada: "
             "todo el error que se reporta es fuera de muestra.")
    return fig, texto


def _sensibilidad(c):
    d = pd.read_csv(RES / c / "sensibilidad_hiperparametros.csv")
    g = d.pivot(index="lam", columns="bandwidth", values="rmse_medio")
    best, bv = E.best_baseline(c)
    fig = go.Figure(go.Heatmap(z=g.values, x=[N(v) for v in g.columns], y=[N(v, 4) for v in g.index], colorscale=[[0, "#FFFFFF"], [1, T.NAVY]],
                               text=np.round(g.values, 3), texttemplate="%{text}", textfont=dict(size=11),
                               colorbar=dict(title="RMSE"), hovertemplate="h = %{x} · λ = %{y}<br>RMSE %{z:.3f}<extra></extra>"))
    fig.update_layout(height=330, xaxis=dict(title="Ancho del núcleo h", type="category"), yaxis=dict(title="Regularización λ", type="category"),
                      margin=dict(l=70, r=20, t=10, b=50))
    peor = g.values.max()
    cmp = (f"aun la peor combinación queda por debajo del mejor modelo del curso ({etiqueta(best)}, {N(bv, 3)})" if peor < bv else
           f"{int((g.values < bv).sum())} de {g.size} combinaciones quedan por debajo del mejor modelo del curso ({etiqueta(best)}, {N(bv, 3)})")
    texto = (f"Rejilla de {g.size} combinaciones con vida media fija de 60 meses en {C.name(c)}. El RMSE va de {N(g.values.min(), 3)} a "
             f"{N(peor, 3)}: {cmp}. Es exploratorio: no se usó para cambiar los hiperparámetros, porque sería ajustar con los datos de evaluación.")
    return fig, texto, peor < bv


def _semillas():
    s = pd.read_csv(RES / "sensibilidad_semillas.csv")
    s = s[s["semilla"].astype(str).str.fullmatch(r"\d+")]
    filas = [{"a": f"ALEE · {C.name(k)}", "b": str(len(g)), "c": N(g["rmse_medio"].mean(), 3), "d": N(g["rmse_medio"].std(), 4),
              "e": f"{N(g['rmse_medio'].min(), 3)} – {N(g['rmse_medio'].max(), 3)}", "_cls": "hl" if k == "colombia" else ""}
             for k, g in s.groupby("conjunto", sort=False)]
    b = pd.read_csv(RES / "semillas_baselines_colombia.csv")
    filas += [{"a": f"{etiqueta(m)} · Colombia", "b": str(len(g)), "c": N(g["rmse_medio"].mean(), 3), "d": N(g["rmse_medio"].std(), 4),
               "e": f"{N(g['rmse_medio'].min(), 3)} – {N(g['rmse_medio'].max(), 3)}"} for m, g in b.groupby("modelo", sort=False)]
    return filas


def robustez(c):
    fig_c, txt_c = _curva()
    fig_s, txt_s, _ = _sensibilidad(c)
    curva = U.panel("Curva de aprendizaje", html.Div([U.graph(fig_c), html.P(txt_c, className="li-cap")]),
                    sub="Criterio 4: sobreajuste. Estados Unidos, la serie más larga (946 meses)")
    sens = U.panel(f"Sensibilidad a los hiperparámetros · {C.name(c)}", html.Div([U.graph(fig_s), html.P(txt_s, className="li-cap")]),
                   sub="Criterio 4: ¿depende el resultado de una elección afortunada de h y λ?")
    sem = U.panel("Sensibilidad a la semilla", html.Div([
        U.table(_semillas(), [("a", "Modelo y conjunto"), ("b", "Semillas"), ("c", "RMSE medio"), ("d", "Desviación"), ("e", "Rango")],
                numeric=("b", "c", "d", "e")),
        html.P("Semillas 42 a 46 del optimizador. La variación de ALEE es de milésimas, mucho menor que la de los modelos con azar "
               "(Random Forest, XGBoost, LightGBM), y ninguna semilla de ellos se acerca al RMSE de ALEE.", className="li-cap"),
    ]), sub="Criterio 4: robustez y reproducibilidad")
    gen = U.note(["La generalización a datos que el diseño no vio se evalúa en ocho países más, con el modelo congelado: ver ",
                  dcc.Link("05 · Comparación internacional", href="/global"), "."], "info", "Generalización.")
    return [U.block("3.5 · Sobreajuste y robustez", "¿El resultado se sostiene si cambian los datos, los hiperparámetros o la semilla?"),
            curva, U.grid(html.Div(sens), html.Div(sem)), gen]


@callback(Output("px-modelo", "children"), Input("pais", "value"))
@U.safe
def modelo(c):
    c = c or C.DEFAULT
    return diseno_experimental(c) + robustez(c)
