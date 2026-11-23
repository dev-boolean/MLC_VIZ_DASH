"""02. Exploratory Analytics: la historia que cuentan los datos del país."""
import dash
import numpy as np
from dash import Input, Output, callback, dcc, html

from labor import catalog as C
from labor import engine as E
from labor import figs as F
from labor import stats as S
from labor import ui as U
from src.dash_data import FEATURE_INFO, RAW_LABELS

dash.register_page(__name__, path="/analytics", name="Exploratory Analytics", title="Exploratory Analytics · LABOR INSIGHTS")
N = C.num
FEAT = {k: v[0] for k, v in FEATURE_INFO.items()}
SHORT = {"td_nacional_lag1": "Tasa del mes t", "td_nacional_lag2": "Tasa de t−1", "td_nacional_lag3": "Tasa de t−2", "td_nacional_lag6": "Tasa de t−5", "td_nacional_lag12": "Tasa de t−11",
         "y": "Tasa del mes siguiente"}

layout = html.Div([
    U.head("02", "EXPLORATORY ANALYTICS", "Lo que los datos", "realmente dicen.",
           "Seis lecturas de la misma serie: cómo evoluciona, qué se repite cada año, cómo se distribuye, qué se mueve con ella, cuánta memoria tiene y cuándo se rompe."),
    html.Div([html.Div([html.Span("Periodo de análisis", className="seg-label"), html.Div(id="ea-range-wrap", className="range")], className="li-filter reveal")], className="li-wrap"),
    html.Div([U.loading(html.Div(id="ea-a")), U.loading(html.Div(id="ea-b")), U.loading(html.Div(id="ea-c"))], className="li-wrap"),
])


@callback(Output("ea-range-wrap", "children"), Input("pais", "value"))
def rng(c):
    s = E.series(c)
    a, b = s.index[0].year, s.index[-1].year
    step = 4 if b - a < 35 else 10
    marks = {y: {"label": str(y)} for y in sorted({a, b, *[y for y in range(a + (-a) % step, b - step // 2, step) if y - a >= step // 2]})}
    return dcc.RangeSlider(id="ea-range", min=a, max=b, step=1, value=[a, b], marks=marks, allowCross=False, pushable=4, tooltip={"placement": "bottom"})


@callback(Output("ea-a", "children"), Input("ea-range", "value"), dash.State("pais", "value"))
@U.safe
def part_a(years, c):
    s = S.clip(c, years)
    if len(s) < 48:
        return U.note("Elija un periodo de al menos cuatro años.", "warn")
    up, dn = S.runs(s)
    sea = S.seasonality(s)
    mx, mn = sea.loc[sea["media"].idxmax()], sea.loc[sea["media"].idxmin()]
    strength = S.seasonal_strength(s)
    roll = s.diff().rolling(36).std().dropna()
    return html.Div([
        U.block("2.1 · TIEMPO", f"La serie de {C.name(c)}", f"{U.source(c)}" + (" Octubre de 2025 no se publicó y se interpoló." if E.frame(c)["interpolado"].any() else "")),
        U.panel("Evolución mensual y tendencia", html.Div([
            U.graph(F.evolution(c, years)),
            U.reading(f"Entre {C.month_long(s.index[0])} y {C.month_long(s.index[-1])} la tasa fue en promedio {N(s.mean())} %, con un máximo de {N(s.max())} % ({C.month_long(s.idxmax())}) y un mínimo de {N(s.min())} % ({C.month_long(s.idxmin())}).",
                      (f"La tendencia subió como máximo {N(up[0], 1)} pp ({C.month_short(up[1])} a {C.month_short(up[2])}) y bajó como máximo {N(dn[0], 1)} pp ({C.month_short(dn[1])} a {C.month_short(dn[2])})." if up else "El periodo es corto para medir tramos de tendencia.")
                      + (f" La volatilidad de los cambios mensuales, en ventanas de tres años, va de {N(roll.min())} a {N(roll.max())} pp." if len(roll) else ""),
                      "El nivel cambia despacio y los choques son bruscos: un buen pronóstico a un mes parte del último dato y lo corrige."),
        ]), sub="Los marcadores naranjas son cambios mensuales atípicos (2.6)"),
        U.block("2.2 · ESTACIONALIDAD", "Lo que se repite cada año"),
        U.grid(
            U.panel("Cambio mensual medio", U.graph(F.seasonal_bars(s)), sub="Intervalo de confianza del 95 %. Es el patrón que usa el experto estacional de ALEE"),
            U.panel("Cada mes frente a la media de su año", U.graph(F.month_heat(s)), sub="Naranja: por encima. Azul: por debajo"),
        ),
        html.Div(U.reading(f"El mayor aumento medio ocurre en {C.MONTHS[int(mx['mes']) - 1]} ({N(mx['media'], 2, True)} pp; sube en el {N(100 * mx['sube'], 0)} % de los años) y la mayor caída en {C.MONTHS[int(mn['mes']) - 1]} ({N(mn['media'], 2, True)} pp).",
                           f"El mes del año explica el {N(100 * strength, 0)} % de la varianza de los cambios mensuales.",
                           "Las series no están desestacionalizadas: parte de lo que parece una noticia mensual es calendario." if strength > 0.2 else "En este país el calendario explica poco: un pronóstico puramente estacional tiene menos ventaja."), className="reveal"),
    ])


def _vars(c):
    return list(RAW_LABELS) if c == "colombia" else ["valor"]


@callback(Output("ea-b", "children"), Input("ea-range", "value"), dash.State("pais", "value"))
@U.safe
def part_b(years, c):
    if len(S.clip(c, years)) < 48:
        return None
    opts = [{"label": RAW_LABELS[k], "value": k} for k in RAW_LABELS] if c == "colombia" else [{"label": "Tasa de desempleo", "value": "valor"}]
    xo = ([{"label": RAW_LABELS[k], "value": f"raw:{k}"} for k in RAW_LABELS if k != "tasa_desempleo_nacional"] if c == "colombia" else [])
    xo += [{"label": SHORT.get(k, FEAT.get(k, k)), "value": f"feat:{k}"} for k in E.features(c).columns if k not in ("mes_sin", "mes_cos", "trend_idx") and not k.endswith("_area")]
    return html.Div([
        U.block("2.3 · DISTRIBUCIÓN", "Qué valores son normales y cuáles no"),
        U.panel("Distribución univariada", html.Div([
            html.Div([html.Label("Variable", htmlFor="ea-var"), dcc.Dropdown(id="ea-var", options=opts, value=opts[0]["value"], clearable=False, searchable=False, className="li-dd")], className="li-ctl"),
            html.Div(id="ea-dist"),
        ]), sub="Colombia tiene seis tasas; los demás países, solo la de desempleo" if c == "colombia" else "Este país solo tiene la tasa de desempleo"),
        U.block("2.4 · RELACIONES", "Qué se mueve con el desempleo", "Una correlación describe que dos series se mueven juntas; no dice que una cause la otra."),
        U.grid(
            U.panel("Matriz de correlación", html.Div([
                U.seg("ea-corr-mode", ([("Tasas publicadas", "raw")] if c == "colombia" else []) + [("Variables del modelo", "feat")], "raw" if c == "colombia" else "feat"),
                html.Div(id="ea-corr"),
            ])),
            U.panel("Explorador de relaciones", html.Div([
                html.Div([html.Label("Comparar con", htmlFor="ea-x"), dcc.Dropdown(id="ea-x", options=xo, value=xo[0]["value"], clearable=False, searchable=False, className="li-dd")], className="li-ctl"),
                html.Div(id="ea-scatter"),
            ])),
        ),
    ])


def _raw(c, years):
    d = E.raw_colombia().set_index("fecha")
    return d[(d.index.year >= years[0]) & (d.index.year <= years[1])]


@callback(Output("ea-dist", "children"), Input("ea-var", "value"), dash.State("ea-range", "value"), dash.State("pais", "value"))
@U.safe
def dist(var, years, c):
    if c == "colombia" and var in RAW_LABELS:
        v, label = _raw(c, years)[var], RAW_LABELS[var]
    else:
        v, label = S.clip(c, years), "Tasa de desempleo"
    d = S.describe(v)
    iqr = d["Q3"] - d["Q1"]
    out = v[(v < d["Q1"] - 1.5 * iqr) | (v > d["Q3"] + 1.5 * iqr)]
    skew = "una cola hacia valores altos" if d["Asimetría"] > 0.5 else "una cola hacia valores bajos" if d["Asimetría"] < -0.5 else "una forma casi simétrica"
    st = html.Div([html.Div([html.Small(k), html.B(N(x) + ("" if k == "Asimetría" else " %" if k != "Desviación" else " pp"))]) for k, x in d.items()], className="li-stats")
    return html.Div([html.Div([U.graph(F.distribution(v, label)), st], className="dist-row"),
                     U.reading(f"La mitad central de los meses está entre {N(d['Q1'])} % y {N(d['Q3'])} %; la mediana es {N(d['Mediana'])} %.",
                               f"La distribución tiene {skew} (asimetría {N(d['Asimetría'])}) y {len(out)} meses fuera de las vallas de 1,5 rangos intercuartílicos.",
                               "Los meses extremos son episodios reales: no se eliminan, porque son los que un modelo debe enfrentar.")])


@callback(Output("ea-corr", "children"), Input("ea-corr-mode", "value"), dash.State("ea-range", "value"), dash.State("pais", "value"))
@U.safe
def corr(mode, years, c):
    if mode == "raw" and c == "colombia":
        df = _raw(c, years)[list(RAW_LABELS)]
        r = df.corr().loc["tasa_desempleo_nacional", "tasa_desempleo_area"]
        return [U.graph(F.corr_heat(df, RAW_LABELS)), U.note(f"El desempleo nacional y el de las 13 áreas tienen r = {N(r)}. La ocupación y la participación nacionales determinan el desempleo del mismo mes por identidad contable (TD = 1 − TO/TGP) y por eso no son variables del modelo.")]
    f = E.features(c).copy()
    f["y"] = E.series(c).shift(-1)
    f = f.dropna()
    f = f[(f.index.year >= years[0]) & (f.index.year <= years[1])]
    cols = ["y"] + [k for k in f.columns if k not in ("y",)]
    f = f[cols]
    f = f.loc[:, f.std() > 0]
    r = f.corr()["y"].drop("y").abs().sort_values(ascending=False)
    lab = {k: SHORT.get(k, FEAT.get(k, k)) for k in f.columns}
    return [U.graph(F.corr_heat(f, lab, 520, values=False)), U.note(f"La variable más correlacionada con la tasa del mes siguiente es «{lab[r.index[0]]}» (|r| = {N(r.iloc[0])}). Los rezagos y las medias móviles se parecen mucho entre sí: miden casi lo mismo.")]


@callback(Output("ea-scatter", "children"), Input("ea-x", "value"), dash.State("ea-range", "value"), dash.State("pais", "value"))
@U.safe
def scat(key, years, c):
    kind, k = key.split(":")
    if kind == "raw":
        d = _raw(c, years)
        x, y, lx, ly, sx = d[k], d["tasa_desempleo_nacional"], RAW_LABELS[k], "Desempleo nacional", " %"
    else:
        f = E.features(c)
        x = f[k]
        y = E.series(c).shift(-1)
        ok = x.notna() & y.notna() & (x.index.year >= years[0]) & (x.index.year <= years[1])
        x, y, lx, ly, sx = x[ok], y[ok], SHORT.get(k, FEAT.get(k, k)), "Tasa del mes siguiente", " pp" if "std" in k else " %"
    r = np.corrcoef(x, y)[0, 1]
    how = "fuerte" if abs(r) > 0.7 else "moderada" if abs(r) > 0.4 else "débil"
    return [U.graph(F.scatter(x, y, x.index.year, lx, ly, sx, " %")), U.note(f"La relación lineal es {how} y {'directa' if r > 0 else 'inversa'} (r = {N(r)}) en {len(x)} meses.")]


@callback(Output("ea-c", "children"), Input("ea-range", "value"), dash.State("pais", "value"))
@U.safe
def part_c(years, c):
    s = S.clip(c, years)
    if len(s) < 48:
        return None
    d, lo, hi, an = S.anomalies(s)
    st = S.stationarity(c, years)
    dd = s.diff().dropna()
    a1, a12 = S.acf_at(s, 1), S.acf_at(dd, 12)
    rows = [{"a": n, "b": ("< 0,0001" if p < 1e-4 else N(p, 4)), "c": ("≤ 0,01" if k <= 0.01 else "≥ 0,10" if k >= 0.1 else N(k, 3)), "d": v} for n, p, k, v in st]
    big = an.reindex(an.abs().sort_values(ascending=False).index).head(8)
    an_rows = [{"a": C.month_long(i).capitalize(), "b": f"{N(v, 2, True)} pp", "c": f"{N(s.loc[i])} %"} for i, v in big.items()]
    return html.Div([
        U.block("2.5 · MEMORIA", "Cuánto recuerda la serie de sí misma"),
        U.panel("Autocorrelación del nivel y del cambio mensual", html.Div([
            U.graph(F.acf_pair(s)),
            U.reading(f"El nivel tiene autocorrelación de {N(a1)} en el primer rezago; el cambio mensual, {N(a12)} en el rezago 12.",
                      f"En niveles, la serie resulta: {st[0][3].lower()}. La primera diferencia: {st[1][3].lower()}.",
                      "Persistencia y calendario son las dos estructuras que ALEE incorpora directamente con sus expertos."),
            U.table(rows, [("a", "Serie"), ("b", "ADF (H0: raíz unitaria), p"), ("c", "KPSS (H0: estacionaria), p"), ("d", "Lectura")], numeric=("b", "c")),
        ]), sub="Bandas de ±1,96/√n. En violeta, los rezagos 12 y 24"),
        U.block("2.6 · ANOMALÍAS", "Cuándo se rompe el patrón"),
        U.panel("Cambios mensuales y vallas externas de Tukey", html.Div([
            U.graph(F.anomaly_fig(s)),
            U.table(an_rows, [("a", "Mes"), ("b", "Cambio"), ("c", "Tasa")], numeric=("b", "c")) if an_rows else U.note("Ningún cambio mensual queda fuera de las vallas en este periodo."),
            U.reading(f"{len(an)} de {len(d)} cambios mensuales caen fuera de [{N(lo)}; {N(hi)}] pp.",
                      f"El más grande es el de {C.month_long(an.abs().idxmax())} ({N(an[an.abs().idxmax()], 2, True)} pp)." if len(an) else "El periodo no tiene rupturas extremas.",
                      "Decisión metodológica: los atípicos se marcan, pero no se borran ni se recortan. Con pocos antecedentes parecidos, ALEE se apoya más en sus pesos globales."),
        ]), sub="Regla del paper: más de tres rangos intercuartílicos por fuera de los cuartiles"),
    ])
