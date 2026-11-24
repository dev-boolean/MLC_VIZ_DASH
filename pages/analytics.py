"""02. Exploratory Analytics: análisis exploratorio de Colombia, el problema principal de la investigación."""
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
CO = "colombia"  # el EDA es el de Colombia: el selector de país no cambia esta sección
FEAT = {k: v[0] for k, v in FEATURE_INFO.items()}
SHORT = {"td_nacional_lag1": "Tasa del mes t", "td_nacional_lag2": "Tasa de t−1", "td_nacional_lag3": "Tasa de t−2", "td_nacional_lag6": "Tasa de t−5", "td_nacional_lag12": "Tasa de t−11",
         "y": "Tasa del mes siguiente"}
_S = E.series(CO)
A, B = _S.index[0].year, _S.index[-1].year


def interp(measure, pattern, evidence, modeling):
    """Interpretación estadística con la misma estructura en todas las subsecciones."""
    rows = [("Qué se mide", measure), ("Qué patrón se observa", pattern), ("Evidencia", evidence), ("Implicación para el modelo", modeling)]
    return html.Dl([x for t, v in rows for x in (html.Dt(t), html.Dd(v))], className="li-interp")


def stats_row(items):
    return html.Div([html.Div([html.Small(k), html.B(v)]) for k, v in items], className="li-stats")


layout = html.Div([
    U.head("02", "Exploratory Analytics", "El mercado laboral colombiano,", "en seis lecturas.",
           "La base analítica de la investigación: cómo evoluciona la tasa de desempleo de Colombia, qué se repite cada año, cómo se distribuye, qué se mueve con ella, cuánta memoria tiene y cuándo se rompe su patrón."),
    html.Div([
        U.note([f"Esta sección analiza Colombia ({C.month_long(_S.index[0])} a {C.month_long(_S.index[-1])}), el problema principal del estudio. El selector de país no la modifica; "
                "la evolución de los otros ocho países está en Global Intelligence. ", U.source(CO)], "info", "Alcance."),
        html.Div([html.Span("Periodo de análisis", className="seg-label"),
                  html.Div(dcc.RangeSlider(id="ea-range", min=A, max=B, step=1, value=[A, B], marks={y: {"label": str(y)} for y in range(A, B + 1, 4)} | {B: {"label": str(B)}},
                                           allowCross=False, pushable=4, tooltip={"placement": "bottom"}), className="range")], className="li-filter reveal"),
        html.Nav([dcc.Link(t, href=f"#{i}") for i, t in [("s21", "2.1 Evolución"), ("s22", "2.2 Estacionalidad"), ("s23", "2.3 Distribución"), ("s24", "2.4 Relaciones"),
                                                        ("s25", "2.5 Autocorrelación"), ("s26", "2.6 Rupturas")]], className="li-subnav", **{"aria-label": "Subsecciones"}),
        U.loading(html.Div(id="ea-a")), U.loading(html.Div(id="ea-b")), U.loading(html.Div(id="ea-c")),
    ], className="li-wrap"),
])


@callback(Output("ea-a", "children"), Input("ea-range", "value"))
@U.safe
def part_a(years):
    c = CO
    s = S.clip(c, years)
    if len(s) < 48:
        return U.note("Elija un periodo de al menos cuatro años.", "warn")
    up, dn = S.runs(s)
    sea = S.seasonality(s)
    mx, mn = sea.loc[sea["media"].idxmax()], sea.loc[sea["media"].idxmin()]
    strength = S.seasonal_strength(s)
    roll = s.diff().rolling(36).std().dropna()
    sig = sea[(sea["media"] - sea["h"] > 0) | (sea["media"] + sea["h"] < 0)]
    return html.Div([
        html.Section([
            U.block("2.1 · Evolución y tendencia", "Cómo se ha movido la tasa"),
            U.panel("Evolución mensual y tendencia", html.Div([
                U.graph(F.evolution(c, years, 460)),
                stats_row([("Media", f"{N(s.mean())} %"), ("Máximo", f"{N(s.max())} % · {C.month_short(s.idxmax())}"), ("Mínimo", f"{N(s.min())} % · {C.month_short(s.idxmin())}"),
                           ("Mayor subida de la tendencia", f"{N(up[0], 1)} pp" if up else "—"), ("Mayor bajada", f"{N(dn[0], 1)} pp" if dn else "—")]),
                interp("La tasa de desempleo nacional mensual y su tendencia (media móvil centrada de 12 meses, la de la descomposición clásica).",
                       f"Entre {C.month_long(s.index[0])} y {C.month_long(s.index[-1])} la tasa promedió {N(s.mean())} %. " +
                       (f"La tendencia subió como máximo {N(up[0], 1)} pp ({C.month_short(up[1])} a {C.month_short(up[2])}) y bajó como máximo {N(dn[0], 1)} pp ({C.month_short(dn[1])} a {C.month_short(dn[2])})." if up else ""),
                       (f"La volatilidad de los cambios mensuales, en ventanas de tres años, va de {N(roll.min())} a {N(roll.max())} pp: el nivel cambia despacio y los choques son bruscos." if len(roll) else "El periodo es corto para medir la volatilidad en ventanas de tres años."),
                       "Un buen pronóstico a un mes parte del último dato y lo corrige: es la idea del experto de persistencia y la base de los otros dos."),
            ]), sub="Las cruces rojas son cambios mensuales atípicos (ver 2.6)"),
        ], id="s21"),
        html.Section([
            U.block("2.2 · Estacionalidad", "Lo que se repite cada año"),
            U.grid(
                U.panel("Cambio mensual medio por mes del año", U.graph(F.seasonal_bars(s, 380)), sub="Barras de error: intervalo de confianza t del 95 %"),
                U.panel("Cada mes frente a la media de su año", U.graph(F.month_heat(s, 380)), sub="Ocre: por encima de la media del año. Azul: por debajo"),
            ),
            stats_row([("Mayor aumento medio", f"{C.MONTHS[int(mx['mes']) - 1]} · {N(mx['media'], 2, True)} pp"), ("Mayor caída media", f"{C.MONTHS[int(mn['mes']) - 1]} · {N(mn['media'], 2, True)} pp"),
                       ("Varianza explicada por el mes", f"{N(100 * strength, 0)} %"), ("Meses con IC 95 % sin el cero", f"{len(sig)} de {len(sea)}")]),
            interp("El cambio mensual medio de cada mes del año y la proporción de la varianza de los cambios que explica el mes (R² de las medias por mes).",
                   f"El mayor aumento medio ocurre en {C.MONTHS[int(mx['mes']) - 1]} ({N(mx['media'], 2, True)} pp; sube en el {N(100 * mx['sube'], 0)} % de los años) y la mayor caída en {C.MONTHS[int(mn['mes']) - 1]} ({N(mn['media'], 2, True)} pp).",
                   f"El mes del año explica el {N(100 * strength, 0)} % de la varianza de los cambios mensuales, y en {len(sig)} de {len(sea)} meses el intervalo de confianza del 95 % del cambio medio no contiene el cero.",
                   "La serie no está desestacionalizada: parte de lo que parece una noticia mensual es calendario. El experto estacional de ALEE suma al último dato justamente este cambio típico del mes."),
        ], id="s22"),
    ])


@callback(Output("ea-b", "children"), Input("ea-range", "value"))
@U.safe
def part_b(years):
    c = CO
    if len(S.clip(c, years)) < 48:
        return None
    opts = [{"label": RAW_LABELS[k], "value": k} for k in RAW_LABELS]
    xo = [{"label": RAW_LABELS[k], "value": f"raw:{k}"} for k in RAW_LABELS if k != "tasa_desempleo_nacional"]
    xo += [{"label": SHORT.get(k, FEAT.get(k, k)), "value": f"feat:{k}"} for k in E.features(c).columns if k not in ("mes_sin", "mes_cos", "trend_idx") and not k.endswith("_area")]
    return html.Div([
        html.Section([
            U.block("2.3 · Distribución", "Qué valores son habituales y cuáles no"),
            U.panel("Distribución univariada", html.Div([
                html.Div([html.Label("Variable", htmlFor="ea-var"), dcc.Dropdown(id="ea-var", options=opts, value=opts[0]["value"], clearable=False, searchable=False, className="li-dd")], className="li-ctl"),
                html.Div(id="ea-dist"),
            ]), sub="Las seis tasas publicadas: desempleo, ocupación y participación, nacionales y de las 13 áreas"),
        ], id="s23"),
        html.Section([
            U.block("2.4 · Correlaciones y relaciones", "Qué se mueve con el desempleo", "Una correlación describe que dos series se mueven juntas; no dice que una cause la otra."),
            U.grid(
                U.panel("Matriz de correlación", html.Div([
                    U.seg("ea-corr-mode", [("Tasas publicadas", "raw"), ("Variables del modelo", "feat")], "raw"),
                    html.Div(id="ea-corr"),
                ])),
                U.panel("Explorador de relaciones", html.Div([
                    html.Div([html.Label("Comparar con", htmlFor="ea-x"), dcc.Dropdown(id="ea-x", options=xo, value=xo[0]["value"], clearable=False, searchable=False, className="li-dd")], className="li-ctl"),
                    html.Div(id="ea-scatter"),
                ])),
            ),
        ], id="s24"),
    ])


def _raw(years):
    d = E.raw_colombia().set_index("fecha")
    return d[(d.index.year >= years[0]) & (d.index.year <= years[1])]


@callback(Output("ea-dist", "children"), Input("ea-var", "value"), dash.State("ea-range", "value"))
@U.safe
def dist(var, years):
    v, label = _raw(years)[var], RAW_LABELS[var]
    d = S.describe(v)
    iqr = d["Q3"] - d["Q1"]
    out = v[(v < d["Q1"] - 1.5 * iqr) | (v > d["Q3"] + 1.5 * iqr)]
    skew = "una cola hacia valores altos" if d["Asimetría"] > 0.5 else "una cola hacia valores bajos" if d["Asimetría"] < -0.5 else "una forma casi simétrica"
    st = html.Div([html.Div([html.Small(k), html.B(N(x) + ("" if k == "Asimetría" else " %" if k != "Desviación" else " pp"))]) for k, x in d.items()], className="li-stats col")
    return html.Div([html.Div([U.graph(F.distribution(v, label, 380)), st], className="dist-row"),
                     interp(f"La distribución de «{label}» en {len(v)} meses: histograma, densidad por núcleo y diagrama de caja.",
                            f"La mitad central de los meses está entre {N(d['Q1'])} % y {N(d['Q3'])} %; la mediana es {N(d['Mediana'])} % y la distribución tiene {skew}.",
                            f"Asimetría de {N(d['Asimetría'])} y {len(out)} meses fuera de las vallas de 1,5 rangos intercuartílicos.",
                            "Los meses extremos son episodios reales: no se eliminan ni se recortan, porque son los que un modelo de pronóstico debe enfrentar.")])


@callback(Output("ea-corr", "children"), Input("ea-corr-mode", "value"), dash.State("ea-range", "value"))
@U.safe
def corr(mode, years):
    c = CO
    if mode == "raw":
        df = _raw(years)[list(RAW_LABELS)]
        r = df.corr().loc["tasa_desempleo_nacional", "tasa_desempleo_area"]
        return [U.graph(F.corr_heat(df, RAW_LABELS, 460)), U.note(f"El desempleo nacional y el de las 13 áreas tienen r = {N(r)}. La ocupación y la participación nacionales determinan el desempleo del mismo mes por identidad contable (TD = 1 − TO/TGP) y por eso no son variables del modelo.")]
    f = E.features(c).copy()
    f["y"] = E.series(c).shift(-1)
    f = f.dropna()
    f = f[(f.index.year >= years[0]) & (f.index.year <= years[1])]
    f = f[["y"] + [k for k in f.columns if k != "y"]]
    f = f.loc[:, f.std() > 0]
    r = f.corr()["y"].drop("y").abs().sort_values(ascending=False)
    lab = {k: SHORT.get(k, FEAT.get(k, k)) for k in f.columns}
    return [U.graph(F.corr_heat(f, lab, 560, values=False)), U.note(f"La variable más correlacionada con la tasa del mes siguiente es «{lab[r.index[0]]}» (|r| = {N(r.iloc[0])}). Los rezagos y las medias móviles se parecen mucho entre sí: miden casi lo mismo.")]


@callback(Output("ea-scatter", "children"), Input("ea-x", "value"), dash.State("ea-range", "value"))
@U.safe
def scat(key, years):
    c = CO
    kind, k = key.split(":")
    if kind == "raw":
        d = _raw(years)
        x, y, lx, ly, sx = d[k], d["tasa_desempleo_nacional"], RAW_LABELS[k], "Desempleo nacional", " %"
    else:
        f = E.features(c)
        x = f[k]
        y = E.series(c).shift(-1)
        ok = x.notna() & y.notna() & (x.index.year >= years[0]) & (x.index.year <= years[1])
        x, y, lx, ly, sx = x[ok], y[ok], SHORT.get(k, FEAT.get(k, k)), "Tasa del mes siguiente", " pp" if "std" in k else " %"
    r = np.corrcoef(x, y)[0, 1]
    how = "fuerte" if abs(r) > 0.7 else "moderada" if abs(r) > 0.4 else "débil"
    return [U.graph(F.scatter(x, y, x.index.year, lx, ly, sx, " %", 460)), U.note(f"La relación lineal es {how} y {'directa' if r > 0 else 'inversa'} (r = {N(r)}) en {len(x)} meses. El color indica el año.")]


@callback(Output("ea-c", "children"), Input("ea-range", "value"))
@U.safe
def part_c(years):
    c = CO
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
        html.Section([
            U.block("2.5 · Autocorrelación y dependencia temporal", "Cuánto recuerda la serie de sí misma"),
            U.panel("Autocorrelación del nivel y del cambio mensual", html.Div([
                U.graph(F.acf_pair(s, 360)),
                stats_row([("ACF del nivel, rezago 1", N(a1)), ("ACF del cambio, rezago 12", N(a12)), ("Nivel", st[0][3]), ("Primera diferencia", st[1][3])]),
                U.table(rows, [("a", "Serie"), ("b", "ADF (H0: raíz unitaria), p"), ("c", "KPSS (H0: estacionaria), p"), ("d", "Lectura")], numeric=("b", "c")),
                interp("La autocorrelación del nivel y del cambio mensual, y las pruebas de raíz unitaria ADF y de estacionariedad KPSS.",
                       f"El nivel tiene autocorrelación de {N(a1)} en el primer rezago y decae despacio; la del cambio mensual se concentra en el rezago 12 ({N(a12)}).",
                       f"En niveles, la serie resulta: {st[0][3].lower()}. La primera diferencia: {st[1][3].lower()}. Bandas de ±1,96/√n.",
                       "Persistencia y calendario son las dos estructuras que ALEE incorpora directamente con sus expertos. Por la dependencia temporal, las particiones de validación respetan el orden del tiempo."),
            ]), sub="Los rezagos 12 y 24 se marcan en azul marino"),
        ], id="s25"),
        html.Section([
            U.block("2.6 · Cambios estructurales y rupturas", "Cuándo se rompe el patrón"),
            U.panel("Cambios mensuales y vallas externas de Tukey", html.Div([
                U.graph(F.anomaly_fig(s, 380)),
                U.table(an_rows, [("a", "Mes"), ("b", "Cambio"), ("c", "Tasa")], numeric=("b", "c")) if an_rows else U.note("Ningún cambio mensual queda fuera de las vallas en este periodo."),
                interp("Los cambios mensuales que caen fuera de las vallas externas de Tukey (más de tres rangos intercuartílicos por fuera de los cuartiles), la regla del paper.",
                       f"{len(an)} de {len(d)} cambios mensuales caen fuera de [{N(lo)}; {N(hi)}] pp." + (f" El más grande es el de {C.month_long(an.abs().idxmax())} ({N(an[an.abs().idxmax()], 2, True)} pp)." if len(an) else ""),
                       "La regla es descriptiva: marca rupturas del patrón mensual, no prueba un cambio estructural de la media.",
                       "Los atípicos se marcan, pero no se borran ni se recortan. Con pocos antecedentes parecidos, ALEE se apoya más en su combinación global; por eso los mayores errores se concentran en esos meses."),
            ]), sub="Regla del paper: más de tres rangos intercuartílicos por fuera de los cuartiles"),
        ], id="s26"),
    ])
