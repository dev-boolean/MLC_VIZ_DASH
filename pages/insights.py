"""06. Insights & Conclusions: síntesis de la investigación, limitaciones, referencias e informe."""
import re
from functools import lru_cache

import dash
from dash import Input, Output, callback, dcc, html

from labor import catalog as C
from labor import engine as E
from labor import figs as F
from labor import stats as S
from labor import ui as U
from src import config

dash.register_page(__name__, path="/insights", name="Insights & Conclusions", title="Insights & Conclusions · LABOR INSIGHTS")
N = C.num
TEX = config.ROOT_DIR / "paper" / "EDA_Mercado_Laboral_Colombiano_ALEE.tex"


def _parts(t):
    t = t.replace(r"\={a}", "ā").replace(r"\_", "_")
    t = re.sub(r"\\textit\{([^}]*)\}", r"\1", t)
    t = t.replace(r"\&", "&").replace("--", "–").replace(r"\,", " ").replace("``", "«").replace("''", "»")
    t = re.sub(r"\\v\{?([a-zA-Z])\}?", lambda m: {"s": "š", "c": "č"}.get(m.group(1), m.group(1)), t).replace(r"\ ", " ")
    out, pos = [], 0
    for m in re.finditer(r"\\url\{([^}]*)\}", t):
        out += [t[pos:m.start()], html.A(m.group(1), href=m.group(1), target="_blank", rel="noopener")]
        pos = m.end()
    return out + [t[pos:]]


def references():
    body = TEX.read_text(encoding="utf-8").split(r"\begin{thebibliography}")[1].split(r"\end{thebibliography}")[0]
    return [html.Li(_parts(e.strip())) for e in re.split(r"\\bibitem\{[^}]*\}", body)[1:] if e.strip()]


@lru_cache(maxsize=None)
def findings():
    """Cifras de la síntesis, calculadas con los resultados guardados (nada se escribe a mano)."""
    rmse = E.metric_table()
    best = rmse.drop(columns="ALEE").min(axis=1)
    red = 1 - rmse["ALEE"] / best
    t = E.result("pruebas_por_baseline")
    b = E.result("pruebas_entre_conjuntos")
    abl = E.result("ablacion")
    simple = abl[(abl["variante"] == "Promedio simple") & (abl["dif_perdida"] < 0) & (abl["dm_p_holm"] < 0.05)]
    loc = abl[(abl["variante"] == "Sin localización") & (abl["dm_p_holm"] < 0.05)]["conjunto"].map(C.name).tolist()
    seas = abl[(abl["variante"] == "Solo estacional") & (abl["dm_p_holm"] >= 0.05)]["conjunto"].map(C.name).tolist()
    strength = {c: S.seasonal_strength(E.series(c)) for c in C.COUNTRIES}
    cov = {c: E.conformal(c)["coverage"] for c in C.COUNTRIES}
    return dict(red=red, wins=int((rmse.idxmin(axis=1) == "ALEE").sum()), n_sig=int((t["dm_p_holm"] < 0.05).sum()), n_tests=len(t), simple=len(simple), loc=loc, seas=seas,
                strength=strength, cov=cov, n_test=sum(len(E.oof(c)) for c in C.COUNTRIES), wil=float(b["wilcoxon_p_holm"].max()),
                conover_ns=b[b["conover_p_holm"] >= 0.05]["baseline"].map(C.MODEL_LABELS).tolist())


def _list(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " y " + xs[-1] if xs else "ningún país"


def questions(R):
    st, cov, red = R["strength"], R["cov"], R["red"]
    hi, lo = max(st, key=st.get), min(st, key=st.get)
    return [
        ("¿Qué aprendimos del desempleo?",
         "Es persistente y estacional, con choques escasos pero enormes.",
         f"El mes del año explica desde el {N(100 * st[lo], 0)} % ({C.name(lo)}) hasta el {N(100 * st[hi], 0)} % ({C.name(hi)}) de la varianza de los cambios mensuales. Los choques, como el de 2020, concentran los mayores errores de todos los modelos.",
         ("Exploratory Analytics", "/analytics")),
        ("¿Qué capacidad predictiva tiene ALEE?",
         f"Tuvo el menor RMSE en {R['wins']} de 9 países.",
         f"En {R['n_test']} meses de prueba fuera de muestra, con entre {N(100 * red.min(), 0)} % y {N(100 * red.max(), 0)} % menos error que el mejor modelo de comparación de cada país. "
         f"Sus intervalos del 90 % cubrieron entre el {N(100 * min(cov.values()), 1)} % y el {N(100 * max(cov.values()), 1)} % de los casos medidos hacia adelante.",
         ("Model Intelligence", "/modelo")),
        ("¿Qué ventajas presenta?",
         "Incorpora la persistencia y el calendario en lugar de aprenderlos de pocos datos, y cada pronóstico se explica con exactitud.",
         f"El pronóstico se descompone sin aproximaciones en el aporte de tres expertos. La combinación aprendida supera con significancia al promedio simple en {R['simple']} de 9 países.",
         ("Model Intelligence", "/modelo")),
        ("¿Cómo se comporta internacionalmente?",
         "La ventaja se sostiene en los ocho países evaluados con el diseño congelado.",
         f"Diebold-Mariano con Holm rechaza la igualdad en {R['n_sig']} de {R['n_tests']} comparaciones país-modelo; entre países, Wilcoxon da p = {N(R['wil'], 3)}. Cada país se ajusta con su propia serie: generaliza el método, no un modelo único.",
         ("Global Intelligence", "/global")),
        ("¿Qué aplicaciones puede tener?",
         "Una referencia académica a un mes, con su margen de error explícito.",
         "Sirve como punto de comparación para analistas, como monitor internacional y como banco de pruebas para otras reglas de combinación de pronósticos. No sustituye a las cifras oficiales ni a las decisiones de política.",
         ("Unemployment Forecast", "/forecast")),
        ("¿Cuáles son sus limitaciones?",
         "La idea central, la localización por estado, aporta menos de lo esperado.",
         f"Solo aporta con significancia en {_list(R['loc'])}. El experto estacional por sí solo no se distingue de ALEE en {len(R['seas'])} de 9 países. ALEE reacciona tarde al inicio de una crisis y solo pronostica un mes.",
         ("Limitaciones", "#limitaciones")),
    ]


def limitations(R):
    return [
        ("Teóricas", ["Las propiedades demostradas garantizan solución única, descenso monótono y un error que nunca supera al del peor experto; no garantizan superar al mejor experto.",
                      "El intervalo conforme supone errores intercambiables. Con errores de series de tiempo, dependientes entre sí, la cobertura del 90 % no está garantizada: se mide hacia adelante."]),
        ("Empíricas", [f"La localización por estado solo aporta con significancia en {_list(R['loc'])}; la mayor parte de la ventaja viene de los expertos.",
                       f"El experto estacional por sí solo no se distingue de ALEE en {len(R['seas'])} de 9 países ({_list(R['seas'])}).",
                       "ALEE reacciona tarde al inicio de una crisis: el estado se construye con meses ya observados, y con pocos meses parecidos la combinación vuelve a la global."]),
        ("De los datos", ["Se usa la versión vigente de las series, no la publicada en tiempo real cada mes: puede quedar fuga de información residual por revisiones.",
                          "Octubre de 2025 no se publicó en Estados Unidos y se interpoló.",
                          "Los países adicionales no tienen indicadores de área: en ellos todos los modelos usan 14 variables y no 17."]),
        ("De generalización", [f"Nueve conjuntos dan una potencia moderada a las pruebas entre países" + (f": Conover no distingue a ALEE de {_list([x.lower() for x in R['conover_ns']])}." if R["conover_ns"] else "."),
                               "Cada país tiene su propio ALEE: se comprueba que el método generaliza, no que un modelo entrenado en Colombia sirva en otro país.",
                               "La comparación incluye los modelos del curso y LightGBM, no todos los métodos de series de tiempo que existen."]),
        ("Metodológicas", ["En Colombia se evaluaron diez iteraciones de diseño sobre los mismos 240 meses, lo que puede volver optimista el resultado en ese conjunto; los ocho países adicionales son la evidencia más confiable.",
                           "Varias piezas del análisis se agregaron después de conocer los resultados, aunque ninguna modifica las predicciones de ALEE.",
                           "La prueba bayesiana con ROPE supone particiones intercambiables y se aplica sobre cinco pliegues temporales dependientes: sus probabilidades son orientativas."]),
    ]


def key_limitations(R):
    """Las cuatro limitaciones que más pesan en la lectura de los resultados (el resto queda desplegable)."""
    return [f"La localización por estado, la idea central, solo aporta con significancia en {_list(R['loc'])}: la ventaja viene sobre todo de los expertos.",
            f"El experto estacional por sí solo no se distingue de ALEE en {len(R['seas'])} de 9 países.",
            "ALEE reacciona tarde al inicio de una crisis y solo pronostica un mes.",
            "En Colombia se probaron diez iteraciones de diseño sobre los mismos meses: los ocho países adicionales, con el diseño congelado, son la evidencia más confiable."]


QUESTION = ("¿Puede un modelo de combinación de expertos, con pesos que dependen del estado del mercado laboral, pronosticar la tasa "
            "de desempleo del mes siguiente mejor que los modelos de regresión estudiados en clase?")


def answer(R):
    red = R["red"]
    yes = "Sí. " if R["wins"] == 9 else ""
    return (f"{yes}ALEE tuvo el menor RMSE en {R['wins']} de 9 países, con entre {N(100 * red.min(), 0)} % y {N(100 * red.max(), 0)} % menos error que el mejor "
            f"modelo de comparación, y la diferencia es significativa en {R['n_sig']} de {R['n_tests']} comparaciones. Con un matiz: la ventaja viene sobre todo "
            f"de los expertos; los pesos por estado solo aportan con significancia en {_list(R['loc'])}.")


def contributions(R):
    return [("Un modelo original", "ALEE combina tres expertos sencillos con pesos que resuelven un problema convexo con solución única, regularizado hacia una combinación global."),
            ("Una validación amplia", f"{R['n_test']} meses fuera de muestra en nueve países, con protocolo pre-registrado; ocho países se evaluaron con el diseño ya congelado."),
            ("Pronósticos explicables", "Cada pronóstico se descompone exactamente en el aporte de cada experto y viene con un intervalo del 90 % cuya cobertura se mide hacia adelante.")]


def real_check():
    """Primer dato oficial publicado después del corte que el modelo pudo pronosticar (Colombia)."""
    v = E.later_validation(C.DEFAULT)
    if v.empty or v["pred"].isna().all():
        return None
    r = v.dropna(subset=["pred"]).iloc[0]
    inside = "dentro del intervalo" if r["dentro"] else "fuera del intervalo"
    return html.Div([
        html.Small("Comprobación con un dato real", className="ans-k"),
        html.P([f"Con datos hasta {C.month_long(r['origen'])}, ALEE pronosticó ", html.B(f"{N(r['pred'])} %", className="est"),
                f" para {C.month_long(r['objetivo'])} (intervalo del 90 %: {N(r['lo'])} a {N(r['hi'])} %). El DANE publicó después ", html.B(f"{N(r['obs'], 1)} %", className="obs"),
                f": {inside}, con un error de {N(r['error'], 2, True)} pp."]),
        html.P(["Es un solo mes: una comprobación, no una evaluación. ", html.A("Boletín del DANE", href=r["fuente"], target="_blank", rel="noopener"), "."], className="li-cap"),
    ], className="ans-check")


FUTURE = ["Variables de estado que reaccionen antes a los saltos bruscos: la volatilidad de tres meses y la brecha frente a la media anual se basan en meses ya observados.",
          "Pronóstico a más de un mes, que exige expertos y estados definidos para cada horizonte.",
          "Más conjuntos de datos, para dar más potencia a las pruebas entre países."]


def layout(**_):
    R = findings()
    red = R["red"]
    return html.Div([
        U.head("06", "Insights & Conclusions", "Lo que deja", "esta investigación.", "La respuesta a la pregunta de investigación, la evidencia que la sostiene y lo que no permite afirmar. Todas las cifras se calculan al cargar la página."),
        html.Div([
            html.Section([
                html.Div([
                    html.Small("Pregunta de investigación", className="ans-k"),
                    html.P(QUESTION, className="ans-q"),
                    html.Small("Respuesta", className="ans-k"),
                    html.P(answer(R), className="ans-a"),
                ], className="ans-main"),
                real_check(),
            ], className="li-panel pad ans reveal"),
            html.Div([html.Div([html.Span(f"{i + 1}", className="contrib-n"), html.H3(t), html.P(x)], className="contrib reveal")
                      for i, (t, x) in enumerate(contributions(R))], className="contrib-grid"),
            U.kpis(U.kpi("Países con el menor RMSE", f"{R['wins']} de 9", "calc", foot="ALEE frente a nueve modelos de comparación"),
                   U.kpi("Reducción del error", f"{N(100 * red.min(), 0)} a {N(100 * red.max(), 0)} %", "calc", foot="RMSE frente al mejor modelo de cada país"),
                   U.kpi("Comparaciones significativas", f"{R['n_sig']} de {R['n_tests']}", "calc", foot="Diebold-Mariano-HLN con Holm"),
                   U.kpi("Meses de prueba", str(R["n_test"]), "calc", foot="Todos fuera de muestra")),
            U.block("6.1 · Seis preguntas", "Respuestas con su evidencia"),
            html.Div([html.Article([
                html.Span(f"{i + 1:02d}", className="q-n"), html.H3(q), html.P(a, className="q-a"),
                html.Div([html.Small("Evidencia"), html.P(ev)], className="q-ev"),
                dcc.Link(f"Ver en {ref[0]} →", href=ref[1], className="li-link") if ref[1].startswith("/") else html.A(f"Ver {ref[0].lower()} →", href=ref[1], className="li-link"),
            ], className="q-card reveal") for i, (q, a, ev, ref) in enumerate(questions(R))], className="q-grid"),
            U.block("6.2 · Evidencia experimental", "Los resultados que sostienen las conclusiones"),
            U.grid(U.panel("Reducción del error por país", U.graph(F.reductions(380)), sub="Frente al mejor modelo de comparación de cada país"),
                   U.panel("Cambio pronosticado por país", U.graph(F.forecast_changes(380)), sub="Con su intervalo del 90 %")),
            U.panel("Comparación internacional de los modelos", U.graph(F.cd_fig(360)), sub="Rango medio en los nueve países (1 = mejor) y diferencia crítica de Nemenyi"),
            html.Section([
                U.block("6.3 · Limitaciones y trabajo futuro", "Lo que esta evidencia no permite afirmar"),
                html.Div([html.Div([html.H3("Las cuatro que más pesan"), html.Ul([html.Li(x) for x in key_limitations(R)])], className="lim-card"),
                          html.Div([html.H3("Trabajo futuro"), html.Ol([html.Li(x) for x in FUTURE])], className="lim-card future")], className="lim-grid reveal"),
                html.H3("Todas las limitaciones, por tipo", className="li-group"),
                html.Div([html.Div([html.H3(t), html.Ul([html.Li(x) for x in xs])], className="lim-card compact") for t, xs in limitations(R)], className="lim-grid all reveal"),
            ], id="limitaciones"),
            html.Section([
                U.block("6.4 · Referencias y documentación", "Fuentes de la investigación"),
                U.panel("Referencias del paper", html.Ol(references(), className="refs cols")),
                html.Div([
                        U.panel("Fuentes de datos", html.Ul([html.Li([html.B(m["name"] + ": "), f"{m['source']}. {m['definition']}."]) for m in C.COUNTRIES.values()], className="li-src")),
                        U.panel("Documentación metodológica", html.Ul([
                            html.Li("Paper: «EDA del mercado laboral colombiano y ALEE» (paper/EDA_Mercado_Laboral_Colombiano_ALEE.tex)."),
                            html.Li("Declaración de datos: DATA_STATEMENT.md."),
                            html.Li("Implementación del modelo: src/original_model.py, con sus propiedades verificadas en las pruebas."),
                            html.Li(["Video del proyecto: ", html.A("YouTube", href="https://www.youtube.com/watch?v=2pBUO-to_KM", target="_blank", rel="noopener"), "."]),
                        ], className="li-src")),
                        U.panel("Informe en PDF", html.Div([
                            html.P("Resumen académico del país seleccionado: indicadores, pronóstico e intervalo, validación, evaluación, conclusiones, limitaciones y referencias, con los datos de la plataforma.", className="li-cap"),
                            html.A("Descargar el informe", id="in-pdf", href="/reporte.pdf?pais=colombia", className="btn-li primary"),
                        ])),
                ], className="li-grid g3"),
            ], id="referencias"),
        ], className="li-wrap"),
        html.Section([
            html.Div([
                html.Span("LABOR INSIGHTS", className="close-k"),
                html.H2("Comprender el mercado laboral. Evaluar la evidencia predictiva."),
                html.P(f"Esta investigación propone ALEE, una combinación local y regularizada de tres expertos sencillos, y la evalúa fuera de muestra en nueve países frente a nueve modelos de comparación. "
                       f"Pronostica un mes adelante, con su intervalo del 90 % y con la evidencia completa de dónde funciona y dónde no."),
                html.P("Andrés Parejo, Santiago Hurtado y Juan Marín · Pregrado en Ciencia de Datos, curso de Machine Learning · Universidad del Norte · 2026", className="close-auth"),
                dcc.Link("Volver a Overview", href="/", className="btn-li ghost"),
            ], className="close-copy reveal"),
        ], className="li-close"),
    ])


@callback(Output("in-pdf", "href"), Input("pais", "value"))
def pdf_link(c):
    return f"/reporte.pdf?pais={c or C.DEFAULT}"
