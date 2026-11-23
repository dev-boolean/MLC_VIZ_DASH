"""07. Insights & Conclusions."""
import re

import dash
from dash import dcc, html

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


def layout(**_):
    rmse = E.metric_table()
    best = rmse.drop(columns="ALEE").min(axis=1)
    red = 1 - rmse["ALEE"] / best
    t = E.result("pruebas_por_baseline")
    abl = E.result("ablacion")
    simple = abl[(abl["variante"] == "Promedio simple") & (abl["dif_perdida"] < 0) & (abl["dm_p_holm"] < 0.05)]
    loc = abl[(abl["variante"] == "Sin localización") & (abl["dm_p_holm"] < 0.05)]["conjunto"].map(C.name).tolist()
    strength = {c: S.seasonal_strength(E.series(c)) for c in C.COUNTRIES}
    hi, lo = max(strength, key=strength.get), min(strength, key=strength.get)
    cov = [E.conformal(c)["coverage"] for c in C.COUNTRIES]
    n_test = sum(len(E.oof(c)) for c in C.COUNTRIES)
    q = [
        ("¿Qué aprendimos del desempleo?", [f"Es persistente y estacional. El mes del año explica desde el {N(100 * strength[lo], 0)} % ({C.name(lo)}) hasta el {N(100 * strength[hi], 0)} % ({C.name(hi)}) de los cambios mensuales.",
                                            "Los choques, como el de 2020, son pocos y enormes: concentran los mayores errores de todos los modelos."]),
        ("¿Qué capacidad predictiva tiene?", [f"En {n_test} meses de prueba, ALEE tuvo el menor RMSE en los nueve países, con entre {N(100 * red.min(), 0)} % y {N(100 * red.max(), 0)} % menos error que el mejor modelo de comparación.",
                                              f"Sus intervalos del 90 % cubrieron entre el {N(100 * min(cov), 0)} % y el {N(100 * max(cov), 0)} % de los casos."]),
        ("¿Qué ventajas presenta?", ["Incorpora la persistencia y el calendario en lugar de aprenderlos de pocos datos.", f"Cada pronóstico se descompone con exactitud en tres expertos. La combinación aprendida supera al promedio simple en {len(simple)} de 9 países."]),
        ("¿Cómo se comporta internacionalmente?", [f"Diebold-Mariano con Holm rechaza la igualdad en {int((t['dm_p_holm'] < 0.05).sum())} de {len(t)} comparaciones. Los ocho países adicionales se evaluaron una sola vez, con el diseño congelado.",
                                                   "Cada país se ajusta con su propia serie: generaliza el método, no un modelo único."]),
        ("¿Qué aplicaciones puede tener?", ["Referencia a un mes para analistas, con margen de error explícito.", "Monitor internacional y banco de pruebas para otras reglas de combinación. Es una herramienta académica: no sustituye a las cifras oficiales."]),
        ("¿Cuáles son sus limitaciones?", [f"La localización por estado solo aporta con significancia en {' y '.join(loc)}. En Colombia, Estados Unidos y otros cuatro países no se distingue del experto estacional solo.",
                                          "Reacciona tarde al inicio de una crisis, solo pronostica un mes y usa la versión vigente de las series. En Colombia se probaron diez iteraciones de diseño sobre los mismos 240 meses."]),
    ]
    return html.Div([
        U.head("07", "INSIGHTS & CONCLUSIONS", "Lo que deja", "este trabajo.", "Seis preguntas, respondidas con los resultados de la evaluación."),
        html.Div([
            U.kpis(U.kpi("Países con el menor RMSE", "9 de 9", "ref", foot="ALEE frente a nueve modelos"), U.kpi("Reducción del error", f"{N(100 * red.min(), 0)} a {N(100 * red.max(), 0)} %", "ref", foot="RMSE frente al mejor modelo de comparación"),
                   U.kpi("Comparaciones significativas", f"{int((t['dm_p_holm'] < 0.05).sum())} de {len(t)}", "ref", foot="Diebold-Mariano-HLN, Holm"), U.kpi("Meses de prueba", str(n_test), "ref", foot="Todos fuera de muestra")),
            html.Div([html.Article([html.Span(f"{i + 1:02d}", className="q-n"), html.H3(a), *[html.P(x) for x in b]], className="q-card reveal") for i, (a, b) in enumerate(q)], className="q-grid"),
            U.grid(U.panel("Reducción del error por país", U.graph(F.reductions())), U.panel("Cambio pronosticado por país", U.graph(F.forecast_changes(360)))),
            html.Details([html.Summary("Referencias del paper"), html.Ol(references(), className="refs")], className="li-panel pad reveal"),
        ], className="li-wrap"),
        html.Section([
            html.Div(className="li-globe big", **{"data-mode": "close"}),
            html.Div([html.P("El futuro del mercado laboral no se observa únicamente en los datos."), html.P("También podemos anticipar su comportamiento.", className="li-accent"),
                      html.Div("UNDERSTAND THE LABOR MARKET. PREDICT WHAT COMES NEXT.", className="close-tag"),
                      dcc.Link("Volver al inicio", href="/", className="btn-li ghost")], className="close-copy reveal"),
        ], className="li-close"),
    ])
