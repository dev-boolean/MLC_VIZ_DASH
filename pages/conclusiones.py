"""Conclusiones del proyecto y referencias (las mismas del paper)."""
import re

import dash
import dash_bootstrap_components as dbc
from dash import html

from src import config
from src import dash_data as D

dash.register_page(__name__, path="/conclusiones", name="Conclusiones", title="Conclusiones · Mercado Laboral Colombiano")

PAPER_TEX = config.ROOT_DIR / "paper" / "EDA_Mercado_Laboral_Colombiano_ALEE.tex"


def card(title, children, cls=""):
    return dbc.Card([dbc.CardHeader(title), dbc.CardBody(children)], className=f"mlc-card {cls}")


def _latex_to_parts(text):
    """Convierte una entrada de bibliografía en LaTeX a texto con enlaces."""
    text = text.replace(r"\={a}", "ā").replace(r"\_", "_")
    text = re.sub(r"\\textit\{([^}]*)\}", r"\1", text)
    text = text.replace(r"\&", "&").replace("--", "–").replace(r"\,", " ").replace("``", "«").replace("''", "»")
    text = re.sub(r"\\v\{?([a-zA-Z])\}?", lambda m: {"s": "š", "c": "č"}.get(m.group(1), m.group(1)), text)
    text = text.replace(r"\ ", " ")
    parts, pos = [], 0
    for m in re.finditer(r"\\url\{([^}]*)\}", text):
        parts.append(text[pos:m.start()])
        parts.append(html.A(m.group(1), href=m.group(1), target="_blank"))
        pos = m.end()
    parts.append(text[pos:])
    return parts


def references():
    """Bibliografía del paper, en el mismo orden."""
    tex = PAPER_TEX.read_text(encoding="utf-8")
    body = tex.split(r"\begin{thebibliography}")[1].split(r"\end{thebibliography}")[0]
    entries = [e.strip() for e in re.split(r"\\bibitem\{[^}]*\}", body)[1:]]
    return [html.Li(_latex_to_parts(e)) for e in entries if e]


def findings():
    rmse = D.paper_table()
    others = rmse.drop(columns="ALEE")
    red = 1 - rmse["ALEE"] / others.min(axis=1)
    tests = D.paper_tests()
    between = D.paper_between()
    abl = D.paper_ablation()
    simple = abl[abl["variante"] == "Promedio simple"]
    simple_worse = int(((simple["dif_perdida"] < 0) & (simple["dm_p_holm"] < 0.05)).sum())
    loc = abl[(abl["variante"] == "Sin localización") & (abl["dm_p_holm"] < 0.05)]["conjunto"].map(D.DATASETS).tolist()
    return {"wins": int((rmse["ALEE"] < others.min(axis=1)).sum()), "n": len(rmse), "red_min": red.min(),
            "red_max": red.max(), "dm_sig": int((tests["dm_p_holm"] < 0.05).sum()), "n_tests": len(tests),
            "wilcoxon": between["wilcoxon_p_holm"].max(), "simple_worse": simple_worse, "loc": loc}


F = findings()
raw = D.raw()
peak = raw.loc[raw["tasa_desempleo_nacional"].idxmax()]

layout = html.Div([
    html.H2("Conclusiones", className="mlc-section-title"),
    dbc.Row([
        dbc.Col(card("Lo que muestran los datos", [
            html.P("La tasa de desempleo nacional es persistente y tiene un patrón estacional marcado: sube en enero y "
                   "baja hacia final de año. Ese patrón es la información más útil para pronosticar el mes siguiente."),
            html.P(f"La pandemia fue un choque extremo: la tasa llegó a {peak['tasa_desempleo_nacional']:.1f} % en "
                   f"{D.month_es(peak['fecha'])}. Esos meses no se eliminaron, porque son parte real de la serie, pero "
                   "concentran los mayores errores de todos los modelos."),
            html.P("El desempleo nacional y el de las 13 áreas metropolitanas se mueven casi juntos, y la ocupación y la "
                   "participación del mismo mes determinan el desempleo por una identidad contable; por eso esas dos "
                   "variables se excluyeron de los modelos.", className="mb-0"),
        ]), lg=6),
        dbc.Col(card("Lo que muestran los modelos", [
            html.P(f"ALEE tiene el menor RMSE en {F['wins']} de los {F['n']} conjuntos, con entre {F['red_min']:.0%} y "
                   f"{F['red_max']:.0%} menos error que el mejor modelo de regresión del curso o LightGBM. Las diferencias "
                   f"son significativas en {F['dm_sig']} de {F['n_tests']} comparaciones (Diebold-Mariano-HLN con Holm) y "
                   f"entre conjuntos (Wilcoxon con Holm, p = {F['wilcoxon']:.3f})."),
            html.P(f"La ventaja viene sobre todo de los expertos, que incorporan la persistencia y la estacionalidad de la "
                   f"serie. La combinación aprendida supera con significancia al promedio simple en {F['simple_worse']} "
                   f"conjuntos, y la localización por estado, que era la idea central del modelo, solo aporta de forma "
                   f"significativa en {' y '.join(F['loc']) or 'ningún conjunto'}."),
            html.P("Para el pronóstico del desempleo a un mes, incorporar la estructura temporal de la serie pesa más que "
                   "la flexibilidad de un modelo genérico.", className="mb-0 fw-semibold"),
        ]), lg=6),
    ]),
    card("Limitaciones", html.Ul([
        html.Li("Nueve conjuntos dan una potencia moderada a las pruebas entre conjuntos."),
        html.Li("En Colombia se evaluaron diez iteraciones de diseño sobre los mismos 240 meses; los ocho países "
                "adicionales, evaluados una sola vez con el modelo congelado, son la evidencia más confiable."),
        html.Li("La prueba bayesiana con ROPE supone pliegues intercambiables y aquí se aplica a pliegues temporales "
                "dependientes: sus probabilidades son orientativas."),
        html.Li("Los datos son la versión vigente de las series y no la publicada en tiempo real cada mes."),
        html.Li("Los países adicionales no tienen indicadores de área: allí todos los modelos usan 14 variables y no 17."),
    ], className="mb-0")),
    html.H2("Referencias", className="mlc-section-title"),
    card("Bibliografía del paper", html.Ol(references(), className="small mb-0")),
])
