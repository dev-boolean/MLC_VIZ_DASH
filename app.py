"""LABOR INSIGHTS: punto de entrada de la plataforma.

El diseño tipo artículo académico vive en diseno/: estilo.css (se carga después de assets/labor.css), tema.py
(paleta y letra de las figuras), li_globe.js (colores del globo) y extras.py (bloques agregados para la rúbrica:
2.7 formulación y preprocesamiento; 3.4 diseño experimental; 3.5 sobreajuste y robustez). Las páginas de pages/ no cambian.

Local:   python app.py                  (http://127.0.0.1:8050)
Render:  gunicorn app:server
"""
import json
import os

import dash
from dash import Input, Output, dcc, html
from flask import Response, abort, jsonify, request

from labor import catalog as C
from labor import engine as E
from labor import theme  # noqa: F401  (registra la plantilla de las figuras)
from labor import ui as U
from diseno import tema  # noqa: F401  (diseño tipo artículo: paleta y letra de las figuras; va antes de importar las páginas)
from src.chatbot import chatbot_bp

ROOT = os.path.dirname(os.path.abspath(__file__))
# Three.js, GSAP, Bootstrap y el mapa del globo van en vendor/ (copias exactas de las versiones de CDN) para que la
# plataforma funcione sin internet en la sustentación. Solo las fuentes de Google vienen de internet; sin ellas se usan
# las de respaldo del sistema.
VENDOR = os.path.join(ROOT, "vendor")
DISENO = os.path.join(ROOT, "diseno")
# El asistente no se ha probado: queda oculto salvo que se active con LI_CHATBOT=1.
CHATBOT = os.environ.get("LI_CHATBOT") == "1"

# Nombres de las secciones en español. Las páginas de pages/ no se modifican: solo cambia cómo se rotulan.
NOMBRES = {"Overview": "Panorama", "Exploratory Analytics": "Análisis exploratorio", "Model Intelligence": "Modelo",
           "Unemployment Forecast": "Pronóstico", "Global Intelligence": "Comparación internacional",
           "Insights & Conclusions": "Conclusiones"}
_head = U.head
U.head = lambda num, kicker, title, accent, lead: _head(num, NOMBRES.get(kicker, kicker), title, accent, lead)

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[
        "/vendor/bootstrap.min.css",
        "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400"
        "&family=IBM+Plex+Serif:ital,wght@0,400;0,500;0,600;1,400&display=swap",
    ],
    # El globo sale de diseno/li_globe.js (misma lógica, colores que se distinguen del fondo blanco).
    external_scripts=["/vendor/three.min.js", "/vendor/gsap.min.js", "/vendor/ScrollTrigger.min.js", "/diseno/li_globe.js"],
    assets_ignore=r"li_globe\.js" if CHATBOT else r"chatbot\.js|li_globe\.js",
    title="LABOR INSIGHTS",
    update_title=None,
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"},
               {"name": "theme-color", "content": "#FFFFFF"}],
)
app.index_string = (app.index_string.replace("<html>", '<html lang="es" data-bs-theme="light">')
                    .replace("{%css%}", '{%css%}\n<link rel="stylesheet" href="/diseno/estilo.css">'))
server = app.server
server.register_blueprint(chatbot_bp)

from diseno import extras  # noqa: E402  (paneles agregados a Pronóstico y Modelo; registra sus callbacks)

for _p in dash.page_registry.values():
    _p["name"] = NOMBRES.get(_p["name"], _p["name"])
    _p["title"] = f"{_p['name']} · LABOR INSIGHTS"
extras.agregar(dash.page_registry)


@server.route("/vendor/<path:name>")
def vendor(name):
    from flask import send_from_directory

    return send_from_directory(VENDOR, name, max_age=86400)


@server.route("/diseno/<any('estilo.css', 'li_globe.js'):name>")
def diseno(name):
    from flask import send_from_directory

    return send_from_directory(DISENO, name, max_age=0)  # sin caché: un cambio de diseño se ve en la siguiente recarga


@server.route("/health")
def health():
    return "ok", 200


@server.route("/api/globo")
def globe_data():
    """Datos del globo: posición, último dato y pronóstico de cada país."""
    out = []
    for c, m in C.COUNTRIES.items():
        f = E.forecast(c)
        out.append({"id": c, "name": m["name"], "iso": m["iso"], "lat": m["lat"], "lon": m["lon"],
                    "obs": C.num(f["observed"]), "pred": C.num(f["pred"]), "last": C.month_short(f["last_date"]),
                    "target": C.month_short(f["target_date"])})
    return jsonify(out)


@server.route("/reporte.pdf")
def report_pdf():
    """Informe académico en PDF del país seleccionado, generado con los datos y resultados de la plataforma."""
    from labor.report import build_report

    c = request.args.get("pais", C.DEFAULT)
    if c not in C.COUNTRIES:
        abort(404)
    pdf = build_report(c)
    name = f"LABOR_INSIGHTS_{C.COUNTRIES[c]['short']}.pdf"
    return Response(pdf, mimetype="application/pdf", headers={"Content-Disposition": f'attachment; filename="{name}"'})


# Los nueve pronósticos se calculan una sola vez, al iniciar.
for _c in C.COUNTRIES:
    E.forecast(_c)


def _intro_data():
    """Serie de Colombia para la introducción animada: la misma fuente del EDA (``E.series``), sin transformar."""
    f = E.frame(C.DEFAULT)
    return json.dumps({"fechas": f["fecha"].dt.strftime("%Y-%m").tolist(), "valores": [float(v) for v in f["valor"]],
                       "interpolados": int(f["interpolado"].sum()), "fuente": C.COUNTRIES[C.DEFAULT]["source"]},
                      ensure_ascii=False).replace("</", "<\\/")


# Va dentro del HTML para que la introducción no necesite otra petición.
app.index_string = app.index_string.replace(
    "{%app_entry%}", f'<script id="li-intro-data" type="application/json">{_intro_data()}</script>\n{{%app_entry%}}')

NAV = [("01", "Panorama", "/"), ("02", "Análisis exploratorio", "/analytics"), ("03", "Modelo", "/modelo"),
       ("04", "Pronóstico", "/forecast"), ("05", "Comparación internacional", "/global"), ("06", "Conclusiones", "/insights")]

sidebar = html.Aside([
    dcc.Link([html.Img(src=app.get_asset_url("logo.png"), alt="Logo del Banco de la República", className="li-logo"),
              html.Span([html.B("LABOR INSIGHTS"), html.Small("Global Labor Market Intelligence")])], href="/", className="li-brand"),
    html.Nav(id="li-nav", className="li-nav", **{"aria-label": "Secciones"}),
    html.Div([
        html.P("Investigación académica", className="li-side-k"),
        html.P("Pronóstico a un mes de la tasa de desempleo con ALEE, un modelo original de combinación de expertos."),
    ], className="li-side-about"),
    html.Div([html.P("Andrés Parejo, Santiago Hurtado y Juan Marín"), html.P("Universidad del Norte · 2026"),
              html.A("Video del proyecto", href="https://www.youtube.com/watch?v=2pBUO-to_KM", target="_blank", rel="noopener"),
              html.Div(html.Button("Ver la introducción", id="li-intro-replay", className="li-replay", title="Repetir la introducción animada"))],
             className="li-credits"),
], className="li-side")

topbar = html.Div([
    html.Button("☰", id="li-burger", className="li-burger", **{"aria-label": "Abrir menú"}),
    html.Div(id="li-crumb", className="li-crumb"),
    html.Div([
        html.Div(id="li-prov"),
        html.Div([
            html.Label("País", htmlFor="pais"),
            dcc.Dropdown(id="pais", options=[{"label": m["name"], "value": c} for c, m in C.COUNTRIES.items()], value=C.DEFAULT,
                         clearable=False, searchable=False, persistence=True, persistence_type="session", className="li-country"),
        ], className="li-country-wrap"),
        html.A("Informe PDF", id="li-pdf", href=f"/reporte.pdf?pais={C.DEFAULT}", className="btn-li ghost sm li-pdf",
               title="Descargar el informe del país seleccionado"),
    ], className="li-top-right"),
    html.Div(html.Div(id="li-progress-bar"), className="li-progress", **{"aria-hidden": "true"}),
], className="li-top")

app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    html.Div([topbar, html.Main(dash.page_container, id="li-main", className="li-main"),
              html.Footer(["Plataforma académica. No sustituye las cifras oficiales. Datos: DANE (GEIH), vía Banco de la República; BLS, vía FRED; "
                           "Eurostat y OCDE."], className="li-foot")], className="li-shell"),
], id="li-app")


@app.callback(Output("li-nav", "children"), Output("li-crumb", "children"), Input("url", "pathname"))
def nav(path):
    path = path or "/"
    links = [dcc.Link([html.Span(n, className="n"), html.Span(t)], href=h, className="on" if h == path else "") for n, t, h in NAV]
    here = next(((n, t) for n, t, h in NAV if h == path), ("", ""))
    return links, [html.Span(here[0], className="n"), html.Span(here[1])]


@app.callback(Output("li-prov", "children"), Output("li-pdf", "href"), Input("pais", "value"))
def prov(c):
    from labor.ui import provenance

    c = c or C.DEFAULT
    return provenance(c), f"/reporte.pdf?pais={c}"


if __name__ == "__main__":
    app.run(debug=os.environ.get("LI_DEBUG") == "1")
