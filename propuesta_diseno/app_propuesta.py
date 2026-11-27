"""Propuesta de diseño de LABOR INSIGHTS, aparte del dashboard actual.

    .venv\\Scripts\\python propuesta_diseno\\app_propuesta.py      (http://127.0.0.1:8051)

Regla: no se pierde nada. Las seis secciones son los mismos módulos de ``pages/``
del dashboard original, importados sin copiarlos ni modificarlos, con las mismas
funciones (selector de país, informe PDF, globo, introducción). Lo que cambia:

* la hoja de estilos ``estilo.css``, que se carga después de ``assets/labor.css``
  y solo cambia el aspecto (tipografía, reglas en vez de tarjetas, etiquetas);
* la paleta y la tipografía de los gráficos (``tema.py``);
* los nombres de la navegación, en español;
* tres paneles nuevos que se suman a lo que ya había (``extras.py``).

Ningún archivo del dashboard original se modifica. Para descartar la propuesta
basta con borrar esta carpeta.
"""
import json
import os
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path[:0] = [str(RAIZ), str(AQUI)]

import tema  # noqa: E402,F401  (debe ir antes de importar las páginas: cambia la paleta de las figuras)
import dash  # noqa: E402
from dash import Input, Output, dcc, html  # noqa: E402
from flask import Response, abort, jsonify, request, send_file, send_from_directory  # noqa: E402

from labor import catalog as C  # noqa: E402
from labor import engine as E  # noqa: E402
from labor import ui as U  # noqa: E402

NOMBRES = {"Overview": "Panorama", "Exploratory Analytics": "Análisis exploratorio", "Model Intelligence": "Modelo",
           "Unemployment Forecast": "Pronóstico", "Global Intelligence": "Comparación internacional",
           "Insights & Conclusions": "Conclusiones"}

# Encabezados de sección con el nombre en español; el resto del encabezado no cambia.
_head = U.head
U.head = lambda num, kicker, title, accent, lead: _head(num, NOMBRES.get(kicker, kicker), title, accent, lead)

VENDOR = RAIZ / "vendor"
CHATBOT = os.environ.get("LI_CHATBOT") == "1"
app = dash.Dash(
    __name__,
    use_pages=True,
    pages_folder=str(RAIZ / "pages"),
    assets_folder=str(RAIZ / "assets"),
    # El globo se sirve desde una copia con otros colores (propuesta_diseno/li_globe.js), para que no se pierda en el fondo blanco.
    assets_ignore=r"li_globe\.js" if CHATBOT else r"chatbot\.js|li_globe\.js",
    external_stylesheets=[
        "/vendor/bootstrap.min.css",
        "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400"
        "&family=IBM+Plex+Serif:ital,wght@0,400;0,500;0,600;1,400&display=swap",
    ],
    external_scripts=["/vendor/three.min.js", "/vendor/gsap.min.js", "/vendor/ScrollTrigger.min.js", "/propuesta/li_globe.js"],
    title="LABOR INSIGHTS",
    update_title=None,
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}, {"name": "theme-color", "content": "#FFFFFF"}],
)
server = app.server
from src.chatbot import chatbot_bp  # noqa: E402

server.register_blueprint(chatbot_bp)

import extras  # noqa: E402  (después de crear la app: agrega los paneles nuevos a las páginas originales)

for p in dash.page_registry.values():
    nombre = NOMBRES.get(p["name"], p["name"])
    p["name"], p["title"] = nombre, f"{nombre} · LABOR INSIGHTS"
extras.agregar(dash.page_registry)


@server.route("/vendor/<path:name>")
def vendor(name):
    return send_from_directory(VENDOR, name, max_age=86400)


@server.route("/propuesta/estilo.css")
def estilo():
    return send_file(AQUI / "estilo.css", mimetype="text/css", max_age=0)


@server.route("/propuesta/li_globe.js")
def globo_js():
    return send_file(AQUI / "li_globe.js", mimetype="text/javascript", max_age=0)


@server.route("/health")
def health():
    return "ok", 200


@server.route("/api/globo")
def globe_data():
    out = []
    for c, m in C.COUNTRIES.items():
        f = E.forecast(c)
        out.append({"id": c, "name": m["name"], "iso": m["iso"], "lat": m["lat"], "lon": m["lon"],
                    "obs": C.num(f["observed"]), "pred": C.num(f["pred"]), "last": C.month_short(f["last_date"]),
                    "target": C.month_short(f["target_date"])})
    return jsonify(out)


@server.route("/reporte.pdf")
def report_pdf():
    from labor.report import build_report

    c = request.args.get("pais", C.DEFAULT)
    if c not in C.COUNTRIES:
        abort(404)
    name = f"LABOR_INSIGHTS_{C.COUNTRIES[c]['short']}.pdf"
    return Response(build_report(c), mimetype="application/pdf", headers={"Content-Disposition": f'attachment; filename="{name}"'})


for _c in C.COUNTRIES:
    E.forecast(_c)


def _intro_data():
    f = E.frame(C.DEFAULT)
    return json.dumps({"fechas": f["fecha"].dt.strftime("%Y-%m").tolist(), "valores": [float(v) for v in f["valor"]],
                       "interpolados": int(f["interpolado"].sum()), "fuente": C.COUNTRIES[C.DEFAULT]["source"]},
                      ensure_ascii=False).replace("</", "<\\/")


app.index_string = (app.index_string
                    .replace("<html>", '<html lang="es" data-bs-theme="light">')
                    .replace("{%css%}", '{%css%}\n<link rel="stylesheet" href="/propuesta/estilo.css">')
                    .replace("{%app_entry%}", f'<script id="li-intro-data" type="application/json">{_intro_data()}</script>\n{{%app_entry%}}'))

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
    c = c or C.DEFAULT
    return U.provenance(c), f"/reporte.pdf?pais={c}"


if __name__ == "__main__":
    app.run(debug=False, port=8051)
