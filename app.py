"""LABOR INSIGHTS: punto de entrada de la plataforma.

Local:   python app.py                  (http://127.0.0.1:8050)
Render:  gunicorn app:server
"""
import os

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
from flask import Response, abort, jsonify, request

from labor import catalog as C
from labor import engine as E
from labor import theme  # noqa: F401  (registra la plantilla de las figuras)
from src.chatbot import chatbot_bp

CDN = "https://cdnjs.cloudflare.com/ajax/libs"
app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600"
        "&family=Source+Serif+4:ital,opsz,wght@0,8..60,500;0,8..60,600;0,8..60,700;1,8..60,500&display=swap",
    ],
    external_scripts=[
        f"{CDN}/three.js/r128/three.min.js",
        f"{CDN}/gsap/3.13.0/gsap.min.js",
        f"{CDN}/gsap/3.13.0/ScrollTrigger.min.js",
    ],
    title="LABOR INSIGHTS",
    update_title=None,
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"},
               {"name": "theme-color", "content": "#F6F7F5"}],
)
app.index_string = app.index_string.replace("<html>", '<html lang="es" data-bs-theme="light">')
server = app.server
server.register_blueprint(chatbot_bp)


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

NAV = [("01", "Overview", "/"), ("02", "Exploratory Analytics", "/analytics"), ("03", "Model Intelligence", "/modelo"),
       ("04", "Unemployment Forecast", "/forecast"), ("05", "Global Intelligence", "/global"), ("06", "Insights & Conclusions", "/insights")]

sidebar = html.Aside([
    dcc.Link([html.Span(className="li-mark", **{"aria-hidden": "true"}),
              html.Span([html.B("LABOR INSIGHTS"), html.Small("Global Labor Market Intelligence")])], href="/", className="li-brand"),
    html.Nav(id="li-nav", className="li-nav", **{"aria-label": "Secciones"}),
    html.Div([
        html.P("Investigación académica", className="li-side-k"),
        html.P("Pronóstico a un mes de la tasa de desempleo con ALEE, un modelo original de combinación de expertos."),
    ], className="li-side-about"),
    html.Div([html.P("Andrés Parejo, Santiago Hurtado y Juan Marín"), html.P("Universidad del Norte · 2026"),
              html.A("Video del proyecto", href="https://www.youtube.com/watch?v=2pBUO-to_KM", target="_blank", rel="noopener")], className="li-credits"),
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
                           "Eurostat y OCDE. El globo de la portada es ilustrativo."], className="li-foot")], className="li-shell"),
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
