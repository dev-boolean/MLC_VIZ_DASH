"""Punto de entrada del dashboard: crea la aplicación multipágina y expone ``server``.

El diseño (video de entrada, encabezado, pestañas y asistente) viene del
dashboard del primer entregable, MLC_VIZ_DASH; el contenido es el del proyecto
de machine learning.

Local:   python app.py                  (http://127.0.0.1:8050)
Render:  gunicorn app:server
"""
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

from src import dash_data as D
from src import dash_plots  # noqa: F401  (registra la plantilla de las figuras)
from src.chatbot import chatbot_bp

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Mercado Laboral Colombiano · Machine Learning",
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server
server.register_blueprint(chatbot_bp)


@server.route("/health")
def health():
    return "ok", 200


# Carga única de datos y modelos al iniciar el servidor (no en cada interacción).
D.dataset()
for _name in D.artifact_names():
    D.model(_name)

AUTHORS = [
    ("Andrés Parejo", "https://github.com/dev-boolean", "https://www.linkedin.com/in/andres-parejo-40a653168/",
     "mailto:aeparejo@uninorte.edu.co"),
    ("Santiago Hurtado", "https://github.com/SHurtado26", "https://www.linkedin.com/in/santiago-hurtado-369687291/",
     "mailto:hurtadosantiago@uninorte.edu.co"),
    ("Juan Marín", None, None, None),
]
PAGES = [("Contexto", "/"), ("EDA", "/eda"), ("Modelos", "/modelos"), ("Resultados", "/resultados"),
         ("Conclusiones", "/conclusiones")]


def author_links():
    out = [html.Span("Autores: ", style={"marginRight": "6px"})]
    for i, (name, gh, li, mail) in enumerate(AUTHORS):
        if i:
            out.append(html.Span(" · ", style={"margin": "0 6px"}))
        out.append(html.A(name, href=gh, target="_blank", className="github-author-link") if gh else html.Span(name))
        if li:
            out.append(html.A("", href=li, target="_blank", className="linkedin-author-link", title=f"LinkedIn de {name}"))
        if mail:
            out.append(html.A("", href=mail, className="email-author-link", title=mail.replace("mailto:", "")))
    return out


splash = html.Div(id="splash", children=[
    html.Video(id="intro-video", src=dash.get_asset_url("intro.mp4"), autoPlay=True, muted=True, playsInline=True,
               className="splash-video"),
    html.Div(id="splash-white", className="splash-white"),
    html.Img(id="flying-logo", src=dash.get_asset_url("logo.png"), alt="", className="flying-logo"),
    html.Button("Saltar intro ▶", id="skip-intro", className="skip-intro"),
], className="splash")

hero = html.Div(className="hero-header", children=[
    html.Div(className="hero-left", children=[
        html.Img(id="logo-target", src=dash.get_asset_url("logo.png"), alt="Logo del proyecto", className="hero-logo"),
    ]),
    html.Div(className="hero-right", children=[
        html.Div("Mercado Laboral Colombiano", className="hero-title"),
        html.Div("Pronóstico de la tasa de desempleo con machine learning y un modelo original, ALEE, evaluado en "
                 "Colombia y ocho países más.", className="hero-subtitle"),
        html.Div(author_links(), className="hero-authors"),
        html.A([html.Span(className="yt-icon"), html.Span("Mira nuestro video de YouTube")],
               href="https://www.youtube.com/watch?v=2pBUO-to_KM", target="_blank", className="hero-youtube-link"),
    ]),
])

app.layout = html.Div([
    dcc.Location(id="url"),
    splash,
    html.Div(id="main-app", className="main-app", children=[
        dbc.Container(fluid=True, className="app-container", children=[
            hero,
            html.Nav(id="main-nav", className="custom-tabs page-tabs"),
            html.Div(dash.page_container, className="mlc-page"),
            html.Footer([
                html.Span("Andrés Parejo, Santiago Hurtado y Juan Marín · Proyecto de Machine Learning, Universidad del "
                          "Norte · "),
                html.Span("Datos: DANE (GEIH), vía Banco de la República, y OCDE, FRED y Eurostat para los países adicionales."),
            ], className="mlc-footer"),
        ]),
    ]),
])


@app.callback(Output("main-nav", "children"), Input("url", "pathname"))
def update_nav(path):
    path = path or "/"
    return [dcc.Link(label, href=href, className="tab tab--selected" if path == href else "tab")
            for label, href in PAGES]


if __name__ == "__main__":
    app.run(debug=True)
