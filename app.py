import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

from utils import cargar_datos
from callbacks import register_callbacks

# =========================
# CARGA DE DATOS
# =========================
df = cargar_datos()

# =========================
# APP
# =========================
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Mercado Laboral Colombiano"
)

@app.server.route('/health')
def health():
    return "ok", 200

server = app.server

# =========================
# LAYOUT
# =========================
app.layout = html.Div([

    # ── SPLASH / INTRO ────────────────────────────────────────────
    html.Div(id="splash", children=[

        html.Video(
            id="intro-video",
            src="/assets/intro.mp4",
            autoPlay=True,
            muted=True,
            playsInline=True,
            style={
                "position": "absolute",
                "inset": "0",
                "width": "100%",
                "height": "100%",
                "objectFit": "cover",
                "transition": "opacity 0.6s ease",
            }
        ),

        # Overlay blanco para la transición
        html.Div(id="splash-white", style={
            "position": "absolute",
            "inset": "0",
            "background": "#fff",
            "opacity": "0",
            "transition": "opacity 0.5s ease",
            "pointerEvents": "none",
        }),

        # Logo volador — parte del centro y vuela al header
        html.Img(
            id="flying-logo",
            src="/assets/logo.png",
            style={
                "position": "fixed",
                "left": "50%",
                "top": "50%",
                "transform": "translate(-50%, -50%)",
                "width": "220px",
                "maxWidth": "60vw",
                "height": "auto",
                "opacity": "0",
                "zIndex": "10000",
                "filter": "drop-shadow(0 10px 30px rgba(0,0,0,0.25))",
                "pointerEvents": "none",
                "transition": "all 1.2s cubic-bezier(0.4, 0, 0.2, 1)",
                "willChange": "left, top, width, transform",
            }
        ),

        # Botón saltar
        html.Button("Saltar intro ▶", id="skip-intro", style={
            "position": "absolute",
            "bottom": "24px",
            "right": "24px",
            "background": "rgba(0,0,0,0.60)",
            "backdropFilter": "blur(8px)",
            "color": "#fff",
            "border": "1px solid rgba(255,255,255,0.25)",
            "padding": "10px 20px",
            "borderRadius": "999px",
            "fontSize": "14px",
            "fontWeight": "500",
            "cursor": "pointer",
            "zIndex": "10001",
            "transition": "all 0.2s",
        }),

    ], style={
        "position": "fixed",
        "inset": "0",
        "zIndex": "9999",
        "background": "#000",
        "overflow": "hidden",
        "transition": "opacity 0.5s ease",
    }),

    # ── APP PRINCIPAL ─────────────────────────────────────────────
    html.Div(id="main-app", style={"opacity": "0"}, children=[

        dbc.Container(fluid=True, className="app-container", children=[

            html.Div(className="hero-header", children=[
                html.Div(className="hero-left", children=[
                    html.Img(
                        id="logo-target",
                        src="/assets/logo.png",
                        className="hero-logo"
                    )
                ]),
                html.Div(className="hero-right", children=[
                    html.Div("Mercado Laboral Colombiano", className="hero-title"),
                    html.Div(
                        "Análisis estadístico, exploratorio y temporal del desempleo, la ocupación y la participación laboral en Colombia.",
                        className="hero-subtitle"
                    ),
                    html.Div([
                        html.Span("Autores: ", style={"marginRight": "6px"}),
                        html.A("Andrés Parejo", href="https://github.com/dev-boolean", target="_blank",
                               className="github-author-link"),
                        html.A("", href="https://www.linkedin.com/in/andres-parejo-40a653168/", target="_blank",
                               className="linkedin-author-link", title="LinkedIn de Andrés Parejo"),
                        html.Span(" & ", style={"margin": "0 6px"}),
                        html.A("Santiago Hurtado", href="https://github.com/SHurtado26", target="_blank",
                               className="github-author-link"),
                        html.A("", href="https://www.linkedin.com/in/santiago-hurtado-369687291/", target="_blank",
                               className="linkedin-author-link", title="LinkedIn de Santiago Hurtado"),
                    ], className="hero-authors"),
                    html.A([
                        html.Span(className="yt-icon"),
                        html.Span("Mira nuestro video de Youtube!")
                    ], href="https://www.youtube.com/watch?v=2pBUO-to_KM",
                       target="_blank", className="hero-youtube-link",
                       title="Mira nuestro video de Youtube!")
                ])
            ]),

            html.Div(className="filter-bar", id="filter-bar-global", children=[
                html.Div([
                    html.Div("Filtro temporal", className="filter-title"),
                    html.Div(
                        "Selecciona el rango de fechas para explorar el comportamiento del mercado laboral. Este filtro aplica a todas las visualizaciones del módulo EDA.",
                        className="filter-sub"
                    )
                ]),
                dcc.DatePickerRange(
                    id="rango-fecha",
                    start_date=df["fecha"].min(),
                    end_date=df["fecha"].max(),
                    display_format="YYYY-MM-DD",
                    minimum_nights=0,
                    clearable=False,
                    style={"borderRadius": "16px"}
                )
            ]),

            dcc.Tabs(
                id="main-tabs",
                value="componentes",
                className="custom-tabs",
                children=[
                    dcc.Tab(label="Componentes",     value="componentes"),
                    dcc.Tab(label="EDA",             value="eda"),
                    dcc.Tab(label="Modelado SARIMA", value="modelo"),
                    dcc.Tab(label="Conclusiones",    value="conclusiones"),
                    dcc.Tab(label="Referencias",     value="referencias"),
                ]
            ),

            dcc.Loading(
                id="loading-main",
                type="circle",
                fullscreen=False,
                children=html.Div(id="main-content")
            )
        ])
    ])
])

# =========================
# CALLBACKS
# =========================
register_callbacks(app, df)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)
