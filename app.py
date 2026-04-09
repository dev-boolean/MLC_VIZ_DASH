import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

from utils import cargar_datos
from callbacks import register_callbacks


df = cargar_datos()

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Mercado Laboral Colombiano"
)

server = app.server

app.layout = dbc.Container(fluid=True, className="app-container", children=[

html.Div(className="hero-header", children=[
    html.Div(className="hero-left", children=[
        html.Img(src="/assets/logo.png", className="hero-logo")
    ]),

    html.Div(className="hero-right", children=[
        html.Div("Mercado Laboral Colombiano", className="hero-title"),
        html.Div(
            "Análisis estadístico, exploratorio y temporal del desempleo, la ocupación y la participación laboral en Colombia.",
            className="hero-subtitle"
        ),
        html.Div(
            "Autores: Andrés Parejo, Santiago Hurtado",
            className="hero-authors"
        )
    ])
]),

    html.Div(className="filter-bar", children=[
        html.Div([
            html.Div("Filtro temporal", className="filter-title"),
            html.Div(
                "Selecciona el rango de fechas para explorar el comportamiento del mercado laboral.",
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
            dcc.Tab(label="Componentes", value="componentes"),
            dcc.Tab(label="EDA", value="eda"),
            dcc.Tab(label="Conclusiones", value="conclusiones"),
            dcc.Tab(label="Referencias", value="referencias"),
        ]
    ),

    dcc.Loading(
        id="loading-main",
        type="circle",
        fullscreen=False,
        children=html.Div(id="main-content")
    )
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