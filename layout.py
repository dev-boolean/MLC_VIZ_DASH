from dash import html, dcc
import dash_bootstrap_components as dbc


def build_layout(min_date, max_date):
    return dbc.Container(fluid=True, className="app-shell", children=[
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div("MERCADO LABORAL", className="sidebar-title"),
                    html.Div("COLOMBIA · DASHBOARD ACADÉMICO", className="sidebar-subtitle"),
                    html.Div("2001–2025", className="sidebar-badge"),
                    html.Hr(),

                    html.Div("NAVEGACIÓN", className="side-mini-title"),

                    dcc.Tabs(
                        id="main-tabs",
                        value="componentes",
                        vertical=True,
                        className="vertical-tabs",
                        children=[
                            dcc.Tab(label="Componentes", value="componentes"),
                            dcc.Tab(label="EDA", value="eda"),
                            dcc.Tab(label="Conclusiones", value="conclusiones"),
                            dcc.Tab(label="Referencias", value="referencias"),
                        ]
                    ),

                    html.Hr(),

                    html.Div([
                        html.Div("Fuente principal: DANE / Banco de la República"),
                        html.Div("Frecuencia: mensual"),
                        html.Div("Cobertura: mercado laboral colombiano"),
                        html.Br(),
                        html.Div("Autores: S. Hurtado & A. Parejo")
                    ], className="sidebar-info")
                ], className="sidebar")
            ], md=2),

            dbc.Col([
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div("Dashboard del mercado laboral colombiano", className="page-title"),
                            html.Div(
                                "Análisis descriptivo, bivariado y temporal de los principales indicadores laborales",
                                className="page-subtitle"
                            ),
                        ], className="page-head-left"),

                        html.Div([
                            html.Div("Filtro de período", className="filter-title"),
                            dcc.DatePickerRange(
                                id="rango-fecha",
                                start_date=min_date,
                                end_date=max_date,
                                min_date_allowed=min_date,
                                max_date_allowed=max_date,
                                display_format="YYYY-MM-DD",
                                className="date-range"
                            ),
                            html.Div(
                                "Este filtro afecta todas las visualizaciones del módulo EDA",
                                className="filter-note"
                            )
                        ], className="global-filter-box")
                    ], className="top-banner"),

                    html.Div(id="main-content", className="main-content")
                ])
            ], md=10)
        ])
    ])