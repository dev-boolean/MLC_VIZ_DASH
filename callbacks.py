from dash import html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

from utils import filtrar_por_fecha, obtener_kpis, generar_alerta, transformar_a_largo
from stats import (
    tabla_descriptiva, tabla_caracteristicas, matriz_corr,
    prueba_adf, descomposicion_desempleo, acf_pacf_data
)
from charts import (
    fig_histograma, fig_boxplot, fig_dispersion, fig_boxplots_caracteristicas,
    fig_corr, fig_scatter_pairs, fig_lineas, fig_comparacion_anual,
    fig_descomposicion, fig_acf, fig_pacf,
    # SARIMA
    fig_comparacion_modelos, fig_ajuste, fig_residuos_diagnostico,
    fig_acf_residuos, fig_pronostico,
)
from model import (
    fit_modelo_final, comparar_modelos, metricas_modelo,
    ljung_box_tests, datos_residuos, pronostico,
    serie_ajustada, resumen_modelo_texto, tabla_pronostico_df,
)


# =========================================================
# COMPONENTES AUXILIARES (sin cambios del Dash original)
# =========================================================
def table_component(df, page_size=10):
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        page_size=page_size,
        style_table={"overflowX": "auto", "borderRadius": "18px", "overflow": "hidden"},
        style_cell={
            "textAlign": "center", "padding": "12px",
            "fontFamily": "Inter, Segoe UI", "fontSize": "13px",
            "border": "1px solid #eef2f7", "backgroundColor": "rgba(255,255,255,0.92)",
            "color": "#334155", "whiteSpace": "normal", "height": "auto"
        },
        style_header={
            "fontWeight": "bold", "backgroundColor": "#f8fafc",
            "border": "1px solid #e5e7eb", "color": "#0f172a", "fontSize": "13px"
        },
        style_data={"border": "1px solid #f1f5f9"},
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "rgba(248,250,252,0.65)"}]
    )


def accordion_block(title, badge, items):
    return html.Details([
        html.Summary([
            html.Span(title, className="acc-title"),
            html.Span(badge, className="acc-badge")
        ]),
        html.Ul([html.Li(x) for x in items], className="acc-list")
    ], className="acc-item")


def hallazgo_card(icono, titulo, texto):
    return html.Div([
        html.Div(icono, className="hallazgo-icon"),
        html.Div([
            html.Div(titulo, className="hallazgo-title"),
            html.Div(texto,  className="hallazgo-text")
        ])
    ], className="hallazgo-card")


def interpretacion(titulo, texto, implicacion=None):
    return html.Div([
        html.Div(titulo, className="interp-title"),
        html.Div(texto,  className="interp-text"),
        html.Div([
            html.Span("Interpretación económica: ", className="interp-label"),
            html.Span(implicacion)
        ], className="interp-implicacion") if implicacion else None
    ], className="interpretacion-box")


def section_title(titulo, subtitulo=None):
    return html.Div([
        html.H3(titulo, className="section-title"),
        html.Div(subtitulo, className="section-subtitle") if subtitulo else None
    ], className="section-head")


def graph_card(fig, height=520):
    fig.update_layout(height=height)
    return html.Div(
        dcc.Graph(
            figure=fig,
            config={"displayModeBar": False, "responsive": True, "scrollZoom": False},
            style={"height": f"{height}px"}
        ),
        className="graph-card"
    )


def graph_pair(fig_left, fig_right, left_height=430, right_height=430):
    fig_left.update_layout(height=left_height)
    fig_right.update_layout(height=right_height)
    return dbc.Row([
        dbc.Col(html.Div(dcc.Graph(figure=fig_left,  config={"displayModeBar": False, "responsive": True}), className="graph-card"), md=6),
        dbc.Col(html.Div(dcc.Graph(figure=fig_right, config={"displayModeBar": False, "responsive": True}), className="graph-card"), md=6),
    ], className="g-4")


def text_block(*children):
    return html.Div(list(children), className="text-block")


# =========================================================
# HELPER: KPI card SARIMA (estilo purple del Shiny)
# =========================================================
def kpi_sarima(label, valor, sufijo="", color="purple", sub=""):
    color_map = {
        "purple": ("#7c3aed", "kpi-card kpi-card-purple"),
        "blue":   ("#0066cc", "kpi-card"),
        "orange": ("#f08c00", "kpi-card kpi-card-orange"),
        "green":  ("#00a854", "kpi-card kpi-card-green"),
    }
    c, cls = color_map.get(color, ("#7c3aed", "kpi-card kpi-card-purple"))
    return html.Div(className=cls, children=[
        html.Div(label, className="kpi-label"),
        html.Div(f"{valor}{sufijo}", className="kpi-value", style={"color": c, "fontWeight": "800", "fontSize": "2em"}),
        html.Div(sub, className="kpi-sub")
    ])


# =========================================================
# TABLA LJUNG-BOX HTML (fiel al Shiny)
# =========================================================
def tabla_ljungbox_html(tests: list):
    filas = []
    for t in tests:
        concl_style = {"color": "#00a854", "fontWeight": "700"} if t["ok"] else {"color": "#e03131", "fontWeight": "700"}
        filas.append(html.Tr([
            html.Td(t["lag"],            style={"padding": "10px 16px", "fontSize": "13px"}),
            html.Td(t["estadistico"],    style={"padding": "10px 16px", "fontSize": "13px"}),
            html.Td(t["p_valor"],        style={"padding": "10px 16px", "fontSize": "13px"}),
            html.Td(html.Span(t["conclusion"], style=concl_style), style={"padding": "10px 16px"}),
        ]))

    return html.Table(
        style={"width": "100%", "borderCollapse": "collapse", "background": "#fff",
               "border": "1px solid #e0e4ea", "borderRadius": "8px", "overflow": "hidden"},
        children=[
            html.Thead(style={"background": "#f8fafc"}, children=[
                html.Tr([
                    html.Th(h, style={"padding": "10px 16px", "fontSize": "12px", "fontWeight": "700",
                                      "textTransform": "uppercase", "letterSpacing": ".5px",
                                      "color": "#1a1a2e", "borderBottom": "2px solid #7c3aed", "textAlign": "left"})
                    for h in ["Rezagos", "Estadístico", "p-valor", "Conclusión"]
                ])
            ]),
            html.Tbody(filas)
        ]
    )


# =========================================================
# MÓDULO SARIMA COMPLETO (tab "modelo")
# =========================================================
def render_modelo(df):
    """Construye todo el contenido del tab Modelado SARIMA."""

    # Ajustar modelo (sobre serie completa, igual que Shiny)
    resultado = fit_modelo_final(df)
    metricas  = metricas_modelo(resultado)
    lb_tests  = ljung_box_tests(resultado)
    resid_data = datos_residuos(resultado)
    df_comp   = comparar_modelos(df)
    fechas_h, obs_h, ajust_h = serie_ajustada(resultado, df)

    return html.Div([
        # ── Header del módulo (igual al Shiny) ──────────────────────────────
        html.Div(className="modelo-header", children=[
            html.H3(children=[
                "✨ Modelado Predictivo ",
                html.Span("SARIMA", className="modelo-badge")
            ]),
            html.P("Modelo SARIMA(2,1,2)(1,1,1)[12] sobre la tasa de desempleo nacional colombiana — 2001 a 2025")
        ]),

        # ── Sub-tabs (igual estructura al Shiny) ────────────────────────────
        dcc.Tabs(
            value="justificacion",
            className="custom-tabs",
            children=[

                # Tab 1: Justificación
                dcc.Tab(label="Justificación", value="justificacion", children=[
                    html.Div(className="content-box", children=[
                        section_title("¿Por qué SARIMA?"),
                        html.Hr(className="ndq-divider"),
                        html.P("El análisis exploratorio desarrollado en el EDA dejó evidencia clara sobre las características estructurales de la serie tasa_desempleo_nacional: no estacionariedad en niveles (p-valor ADF > 0.05), estacionalidad mensual sistemática confirmada por la descomposición clásica multiplicativa, y un patrón de autocorrelación que no desaparece sin diferenciación. Estas condiciones descartan el uso de un ARIMA simple."),
                        html.Div(className="interpretacion-box", style={"borderLeft": "3px solid #7c3aed"}, children=[
                            html.B("Forma general del modelo SARIMA", style={"color": "#4c1d95"}),
                            html.Br(), html.Br(),
                            html.P("Φ_P(B¹²) · φ_p(B) · (1−B)ᵈ · (1−B¹²)ᴰ · yₜ  =  Θ_Q(B¹²) · θ_q(B) · εₜ"),
                            html.Br(),
                            html.P("donde B es el operador de rezago, d es el orden de diferenciación regular, D es el orden de diferenciación estacional y s = 12 para datos mensuales.")
                        ]),
                        html.P("Un modelo ARIMA(p,d,q) clásico solo contempla diferenciación regular pero no captura la dependencia estacional periódica (cada 12 meses). Al ajustar ARIMA(1,1,1) sobre la serie de entrenamiento se obtiene un AIC = 895.39 y RMSE = 1.53, resultados claramente inferiores a cualquier especificación estacional."),
                        html.Hr(className="ndq-divider"),
                        html.H4("Evidencia del EDA que orienta al modelo", className="subsection-title"),
                        dbc.Row([
                            dbc.Col(html.Div(className="kpi-card kpi-card-purple", children=[
                                html.Div("No Estacionariedad", className="kpi-label"),
                                html.Div("p > 0.05",           style={"color": "#7c3aed", "fontWeight": "800", "fontSize": "1.7em"}),
                                html.Div("ADF — requiere d ≥ 1", className="kpi-sub")
                            ]), md=4),
                            dbc.Col(html.Div(className="kpi-card kpi-card-purple", children=[
                                html.Div("Estacionalidad", className="kpi-label"),
                                html.Div("s = 12",         style={"color": "#7c3aed", "fontWeight": "800", "fontSize": "1.7em"}),
                                html.Div("Ciclo anual comprobado", className="kpi-sub")
                            ]), md=4),
                            dbc.Col(html.Div(className="kpi-card kpi-card-purple", children=[
                                html.Div("Ruptura estructural", className="kpi-label"),
                                html.Div("2020",               style={"color": "#7c3aed", "fontWeight": "800", "fontSize": "1.7em"}),
                                html.Div("COVID-19 — pico >21%", className="kpi-sub")
                            ]), md=4),
                        ], className="g-4")
                    ])
                ]),

                # Tab 2: Comparación de modelos
                dcc.Tab(label="Comparación de modelos", value="comparacion", children=[
                    html.Div(className="content-box", children=[
                        section_title("Selección del mejor modelo"),
                        html.Hr(className="ndq-divider"),
                        html.P("Se evaluaron cinco modelos candidatos usando los primeros 270 meses como entrenamiento (2001–2023) y los 30 restantes como validación. La comparación se realizó por AIC, BIC y RMSE fuera de muestra."),
                        html.Div(className="interpretacion-box", children=[
                            html.P("El modelo seleccionado es el SARIMA(2,1,2)(1,1,1)[12] por obtener el menor RMSE en validación. Aunque su AIC es ligeramente superior al de las especificaciones más parsimoniosas, la ganancia en precisión predictiva fuera de muestra justifica la especificación más rica dado el horizonte de pronóstico y los choques estructurales presentes en la serie.")
                        ]),
                        html.Br(),
                        table_component(df_comp),
                        html.Br(),
                        graph_card(fig_comparacion_modelos(df_comp), height=400)
                    ])
                ]),

                # Tab 3: Modelo ajustado
                dcc.Tab(label="Modelo ajustado", value="ajustado", children=[
                    html.Div(className="content-box", children=[
                        section_title("SARIMA(2,1,2)(1,1,1)[12] — Ajuste completo"),
                        html.Hr(className="ndq-divider"),
                        html.P("El modelo se ajustó sobre la serie completa de observaciones (enero 2001 – diciembre 2025). Los coeficientes del componente regular AR(2) y MA(2) capturan la dinámica de corto plazo del desempleo mes a mes, mientras que los términos estacionales SAR(1) y SMA(1) modelan la dependencia periódica de 12 meses."),
                        html.Br(),
                        dbc.Row([
                            dbc.Col(kpi_sarima("RMSE", metricas["RMSE"], color="purple", sub="Error cuadrático medio"), md=3),
                            dbc.Col(kpi_sarima("MAE",  metricas["MAE"],  color="blue",   sub="Error absoluto medio"),    md=3),
                            dbc.Col(kpi_sarima("MAPE", metricas["MAPE"], sufijo="%", color="orange", sub="Error porcentual medio"), md=3),
                            dbc.Col(kpi_sarima("AIC",  metricas["AIC"],  color="green",  sub="Criterio Akaike"),          md=3),
                        ], className="g-4"),
                        html.Br(),
                        html.H4("Serie observada vs. valores ajustados", className="subsection-title"),
                        graph_card(fig_ajuste(fechas_h, obs_h, ajust_h), height=440),
                        html.Div(className="interpretacion-box", children=[
                            html.P("La línea punteada naranja sigue de cerca a la observada durante la mayor parte del período 2001–2025, lo que confirma un buen ajuste en muestra. El único tramo de desajuste visible corresponde al pico de 2020, donde el desempleo superó el 20% como consecuencia del cierre económico por la pandemia: un choque exógeno que ningún modelo univariado de series de tiempo puede modelar sin información externa adicional.")
                        ]),
                        html.Br(),
                        html.H4("Resumen del modelo", className="subsection-title"),
                        html.Pre(resumen_modelo_texto(resultado),
                                 style={"background": "#f8fafc", "color": "#1a1a2e",
                                        "border": "1px solid #e0e4ea", "borderRadius": "6px",
                                        "fontSize": "12px", "padding": "16px",
                                        "overflowX": "auto", "whiteSpace": "pre-wrap"})
                    ])
                ]),

                # Tab 4: Diagnóstico de residuos
                dcc.Tab(label="Diagnóstico de residuos", value="diagnostico", children=[
                    html.Div(className="content-box", children=[
                        section_title("Validación estadística del modelo"),
                        html.Hr(className="ndq-divider"),
                        html.P("Un buen modelo debe dejar residuos que se comporten como ruido blanco: sin autocorrelación, con media cero y distribución aproximadamente normal."),
                        html.Br(),
                        html.H4("Prueba de Ljung-Box", className="subsection-title"),
                        tabla_ljungbox_html(lb_tests),
                        html.Div(className="interpretacion-box", children=[
                            html.P("Los p-valores de la prueba de Ljung-Box superan ampliamente el nivel de significancia del 5%, por lo que no se rechaza la hipótesis nula de ausencia de autocorrelación en los residuos. Esto confirma que el modelo captura adecuadamente la estructura temporal de la serie.")
                        ]),
                        html.Br(),
                        html.H4("ACF de los residuos", className="subsection-title"),
                        graph_card(fig_acf_residuos(resid_data["residuos"]), height=360),
                        html.Br(),
                        html.H4("Normalidad de los residuos", className="subsection-title"),
                        graph_card(fig_residuos_diagnostico(resid_data), height=380),
                        html.Div(className="interpretacion-box", children=[
                            html.P("El QQ-plot muestra que los residuos se ajustan razonablemente a la distribución normal salvo en los extremos, donde los valores de 2020 generan colas algo más pesadas. Esto es coherente con la ruptura estructural de la pandemia, un evento de baja probabilidad que ningún modelo univariado puede anticipar pero que, una vez ocurrido, no contamina la estructura de autocorrelación.")
                        ])
                    ])
                ]),

                # Tab 5: Pronóstico — con controles interactivos
                dcc.Tab(label="Pronóstico 2026", value="pronostico", children=[
                    html.Div(className="content-box", children=[
                        section_title("Proyección a 12 meses — 2026"),
                        html.Hr(className="ndq-divider"),

                        # Controles (replica modelo-control-box del Shiny)
                        html.Div(className="modelo-control-box", children=[
                            html.Div([
                                html.Label("Horizonte de pronóstico", className="control-label"),
                                dcc.Slider(
                                    id="horizonte-fc",
                                    min=6, max=24, value=12, step=6,
                                    marks={
    6:  {"label": "6m",  "style": {"color": "#aab4c8"}},
    12: {"label": "12m", "style": {"color": "#aab4c8"}},
    18: {"label": "18m", "style": {"color": "#aab4c8"}},
    24: {"label": "24m", "style": {"color": "#aab4c8"}},
},
                                    tooltip={"placement": "bottom", "always_visible": False},
                                )
                            ], style={"minWidth": "260px"}),
                            html.Div([
                                html.Label("Banda de confianza", className="control-label"),
                                dcc.Dropdown(
                                    id="nivel-ic",
                                    options=[
                                        {"label": "80% y 95%", "value": "both"},
                                        {"label": "Solo 95%",  "value": "95"},
                                        {"label": "Solo 80%",  "value": "80"},
                                    ],
                                    value="both",
                                    clearable=False,
                                    style={"minWidth": "160px", "color": "#1a1a2e"}
                                )
                            ], style={"minWidth": "200px"}),
                        ]),

                        html.Div(id="pronostico-graph-container"),
                        html.Div(className="interpretacion-box", children=[
                            html.P("El pronóstico muestra una tasa de desempleo que oscila entre 7.5% y 10.9% durante 2026, con un patrón estacional típico: valores más altos al inicio del año (enero) y una moderación hacia el segundo semestre, en línea con el comportamiento histórico de la serie. Los intervalos de confianza se ensanchan progresivamente conforme avanza el horizonte, reflejando la incertidumbre creciente inherente a todo pronóstico de largo plazo.")
                        ]),
                        html.Br(),
                        html.H4("Tabla de pronósticos — valores puntuales e intervalos", className="subsection-title"),
                        html.Div(id="pronostico-tabla-container"),
                    ])
                ]),

                # Tab 6: Conclusiones del modelo
                dcc.Tab(label="Conclusiones del modelo", value="conclusiones-modelo", children=[
                    html.Div(className="content-box", children=[
                        section_title("Síntesis del modelado SARIMA"),
                        html.Hr(className="ndq-divider"),
                        hallazgo_card("🎯", "Modelo seleccionado",
                                      "El SARIMA(2,1,2)(1,1,1)[12] logra el mejor equilibrio entre ajuste en muestra y capacidad predictiva fuera de muestra, con un RMSE de validación de 0.835 frente a 1.53 del ARIMA(1,1,1) sin componente estacional."),
                        hallazgo_card("📊", "Residuos — ruido blanco",
                                      "Las pruebas de Ljung-Box arrojan p-valores superiores al 5% para rezagos 12 y 24, confirmando ausencia de autocorrelación en los residuos. El modelo captura adecuadamente la estructura temporal de la serie."),
                        hallazgo_card("🗓️", "Pronóstico 2026",
                                      "Los pronósticos para 2026 proyectan una tasa de desempleo en el rango 7.5%–10.9%, con un patrón estacional consistente con el comportamiento histórico. Valores más altos en enero y moderación hacia el segundo semestre."),
                        hallazgo_card("⛔", "Limitación — COVID-19",
                                      "El choque de 2020 generó colas más pesadas en los residuos. Ningún modelo univariado puede anticipar este tipo de ruptura estructural exógena sin información adicional. La estimación debe interpretarse como proyección condicional a la dinámica interna de la serie."),
                        hallazgo_card("📐", "Precisión del modelo",
                                      f"Un RMSE de ~{metricas['RMSE']} sobre una serie cuyo rango histórico es 7%–22% representa un error relativo reducido. El MAPE indica que, en promedio, el modelo se desvía menos del 8% del valor real mensual, lo cual es aceptable para una serie macroeconómica con componentes estructurales tan marcados."),
                    ])
                ]),
            ]
        )
    ])


# =========================================================
# CALLBACKS
# =========================================================
def register_callbacks(app, df):

    # ── Ajustar modelo una sola vez al iniciar (igual que reactive() en Shiny) ─
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _resultado_cache  = fit_modelo_final(df)
        _fechas_h, _obs_h, _ = serie_ajustada(_resultado_cache, df)

    # ── Callback principal: renderiza el tab seleccionado ────────────────────
    @app.callback(
        Output("main-content", "children"),
        Input("main-tabs", "value"),
        Input("rango-fecha", "start_date"),
        Input("rango-fecha", "end_date")
    )
    def render_content(tab, start_date, end_date):
        dff     = filtrar_por_fecha(df, start_date, end_date)
        df_long = transformar_a_largo(dff)

        if tab == "componentes":
            return html.Div([
                dcc.Tabs(
                    value="intro",
                    className="custom-tabs",
                    children=[

                        dcc.Tab(label="Introducción", children=[
                            html.Div(className="content-box", children=[
                                section_title("Introducción", "Contexto y propósito del estudio"),
                                text_block(
                                    html.P("El mercado laboral constituye uno de los principales indicadores del desempeño económico de un país, ya que refleja la capacidad de la economía de generar empleo, absorber fuerza de trabajo y sostener ingresos para los hogares. Su estudio permite comprender la interacción entre la oferta laboral y la demanda de trabajo, así como evaluar la evolución general de la actividad económica."),
                                    html.P("En Colombia, el seguimiento de estos fenómenos se realiza principalmente a partir de la Gran Encuesta Integrada de Hogares (GEIH), producida por el DANE y sistematizada en diversas series históricas por el Banco de la República. A partir de estos registros es posible observar no solo el nivel del desempleo, sino también sus relaciones con la participación laboral y la ocupación."),
                                    html.P("Este proyecto analiza la evolución del mercado laboral colombiano entre enero de 2001 y diciembre de 2025, con énfasis en la tasa de desempleo nacional y su comportamiento frente a otros indicadores estructurales del mercado de trabajo.")
                                ),
                                html.Hr(className="ndq-divider"),
                                html.H4("Variables analizadas", className="subsection-title"),

                                accordion_block("Tasa Global de Participación — 13 áreas", "TGP ÁREA", [
                                    "Relación porcentual entre la fuerza laboral y la población en edad de trabajar en las 13 principales áreas metropolitanas.",
                                    "Tipo: cuantitativa continua.", "Escala: razón.", "Unidad: porcentaje.", "Frecuencia: mensual.",
                                    "Interpretación: refleja la presión de la población sobre el mercado laboral urbano."
                                ]),
                                accordion_block("Tasa Global de Participación — Nacional", "TGP NAL", [
                                    "Relación entre la fuerza laboral y la población en edad de trabajar a nivel nacional.",
                                    "Tipo: cuantitativa continua.", "Escala: razón.", "Unidad: porcentaje.", "Frecuencia: mensual.",
                                    "Interpretación: permite evaluar la disposición agregada de la población a participar en el mercado laboral."
                                ]),
                                accordion_block("Tasa de Desempleo — 13 áreas", "TD ÁREA", [
                                    "Proporción de personas desocupadas sobre la fuerza laboral en las principales áreas urbanas.",
                                    "Tipo: cuantitativa continua.", "Escala: razón.", "Unidad: porcentaje.", "Frecuencia: mensual.",
                                    "Interpretación: aproxima la presión laboral urbana y la dificultad de inserción en el empleo."
                                ]),
                                accordion_block("Tasa de Desempleo — Nacional", "TD NAL", [
                                    "Proporción de personas desocupadas sobre la fuerza laboral a nivel nacional.",
                                    "Tipo: cuantitativa continua.", "Escala: razón.", "Unidad: porcentaje.", "Frecuencia: mensual.",
                                    "Interpretación: variable objetivo central del análisis."
                                ]),
                                accordion_block("Tasa de Ocupación — 13 áreas", "TO ÁREA", [
                                    "Relación entre la población ocupada y la población en edad de trabajar en las 13 principales áreas.",
                                    "Tipo: cuantitativa continua.", "Escala: razón.", "Unidad: porcentaje.", "Frecuencia: mensual.",
                                    "Interpretación: mide la absorción efectiva de empleo en el entorno urbano."
                                ]),
                                accordion_block("Tasa de Ocupación — Nacional", "TO NAL", [
                                    "Relación entre población ocupada y población en edad de trabajar a nivel nacional.",
                                    "Tipo: cuantitativa continua.", "Escala: razón.", "Unidad: porcentaje.", "Frecuencia: mensual.",
                                    "Interpretación: mide la capacidad agregada del sistema económico para generar ocupación."
                                ]),

                                html.Hr(className="ndq-divider"),
                                html.Div(className="highlight-box", children=[
                                    html.B("Pregunta de investigación"),
                                    html.Br(), html.Br(),
                                    html.P("¿Cómo ha evolucionado la tasa de desempleo en Colombia a lo largo del tiempo y qué relación presenta con otros indicadores del mercado laboral como la tasa de ocupación y la tasa global de participación?")
                                ])
                            ])
                        ]),

                        dcc.Tab(label="Justificación", children=[
                            html.Div(className="content-box", children=[
                                section_title("Justificación", "Relevancia analítica y aplicada"),
                                text_block(
                                    html.P("El mercado laboral colombiano ha estado expuesto a transformaciones asociadas a ciclos económicos, reformas institucionales, cambios demográficos y choques externos. En este contexto, el desempleo se convierte en un indicador clave no solo para describir el estado de la economía, sino también para evaluar la efectividad de políticas públicas orientadas al bienestar social."),
                                    html.P("La utilidad de este análisis radica en integrar, en una sola herramienta, evidencia descriptiva, relaciones entre variables y comportamiento temporal de largo plazo. Esto permite pasar de una lectura fragmentada de los indicadores a una visión estructurada del sistema laboral colombiano."),
                                    html.P("Adicionalmente, el desarrollo de un dashboard interactivo facilita la comunicación de hallazgos, mejora la trazabilidad de los resultados y fortalece el uso de herramientas reproducibles en análisis económico aplicado.")
                                )
                            ])
                        ]),

                        dcc.Tab(label="Objetivos", children=[
                            html.Div(className="content-box", children=[
                                section_title("Objetivos", "Propósito general y metas específicas"),
                                html.H4("Objetivo general", className="subsection-title"),
                                html.Div(className="highlight-box", children=[
                                    html.P("Analizar la evolución de la tasa de desempleo en Colombia a lo largo del tiempo y su relación con otros indicadores del mercado laboral mediante un dashboard interactivo de análisis estadístico y temporal.")
                                ]),
                                html.H4("Objetivos específicos", className="subsection-title"),
                                html.Ol([
                                    html.Li("Explorar el comportamiento histórico de los principales indicadores del mercado laboral colombiano."),
                                    html.Li("Describir la distribución, dispersión y patrones de las variables laborales mediante análisis exploratorio de datos."),
                                    html.Li("Evaluar relaciones bivariadas y correspondencias estructurales entre desempleo, ocupación y participación."),
                                    html.Li("Analizar la dinámica temporal de la tasa de desempleo mediante herramientas de series de tiempo."),
                                    html.Li("Implementar y evaluar un modelo SARIMA predictivo para estimar la evolución de la tasa de desempleo."),
                                ], className="ordered-list")
                            ])
                        ]),

                        dcc.Tab(label="Marco Teórico", children=[
                            html.Div(className="content-box", children=[
                                section_title("Marco Teórico", "Fundamentos conceptuales del análisis"),
                                text_block(
                                    html.P("El mercado laboral puede entenderse como el espacio en el que interactúan la oferta y la demanda de trabajo. En teoría, el salario actúa como variable de ajuste; sin embargo, en la práctica, el equilibrio se ve afectado por fricciones institucionales, rigideces salariales, informalidad y choques macroeconómicos."),
                                    html.P("Bajo las definiciones de la OIT, la población en edad de trabajar se divide entre quienes participan y quienes no participan del mercado laboral. Dentro de la fuerza laboral, la población se clasifica en ocupada o desocupada. Esta estructura conceptual da origen a los indicadores centrales analizados en este estudio."),
                                    html.P("Desde la perspectiva de series de tiempo, los indicadores laborales no son estáticos: incorporan tendencia, estacionalidad, persistencia y posibles rupturas estructurales. Por ello, su análisis requiere herramientas que permitan distinguir movimientos de largo plazo, fluctuaciones cíclicas y episodios extraordinarios como el choque de 2020. Para este fin se emplea el modelo SARIMA(2,1,2)(1,1,1)[12] (Hyndman & Athanasopoulos, 2021).")
                                )
                            ])
                        ]),

                        dcc.Tab(label="Hipótesis", children=[
                            html.Div(className="content-box", children=[
                                section_title("Hipótesis", "Supuestos analíticos del estudio"),
                                html.H4("Hipótesis principal", className="subsection-title"),
                                html.Div(className="highlight-box", children=[
                                    html.P("La tasa de desempleo nacional presenta una relación inversa y estructural con la tasa de ocupación, además de una relación sistemática con la participación laboral y con la dinámica del desempleo urbano.")
                                ]),
                                html.H4("Hipótesis secundarias", className="subsection-title"),
                                html.Ol([
                                    html.Li("El desempleo de las 13 áreas metropolitanas mantiene una relación positiva fuerte con el desempleo nacional."),
                                    html.Li("La pandemia de COVID-19 introdujo una ruptura coyuntural visible dentro del comportamiento histórico del mercado laboral."),
                                    html.Li("La serie de desempleo nacional presenta componentes de tendencia, estacionalidad y persistencia temporal identificables empíricamente."),
                                ], className="ordered-list")
                            ])
                        ]),

                        dcc.Tab(label="Metodología", children=[
                            html.Div(className="content-box", children=[
                                section_title("Metodología", "Enfoque cuantitativo y flujo de trabajo"),
                                text_block(
                                    html.P("El proyecto adopta un enfoque cuantitativo con énfasis en análisis exploratorio y herramientas de series de tiempo. Se emplean datos mensuales para evaluar tanto el comportamiento aislado de cada variable como su dinámica conjunta y temporal.")
                                ),
                                html.Hr(className="ndq-divider"),
                                html.H4("Fases del proyecto", className="subsection-title"),
                                dbc.Row([
                                    dbc.Col(html.Div(className="fase-card", children=[
                                        html.Div("01", className="fase-num"),
                                        html.Div("Caracterización", className="fase-titulo"),
                                        html.Div("Organización, limpieza e interpretación inicial del conjunto de datos.", className="fase-desc")
                                    ]), md=4),
                                    dbc.Col(html.Div(className="fase-card", children=[
                                        html.Div("02", className="fase-num"),
                                        html.Div("EDA", className="fase-titulo"),
                                        html.Div("Análisis descriptivo, bivariado y temporal de los indicadores laborales.", className="fase-desc")
                                    ]), md=4),
                                    dbc.Col(html.Div(className="fase-card", children=[
                                        html.Div("03", className="fase-num"),
                                        html.Div("SARIMA", className="fase-titulo"),
                                        html.Div("Modelo predictivo estacional con diagnóstico y pronóstico a horizonte variable.", className="fase-desc")
                                    ]), md=4),
                                ], className="g-4")
                            ])
                        ]),
                    ]
                )
            ])

        elif tab == "eda":
            kpis    = obtener_kpis(dff)
            alerta  = generar_alerta(dff)
            corr    = matriz_corr(dff)
            adf     = prueba_adf(dff)
            decomp  = descomposicion_desempleo(dff)
            acf_vals, pacf_vals = acf_pacf_data(dff)

            return html.Div([
                html.Div(className=f"alerta-box {alerta['clase']}", children=[
                    html.Div(alerta["icono"], className="alerta-icon"),
                    html.Div([
                        html.Div(alerta["titulo"], className="alerta-title"),
                        html.Div(alerta["texto"],  className="alerta-text")
                    ])
                ]),

                dbc.Row([
                    dbc.Col(html.Div(className="kpi-card", children=[
                        html.Div("Desempleo promedio", className="kpi-label"),
                        html.Div(f"{kpis['media_desempleo']}%", className="kpi-value blue"),
                        html.Div("Media del período seleccionado", className="kpi-sub")
                    ]), lg=3, md=6),
                    dbc.Col(html.Div(className="kpi-card", children=[
                        html.Div("Mínimo histórico", className="kpi-label"),
                        html.Div(f"{kpis['min_desempleo']}%", className="kpi-value green"),
                        html.Div(kpis["min_fecha"], className="kpi-sub")
                    ]), lg=3, md=6),
                    dbc.Col(html.Div(className="kpi-card", children=[
                        html.Div("Máximo histórico", className="kpi-label"),
                        html.Div(f"{kpis['max_desempleo']}%", className="kpi-value red"),
                        html.Div(kpis["max_fecha"], className="kpi-sub")
                    ]), lg=3, md=6),
                    dbc.Col(html.Div(className="kpi-card", children=[
                        html.Div("Observaciones", className="kpi-label"),
                        html.Div(f"{kpis['obs']}", className="kpi-value"),
                        html.Div("Período filtrado", className="kpi-sub")
                    ]), lg=3, md=6),
                ], className="g-4 mb-4"),

                dcc.Tabs(
                    value="hallazgos",
                    className="custom-tabs",
                    children=[

                        dcc.Tab(label="Hallazgos clave", children=[
                            html.Div(className="content-box", children=[
                                section_title("Hallazgos clave", "Síntesis interpretativa del análisis"),
                                dbc.Row([
                                    dbc.Col(hallazgo_card("📉", "Choque COVID-19",
                                        "El desempleo nacional registra un salto abrupto en 2020, evidenciando una ruptura coyuntural fuerte en el funcionamiento del mercado laboral colombiano."), lg=4, md=12),
                                    dbc.Col(hallazgo_card("🏙️", "Brecha urbano–nacional",
                                        "Las tasas de desempleo de las 13 áreas metropolitanas suelen ubicarse por encima del promedio nacional, lo que sugiere mayor presión estructural en el empleo urbano."), lg=4, md=12),
                                    dbc.Col(hallazgo_card("📈", "Persistencia temporal",
                                        "La serie de desempleo no se comporta como ruido aleatorio: presenta memoria, estacionalidad y componentes estructurales observables."), lg=4, md=12),
                                ], className="g-4")
                            ])
                        ]),

                        dcc.Tab(label="Inspección inicial", children=[
                            html.Div(className="content-box", children=[
                                section_title("Inspección inicial", "Resumen descriptivo del conjunto de datos"),
                                html.P("Esta tabla resume medidas centrales y de dispersión de las variables laborales incluidas en el análisis."),
                                table_component(tabla_descriptiva(dff))
                            ])
                        ]),

                        dcc.Tab(label="Univariado — Desempleo", children=[
                            html.Div(className="content-box", children=[
                                section_title("Análisis univariado: tasa de desempleo nacional"),
                                html.P("El objetivo de esta sección es estudiar el comportamiento individual de la variable central del trabajo: su distribución, dispersión, estabilidad y presencia de valores atípicos."),
                                graph_pair(fig_histograma(dff), fig_boxplot(dff), left_height=430, right_height=430),
                                interpretacion(
                                    "Interpretación del histograma y boxplot",
                                    "La distribución de la tasa de desempleo nacional permite identificar los niveles más frecuentes observados durante el período de estudio.",
                                    "Esto sugiere que el desempleo colombiano presenta un comportamiento relativamente recurrente en condiciones normales, pero vulnerable a choques macroeconómicos."
                                ),
                                graph_card(fig_dispersion(dff), height=500),
                                interpretacion(
                                    "Interpretación de la dispersión temporal",
                                    "La nube de puntos a lo largo del tiempo permite observar la evolución mensual del desempleo sin imponer una estructura suavizada.",
                                    "Desde una perspectiva económica, este patrón muestra que el mercado laboral colombiano combina persistencia estructural con episodios de ajuste fuerte ante contextos adversos."
                                )
                            ])
                        ]),

                        dcc.Tab(label="Univariado — Características", children=[
                            html.Div(className="content-box", children=[
                                section_title("Análisis univariado: variables explicativas"),
                                html.P("Además del desempleo, resulta importante estudiar el comportamiento individual de las variables que describen participación y ocupación."),
                                table_component(tabla_caracteristicas(dff)),
                                html.Div(style={"height": "18px"}),
                                graph_card(fig_boxplots_caracteristicas(df_long), height=540),
                                interpretacion(
                                    "Interpretación comparativa de variables",
                                    "La comparación entre boxplots permite evaluar diferencias en nivel, dispersión y presencia de valores extremos entre los principales indicadores laborales.",
                                    "Esto sugiere que la mayor volatilidad del sistema laboral colombiano se manifiesta con más claridad en el desempleo."
                                )
                            ])
                        ]),

                        dcc.Tab(label="Bivariado", children=[
                            html.Div(className="content-box", children=[
                                section_title("Análisis bivariado", "Relaciones entre indicadores laborales"),
                                html.P("El análisis bivariado permite estudiar la intensidad y dirección de las relaciones entre variables del mercado laboral."),
                                graph_card(fig_corr(corr), height=580),
                                interpretacion(
                                    "Interpretación de la matriz de correlación",
                                    "La matriz de correlación resume la intensidad y dirección de la asociación lineal entre las variables laborales.",
                                    "Estos resultados son consistentes con la lógica económica del mercado laboral."
                                ),
                                graph_card(fig_scatter_pairs(dff), height=920),
                                interpretacion(
                                    "Interpretación de los diagramas de dispersión",
                                    "Los gráficos de dispersión permiten observar si las relaciones entre variables son lineales, no lineales o si presentan agrupamientos particulares.",
                                    "En términos analíticos, estas relaciones refuerzan la lectura de que los indicadores laborales no evolucionan de forma aislada."
                                )
                            ])
                        ]),

                        dcc.Tab(label="Evolución temporal", children=[
                            html.Div(className="content-box", children=[
                                section_title("Evolución temporal", "Lectura dinámica de las series"),
                                html.P("Dado que las observaciones tienen estructura mensual, la dimensión temporal es esencial para interpretar correctamente los movimientos del mercado laboral."),
                                graph_card(fig_lineas(dff, [
                                    "tasa_desempleo_nacional", "tasa_desempleo_area",
                                    "tasa_ocupacion_nacional", "tasa_ocupacion_area",
                                    "tasa_global_participacion_nacional", "tasa_global_participacion_area"
                                ]), height=650),
                                interpretacion(
                                    "Interpretación de la evolución conjunta",
                                    "La representación temporal simultánea de los indicadores laborales permite observar cómo se relacionan entre sí a lo largo del tiempo.",
                                    "Este comportamiento confirma que el desempleo debe interpretarse dentro de una estructura más amplia donde también intervienen la participación y la ocupación."
                                ),
                                graph_card(fig_comparacion_anual(dff), height=560),
                                interpretacion(
                                    "Interpretación de la comparación anual",
                                    "La comparación por años permite reconocer patrones repetitivos y contrastar períodos de estabilidad, deterioro y recuperación.",
                                    "Desde el punto de vista económico, esta lectura ayuda a separar fluctuaciones normales del mercado laboral de episodios excepcionales."
                                )
                            ])
                        ]),

                        dcc.Tab(label="Series de tiempo", children=[
                            html.Div(className="content-box", children=[
                                section_title("Series de tiempo", "Persistencia, estructura y dependencia temporal"),
                                html.H4("Prueba ADF", className="subsection-title"),
                                html.P("La prueba de Dickey-Fuller aumentada se utiliza para evaluar si la serie presenta raíz unitaria, es decir, si puede considerarse no estacionaria."),
                                table_component(adf),
                                interpretacion(
                                    "Interpretación de la prueba ADF",
                                    "La prueba Dickey-Fuller aumentada evalúa si la serie de desempleo presenta raíz unitaria.",
                                    "Su resultado determina si la serie puede analizarse directamente o si requiere transformaciones previas."
                                ),
                                html.H4("Descomposición clásica", className="subsection-title"),
                                graph_card(fig_descomposicion(decomp), height=950),
                                interpretacion(
                                    "Interpretación de la descomposición",
                                    "La descomposición clásica separa la serie observada en tendencia, estacionalidad y residuo.",
                                    "En términos económicos, esta herramienta ayuda a entender si los cambios observados obedecen a transformaciones estructurales o a oscilaciones transitorias."
                                ),
                                html.H4("Autocorrelación y autocorrelación parcial", className="subsection-title"),
                                graph_pair(fig_acf(acf_vals), fig_pacf(pacf_vals), left_height=420, right_height=420),
                                interpretacion(
                                    "Interpretación de ACF y PACF",
                                    "Las funciones de autocorrelación y autocorrelación parcial permiten medir la dependencia entre observaciones a diferentes rezagos.",
                                    "Esto implica que el desempleo colombiano no evoluciona de forma completamente aleatoria, sino que conserva memoria estadística."
                                )
                            ])
                        ])
                    ]
                )
            ])

        elif tab == "modelo":
            # ── TAB SARIMA ──────────────────────────────────────────────────
            return dcc.Loading(
                type="circle",
                children=render_modelo(df)   # usa df completo (no filtrado)
            )

        elif tab == "conclusiones":
            return html.Div(className="content-box", children=[
                section_title("Conclusiones", "Síntesis final del análisis"),
                text_block(
                    html.P("El análisis del mercado laboral colombiano para el período 2001–2025 permitió identificar que la tasa de desempleo nacional presenta una dinámica persistente, sensible a choques macroeconómicos y estrechamente vinculada a otros indicadores estructurales del mercado laboral."),
                    html.Hr(className="ndq-divider"),
                    html.P("En términos descriptivos, la tasa de desempleo se concentra en rangos relativamente estables durante buena parte del período, aunque con episodios de fuerte disrupción. El caso más evidente corresponde a la pandemia de COVID-19, que introdujo un aumento abrupto y transitorio del desempleo, alterando de manera significativa la trayectoria histórica de la serie."),
                    html.Hr(className="ndq-divider"),
                    html.P("El análisis univariado permitió identificar dispersión moderada, presencia de valores atípicos y evidencia de no normalidad estricta, especialmente por el efecto de eventos extraordinarios. Estos valores extremos no fueron eliminados, dado que representan fenómenos económicos reales y aportan valor interpretativo al estudio."),
                    html.Hr(className="ndq-divider"),
                    html.P("Por su parte, el análisis bivariado mostró una relación negativa clara entre desempleo y ocupación, consistente con la teoría económica, así como una alta correspondencia entre las tasas de desempleo urbanas y nacionales."),
                    html.Hr(className="ndq-divider"),
                    html.P("Finalmente, el análisis temporal confirmó que la serie de desempleo incorpora componentes de tendencia, estacionalidad y dependencia serial, orientando el modelado hacia un esquema SARIMA(2,1,2)(1,1,1)[12]. El modelo validó las tres hipótesis secundarias del proyecto, con pronósticos para 2026 que apuntan a una tasa de desempleo en el rango 7.5%–10.9%.")
                )
            ])

        elif tab == "referencias":
            return html.Div(className="content-box", children=[
                section_title("Referencias", "Fuentes conceptuales y estadísticas"),
                html.Ul(style={"lineHeight": "2.1"}, children=[
                    html.Li("Arango, L., Herrera, P., & Posada, C. (2008). Ensayos sobre Política Económica, 26(56), 204–263."),
                    html.Li("Banco de la República de Colombia. Series estadísticas históricas del mercado laboral."),
                    html.Li("Congreso de la República de Colombia. (1945). Ley 6 de 1945."),
                    html.Li("DANE. Gran Encuesta Integrada de Hogares (GEIH)."),
                    html.Li("Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and Practice (3rd ed.)."),
                    html.Li("Mankiw, N. G. (2015). Principles of Economics (8th ed.)."),
                    html.Li("Mincer, J. (1962). Labor force participation of married women."),
                    html.Li("Organización Internacional del Trabajo (OIT). Definiciones estadísticas del mercado laboral.")
                ])
            ])

        return html.Div("Sin contenido.")

    # ── Callback interactivo: pronóstico (horizonte + nivel IC) ─────────────
    @app.callback(
        Output("pronostico-graph-container", "children"),
        Output("pronostico-tabla-container", "children"),
        Input("horizonte-fc", "value"),
        Input("nivel-ic", "value"),
    )
    def actualizar_pronostico(h, nivel):
        if h is None:
            h = 12
        if nivel is None:
            nivel = "both"

        # Usar modelo ya ajustado (no recalcular en cada interacción)
        resultado = _resultado_cache
        fc_dict   = pronostico(resultado, h=h, nivel=nivel)
        fechas_h  = _fechas_h
        obs_h     = _obs_h

        fig = fig_pronostico(fechas_h, obs_h, fc_dict, h, nivel)
        fig.update_layout(height=480)

        df_tabla = tabla_pronostico_df(fc_dict, nivel)

        tabla = dash_table.DataTable(
            data=df_tabla.to_dict("records"),
            columns=[{"name": c, "id": c} for c in df_tabla.columns],
            page_size=12,
            style_table={"overflowX": "auto", "borderRadius": "18px", "overflow": "hidden"},
            style_cell={
                "textAlign": "center", "padding": "12px",
                "fontFamily": "Inter, Segoe UI", "fontSize": "13px",
                "border": "1px solid #eef2f7", "backgroundColor": "rgba(255,255,255,0.92)",
                "color": "#334155",
            },
            style_header={
                "fontWeight": "bold", "backgroundColor": "#f8fafc",
                "border": "1px solid #e5e7eb", "color": "#0f172a", "fontSize": "13px"
            },
            style_data_conditional=[
                {"if": {"column_id": "Pronóstico"}, "color": "#0066cc", "fontWeight": "bold"},
                {"if": {"row_index": "odd"}, "backgroundColor": "rgba(248,250,252,0.65)"}
            ]
        )

        return (
            html.Div(dcc.Graph(figure=fig, config={"displayModeBar": False, "responsive": True},
                               style={"height": "480px"}), className="graph-card"),
            tabla
        )
