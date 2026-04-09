import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========================================================
# ESTILO GLOBAL
# =========================================================
def aplicar_estilo(fig, titulo=None):
    fig.update_layout(
        title={
            "text": titulo if titulo else "",
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 20, "family": "Inter, Segoe UI", "color": "#0f172a"}
        },
        paper_bgcolor="rgba(255,255,255,0.0)",
        plot_bgcolor="rgba(248,250,252,0.92)",
        font=dict(family="Inter, Segoe UI", size=13, color="#334155"),
        margin=dict(l=40, r=30, t=80, b=70),
        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
            font_family="Inter"
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.12,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.0)",
            font=dict(size=11),
            itemwidth=40
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.18)",
            zeroline=False,
            showline=False,
            tickangle=-25,
            automargin=True
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.18)",
            zeroline=False,
            showline=False,
            automargin=True
        )
    )
    return fig


# =========================================================
# UNIVARIADO
# =========================================================
def fig_histograma(df):
    fig = px.histogram(
        df,
        x="tasa_desempleo_nacional",
        nbins=25,
        marginal="box"
    )
    fig.update_traces(opacity=0.9, marker_line_width=0)
    return aplicar_estilo(fig, "Distribución de la tasa de desempleo nacional")


def fig_boxplot(df):
    fig = px.box(
        df,
        y="tasa_desempleo_nacional",
        points="all"
    )
    fig.update_traces(jitter=0.35, pointpos=-1.8)
    return aplicar_estilo(fig, "Boxplot de la tasa de desempleo nacional")


def fig_dispersion(df):
    fig = px.scatter(
        df,
        x="fecha",
        y="tasa_desempleo_nacional"
    )
    fig.update_traces(marker=dict(size=8, opacity=0.72))
    fig.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Tasa de desempleo",
        xaxis=dict(
            tickformat="%Y",
            tickangle=-25
        )
    )
    return aplicar_estilo(fig, "Dispersión temporal del desempleo nacional")


def fig_boxplots_caracteristicas(df_long):
    # Detectar automáticamente qué columna usar como categoría
    if "serie" in df_long.columns:
        eje_x = "serie"
    elif "indicador" in df_long.columns:
        eje_x = "indicador"
    elif "variable" in df_long.columns:
        eje_x = "variable"
    else:
        raise ValueError(
            f"No se encontró una columna categórica válida en df_long. "
            f"Columnas disponibles: {df_long.columns.tolist()}"
        )

    fig = px.box(
        df_long,
        x=eje_x,
        y="valor",
        points=False
    )

    fig.update_layout(
        xaxis_title="",
        yaxis_title="Valor"
    )

    fig.update_xaxes(
        tickangle=-35,
        automargin=True
    )

    return aplicar_estilo(fig, "Comparación de distribución entre variables laborales")


# =========================================================
# BIVARIADO
# =========================================================
def fig_corr(corr):
    etiquetas = [c.replace("_", " ").replace("Tasa ", "").title() for c in corr.columns]

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        x=etiquetas,
        y=etiquetas
    )

    fig.update_layout(
        coloraxis_colorbar=dict(title="Correlación"),
        margin=dict(l=90, r=30, t=80, b=90)
    )

    fig.update_xaxes(tickangle=-35, automargin=True)
    fig.update_yaxes(automargin=True)

    return aplicar_estilo(fig, "Matriz de correlación")


def fig_scatter_pairs(df):
    variables = [
        "tasa_desempleo_nacional",
        "tasa_desempleo_area",
        "tasa_ocupacion_nacional",
        "tasa_ocupacion_area",
        "tasa_global_participacion_nacional",
        "tasa_global_participacion_area"
    ]

    disponibles = [v for v in variables if v in df.columns]

    # Renombres cortos para que no se encimen
    nombres_cortos = {
        "tasa_desempleo_nacional": "TD Nacional",
        "tasa_desempleo_area": "TD 13 áreas",
        "tasa_ocupacion_nacional": "TO Nacional",
        "tasa_ocupacion_area": "TO 13 áreas",
        "tasa_global_participacion_nacional": "TGP Nacional",
        "tasa_global_participacion_area": "TGP 13 áreas"
    }

    dff = df[disponibles].rename(columns=nombres_cortos)

    fig = px.scatter_matrix(
        dff,
        dimensions=list(dff.columns)
    )

    fig.update_traces(
        diagonal_visible=False,
        marker=dict(size=5, opacity=0.55)
    )

    fig.update_layout(
        height=950,
        margin=dict(l=90, r=40, t=80, b=80)
    )

    # Ajustar tamaño de labels y evitar amontonamiento
    fig.update_xaxes(
        tickangle=-35,
        tickfont=dict(size=10)
    )
    fig.update_yaxes(
        tickfont=dict(size=10)
    )

    return aplicar_estilo(fig, "Relaciones bivariadas entre indicadores laborales")


# =========================================================
# TEMPORAL
# =========================================================
def fig_lineas(df, columnas):
    fig = go.Figure()

    nombres_bonitos = {
        "tasa_desempleo_nacional": "Desempleo Nacional",
        "tasa_desempleo_area": "Desempleo 13 Áreas",
        "tasa_ocupacion_nacional": "Ocupación Nacional",
        "tasa_ocupacion_area": "Ocupación 13 Áreas",
        "tasa_global_participacion_nacional": "Participación Nacional",
        "tasa_global_participacion_area": "Participación 13 Áreas",
    }

    for col in columnas:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["fecha"],
                y=df[col],
                mode="lines",
                name=nombres_bonitos.get(col, col.replace("_", " ").title()),
                line=dict(width=3),
                hovertemplate="%{x|%Y-%m}<br>%{y:.2f}%<extra></extra>"
            ))

    fig.update_layout(
        title={
            "text": "Evolución temporal de indicadores laborales",
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 20, "family": "Inter, Segoe UI", "color": "#0f172a"}
        },
        xaxis_title="Fecha",
        yaxis_title="Porcentaje",
        paper_bgcolor="rgba(255,255,255,0.0)",
        plot_bgcolor="rgba(248,250,252,0.92)",
        font=dict(family="Inter, Segoe UI", size=13, color="#334155"),

        # 👇 MÁS ESPACIO ARRIBA
        margin=dict(l=40, r=30, t=120, b=60),

        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
            font_family="Inter"
        ),

        # 👇 LEYENDA MÁS ABAJO DEL TÍTULO
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.0)",
            font=dict(size=12)
        ),

        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.18)",
            zeroline=False,
            showline=False,
            tickformat="%Y",
            tickangle=0,
            automargin=True
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.18)",
            zeroline=False,
            showline=False,
            automargin=True
        ),

        height=620
    )

    return fig


def fig_comparacion_anual(df):
    dff = df.copy()
    dff["anio"] = dff["fecha"].dt.year.astype(str)
    dff["mes"] = dff["fecha"].dt.month

    fig = px.line(
        dff,
        x="mes",
        y="tasa_desempleo_nacional",
        color="anio",
        markers=False
    )

    fig.update_traces(
        line=dict(width=2.5),
        hovertemplate="Mes %{x}<br>%{y:.2f}%<extra></extra>"
    )

    fig.update_layout(
        title={
            "text": "Comparación anual del desempleo nacional",
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 20, "family": "Inter, Segoe UI", "color": "#0f172a"}
        },
        xaxis_title="Mes",
        yaxis_title="Tasa de desempleo",
        paper_bgcolor="rgba(255,255,255,0.0)",
        plot_bgcolor="rgba(248,250,252,0.92)",
        font=dict(family="Inter, Segoe UI", size=13, color="#334155"),

        # 👇 MÁS ESPACIO ARRIBA PARA QUE NO TOQUE EL TÍTULO
        margin=dict(l=40, r=30, t=120, b=60),

        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
            font_family="Inter"
        ),

        # 👇 LEYENDA BAJO EL TÍTULO Y BIEN ORDENADA
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.0)",
            font=dict(size=11),
            title=""
        ),

        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.18)",
            zeroline=False,
            showline=False,
            tickmode="array",
            tickvals=[1,2,3,4,5,6,7,8,9,10,11,12],
            ticktext=["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"],
            automargin=True
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.18)",
            zeroline=False,
            showline=False,
            automargin=True
        ),

        height=620
    )

    return fig


# =========================================================
# SERIES DE TIEMPO
# =========================================================
def fig_descomposicion(decomp):
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Serie observada", "Tendencia", "Estacionalidad", "Residuo")
    )

    fig.add_trace(go.Scatter(x=decomp.observed.index, y=decomp.observed, mode="lines", name="Observada"), row=1, col=1)
    fig.add_trace(go.Scatter(x=decomp.trend.index, y=decomp.trend, mode="lines", name="Tendencia"), row=2, col=1)
    fig.add_trace(go.Scatter(x=decomp.seasonal.index, y=decomp.seasonal, mode="lines", name="Estacionalidad"), row=3, col=1)
    fig.add_trace(go.Scatter(x=decomp.resid.index, y=decomp.resid, mode="lines", name="Residuo"), row=4, col=1)

    fig.update_traces(line=dict(width=2.5))
    fig.update_layout(
        height=900,
        showlegend=False
    )

    fig = aplicar_estilo(fig, "Descomposición clásica de la serie de desempleo")
    return fig


def fig_acf(acf_vals):
    lags = list(range(len(acf_vals)))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=lags,
        y=acf_vals,
        name="ACF"
    ))
    fig.add_hline(y=0, line_width=1)
    fig.update_layout(
        xaxis_title="Rezago",
        yaxis_title="Autocorrelación"
    )
    return aplicar_estilo(fig, "Función de autocorrelación (ACF)")


def fig_pacf(pacf_vals):
    lags = list(range(len(pacf_vals)))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=lags,
        y=pacf_vals,
        name="PACF"
    ))
    fig.add_hline(y=0, line_width=1)
    fig.update_layout(
        xaxis_title="Rezago",
        yaxis_title="Autocorrelación parcial"
    )
    return aplicar_estilo(fig, "Función de autocorrelación parcial (PACF)")