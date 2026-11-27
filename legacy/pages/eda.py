"""Sección 2: Análisis exploratorio con filtros globales.

El rango de fechas y la cobertura (nacional o 13 áreas) actualizan todos los
gráficos de la página. Cada gráfico va acompañado de una frase de
interpretación calculada con los datos filtrados.
"""
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, callback, dcc, html

from src import config
from src import dash_data as D
from src import dash_plots as P

dash.register_page(__name__, path="/eda", name="EDA", title="EDA · Mercado Laboral Colombiano")

RAW = D.raw()
L = D.RAW_LABELS
COVERAGE = {
    "nacional": ["tasa_desempleo_nacional", "tasa_ocupacion_nacional", "tasa_global_participacion_nacional"],
    "area": ["tasa_desempleo_area", "tasa_ocupacion_area", "tasa_global_participacion_area"],
}
TARGET = "tasa_desempleo_nacional"
VAR_OPTIONS = [{"label": v, "value": k} for k, v in L.items()]
COLOR_OPTIONS = [
    {"label": "Periodo (antes, durante y después de 2020)", "value": "periodo"},
    {"label": "Categoría de desempleo (Bajo, Medio, Alto)", "value": "categoria"},
    {"label": "Año", "value": "anio_num"},
]


def card(title, body_id, insight_id, extra=None):
    return dbc.Card([
        dbc.CardHeader(title),
        # Un componente de Dash sin hijos tiene len() == 0, así que no se puede usar `extra or ...`.
        dbc.CardBody([extra if extra is not None else html.Div(), dcc.Loading(dcc.Graph(id=body_id, config={"displaylogo": False})),
                      html.Div(id=insight_id, className="mlc-insight")]),
    ], className="mlc-card")


def filtered(start, end):
    df = RAW[(RAW["fecha"] >= pd.Timestamp(start)) & (RAW["fecha"] <= pd.Timestamp(end))].copy()
    df["periodo"] = np.where(df["fecha"] < "2020-03-01", "Antes de 2020",
                             np.where(df["fecha"] <= "2021-06-30", "Choque 2020-2021", "Después"))
    df["categoria"] = pd.cut(df[TARGET], [-np.inf, 10, 14, np.inf], labels=config.CLASS_LABELS, right=False)
    df["anio_num"] = df["fecha"].dt.year
    return df


filters = dbc.Card(dbc.CardBody(dbc.Row([
    dbc.Col([
        html.Label("Rango de fechas", htmlFor="eda-fechas", className="fw-semibold"),
        dcc.DatePickerRange(id="eda-fechas", start_date=RAW["fecha"].min().date(), end_date=RAW["fecha"].max().date(),
                            min_date_allowed=RAW["fecha"].min().date(), max_date_allowed=RAW["fecha"].max().date(),
                            display_format="YYYY-MM", className="d-block"),
    ], md=4),
    dbc.Col([
        html.Label("Años (atajo)", htmlFor="eda-anios", className="fw-semibold"),
        dcc.RangeSlider(id="eda-anios", min=2001, max=2025, step=1, value=[2001, 2025],
                        marks={y: str(y) for y in range(2001, 2026, 4)}, tooltip={"placement": "bottom"}),
    ], md=4),
    dbc.Col([
        html.Label("Cobertura", htmlFor="eda-cobertura", className="fw-semibold"),
        dcc.Dropdown(id="eda-cobertura", clearable=False, value="nacional",
                     options=[{"label": "Total nacional", "value": "nacional"}, {"label": "13 áreas metropolitanas", "value": "area"}]),
    ], md=4),
], className="g-3")), className="mlc-card mlc-filters")

layout = html.Div([
    html.H2("Análisis exploratorio", className="mlc-section-title"),
    html.P("Los filtros de esta barra actualizan todos los gráficos. Las zonas en ámbar marcan el choque de la pandemia (marzo de 2020 a junio de 2021)."),
    filters,
    card("Evolución de los indicadores", "eda-serie", "eda-serie-txt"),
    dbc.Row([
        dbc.Col(card("Distribución de la variable objetivo", "eda-objetivo", "eda-objetivo-txt"), lg=6),
        dbc.Col(card("Balance de clases (clasificación)", "eda-clases", "eda-clases-txt"), lg=6),
    ]),
    dbc.Row([
        dbc.Col(card("Distribución univariada", "eda-uni", "eda-uni-txt", extra=dcc.Dropdown(
            id="eda-uni-var", options=VAR_OPTIONS, value="tasa_desempleo_area", clearable=False,
            className="mb-2")), lg=6),
        dbc.Col(card("Valores atípicos", "eda-atipicos", "eda-atipicos-txt"), lg=6),
    ]),
    card("Explorador bivariado", "eda-bi", "eda-bi-txt", extra=dbc.Row([
        dbc.Col([html.Label("Eje x", htmlFor="eda-x"), dcc.Dropdown(id="eda-x", options=VAR_OPTIONS, value="tasa_desempleo_area", clearable=False)], md=4),
        dbc.Col([html.Label("Eje y", htmlFor="eda-y"), dcc.Dropdown(id="eda-y", options=VAR_OPTIONS, value=TARGET, clearable=False)], md=4),
        dbc.Col([html.Label("Color", htmlFor="eda-color"), dcc.Dropdown(id="eda-color", options=COLOR_OPTIONS, value="periodo", clearable=False)], md=4),
    ], className="mb-2 g-2")),
    dbc.Row([
        dbc.Col(card("Matriz de correlación", "eda-corr", "eda-corr-txt"), lg=7),
        dbc.Col(dbc.Card([dbc.CardHeader("Valores faltantes"), dbc.CardBody(id="eda-faltantes")], className="mlc-card"), lg=5),
    ]),
    html.H3("Serie de tiempo de la variable objetivo", className="mlc-section-title"),
    card("Descomposición", "eda-desc", "eda-desc-txt"),
    card("Autocorrelación", "eda-acf", "eda-acf-txt"),
    dbc.Card([dbc.CardHeader("Pruebas de estacionariedad"), dbc.CardBody(id="eda-estacionariedad")], className="mlc-card"),
])


@callback(Output("eda-fechas", "start_date"), Output("eda-fechas", "end_date"), Input("eda-anios", "value"),
          prevent_initial_call=True)
def years_to_dates(years):
    return f"{years[0]}-01-01", f"{years[1]}-12-31"


@callback(
    Output("eda-serie", "figure"), Output("eda-serie-txt", "children"),
    Output("eda-objetivo", "figure"), Output("eda-objetivo-txt", "children"),
    Output("eda-clases", "figure"), Output("eda-clases-txt", "children"),
    Output("eda-corr", "figure"), Output("eda-corr-txt", "children"),
    Output("eda-faltantes", "children"),
    Input("eda-fechas", "start_date"), Input("eda-fechas", "end_date"), Input("eda-cobertura", "value"),
)
def update_main(start, end, coverage):
    df = filtered(start, end)
    if len(df) < 3:
        msg = "Seleccione un rango de al menos tres meses."
        e = P.empty_figure(msg)
        return e, msg, e, msg, e, msg, e, msg, msg
    cols = COVERAGE[coverage]
    cov_label = "nacional" if coverage == "nacional" else "de las 13 áreas"
    serie = P.time_series(df, cols, L, f"Tasas del mercado laboral, cobertura {cov_label}")
    d = df[cols[0]]
    i_max, i_min = d.idxmax(), d.idxmin()
    serie_txt = (f"Entre {D.month_es(df['fecha'].min())} y {D.month_es(df['fecha'].max())} el desempleo {cov_label} "
                 f"promedió {d.mean():.1f} %, con un máximo de {d.max():.1f} % ({D.month_es(df.loc[i_max, 'fecha'])}) y un "
                 f"mínimo de {d.min():.1f} % ({D.month_es(df.loc[i_min, 'fecha'])}). La ocupación se mueve en sentido "
                 f"contrario al desempleo (correlación {df[cols[0]].corr(df[cols[1]]):.2f}).")

    y = df[TARGET]
    obj = P.histogram_box(y, "Desempleo nacional", "Distribución de la tasa de desempleo nacional",
                          thresholds=[(10, "10 %"), (14, "14 %")])
    skew = y.skew()
    obj_txt = (f"La mediana es {y.median():.1f} % y la media {y.mean():.1f} %; la asimetría de {skew:.2f} indica "
               f"{'una cola larga hacia valores altos' if skew > 0.3 else 'una distribución aproximadamente simétrica'}. "
               f"Las líneas marcan los cortes de las categorías de clasificación.")

    counts = df["categoria"].value_counts().reindex(config.CLASS_LABELS).fillna(0).astype(int)
    clases = P.class_balance(counts, "Meses por categoría de desempleo")
    minority = counts.idxmin()
    clases_txt = (f"La clase menos frecuente es «{minority}» ({counts.min()} de {counts.sum()} meses, "
                  f"{counts.min() / max(counts.sum(), 1) * 100:.0f} %). Ese desbalance es real, no forzado, y por eso "
                  f"el segundo entregable comparó SMOTE, ADASYN y pesos de clase.")

    corr = df[list(L)].corr()
    corr_fig = P.correlation_heatmap(corr, L)
    r_area = corr.loc[TARGET, "tasa_desempleo_area"]
    corr_txt = (f"El desempleo nacional y el de las 13 áreas tienen r = {r_area:.2f} en el periodo elegido. "
                "La ocupación y la participación nacionales no se usan como predictoras del mismo mes porque "
                "TD = 1 − TO/TGP: el modelo solo despejaría una identidad contable.")

    na = df[list(L)].isna().sum()
    faltantes = [
        html.P(f"{int(na.sum())} valores faltantes en {len(df)} meses y {len(L)} series."),
        dbc.Table.from_dataframe(pd.DataFrame({"Serie": [L[c] for c in na.index], "Faltantes": na.values}),
                                 striped=True, bordered=False, hover=True, size="sm"),
        html.Div("La serie del Banco de la República está completa en todo el periodo; los primeros 11 meses "
                 "se pierden solo al construir los rezagos de 12 meses.", className="mlc-insight"),
    ]
    return serie, serie_txt, obj, obj_txt, clases, clases_txt, corr_fig, corr_txt, faltantes


@callback(
    Output("eda-uni", "figure"), Output("eda-uni-txt", "children"),
    Output("eda-atipicos", "figure"), Output("eda-atipicos-txt", "children"),
    Input("eda-fechas", "start_date"), Input("eda-fechas", "end_date"), Input("eda-uni-var", "value"),
)
def update_univariate(start, end, var):
    df = filtered(start, end)
    if len(df) < 3:
        e = P.empty_figure("Seleccione un rango de al menos tres meses.")
        return e, "", e, ""
    v = df[var]
    uni = P.histogram_box(v, L[var], f"Distribución de {L[var].lower()}")
    uni_txt = (f"{L[var]}: media {v.mean():.2f} %, desviación {v.std():.2f} pp, rango de {v.min():.1f} % a {v.max():.1f} %.")
    fig, n_out = P.outliers_figure(df, var, L[var])
    out_txt = (f"Con la regla de 1.5 veces el rango intercuartílico hay {n_out} meses atípicos"
               + (", concentrados en el choque de 2020." if n_out and df.loc[(df[var] - v.median()).abs().idxmax(), 'periodo'] == "Choque 2020-2021" else ".")
               + " No se eliminan: son episodios reales que el modelo debe aprender a pronosticar.")
    return uni, uni_txt, fig, out_txt


@callback(
    Output("eda-bi", "figure"), Output("eda-bi-txt", "children"),
    Input("eda-fechas", "start_date"), Input("eda-fechas", "end_date"),
    Input("eda-x", "value"), Input("eda-y", "value"), Input("eda-color", "value"),
)
def update_bivariate(start, end, x, y, color):
    df = filtered(start, end)
    if len(df) < 3:
        return P.empty_figure("Seleccione un rango de al menos tres meses."), ""
    labels = dict(L, anio_num="Año")
    fig = P.scatter_explorer(df, x, y, color, labels)
    r = df[x].corr(df[y]) if x != y else 1.0
    strength = "fuerte" if abs(r) > 0.7 else "moderada" if abs(r) > 0.4 else "débil"
    txt = (f"La relación lineal entre {labels[x].lower()} y {labels[y].lower()} es {strength} (r = {r:.2f}). "
           "Colorear por periodo permite ver si el choque de 2020 cambia la relación.")
    return fig, txt


@callback(
    Output("eda-desc", "figure"), Output("eda-desc-txt", "children"),
    Output("eda-acf", "figure"), Output("eda-acf-txt", "children"),
    Output("eda-estacionariedad", "children"),
    Input("eda-fechas", "start_date"), Input("eda-fechas", "end_date"), Input("eda-cobertura", "value"),
)
def update_time_series(start, end, coverage):
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, kpss

    df = filtered(start, end)
    col = COVERAGE[coverage][0]
    if len(df) < 36:
        e = P.empty_figure("La descomposición y las pruebas necesitan al menos 36 meses.")
        return e, "", e, "", "Seleccione un rango de al menos 36 meses."
    s = df.set_index("fecha")[col]
    dec = seasonal_decompose(s, model="additive", period=12)
    seas = dec.seasonal.groupby(dec.seasonal.index.month).mean()
    desc_txt = (f"El componente estacional va de {seas.min():+.2f} pp ({D.MONTHS_ES[seas.idxmin() - 1]}) a "
                f"{seas.max():+.2f} pp ({D.MONTHS_ES[seas.idxmax() - 1]}): el desempleo sube cada comienzo de año. "
                "Esa regularidad explica por qué un pronóstico estacional sencillo es tan difícil de superar.")
    acf_fig = P.acf_pacf_figure(s.values)
    from src.explain import acf as _acf

    a = _acf(s.values, 12)
    acf_txt = (f"La autocorrelación decae muy despacio (rezago 1: {a[1]:.2f}; rezago 12: {a[12]:.2f}): la serie es muy "
               "persistente, típica de un proceso con raíz unitaria o cercano a ella.")
    rows = []
    for name, series in [("En niveles", s), ("Primera diferencia", s.diff().dropna()),
                         ("Diferencia estacional (12)", s.diff(12).dropna())]:
        adf_p = adfuller(series, autolag="AIC")[1]
        kpss_p = kpss(series, regression="c", nlags="auto")[1]
        rows.append({"Serie": name, "ADF (H0: raíz unitaria) p": f"{adf_p:.4f}", "KPSS (H0: estacionaria) p": f"{kpss_p:.3f}",
                     "Conclusión": "Estacionaria" if adf_p < 0.05 and kpss_p > 0.05 else "No estacionaria" if adf_p >= 0.05 and kpss_p <= 0.05 else "Evidencia mixta"})
    table = pd.DataFrame(rows)
    est = [dbc.Table.from_dataframe(table, striped=True, hover=True, size="sm"),
           html.Div("ADF y KPSS se complementan porque sus hipótesis nulas son opuestas. KPSS reporta p-valores "
                    "acotados entre 0.01 y 0.1 (fuera de ese rango, el valor es el límite de su tabla).", className="mlc-insight")]
    return dec_fig(dec, L[col]), desc_txt, acf_fig, acf_txt, est


def dec_fig(dec, label):
    return P.decomposition_figure(dec, label.lower())
