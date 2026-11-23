"""Figuras Plotly del dashboard, con una plantilla y una paleta comunes.

La paleta tiene cinco colores (la guía pide un máximo de cinco). El color
nunca es la única señal: las series se distinguen también por tipo de línea
o de marcador, y cada figura lleva título, ejes con unidades y texto en el
``hover``.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from labor import theme  # noqa: F401  (plantilla oscura común)

from .explain import acf, pacf

NAVY, BLUE, VIOLET, SLATE, AMBER = "#F3F7FC", "#3984FF", "#9565FF", "#A7B5C9", "#FFB547"
PALETTE = [BLUE, VIOLET, AMBER, SLATE, NAVY]
DASHES = ["solid", "dash", "dot", "dashdot", "longdash"]
SYMBOLS = ["circle", "diamond", "square", "triangle-up", "x"]
COVID = ("2020-03-01", "2021-06-30")




def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=14, color=SLATE), x=0.5, y=0.5,
                       xref="paper", yref="paper")
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), height=300)
    return fig


def _shade_covid(fig, start, end):
    s, e = max(pd.Timestamp(COVID[0]), start), min(pd.Timestamp(COVID[1]), end)
    if s < e:
        fig.add_vrect(x0=s, x1=e, fillcolor=AMBER, opacity=0.12, line_width=0,
                      annotation_text="Choque 2020", annotation_position="top left", annotation_font_size=11)


# --------------------------------------------------------------------- EDA
def time_series(df: pd.DataFrame, cols: list, labels: dict, title: str) -> go.Figure:
    fig = go.Figure()
    for i, c in enumerate(cols):
        fig.add_trace(go.Scatter(x=df["fecha"], y=df[c], name=labels[c], mode="lines",
                                 line=dict(color=PALETTE[i % 5], dash=DASHES[i % 5], width=2),
                                 hovertemplate="%{x|%b %Y}: %{y:.2f} %<extra>" + labels[c] + "</extra>"))
    _shade_covid(fig, df["fecha"].min(), df["fecha"].max())
    fig.update_layout(title=title, yaxis_title="Tasa (%)", xaxis_title="Mes", height=420, hovermode="x unified")
    return fig


def histogram_box(values: pd.Series, label: str, title: str, thresholds=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values, nbinsx=30, marker_color=BLUE, opacity=0.85, name="Meses",
                               hovertemplate="%{x:.1f} %: %{y} meses<extra></extra>"))
    for t, name in (thresholds or []):
        fig.add_vline(x=t, line_dash="dash", line_color=AMBER, annotation_text=name, annotation_font_size=11)
    fig.update_layout(title=title, xaxis_title=f"{label} (%)", yaxis_title="Número de meses", height=380, bargap=0.05,
                      showlegend=False)
    return fig


def class_balance(counts: pd.Series, title: str) -> go.Figure:
    pct = counts / counts.sum() * 100
    fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=[BLUE, SLATE, AMBER][: len(counts)],
                           text=[f"{v} ({p:.0f} %)" for v, p in zip(counts.values, pct.values)], textposition="outside",
                           hovertemplate="%{x}: %{y} meses<extra></extra>"))
    fig.update_layout(title=title, xaxis_title="Categoría del desempleo del mes", yaxis_title="Número de meses",
                      height=380, yaxis_range=[0, counts.max() * 1.2])
    return fig


def scatter_explorer(df, x, y, color, labels) -> go.Figure:
    fig = go.Figure()
    if color in ("anio", "periodo", "categoria"):
        for i, (g, sub) in enumerate(df.groupby(color, observed=True)):
            fig.add_trace(go.Scatter(x=sub[x], y=sub[y], mode="markers", name=str(g),
                                     marker=dict(color=PALETTE[i % 5], symbol=SYMBOLS[i % 5], size=7, opacity=0.8),
                                     hovertemplate=f"{labels[x]}: %{{x:.2f}}<br>{labels[y]}: %{{y:.2f}}<extra>{g}</extra>"))
    else:
        fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="markers",
                                 marker=dict(color=df[color], colorscale=[[0, "#26467F"], [1, NAVY]], size=7,
                                             colorbar=dict(title=labels.get(color, color)), showscale=True),
                                 hovertemplate=f"{labels[x]}: %{{x:.2f}}<br>{labels[y]}: %{{y:.2f}}<extra></extra>"))
    fig.update_layout(title=f"{labels[y]} frente a {labels[x]}", xaxis_title=f"{labels[x]} (%)",
                      yaxis_title=f"{labels[y]} (%)", height=430)
    return fig


def correlation_heatmap(corr: pd.DataFrame, labels: dict) -> go.Figure:
    names = [labels.get(c, c) for c in corr.columns]
    fig = go.Figure(go.Heatmap(z=corr.values, x=names, y=names, zmin=-1, zmax=1,
                               colorscale=[[0, AMBER], [0.5, "#101B32"], [1, BLUE]],
                               text=np.round(corr.values, 2), texttemplate="%{text}", colorbar=dict(title="r"),
                               hovertemplate="%{y} y %{x}: r = %{z:.3f}<extra></extra>"))
    fig.update_layout(title="Correlación de Pearson entre las tasas", height=460, xaxis=dict(tickangle=-30))
    return fig


def outliers_figure(df, col, label) -> go.Figure:
    v = df[col]
    q1, q3 = v.quantile([0.25, 0.75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    out = df[(v < lo) | (v > hi)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["fecha"], y=v, mode="lines", name=label, line=dict(color=BLUE)))
    fig.add_trace(go.Scatter(x=out["fecha"], y=out[col], mode="markers", name="Atípico (regla de 1.5 RIC)",
                             marker=dict(color=AMBER, size=9, symbol="x"),
                             hovertemplate="%{x|%b %Y}: %{y:.2f} %<extra>atípico</extra>"))
    for b, n in [(lo, "límite inferior"), (hi, "límite superior")]:
        fig.add_hline(y=b, line_dash="dot", line_color=SLATE, annotation_text=n, annotation_font_size=10)
    fig.update_layout(title=f"Valores atípicos de {label}", yaxis_title="Tasa (%)", xaxis_title="Mes", height=380)
    return fig, len(out)


def decomposition_figure(dec, label) -> go.Figure:
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Tendencia", "Componente estacional", "Residuo"))
    for i, (comp, color) in enumerate([(dec.trend, BLUE), (dec.seasonal, VIOLET), (dec.resid, SLATE)], start=1):
        fig.add_trace(go.Scatter(x=comp.index, y=comp.values, mode="lines", line=dict(color=color), showlegend=False,
                                 hovertemplate="%{x|%b %Y}: %{y:.2f}<extra></extra>"), row=i, col=1)
    fig.update_yaxes(title_text="pp", row=2, col=1)
    fig.update_layout(title=f"Descomposición aditiva (periodo 12) de {label}", height=560)
    return fig


def acf_pacf_figure(x, nlags=24) -> go.Figure:
    from plotly.subplots import make_subplots

    n = len(x)
    band = 1.96 / np.sqrt(n)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Autocorrelación (ACF)", "Autocorrelación parcial (PACF)"))
    for j, vals in enumerate([acf(x, nlags), pacf(x, nlags)], start=1):
        lags = np.arange(len(vals))
        fig.add_trace(go.Bar(x=lags[1:], y=vals[1:], marker_color=BLUE, showlegend=False,
                             hovertemplate="rezago %{x}: %{y:.3f}<extra></extra>"), row=1, col=j)
        for s in (band, -band):
            fig.add_hline(y=s, line_dash="dash", line_color=AMBER, row=1, col=j)
    fig.update_xaxes(title_text="Rezago (meses)")
    fig.update_yaxes(title_text="Correlación", range=[-1, 1])
    fig.update_layout(title="Dependencia temporal de la serie (bandas: ±1.96/√n)", height=380)
    return fig


# ------------------------------------------------------------------ modelos
def shap_bar(shap: dict, labels: dict, title: str, top: int = 10) -> go.Figure:
    s = pd.Series(shap).sort_values(key=np.abs, ascending=False).head(top)[::-1]
    fig = go.Figure(go.Bar(x=s.values, y=[labels.get(k, k) for k in s.index], orientation="h",
                           marker_color=[BLUE if v >= 0 else AMBER for v in s.values],
                           marker_pattern_shape=["" if v >= 0 else "/" for v in s.values],
                           hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
    fig.update_layout(title=title, xaxis_title="Contribución a la salida (valor de Shapley)", height=380,
                      margin=dict(l=200))
    return fig


def shap_summary(phi: np.ndarray, X: pd.DataFrame, labels: dict, title: str, top: int = 10) -> go.Figure:
    order = np.argsort(np.abs(phi).mean(axis=0))[::-1][:top][::-1]
    fig = go.Figure()
    rng = np.random.default_rng(0)
    for rank, j in enumerate(order):
        col = X.columns[j]
        v = X[col].values
        norm = (v - v.min()) / (np.ptp(v) or 1.0)
        fig.add_trace(go.Scatter(
            x=phi[:, j], y=rank + rng.uniform(-0.25, 0.25, len(v)), mode="markers", showlegend=False,
            marker=dict(color=norm, colorscale=[[0, "#28D8F0"], [1, VIOLET]], size=6, opacity=0.75,
                        cmin=0, cmax=1, colorbar=dict(title="Valor de la<br>variable", tickvals=[0, 1],
                                                      ticktext=["bajo", "alto"]) if rank == 0 else None,
                        showscale=rank == 0),
            customdata=v, hovertemplate=f"{labels.get(col, col)}<br>valor %{{customdata:.2f}}<br>Shapley %{{x:+.3f}}<extra></extra>"))
    fig.update_layout(title=title, xaxis_title="Valor de Shapley (impacto en la salida)", height=460,
                      yaxis=dict(tickvals=list(range(len(order))), ticktext=[labels.get(X.columns[j], X.columns[j]) for j in order]),
                      margin=dict(l=200))
    fig.add_vline(x=0, line_color=SLATE)
    return fig


def real_vs_pred(df, pred_col, label) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["fecha"], y=df["y_true"], name="Observado", mode="lines", line=dict(color=NAVY)))
    fig.add_trace(go.Scatter(x=df["fecha"], y=df[pred_col], name=f"Pronóstico {label}", mode="lines",
                             line=dict(color=BLUE, dash="dash")))
    for k in sorted(df["fold"].unique())[1:]:
        fig.add_vline(x=df.loc[df["fold"] == k, "fecha"].min(), line_dash="dot", line_color=SLATE)
    _shade_covid(fig, df["fecha"].min(), df["fecha"].max())
    fig.update_layout(title="Observado frente a pronosticado (fuera de muestra; líneas punteadas: pliegues)",
                      yaxis_title="Tasa de desempleo (%)", xaxis_title="Mes pronosticado", height=400, hovermode="x unified")
    return fig


def residual_panels(resid, fitted) -> go.Figure:
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=3, subplot_titles=("Residuo frente a pronóstico", "Histograma de residuos",
                                                        "ACF de los residuos"))
    fig.add_trace(go.Scatter(x=fitted, y=resid, mode="markers", marker=dict(color=BLUE, size=6, opacity=0.7),
                             showlegend=False, hovertemplate="pronóstico %{x:.2f}<br>residuo %{y:+.2f}<extra></extra>"), 1, 1)
    fig.add_hline(y=0, line_color=SLATE, row=1, col=1)
    fig.add_trace(go.Histogram(x=resid, nbinsx=30, marker_color=VIOLET, showlegend=False), 1, 2)
    a = acf(resid, 24)
    fig.add_trace(go.Bar(x=np.arange(1, 25), y=a[1:], marker_color=BLUE, showlegend=False), 1, 3)
    for s in (1.96 / np.sqrt(len(resid)), -1.96 / np.sqrt(len(resid))):
        fig.add_hline(y=s, line_dash="dash", line_color=AMBER, row=1, col=3)
    fig.update_xaxes(title_text="Pronóstico (%)", row=1, col=1)
    fig.update_xaxes(title_text="Residuo (pp)", row=1, col=2)
    fig.update_xaxes(title_text="Rezago", row=1, col=3)
    fig.update_yaxes(title_text="Residuo (pp)", row=1, col=1)
    fig.update_yaxes(title_text="Meses", row=1, col=2)
    fig.update_layout(height=360, title="Diagnóstico de residuos")
    return fig


def confusion_figure(cm: np.ndarray, labels: list) -> go.Figure:
    pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100
    text = [[f"{cm[i, j]}<br>({pct[i, j]:.0f} %)" for j in range(len(labels))] for i in range(len(labels))]
    fig = go.Figure(go.Heatmap(z=pct, x=labels, y=labels, colorscale=[[0, "#101B32"], [1, BLUE]], zmin=0, zmax=100,
                               text=text, texttemplate="%{text}", colorbar=dict(title="% de la fila"),
                               hovertemplate="real %{y}, predicho %{x}<extra></extra>"))
    fig.update_layout(title="Matriz de confusión (fuera de muestra)", xaxis_title="Clase predicha",
                      yaxis_title="Clase real", height=380, yaxis=dict(autorange="reversed"))
    return fig


def curves_figure(curves: dict, kind: str) -> go.Figure:
    fig = go.Figure()
    for i, (name, (x, y, score)) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"{name} ({'AUC' if kind == 'roc' else 'AP'} {score:.2f})".replace(".", ","),
                                 line=dict(color=PALETTE[i], dash=DASHES[i], width=2)))
    if kind == "roc":
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar", line=dict(color=SLATE, dash="dot")))
        fig.update_layout(title="Curvas ROC (una clase contra el resto)", xaxis_title="Tasa de falsos positivos",
                          yaxis_title="Tasa de verdaderos positivos")
    else:
        fig.update_layout(title="Curvas de precisión y exhaustividad", xaxis_title="Exhaustividad (recall)",
                          yaxis_title="Precisión")
    fig.update_layout(height=380, xaxis_range=[0, 1], yaxis_range=[0, 1.02])
    return fig


def calibration_figure(bins: dict) -> go.Figure:
    fig = go.Figure()
    for i, (name, (p, f, ece)) in enumerate(bins.items()):
        fig.add_trace(go.Scatter(x=p, y=f, mode="lines+markers", name=f"{name} (ECE {ece:.3f})",
                                 line=dict(color=PALETTE[i], dash=DASHES[i]), marker=dict(symbol=SYMBOLS[i])))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Calibración perfecta", line=dict(color=SLATE, dash="dot")))
    fig.update_layout(title="Diagrama de calibración", xaxis_title="Probabilidad predicha",
                      yaxis_title="Frecuencia observada", height=380, xaxis_range=[0, 1], yaxis_range=[0, 1])
    return fig


def cd_diagram(avg_ranks: dict, cd: float, labels: dict) -> go.Figure:
    s = pd.Series(avg_ranks).sort_values()
    k = len(s)
    fig = go.Figure()
    fig.add_shape(type="line", x0=1, x1=k, y0=0, y1=0, line=dict(color=NAVY, width=2))
    for r in range(1, k + 1):
        fig.add_annotation(x=r, y=0, text=str(r), showarrow=False, yshift=-14, font=dict(size=11, color=SLATE))
    for i, (m, r) in enumerate(s.items()):
        y = 0.35 + 0.22 * (i % 4)
        fig.add_trace(go.Scatter(x=[r, r], y=[0, y], mode="lines", line=dict(color=SLATE, width=1), showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[r], y=[y], mode="markers+text", text=[f"{labels.get(m, m)} ({r:.2f})"],
                                 textposition="top center", marker=dict(color=BLUE, size=8), showlegend=False,
                                 hovertemplate=f"{labels.get(m, m)}: rango promedio {r:.2f}<extra></extra>"))
    x0 = s.iloc[0]
    fig.add_shape(type="line", x0=x0, x1=x0 + cd, y0=-0.35, y1=-0.35, line=dict(color=AMBER, width=4))
    fig.add_annotation(x=x0 + cd / 2, y=-0.35, text=f"Diferencia crítica de Nemenyi = {cd:.2f}", showarrow=False,
                       yshift=-14, font=dict(size=11, color=AMBER))
    fig.update_layout(title="Diagrama de diferencia crítica (rango 1 = mejor)", height=360,
                      xaxis=dict(range=[0.5, k + 0.5], visible=False), yaxis=dict(range=[-0.7, 1.3], visible=False))
    return fig


def convergence_figure(history: list) -> go.Figure:
    h = np.asarray(history)
    fig = go.Figure(go.Scatter(x=np.arange(len(h)), y=h - h.min() + 1e-12, mode="lines+markers",
                               line=dict(color=VIOLET), marker=dict(size=4),
                               hovertemplate="iteración %{x}: F − F* = %{y:.2e}<extra></extra>"))
    fig.update_layout(title="Convergencia del descenso de espejo (objetivo menos su mínimo)", xaxis_title="Iteración",
                      yaxis_title="F(wₜ) − F(w*)", yaxis_type="log", height=340)
    return fig


def relative_heatmap(table: pd.DataFrame, row_labels: dict, col_labels: dict) -> go.Figure:
    """Error de cada modelo relativo al de ALEE (1 = igual que ALEE), por conjunto de datos."""
    rel = table.div(table["ALEE"], axis=0)
    fig = go.Figure(go.Heatmap(
        z=rel.values, x=[col_labels.get(c, c) for c in rel.columns], y=[row_labels.get(r, r) for r in rel.index],
        text=table.round(3).values, texttemplate="%{text}", textfont=dict(size=10),
        colorscale=[[0, "#101B32"], [1, VIOLET]], zmin=1, zmax=max(2.0, float(np.nanmax(rel.values))),
        colorbar=dict(title="Error / ALEE"),
        hovertemplate="%{y} · %{x}<br>RMSE %{text}<br>%{z:.2f} veces el de ALEE<extra></extra>"))
    fig.update_layout(title="RMSE medio por pliegue en los nueve conjuntos (color: veces el error de ALEE)",
                      height=460, yaxis=dict(autorange="reversed"), margin=dict(l=110, r=20, t=60, b=80))
    return fig


def ablation_figure(abl: pd.DataFrame, order, dataset_label: str) -> go.Figure:
    """RMSE de cada variante de ALEE en un conjunto; las barras rayadas difieren con significancia de ALEE completo."""
    d = abl.set_index("variante").reindex([v for v in order if v in set(abl["variante"])])
    sig = d["dm_p_holm"].fillna(1) < 0.05
    colors = [VIOLET if v == "ALEE completo" else (AMBER if s else SLATE) for v, s in zip(d.index, sig)]
    fig = go.Figure(go.Bar(
        x=d["RMSE"], y=d.index, orientation="h", marker=dict(color=colors, pattern_shape=["/" if s else "" for s in sig]),
        error_x=dict(type="data", array=d["rmse_sd"], color=NAVY, thickness=1),
        customdata=np.c_[d["dm_p_holm"].fillna(np.nan)],
        hovertemplate="%{y}<br>RMSE %{x:.3f}<br>p (Holm) frente a ALEE completo: %{customdata[0]:.3g}<extra></extra>"))
    fig.update_layout(title=f"Ablación en {dataset_label} (rayado ámbar: diferencia significativa, Holm p < 0.05)",
                      xaxis_title="RMSE medio por pliegue (pp)", height=380, yaxis=dict(autorange="reversed"),
                      margin=dict(l=140, r=20, t=60, b=50))
    return fig


def regime_figure(reg: pd.DataFrame, labels: dict, dataset_label: str) -> go.Figure:
    """RMSE por tercil de volatilidad reciente para cada modelo en un conjunto."""
    fig = go.Figure()
    order = ["baja", "media", "alta"]
    for i, (m, g) in enumerate(reg.groupby("modelo", sort=False)):
        g = g.set_index("volatilidad").reindex(order)
        is_alee = m == "ALEE"
        fig.add_trace(go.Scatter(
            x=["Baja", "Media", "Alta"], y=g["rmse"], name=labels.get(m, m), mode="lines+markers",
            line=dict(color=VIOLET if is_alee else SLATE, width=3 if is_alee else 1, dash="solid" if is_alee else DASHES[i % 5]),
            marker=dict(symbol="diamond" if is_alee else SYMBOLS[i % 5], size=10 if is_alee else 6),
            opacity=1 if is_alee else 0.6,
            hovertemplate=f"{labels.get(m, m)}<br>volatilidad %{{x}}: RMSE %{{y:.3f}}<extra></extra>"))
    fig.update_layout(title=f"Error por régimen de volatilidad en {dataset_label}", xaxis_title="Volatilidad reciente (tercil)",
                      yaxis_title="RMSE (pp)", height=400)
    return fig
