"""Figuras de la plataforma."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.explain import acf, pacf

from . import catalog as C
from . import engine as E
from . import stats as S
from .theme import ALERT, BLUE, CARD, CYAN, DASHES, DIM, ELEV, LINE, MONO, MUTED, NAVY, OCHRE, ORANGE, PRED, SERIES, TEXT, VIOLET, base, rgba

N = C.num
EXPERT_COLORS = {"persistencia": DIM, "nucleo": BLUE, "estacional": OCHRE}
BAND = rgba(PRED, 0.16)  # intervalo del 90 %: mismo tono que el pronóstico, más claro


def _dates(idx):
    return [C.month_long(d) for d in idx]


def _hline(fig, y, text=None, color=DIM, dash="dot", **kw):
    fig.add_hline(y=y, line=dict(color=color, width=1, dash=dash), annotation_text=text,
                  annotation_font=dict(size=11, color=MUTED), **kw)


# ------------------------------------------------------------------ pronóstico
def forecast_main(c, months=120, height=430, band=True, posterior=False):
    """Serie observada, corte y segmento del mes siguiente. El segmento predictivo mide un mes: no se extiende."""
    f, s = E.forecast(c), E.series(c)
    s = s if not months else s.iloc[-months:]
    fig = base(height)
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name="Observado", line=dict(color=BLUE, width=2.4),
                             customdata=_dates(s.index), hovertemplate="%{customdata}<br><b>%{y:.2f} %</b> observado<extra></extra>"))
    x0, x1 = f["last_date"], f["target_date"]
    if band:
        fig.add_trace(go.Scatter(x=[x0, x1, x1, x0], y=[f["observed"], f["hi"], f["lo"], f["observed"]], fill="toself", mode="lines",
                                 line=dict(width=0), fillcolor=BAND, name="Intervalo del 90 %", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[x1, x1], y=[f["lo"], f["hi"]], mode="lines+markers", line=dict(color=PRED, width=1.6),
                                 marker=dict(symbol="line-ew", size=12, line=dict(color=PRED, width=1.6)), showlegend=False,
                                 hovertemplate="Intervalo del 90 %: %{y:.2f} %<extra></extra>"))
    fig.add_trace(go.Scatter(x=[x0, x1], y=[f["observed"], f["pred"]], mode="lines", name="Pronóstico de ALEE", showlegend=False,
                             line=dict(color=PRED, width=2.6), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[x0], y=[f["observed"]], mode="markers", marker=dict(color=BLUE, size=9, line=dict(color=CARD, width=2)),
                             showlegend=False, customdata=[C.month_long(x0)],
                             hovertemplate="Último dato: %{customdata}<br><b>%{y:.2f} %</b> observado<extra></extra>"))
    fig.add_trace(go.Scatter(x=[x1], y=[f["pred"]], mode="markers", name="Pronóstico de ALEE",
                             marker=dict(color=PRED, size=13, symbol="diamond", line=dict(color=CARD, width=1.5)),
                             customdata=[C.month_long(x1)], hovertemplate="%{customdata}<br><b>%{y:.2f} %</b> pronosticado<extra></extra>"))
    later = E.later_observations(c) if posterior else None
    xmax = x1
    if later is not None:
        fig.add_trace(go.Scatter(x=later["fecha"], y=later["tasa_desempleo_nacional"], mode="markers", name="Publicado después (no usado)",
                                 marker=dict(color=CARD, size=9, line=dict(color=OCHRE, width=1.8)),
                                 customdata=_dates(later["fecha"]), hovertemplate="%{customdata}<br>%{y:.1f} % publicado después del corte<extra></extra>"))
        xmax = later["fecha"].max()
    fig.add_vline(x=x0, line=dict(color=MUTED, width=1, dash="dot"))
    fig.add_annotation(x=x0, y=1, yref="paper", text="Último dato", showarrow=False, xanchor="right", xshift=-8, yanchor="top",
                       font=dict(family=MONO, size=11, color=MUTED))
    span = (s.index[-1] - s.index[0]).days
    fig.add_annotation(x=x1, y=f["pred"], text=f"<b>{N(f['pred'])} %</b>", showarrow=False, xanchor="left", xshift=12,
                       font=dict(family=MONO, size=15, color=PRED))
    pad = pd.Timedelta(days=max(40, int(span * 0.085)))
    ys = list(s.values) + [f["lo"], f["hi"]] + (list(later["tasa_desempleo_nacional"]) if later is not None else [])
    m = (max(ys) - min(ys)) * 0.12
    fig.update_xaxes(range=[s.index[0] - pd.Timedelta(days=15), max(x1 + pad, xmax + pd.Timedelta(days=30))], tickformat="%Y" if span > 1100 else "%m/%Y")
    fig.update_yaxes(ticksuffix=" %", range=[max(0, min(ys) - m), max(ys) + m])
    fig.update_layout(margin=dict(l=56, r=14, t=16, b=34), showlegend=False)
    return fig


def sparkline(c, months=60, color=BLUE):
    s = E.series(c).iloc[-months:]
    f = E.forecast(c)
    fig = base(54, margin=dict(l=0, r=0, t=4, b=4), showlegend=False)
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", line=dict(color=color, width=1.6), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[f["last_date"], f["target_date"]], y=[f["observed"], f["pred"]], mode="lines+markers",
                             line=dict(color=CYAN, width=2), marker=dict(size=[0, 6], color=CYAN), hoverinfo="skip"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def votes(c, e=None, base_e=None, height=250):
    """Pronóstico de cada experto y pesos con los que se combinan."""
    e = e or E.forecast(c)
    names = list(C.EXPERT_LABELS)
    fig = base(height, margin=dict(l=150, r=90, t=8, b=36), showlegend=False)
    for k, n in enumerate(names):
        fig.add_trace(go.Scatter(x=[e["last"], e["experts"][k]], y=[C.EXPERT_LABELS[n]] * 2, mode="lines",
                                 line=dict(color=EXPERT_COLORS[n], width=2), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[e["experts"][k]], y=[C.EXPERT_LABELS[n]], mode="markers",
                                 marker=dict(size=10 + 30 * e["weights"][k], color=rgba(EXPERT_COLORS[n], 0.85), line=dict(color=CARD, width=1)),
                                 hovertemplate=f"{C.EXPERT_LABELS[n]}<br>propone %{{x:.2f}} %<br>peso {N(100 * e['weights'][k], 1)} %<extra></extra>"))
        fig.add_annotation(x=1.0, xref="paper", y=C.EXPERT_LABELS[n], text=f"{N(e['experts'][k])} %<br><span style='color:{MUTED}'>peso {N(100 * e['weights'][k], 0)} %</span>",
                           showarrow=False, xanchor="left", xshift=8, align="left", font=dict(family=MONO, size=12, color=TEXT))
    fig.add_vline(x=e["last"], line=dict(color=BLUE, width=1.5, dash="dot"))
    fig.add_vline(x=e["pred"], line=dict(color=PRED, width=2))
    fig.update_xaxes(ticksuffix=" %")
    fig.update_yaxes(tickfont=dict(family="Inter", size=13, color=TEXT), autorange="reversed", gridcolor="rgba(0,0,0,0)")
    return fig


def contribution(e, height=300):
    """Descomposición exacta: último dato + aporte de cada experto = pronóstico."""
    names = ["Último<br>dato", "Persis-<br>tencia", "Núcleo", "Estacional", "Pronóstico"]
    vals = [e["last"], *e["contrib"], e["pred"]]
    cum = np.cumsum(vals[:4])
    bases = [0, cum[0], cum[1], cum[2], 0]
    fig = base(height, showlegend=False, bargap=0.28)
    fig.add_trace(go.Bar(x=names, y=vals, base=bases, marker=dict(color=[BLUE, EXPERT_COLORS["persistencia"], EXPERT_COLORS["nucleo"], EXPERT_COLORS["estacional"], PRED]),
                         customdata=[f"{N(v)} %" if k in (0, 4) else f"{N(v, 2, True)} pp" for k, v in enumerate(vals)], hovertemplate="%{x}: %{customdata}<extra></extra>"))
    for k, v in enumerate(vals):
        fig.add_annotation(x=names[k], y=max(bases[k], bases[k] + v), text=f"{N(v)} %" if k in (0, 4) else f"{N(v, 2, True)} pp", showarrow=False, yshift=11,
                           font=dict(family=MONO, size=12, color=TEXT))
    fig.update_xaxes(tickangle=0, tickfont=dict(family="Inter", size=12, color=MUTED))
    for k in range(4):
        fig.add_shape(type="line", x0=k, x1=k + 1, y0=cum[k], y1=cum[k], line=dict(color=MUTED, width=1, dash="dot"))
    lo, hi = min(e["last"], e["pred"]), max(e["last"], e["pred"])
    m = max((hi - lo) * 0.45, hi * 0.04)
    fig.update_yaxes(range=[lo - m, hi + m], ticksuffix=" %")
    fig.update_layout(margin=dict(l=56, r=12, t=14, b=40))
    return fig


def track(c, n=36, height=360):
    """Pronósticos ya hechos frente a lo que ocurrió, fuera de muestra, con el intervalo disponible en cada origen."""
    d = E.recent_track(c, n)
    fig = base(height, hovermode="x unified")
    fig.add_trace(go.Scatter(x=d["fecha"], y=d["ALEE"] + d["hw"], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=d["fecha"], y=d["ALEE"] - d["hw"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=BAND,
                             name="Intervalo del 90 % (solo con errores anteriores)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d["fecha"], y=d["y_true"], mode="lines+markers", name="Observado", line=dict(color=BLUE, width=2.2), marker=dict(size=5),
                             hovertemplate="Observado %{y:.2f} %<extra></extra>"))
    fig.add_trace(go.Scatter(x=d["fecha"], y=d["ALEE"], mode="lines+markers", name="Pronóstico de ALEE", line=dict(color=PRED, width=2, dash="dot"), marker=dict(size=5, symbol="diamond"),
                             hovertemplate="Pronóstico %{y:.2f} %<extra></extra>"))
    out = d[~d["dentro"] & d["hw"].notna()]
    if len(out):
        fig.add_trace(go.Scatter(x=out["fecha"], y=out["y_true"], mode="markers", name="Fuera del intervalo", marker=dict(color=CARD, size=11, line=dict(color=ALERT, width=2)),
                                 hovertemplate="Fuera del intervalo<extra></extra>"))
    fig.update_yaxes(ticksuffix=" %")
    fig.update_xaxes(tickformat="%m/%Y")
    return fig


# ------------------------------------------------------------------------ EDA
def evolution(c, years=None, height=430, show_trend=True, marks=True):
    s = S.clip(c, years)
    fig = base(height, hovermode="x unified")
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name="Tasa observada", line=dict(color=BLUE, width=1.8), hovertemplate="%{y:.2f} %<extra>Observada</extra>"))
    if show_trend and len(s) >= 36:
        t = S.trend(s)
        fig.add_trace(go.Scatter(x=t.index, y=t.values, mode="lines", name="Tendencia (media centrada de 12 meses)", line=dict(color=NAVY, width=2.6),
                                 hovertemplate="%{y:.2f} %<extra>Tendencia</extra>"))
    if marks:
        _, _, _, an = S.anomalies(s)
        if len(an):
            fig.add_trace(go.Scatter(x=an.index, y=s.loc[an.index], mode="markers", name="Cambio atípico", marker=dict(color=ALERT, size=9, symbol="x"),
                                     customdata=[N(v, 2, True) for v in an.values], hovertemplate="Cambio de %{customdata} pp<extra>Atípico</extra>"))
        for i, tag in [(s.idxmax(), "Máximo"), (s.idxmin(), "Mínimo")]:
            fig.add_annotation(x=i, y=s[i], text=f"{tag}: {N(s[i], 1)} %<br>{C.month_short(i)}", showarrow=True, arrowhead=0, arrowcolor=MUTED, ax=0,
                               ay=-34 if tag == "Máximo" else 34, font=dict(size=11, color=TEXT), bgcolor=rgba(ELEV, 0.9), borderpad=4)
    fig.update_yaxes(ticksuffix=" %")
    r = s.max() - s.min()
    fig.update_yaxes(range=[max(0, s.min() - r * 0.18), s.max() + r * 0.2])
    fig.update_xaxes(tickformat="%Y")
    return fig


def seasonal_bars(s, height=360):
    d = S.seasonality(s)
    fig = base(height, showlegend=False)
    fig.add_trace(go.Bar(x=[C.MON[m - 1] for m in d["mes"]], y=d["media"], marker=dict(color=[BLUE if v >= 0 else rgba(BLUE, 0.45) for v in d["media"]]),
                         error_y=dict(type="data", array=d["h"], color=TEXT, thickness=1.2, width=4),
                         customdata=np.c_[100 * d["sube"], d["n"]],
                         hovertemplate="%{x}: %{y:+.2f} pp<br>sube en el %{customdata[0]:.0f} % de %{customdata[1]:.0f} años<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_yaxes(title="Cambio frente al mes anterior (pp)")
    return fig


def month_heat(s, height=360):
    t = s.groupby([s.index.year, s.index.month]).mean().unstack()
    z = t.sub(t.mean(axis=1), axis=0)
    lim = float(np.nanquantile(np.abs(z.values), 0.98))
    fig = base(height, margin=dict(l=52, r=10, t=10, b=34))
    fig.add_trace(go.Heatmap(z=z.values, x=[C.MON[m - 1] for m in t.columns], y=t.index, zmin=-lim, zmax=lim, customdata=t.values,
                             colorscale=[[0, BLUE], [0.5, CARD], [1, OCHRE]], colorbar=dict(title=dict(text="pp frente a<br>su año", font=dict(size=11)), thickness=10, outlinewidth=0, tickfont=dict(family=MONO, size=11, color=MUTED)),
                             hovertemplate="%{x} %{y}: %{customdata:.2f} %<br>%{z:+.2f} pp frente a la media del año<extra></extra>"))
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    return fig


def distribution(v, label, height=360):
    from scipy.stats import gaussian_kde

    fig = make_subplots(rows=2, cols=1, row_heights=[0.2, 0.8], shared_xaxes=True, vertical_spacing=0.03)
    fig.add_trace(go.Box(x=v, marker=dict(color=ORANGE, size=5), line=dict(color=BLUE, width=1.6), fillcolor=rgba(BLUE, 0.2), name="", boxpoints="outliers",
                         hovertemplate="%{x:.2f} %<extra></extra>"), 1, 1)
    fig.add_trace(go.Histogram(x=v, nbinsx=34, histnorm="probability density", marker=dict(color=rgba(BLUE, 0.55), line=dict(color=BLUE, width=1)),
                               hovertemplate="%{x} %: densidad %{y:.3f}<extra></extra>"), 2, 1)
    xs = np.linspace(v.min(), v.max(), 200)
    fig.add_trace(go.Scatter(x=xs, y=gaussian_kde(v)(xs), mode="lines", line=dict(color=TEXT, width=2), hoverinfo="skip"), 2, 1)
    for val, tag in [(v.mean(), "media"), (v.median(), "mediana")]:
        fig.add_vline(x=val, line=dict(color=CYAN if tag == "media" else MUTED, width=1.5, dash="dash"), row=2, col=1)
    fig.update_layout(height=height, showlegend=False, bargap=0.04, margin=dict(l=52, r=14, t=8, b=44))
    fig.update_xaxes(ticksuffix=" %", row=2, col=1, title=label)
    fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_yaxes(title="Densidad", row=2, col=1)
    return fig


def corr_heat(df, labels, height=430, values=True):
    r = df.corr()
    names = [labels.get(k, k) for k in r.columns]
    fig = base(height, margin=dict(l=10, r=10, t=10, b=10))
    fig.add_trace(go.Heatmap(z=r.values, x=names, y=names, zmin=-1, zmax=1, colorscale=[[0, ORANGE], [0.5, CARD], [1, BLUE]],
                             text=[[N(v) for v in row] for row in r.values], texttemplate="%{text}" if values else None, textfont=dict(family=MONO, size=11),
                             colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(family=MONO, size=11, color=MUTED)),
                             hovertemplate="%{y}<br>%{x}<br>r = %{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", tickfont=dict(family="Inter", size=12))
    fig.update_xaxes(tickangle=-35, tickfont=dict(family="Inter", size=12))
    return fig


def scatter(x, y, years, lx, ly, sx="", sy="", height=430):
    fig = base(height)
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(size=7, color=years, colorscale=[[0, "#C9D6E0"], [0.6, BLUE], [1, NAVY]], opacity=0.85, showscale=True,
                             colorbar=dict(title=dict(text="Año", font=dict(size=11)), thickness=10, outlinewidth=0, tickfont=dict(family=MONO, size=11, color=MUTED), tickformat="d")),
                             customdata=[C.month_long(d) for d in x.index], hovertemplate="%{customdata}<br>" + lx + ": %{x:.2f}<br>" + ly + ": %{y:.2f}<extra></extra>"))
    b, a = np.polyfit(x, y, 1) if x.std() > 0 else (0, y.mean())
    xs = np.array([x.min(), x.max()])
    fig.add_trace(go.Scatter(x=xs, y=a + b * xs, mode="lines", line=dict(color=TEXT, width=1.6, dash="dash"), hoverinfo="skip", showlegend=False))
    fig.update_xaxes(title=lx, ticksuffix=sx)
    fig.update_yaxes(title=ly, ticksuffix=sy)
    fig.update_layout(showlegend=False)
    return fig


def acf_pair(s, height=330):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Nivel", "Cambio mensual"), horizontal_spacing=0.08)
    for j, x in enumerate([s.values, s.diff().dropna().values], start=1):
        a = acf(x, 25)
        b = 1.96 / np.sqrt(len(x))
        fig.add_trace(go.Bar(x=np.arange(1, 26), y=a[1:], marker=dict(color=[VIOLET if k in (12, 24) else BLUE for k in range(1, 26)]), hovertemplate="Rezago %{x}: %{y:.2f}<extra></extra>"), 1, j)
        for v in (b, -b):
            fig.add_hline(y=v, line=dict(color=ORANGE, width=1, dash="dash"), row=1, col=j)
    fig.update_layout(height=height, showlegend=False, margin=dict(l=48, r=10, t=34, b=44))
    fig.update_annotations(font=dict(size=13, color=MUTED))
    fig.update_xaxes(title="Rezago (meses)")
    fig.update_yaxes(range=[-0.6, 1])
    return fig


def anomaly_fig(s, height=360):
    d, lo, hi, an = S.anomalies(s)
    fig = base(height)
    fig.add_trace(go.Bar(x=d.index, y=d.values, marker=dict(color=rgba(BLUE, 0.65), line=dict(width=0)), name="Cambio mensual", customdata=_dates(d.index),
                         hovertemplate="%{customdata}<br>%{y:+.2f} pp<extra></extra>"))
    if len(an):
        fig.add_trace(go.Scatter(x=an.index, y=an.values, mode="markers", name="Fuera de las vallas", marker=dict(color=ALERT, size=10, symbol="x"), customdata=_dates(an.index),
                                 hovertemplate="%{customdata}<br>%{y:+.2f} pp<extra></extra>"))
    _hline(fig, lo, color=ORANGE, dash="dash")
    _hline(fig, hi, "Valla externa (3 RIC)", color=ORANGE, dash="dash", annotation_position="top left")
    fig.update_yaxes(title="Cambio mensual (pp)")
    fig.update_xaxes(tickformat="%Y")
    fig.update_layout(bargap=0)
    return fig


# --------------------------------------------------------------------- modelo
def weights_time(c, height=340):
    d = E.oof(c)
    fig = base(height, hovermode="x unified")
    for n in C.EXPERT_LABELS:
        fig.add_trace(go.Scatter(x=d["fecha"], y=d[f"w_{n}"], mode="lines", stackgroup="w", name=C.EXPERT_LABELS[n], line=dict(width=0.6, color=EXPERT_COLORS[n]),
                                 fillcolor=rgba(EXPERT_COLORS[n], 0.75), hovertemplate="%{y:.0%}<extra>" + C.EXPERT_LABELS[n] + "</extra>"))
    for k in range(2, 6):
        fig.add_vline(x=d.loc[d["pliegue"] == k, "fecha"].min(), line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    fig.update_xaxes(tickformat="%Y")
    return fig


def pred_vs_obs(c, other, height=400):
    d = E.oof(c)
    fig = base(height, hovermode="x unified")
    fig.add_trace(go.Scatter(x=d["fecha"], y=d["y_true"], mode="lines", name="Observado", line=dict(color=BLUE, width=2.2), hovertemplate="%{y:.2f} %<extra>Observado</extra>"))
    fig.add_trace(go.Scatter(x=d["fecha"], y=d[other], mode="lines", name=C.MODEL_LABELS[other], line=dict(color=OCHRE, width=1.4), hovertemplate="%{y:.2f} %<extra>" + C.MODEL_LABELS[other] + "</extra>"))
    fig.add_trace(go.Scatter(x=d["fecha"], y=d["ALEE"], mode="lines", name="ALEE", line=dict(color=PRED, width=1.8, dash="dot"), hovertemplate="%{y:.2f} %<extra>ALEE</extra>"))
    for k in range(2, 6):
        fig.add_vline(x=d.loc[d["pliegue"] == k, "fecha"].min(), line=dict(color=LINE, width=1))
    fig.update_yaxes(ticksuffix=" %")
    fig.update_xaxes(tickformat="%Y")
    return fig


def fold_errors(c, metric="RMSE", height=340):
    m = E.metrics()
    m = m[m["conjunto"] == c].pivot(index="pliegue", columns="modelo", values=metric)
    d = E.oof(c)
    per = [f"{g['fecha'].min().year}–{g['fecha'].max().year}" for _, g in d.groupby("pliegue")]
    best, _ = E.best_baseline(c)
    fig = base(height)
    for mod, col in [("ALEE", NAVY), (best, OCHRE)]:
        fig.add_trace(go.Bar(x=per, y=m[mod], name=C.MODEL_LABELS[mod], marker=dict(color=col), hovertemplate="%{x}: %{y:.3f}<extra>" + C.MODEL_LABELS[mod] + "</extra>"))
    fig.update_yaxes(title=f"{metric} (pp)" if metric in ("RMSE", "MAE") else metric)
    fig.update_layout(barmode="group", bargap=0.3)
    return fig


def model_bars(c, metric="RMSE", height=400):
    row = E.metric_table(metric).loc[c].sort_values(ascending=metric != "R2")[::-1]
    fig = base(height, showlegend=False, margin=dict(l=170, r=56, t=8, b=40))
    fig.add_trace(go.Bar(x=row.values, y=[C.MODEL_LABELS[m] for m in row.index], orientation="h", marker=dict(color=[VIOLET if m == "ALEE" else rgba(BLUE, 0.45) for m in row.index]),
                         text=[N(v, 3) for v in row.values], textposition="outside", textfont=dict(family=MONO, size=12, color=TEXT), cliponaxis=False, hovertemplate="%{y}: %{text}<extra></extra>"))
    fig.update_yaxes(tickfont=dict(family="Inter", size=13, color=TEXT), gridcolor="rgba(0,0,0,0)")
    lo, hi = min(0, row.min()), max(row.max(), 0)
    fig.update_xaxes(range=[lo * 1.15, hi * 1.16])
    return fig


def permutation_fig(height=300):
    d = pd.read_csv(E.ART / "resultados" / "importancia_permutacion_alee.csv").head(8)[::-1]
    from src.dash_data import FEATURE_INFO

    fig = base(height, showlegend=False, margin=dict(l=190, r=40, t=8, b=44))
    fig.add_trace(go.Bar(x=d["aumento_rmse_medio"], y=[FEATURE_INFO.get(v, (v,))[0] for v in d["variable"]], orientation="h", marker=dict(color=VIOLET),
                         error_x=dict(type="data", array=d["aumento_rmse_sd"], color=TEXT, thickness=1.1, width=3), hovertemplate="%{y}: +%{x:.2f} pp de RMSE<extra></extra>"))
    fig.update_xaxes(title="Aumento del RMSE al permutar la variable (pp)")
    fig.update_yaxes(tickfont=dict(family="Inter", size=13, color=TEXT), gridcolor="rgba(0,0,0,0)")
    return fig


def state_map(c, height=380):
    """Dónde está el mes actual en el espacio de estados que usa ALEE para localizar."""
    f = E.features(c).dropna()
    vol, gap = f["td_nacional_rollstd3"], f["td_nacional_lag1"] - f["td_nacional_rollmean12"]
    e = E.forecast(c)
    fig = base(height)
    fig.add_trace(go.Scatter(x=vol, y=gap, mode="markers", name="Meses anteriores", marker=dict(size=6, color=rgba(BLUE, 0.45)), customdata=_dates(f.index),
                             hovertemplate="%{customdata}<br>volatilidad %{x:.2f} pp<br>brecha %{y:+.2f} pp<extra></extra>"))
    nb = e["neighbours"]
    fig.add_trace(go.Scatter(x=vol.loc[nb.index], y=gap.loc[nb.index], mode="markers", name="Los 8 meses que más pesan hoy", marker=dict(size=8 + 60 * nb.values, color=rgba(VIOLET, 0.8), line=dict(color=TEXT, width=1)),
                             customdata=[f"{C.month_long(d)}<br>peso {N(100 * w, 1)} %" for d, w in nb.items()], hovertemplate="%{customdata}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[vol.iloc[-1]], y=[gap.iloc[-1]], mode="markers", name="Mes actual", marker=dict(size=15, color=CYAN, symbol="diamond", line=dict(color=TEXT, width=1.5)), hovertemplate="Mes actual<extra></extra>"))
    fig.update_xaxes(title="Volatilidad de los últimos 3 meses (pp)")
    fig.update_yaxes(title="Brecha frente a la media de 12 meses (pp)")
    return fig


def ablation(c, height=360):
    d = E.result("ablacion")
    d = d[d["conjunto"] == c].set_index("variante")
    order = ["ALEE completo", "Sin localización", "Sin λ adaptativa", "Previo uniforme", "Promedio simple", "Solo persistencia", "Solo núcleo", "Solo estacional"]
    d = d.reindex(order)[::-1]
    sig = d["dm_p_holm"].fillna(1) < 0.05
    col = [VIOLET if v == "ALEE completo" else ORANGE if s else DIM for v, s in zip(d.index, sig)]
    fig = base(height, showlegend=False, margin=dict(l=140, r=56, t=8, b=44))
    fig.add_trace(go.Bar(x=d["RMSE"], y=d.index, orientation="h", marker=dict(color=col, pattern=dict(shape=["/" if s else "" for s in sig], fgcolor=CARD)),
                         text=[N(v, 3) for v in d["RMSE"]], textposition="outside", textfont=dict(family=MONO, size=12, color=TEXT), cliponaxis=False,
                         hovertemplate="%{y}: RMSE %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title="RMSE medio por pliegue (pp)", range=[0, d["RMSE"].max() * 1.16])
    fig.update_yaxes(tickfont=dict(family="Inter", size=13, color=TEXT), gridcolor="rgba(0,0,0,0)")
    return fig


def regimes(c, height=340):
    d = E.result("errores_por_regimen")
    d = d[(d["conjunto"] == c) & d["modelo"].isin(C.MODEL_LABELS)]
    best, _ = E.best_baseline(c)
    fig = base(height)
    for m, g in d.groupby("modelo", sort=False):
        g = g.set_index("volatilidad").reindex(["baja", "media", "alta"])
        key = m in ("ALEE", best)
        fig.add_trace(go.Scatter(x=["Baja", "Media", "Alta"], y=g["rmse"], mode="lines+markers", name=C.MODEL_LABELS[m], showlegend=key,
                                 line=dict(color=NAVY if m == "ALEE" else (OCHRE if m == best else rgba(DIM, 0.5)), width=3 if m == "ALEE" else 2 if key else 1),
                                 marker=dict(size=9 if key else 4), hovertemplate=C.MODEL_LABELS[m] + "<br>%{x}: %{y:.3f} pp<extra></extra>"))
    fig.update_yaxes(title="RMSE (pp)", rangemode="tozero")
    fig.update_xaxes(title="Volatilidad de los últimos 3 meses (tercil)")
    return fig


# --------------------------------------------------------------------- global
STATUS_COLORS = {"dev": NAVY, "eval": PRED}


def world_map(selected, height=390):
    """Mapa 2D: qué países tienen datos, ALEE propio y evaluación. El color indica el papel en el estudio."""
    fig = base(height, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    codes = {c: m["short"] for c, m in C.COUNTRIES.items()}
    for role, label in [("dev", "Desarrollo del diseño y evaluación"), ("eval", "Evaluación con el diseño congelado")]:
        cs = [c for c in C.COUNTRIES if (c == "colombia") == (role == "dev")]
        fig.add_trace(go.Choropleth(
            locations=[codes[c] for c in cs], z=[1] * len(cs), locationmode="ISO-3", customdata=cs, name=label,
            colorscale=[[0, STATUS_COLORS[role]], [1, STATUS_COLORS[role]]], showscale=False, zmin=0, zmax=1,
            marker=dict(line=dict(color=[NAVY if c == selected else CARD for c in cs], width=[2.6 if c == selected else 0.8 for c in cs]),
                        opacity=[1 if c == selected else 0.72 for c in cs]),
            text=[C.name(c) for c in cs],
            hovertemplate="<b>%{text}</b><br>Datos disponibles · ALEE propio entrenado · evaluado<br>" + label + "<extra></extra>"))
    fig.update_geos(projection_type="natural earth", showcountries=True, countrycolor="#D5DCE0", showcoastlines=False, showland=True, landcolor="#E9EDEC",
                    showocean=True, oceancolor=CARD, showlakes=False, showframe=False, lataxis_range=[-58, 84], bgcolor="rgba(0,0,0,0)")
    return fig


# --------------------------------------------------------------------- global (comparación)
def compare(countries, mode, years, height=440):
    fig = base(height, hovermode="x unified")
    for i, c in enumerate(countries):
        full = E.series(c)
        if mode == "z":
            v = (full - full.mean()) / full.std()
        elif mode == "yoy":
            v = full.diff(12)
        else:
            v = full
        v = v[(v.index.year >= years[0]) & (v.index.year <= years[1])]
        k = list(C.COUNTRIES).index(c)
        fig.add_trace(go.Scatter(x=v.index, y=v.values, mode="lines", name=C.name(c), line=dict(color=SERIES[k], width=2, dash=DASHES[k]),
                                 hovertemplate="%{y:.2f}<extra>" + C.name(c) + "</extra>"))
    if mode != "level":
        fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_yaxes(title={"level": "Tasa de desempleo (%)", "z": "Desviaciones estándar frente a su propia media", "yoy": "Cambio frente a 12 meses antes (pp)"}[mode])
    fig.update_xaxes(tickformat="%Y")
    return fig


def forecast_changes(height=380):
    cs = sorted(C.COUNTRIES, key=lambda c: E.forecast(c)["change"])
    fig = base(height, showlegend=False, margin=dict(l=120, r=30, t=8, b=44))
    for c in cs:
        f, hw = E.forecast(c), E.conformal(c)["hw"]
        fig.add_trace(go.Scatter(x=[f["change"] - hw, f["change"] + hw], y=[C.name(c)] * 2, mode="lines", line=dict(color=rgba(PRED, 0.35), width=6), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[f["change"]], y=[C.name(c)], mode="markers", marker=dict(size=12, color=PRED, symbol="diamond", line=dict(color=CARD, width=1.5)),
                                 hovertemplate=f"{C.name(c)}<br>{C.month_long(f['target_date'])}: %{{x:+.2f}} pp<br>intervalo ±{N(hw)} pp<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    fig.update_xaxes(title="Cambio pronosticado frente al último dato (pp)")
    fig.update_yaxes(tickfont=dict(family="Inter", size=13, color=TEXT), gridcolor="rgba(0,0,0,0)")
    return fig


def relative_heat(metric="RMSE", height=420):
    t = E.metric_table(metric)
    rel = t.div(t["ALEE"], axis=0)
    txt = [[N(v, 2 if abs(v) >= 10 else 3) for v in row] for row in t.values]
    fig = base(height, margin=dict(l=110, r=10, t=8, b=90))
    fig.add_trace(go.Heatmap(z=np.log2(rel.values), x=[C.MODEL_LABELS[m] for m in t.columns], y=[C.name(c) for c in t.index], text=txt, texttemplate="%{text}", textfont=dict(family=MONO, size=12),
                             customdata=rel.values, colorscale=[[0, "#FFFFFF"], [0.33, "#C9D6E0"], [1, "#5C7A93"]], zmin=0, zmax=3, showscale=False,
                             hovertemplate="%{y}, %{x}<br>" + metric + " %{text}<br>%{customdata:.2f} veces el de ALEE<extra></extra>"))
    fig.update_yaxes(autorange="reversed", tickfont=dict(family="Inter", size=13, color=TEXT))
    fig.update_xaxes(tickangle=-30, tickfont=dict(family="Inter", size=12))
    return fig


def reductions(height=360):
    t = E.metric_table()
    r = (1 - t["ALEE"] / t.drop(columns="ALEE").min(axis=1)).sort_values()
    fig = base(height, showlegend=False, margin=dict(l=120, r=60, t=8, b=40))
    fig.add_trace(go.Bar(x=100 * r.values, y=[C.name(c) for c in r.index], orientation="h", marker=dict(color=VIOLET), text=[f"{N(100 * v, 0)} %" for v in r.values], textposition="outside",
                         textfont=dict(family=MONO, size=13, color=TEXT), cliponaxis=False, customdata=[C.MODEL_LABELS[E.best_baseline(c)[0]] for c in r.index],
                         hovertemplate="%{y}: %{text} menos RMSE que %{customdata}<extra></extra>"))
    fig.update_xaxes(title="Reducción del RMSE frente al mejor modelo de comparación (%)", range=[0, 40])
    fig.update_yaxes(tickfont=dict(family="Inter", size=13, color=TEXT), gridcolor="rgba(0,0,0,0)")
    return fig


def cd_fig(height=300):
    ranks = E.metric_table().rank(axis=1).mean().sort_values()
    from src.stats_tests import critical_difference

    cd = critical_difference(len(ranks), len(C.COUNTRIES))
    fig = base(height, showlegend=False, margin=dict(l=170, r=30, t=8, b=44))
    fig.add_vrect(x0=1, x1=1 + cd, fillcolor=rgba(OCHRE, 0.12), line_width=0)
    fig.add_trace(go.Scatter(x=ranks.values, y=[C.MODEL_LABELS[m] for m in ranks.index], mode="markers", marker=dict(size=12, color=[NAVY if m == "ALEE" else OCHRE for m in ranks.index], line=dict(color=CARD, width=1)),
                             hovertemplate="%{y}: rango medio %{x:.2f}<extra></extra>"))
    fig.update_xaxes(title=f"Rango medio (1 = mejor). Diferencia crítica: {N(cd)}", range=[0.6, 10.4], dtick=1)
    fig.update_yaxes(autorange="reversed", tickfont=dict(family="Inter", size=13, color=TEXT))
    return fig
