"""Paleta y plantilla común de Plotly.

Semántica: azul = histórico, cian = predicción, violeta = modelo, naranja =
incertidumbre y advertencias, gris = referencia. Verde y rojo no se usan para
subidas o bajadas del desempleo, que no son buenas o malas por sí solas.
"""
import plotly.graph_objects as go
import plotly.io as pio

BG, BG2, CARD, ELEV, LINE = "#070B18", "#0C1426", "#101B32", "#16243A", "#26364D"
TEXT, MUTED = "#F3F7FC", "#A7B5C9"
BLUE, CYAN, VIOLET, GREEN, ORANGE, RED = "#3984FF", "#28D8F0", "#9565FF", "#24D69A", "#FFB547", "#FF6178"
DIM = "#6B7C96"
SERIES = [BLUE, CYAN, VIOLET, ORANGE, GREEN, RED, "#8FB8FF", "#C9A9FF", "#F3F7FC"]
DASHES = ["solid", "solid", "solid", "dash", "dash", "dash", "dot", "dot", "dot"]
SANS = "Inter, Segoe UI, Arial, sans-serif"
MONO = "JetBrains Mono, Consolas, monospace"


def rgba(h, a):
    h = h.lstrip("#")
    return f"rgba({int(h[:2], 16)},{int(h[2:4], 16)},{int(h[4:], 16)},{a})"


_axis = dict(gridcolor=rgba(LINE, 0.55), gridwidth=1, zeroline=False, linecolor=LINE, tickfont=dict(family=MONO, size=12, color=MUTED),
             title=dict(font=dict(size=13, color=MUTED), standoff=10), automargin=True, ticks="")
pio.templates["li"] = go.layout.Template(layout=dict(
    font=dict(family=SANS, size=13, color=TEXT), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=SERIES,
    separators=",.", margin=dict(l=52, r=20, t=18, b=44), hoverlabel=dict(bgcolor=ELEV, bordercolor=LINE, font=dict(family=SANS, size=13, color=TEXT)),
    legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(size=12, color=MUTED), bgcolor="rgba(0,0,0,0)"),
    xaxis=_axis, yaxis=_axis, hovermode="closest",
    annotationdefaults=dict(font=dict(color=MUTED, size=12)),
))
pio.templates.default = "li"


def base(height=380, **kw) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(height=height, **kw)
    return fig


def empty(msg, height=300):
    fig = base(height)
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=14, color=MUTED))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig
