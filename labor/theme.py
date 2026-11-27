"""Paleta y plantilla común de Plotly (estilo académico claro).

Semántica, la misma en toda la plataforma:
  OBS   azul acero  -> datos observados (historia publicada)
  PRED  verde azulado -> pronóstico de ALEE y su intervalo del 90 %
  NAVY  azul marino -> ALEE como modelo (barras, rangos, matrices)
  OCHRE ocre        -> modelos de comparación, referencias y avisos
  ALERT rojo apagado -> atípicos y advertencias
  DIM   gris        -> el resto de modelos y elementos de contexto
Verde y rojo no se usan para subidas o bajadas del desempleo, que no son
buenas o malas por sí solas.
"""
import plotly.graph_objects as go
import plotly.io as pio

# Superficies y texto
BG, BG2, CARD, ELEV, LINE = "#F6F7F5", "#EEF1F0", "#FFFFFF", "#F3F5F4", "#E1E6E8"
TEXT, MUTED = "#172536", "#687585"
GRID = "#ECEFF1"

# Significado
NAVY, OBS, PRED, OCHRE, ALERT, DIM = "#183B56", "#47799D", "#258477", "#C18B43", "#B85C5C", "#9AA5B1"

# Nombres anteriores, conservados para no duplicar figuras: ahora apuntan al significado.
BLUE, CYAN, VIOLET, ORANGE, RED = OBS, PRED, NAVY, OCHRE, ALERT
GREEN = "#5E8C6A"

# Nueve países: tonos sobrios y distinguibles, más el tipo de línea.
SERIES = [NAVY, PRED, OCHRE, OBS, ALERT, "#7A6A9C", "#5E8C6A", "#8A6B4F", "#5C6B7A"]
DASHES = ["solid", "solid", "solid", "dash", "dash", "dash", "dot", "dot", "dot"]
SANS = "Inter, Segoe UI, Arial, sans-serif"
MONO = "JetBrains Mono, Consolas, monospace"
SERIF = "Source Serif 4, Georgia, serif"


def rgba(h, a):
    h = h.lstrip("#")
    return f"rgba({int(h[:2], 16)},{int(h[2:4], 16)},{int(h[4:], 16)},{a})"


_axis = dict(gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=LINE, showline=True, tickfont=dict(family=MONO, size=12, color=MUTED),
             title=dict(font=dict(size=13, color=MUTED), standoff=10), automargin=True, ticks="")
pio.templates["li"] = go.layout.Template(layout=dict(
    font=dict(family=SANS, size=13, color=TEXT), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=SERIES,
    separators=",.", margin=dict(l=52, r=20, t=18, b=44), hoverlabel=dict(bgcolor=CARD, bordercolor=LINE, font=dict(family=SANS, size=13, color=TEXT)),
    legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(size=12.5, color=MUTED), bgcolor="rgba(0,0,0,0)"),
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
