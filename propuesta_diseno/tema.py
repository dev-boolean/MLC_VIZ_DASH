"""Paleta y tipografía de los gráficos de la propuesta.

Mantiene la misma semántica de color del original (observado, pronosticado,
ALEE, comparación, advertencia); solo cambia los tonos y la letra. Se aplica
antes de importar ``labor.figs``, que copia los colores al cargarse, y sobre
``src.dash_plots``, que los lee al dibujar.
"""
import plotly.graph_objects as go
import plotly.io as pio

from labor import theme as T

SANS = "IBM Plex Sans, Segoe UI, Arial, sans-serif"
SERIF = "IBM Plex Serif, Georgia, serif"

for k, v in dict(BG="#FFFFFF", BG2="#F6F7F8", CARD="#FFFFFF", ELEV="#F7F8F9", LINE="#D9DCE0", TEXT="#1B1F24", MUTED="#6B7280",
                 GRID="#ECEEF1", NAVY="#0F2A47", OBS="#3A4552", PRED="#1D4E89", OCHRE="#A8742A", ALERT="#B4441F",
                 DIM="#A3AAB4", GREEN="#4F7A5A").items():
    setattr(T, k, v)
T.BLUE, T.CYAN, T.VIOLET, T.ORANGE, T.RED = T.OBS, T.PRED, T.NAVY, T.OCHRE, T.ALERT
T.SERIES = [T.NAVY, T.PRED, T.OCHRE, T.OBS, T.ALERT, "#6B5B8C", "#4F7A5A", "#7A5E45", "#5C6B7A"]
T.SANS, T.MONO, T.SERIF = SANS, SANS, SERIF

_eje = dict(gridcolor=T.GRID, gridwidth=1, zeroline=False, linecolor=T.TEXT, linewidth=0.8, showline=True, ticks="outside",
            ticklen=4, tickcolor=T.TEXT, tickfont=dict(family=SANS, size=12, color=T.MUTED),
            title=dict(font=dict(size=12.5, color=T.MUTED), standoff=10), automargin=True)
pio.templates["li"] = go.layout.Template(layout=dict(
    font=dict(family=SANS, size=13, color=T.TEXT), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=T.SERIES,
    separators=",.", margin=dict(l=52, r=20, t=18, b=44),
    hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=T.LINE, font=dict(family=SANS, size=12.5, color=T.TEXT)),
    legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(size=12.5, color=T.MUTED), bgcolor="rgba(0,0,0,0)"),
    xaxis=_eje, yaxis={**_eje, "showline": False, "ticks": ""}, hovermode="closest",
    annotationdefaults=dict(font=dict(color=T.MUTED, size=12)),
))
pio.templates.default = "li"

from src import dash_plots as P  # noqa: E402

P.NAVY, P.BLUE, P.VIOLET, P.SLATE, P.AMBER = T.NAVY, T.OBS, T.PRED, "#6B7280", T.OCHRE
P.PALETTE = [P.BLUE, P.VIOLET, P.AMBER, P.SLATE, P.NAVY]
