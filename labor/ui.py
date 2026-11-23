"""Componentes visuales reutilizables."""
import dash_bootstrap_components as dbc
from dash import ctx, dcc, html
from dash.exceptions import PreventUpdate

from . import catalog as C
from . import engine as E

N = C.num
KIND = {"obs": "Dato observado", "est": "Dato estimado", "calc": "Variación calculada", "ref": "Referencia"}


def graph(fig=None, id=None, cls="", **kw):
    cfg = {"displaylogo": False, "displayModeBar": False, "responsive": True}
    kw.setdefault("style", {"height": f"{fig.layout.height}px"} if fig is not None and fig.layout.height else None)
    args = dict(config=cfg, className=f"li-graph {cls}", **kw)
    if fig is not None:
        args["figure"] = fig
    if id:
        args["id"] = id
    return dcc.Graph(**args)


def panel(title, children, sub=None, cls="", tools=None):
    head = html.Div([html.Div([html.H3(title), html.P(sub, className="li-sub") if sub else None]), tools], className="li-panel-head")
    return html.Section([head, html.Div(children, className="li-panel-body")], className=f"li-panel reveal {cls}")


def head(num, kicker, title, accent, lead):
    """Encabezado de sección: número en contorno, antetítulo, titular con remate y bajada."""
    return html.Header([
        html.Div(num, className="li-ghost", **{"aria-hidden": "true"}),
        html.Div([html.Span(className="li-dash"), f"{num}  ·  {kicker}"], className="li-kicker"),
        html.H1([html.Span(title + " "), html.Span(accent, className="li-accent")], className="li-title split"),
        html.P(lead, className="li-lead"),
    ], className="li-head")


def block(tag, title, lead=None):
    return html.Div([html.Span(tag, className="li-tag"), html.H2(title), html.P(lead) if lead else None], className="li-block reveal")


def kpi(label, value, kind, unit="", foot=None, dec=None, accent=""):
    """Tarjeta de indicador. Si ``value`` es número, el contador se anima."""
    if isinstance(value, (int, float)):
        val = html.Span(N(value, dec if dec is not None else 2, kind == "calc"), className="num",
                        **{"data-count": f"{value:.6f}", "data-dec": str(dec if dec is not None else 2), "data-sign": "1" if kind == "calc" else "0"})
    else:
        val = html.Span(value, className="num txt")
    return html.Div([
        html.Div([html.Span(label, className="k-label"), html.Span(KIND[kind], className=f"chip {kind}")], className="k-top"),
        html.Div([val, html.Span(unit, className="unit") if unit else None], className=f"k-val {accent or kind}"),
        html.Div(foot, className="k-foot") if foot else None,
    ], className="li-kpi reveal")


def kpis(*items):
    return html.Div(items, className="li-kpis")


def reading(obs, finding, why):
    cols = [("¿Qué observamos?", obs), ("¿Cuál es el hallazgo?", finding), ("¿Por qué importa para el desempleo?", why)]
    return html.Div([html.Div([html.H4(t), html.P(x)]) for t, x in cols], className="li-reading")


def note(children, kind="info", title=None):
    return html.Div([html.B(title + " ") if title else None, html.Span(children)], className=f"li-note {kind}", role="note")


def seg(id, options, value, label=None):
    return html.Div([html.Span(label, className="seg-label") if label else None,
                     dbc.RadioItems(id=id, options=[{"label": l, "value": v} for l, v in options], value=value, className="seg",
                                    inputClassName="btn-check", labelClassName="seg-btn", labelCheckedClassName="on")], className="seg-wrap")


def table(rows, cols, numeric=(), cls=""):
    """Tabla HTML. ``rows``: lista de dict; ``cols``: lista de (clave, título)."""
    return html.Div(html.Table([
        html.Thead(html.Tr([html.Th(t, className="n" if k in numeric else "") for k, t in cols])),
        html.Tbody([html.Tr([html.Td(r.get(k, ""), className="n" if k in numeric else "") for k, _ in cols], className=r.get("_cls", "")) for r in rows]),
    ], className="li-table"), className=f"li-table-wrap {cls}", tabIndex=0)


def grid(*children, cols="2"):
    return html.Div(list(children), className=f"li-grid g{cols}")


def source(c):
    m = C.COUNTRIES[c]
    return f"Fuente: {m['source']}. {m['definition']}."


def provenance(c):
    f = E.forecast(c)
    return html.Div([html.Span("Último dato ", className="pl"), html.Span(C.month_short(f["last_date"]), className="pv obs"),
                     html.Span("Pronóstico ", className="pl"), html.Span(C.month_short(f["target_date"]), className="pv est")], className="li-prov")


def loading(child, **kw):
    return dcc.Loading(child, type="dot", color="#28D8F0", delay_show=350, overlay_style={"visibility": "visible", "opacity": 0.45}, **kw)


def safe(fn):
    """Envuelve el contenido de una sección: si algo falla, muestra un mensaje legible en vez de un hueco."""
    import functools
    import logging

    @functools.wraps(fn)
    def inner(*a, **k):
        try:
            return fn(*a, **k)
        except PreventUpdate:
            raise
        except Exception:
            logging.getLogger("labor").exception("Error al construir %s", fn.__name__)
            msg = note("No se pudo calcular esta sección con los datos de este país. Pruebe con otro país o recargue la página.", "warn", "Algo falló.")
            try:
                n = len(ctx.outputs_list) if isinstance(ctx.outputs_list, list) else 1
            except Exception:
                n = 1
            return msg if n == 1 else [msg] + [None] * (n - 1)
    return inner
