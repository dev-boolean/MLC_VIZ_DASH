"""Paneles agregados a las páginas originales, sin quitarles nada.

* Pronóstico: «Anatomía del pronóstico» (qué dice cada experto, cuánto pesa y cuánto aporta)
  y «Bitácora» (cada mes reciente: lo que dijo el modelo, su intervalo y lo que pasó).
* Modelo: «Mapa de estados» (a qué meses del pasado se parece el mes actual).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, html

from labor import catalog as C
from labor import engine as E
from labor import theme as T
from labor import ui as U

N = C.num
EXPERTOS = ["Persistencia", "Núcleo del cambio", "Cambio estacional"]
NUEVO = "Lectura complementaria de los paneles anteriores"


def agregar(registro):
    """Añade un contenedor al final de las páginas Pronóstico y Modelo; sus contenidos dependen del país."""
    for p in registro.values():
        if p["path"] == "/forecast":
            p["layout"].children.append(html.Div(U.loading(html.Div(id="px-pronostico")), className="li-wrap px-extra"))
        elif p["path"] == "/modelo":
            p["layout"].children.append(html.Div(U.loading(html.Div(id="px-modelo")), className="li-wrap px-extra"))


# ------------------------------------------------------------------ pronóstico
def _recta(c, f):
    W, F = f["weights"], f["experts"]
    val = E.later_validation(c)
    real = val.dropna(subset=["pred"]) if len(val) else val
    xs = list(F) + [f["lo"], f["hi"]] + (list(real["obs"]) if len(real) else [])
    a, b = min(xs), max(xs)
    pad = (b - a) * 0.08 + 0.05
    fig = go.Figure()
    fig.add_shape(type="rect", x0=f["lo"], x1=f["hi"], y0=-0.22, y1=0.22, fillcolor=T.PRED, opacity=0.12, line_width=0)
    fig.add_shape(type="line", x0=a - pad, x1=b + pad, y0=0, y1=0, line=dict(color=T.TEXT, width=0.8))
    for j, i in enumerate(np.argsort(F)):
        fig.add_trace(go.Scatter(x=[F[i]], y=[0], mode="markers", showlegend=False,
                                 marker=dict(size=7 + 26 * W[i], color=T.NAVY if W[i] == W.max() else T.DIM, opacity=0.7),
                                 hovertemplate=f"{EXPERTOS[i]}: %{{x:.2f}} % · peso {W[i] * 100:.0f} %<extra></extra>"))
        fig.add_annotation(x=F[i], y=0.6 + 0.3 * (j % 2), text=f"{EXPERTOS[i]} {N(F[i])}", showarrow=False, bgcolor="#FFFFFF",
                           font=dict(size=12, color=T.MUTED))
    fig.add_shape(type="line", x0=f["pred"], x1=f["pred"], y0=-0.42, y1=0.42, line=dict(color=T.PRED, width=2.5))
    fig.add_annotation(x=f["pred"], y=-0.64, text=f"<b>ALEE {N(f['pred'])}</b>", showarrow=False, font=dict(size=12.5, color=T.PRED))
    for _, r in real.iterrows():
        der = r["obs"] >= f["pred"]
        fig.add_trace(go.Scatter(x=[r["obs"]], y=[-0.3], mode="markers", showlegend=False, marker=dict(symbol="triangle-up", size=11, color=T.ALERT),
                                 hovertemplate="Publicado después: %{x:.1f} %<extra></extra>"))
        fig.add_annotation(x=r["obs"], y=-0.64, text=f"Observado {N(r['obs'], 1)}", showarrow=False, font=dict(size=12, color=T.ALERT),
                           xanchor="left" if der else "right", xshift=8 if der else -8)
    fig.update_layout(height=200, margin=dict(l=14, r=14, t=6, b=30), yaxis=dict(visible=False, range=[-0.85, 1.05]),
                      xaxis=dict(range=[a - pad, b + pad], ticksuffix=" %", showgrid=False))
    return fig


def _bitacora(c):
    t = E.recent_track(c, 12)
    filas = [dict(mes=C.month_short(r["fecha"]).capitalize(), pred=r["ALEE"], obs=r["y_true"], hw=r["hw"], real=False) for _, r in t.iterrows()]
    val = E.later_validation(c)
    for _, r in (val.dropna(subset=["pred"]) if len(val) else val).iterrows():
        filas.append(dict(mes=C.month_short(r["objetivo"]).capitalize(), pred=r["pred"], obs=r["obs"], hw=(r["hi"] - r["lo"]) / 2, real=True))
    n = len(filas)
    ys = [n - 1 - i - (0.6 if r["real"] else 0) for i, r in enumerate(filas)]
    vals = [v for r in filas for v in (r["pred"] - r["hw"], r["pred"] + r["hw"], r["obs"])]
    a, b = min(vals), max(vals)
    fig = go.Figure()
    for r, y in zip(filas, ys):
        dentro = abs(r["obs"] - r["pred"]) <= r["hw"]
        fig.add_shape(type="rect", x0=r["pred"] - r["hw"], x1=r["pred"] + r["hw"], y0=y - 0.28, y1=y + 0.28, fillcolor=T.PRED,
                      opacity=0.26 if r["real"] else 0.13, line_width=0)
        fig.add_shape(type="line", x0=r["pred"], x1=r["pred"], y0=y - 0.36, y1=y + 0.36, line=dict(color=T.PRED, width=2))
        fig.add_trace(go.Scatter(x=[r["obs"]], y=[y], mode="markers", showlegend=False,
                                 marker=dict(size=8, color=T.OBS if dentro else "#FFFFFF", line=dict(color=T.OBS if dentro else T.ALERT, width=2)),
                                 hovertemplate=f"{r['mes']}: pronóstico {N(r['pred'])} %, observado {N(r['obs'])} %<extra></extra>"))
        err = r["obs"] - r["pred"]
        color = T.ALERT if r["real"] else T.TEXT
        for dx, txt, al in [(-280, r["mes"], "left"), (-160, N(r["pred"]), "right"), (-100, N(r["obs"]), "right"), (-28, N(err, 2, True), "right")]:
            fig.add_annotation(x=0, xref="paper", xshift=dx, y=y, text=txt, showarrow=False, xanchor=al,
                               font=dict(size=12.5, color=(T.MUTED if dentro else T.ALERT) if dx == -28 else color))
    top = max(ys) + 0.9
    for dx, txt, al in [(-280, "Mes", "left"), (-160, "Pronóstico", "right"), (-100, "Observado", "right"), (-28, "Error", "right")]:
        fig.add_annotation(x=0, xref="paper", xshift=dx, y=top, text=txt, showarrow=False, xanchor=al, font=dict(size=12, color=T.MUTED))
    if any(r["real"] for r in filas):
        fig.add_shape(type="line", xref="paper", x0=0, x1=1, y0=min(ys) + 0.55, y1=min(ys) + 0.55, line=dict(color=T.TEXT, width=0.8))
        fig.add_annotation(x=0, xref="paper", xshift=-280, y=min(ys) - 0.78, xanchor="left", showarrow=False,
                           text="Pronóstico real: hecho antes de que se publicara el dato", font=dict(size=12, color=T.ALERT))
    pad = (b - a) * 0.05
    fig.update_layout(height=80 + 27 * n, margin=dict(l=290, r=14, t=8, b=32),
                      yaxis=dict(visible=False, range=[min(ys) - 1.2, top + 0.5]),
                      xaxis=dict(range=[a - pad, b + pad], ticksuffix=" %", showgrid=True))
    dentro = sum(abs(r["obs"] - r["pred"]) <= r["hw"] for r in filas if not r["real"])
    return fig, dentro, sum(not r["real"] for r in filas)


@callback(Output("px-pronostico", "children"), Input("pais", "value"))
@U.safe
def pronostico(c):
    c = c or C.DEFAULT
    f = E.forecast(c)
    W, F = f["weights"], f["experts"]
    dom = int(np.argmax(W))
    antes, despues = C.MONTHS[f["last_date"].month - 1], C.MONTHS[f["target_date"].month - 1]
    que = ["Repetir el último dato observado", "Cambio mensual de los meses con un estado parecido",
           f"Cambio medio de {antes} a {despues} en los años anteriores"]
    filas = [{"a": EXPERTOS[i], "b": que[i], "c": f"{N(F[i])} %", "d": f"{N(W[i] * 100, 1)} %",
              "e": html.Span(className="px-bar" + (" on" if i == dom else ""), style={"width": f"{max(W[i] * 160, 3):.0f}px"}),
              "f": N(W[i] * F[i]), "_cls": "hl" if i == dom else ""} for i in range(3)]
    filas.append({"a": html.B("Pronóstico"), "b": "Suma de los aportes", "c": "", "d": "100 %", "e": "", "f": html.B(f"{N(f['pred'])} %")})
    anatomia = U.panel(f"Anatomía del pronóstico: ¿de dónde sale el {N(f['pred'])} %?", html.Div([
        U.table(filas, [("a", "Experto"), ("b", "Qué propone"), ("c", "Pronóstico del experto"), ("d", "Peso"), ("e", ""), ("f", "Aporte (peso × pronóstico)")],
                numeric=("c", "d", "f")),
        U.graph(_recta(c, f)),
        U.legend(("Expertos (tamaño: peso)", "model"), ("Pronóstico de ALEE", "est"), ("Intervalo del 90 %", "unc"),
                 *([("Dato publicado después del corte", "later")] if E.later_observations(c) is not None else [])),
        html.P([f"En {C.month_long(f['target_date'])} pesa sobre todo el experto «{EXPERTOS[dom].lower()}» ({N(W[dom] * 100, 0)} %). "
                f"Sin localizar, los pesos serían {' / '.join(N(p * 100, 0) for p in f['prior'])} %; la diferencia viene de los meses parecidos "
                f"(tamaño efectivo de la muestra local: {N(f['n_eff'], 1)}, frente a {N(f['n_ref'], 1)} de referencia). "
                "Es la misma descomposición del panel «Qué empuja el pronóstico», expresada en niveles en vez de cambios."], className="li-cap"),
    ]), sub=NUEVO, cls="px-panel")

    fig, dentro, total = _bitacora(c)
    bitacora = U.panel("Bitácora de los últimos doce pronósticos", html.Div([
        U.graph(fig),
        U.legend(("Intervalo del 90 % disponible en su momento", "unc"), ("Pronóstico", "est"), ("Observado", "obs")),
        html.P(f"{dentro} de {total} datos cayeron dentro del intervalo. Los círculos vacíos marcan los meses que quedaron por fuera. "
               "Es la misma información del «Historial reciente fuera de muestra», ordenada como un registro de aciertos y errores.",
               className="li-cap"),
    ]), sub=NUEVO, cls="px-panel")
    return [U.block("Lectura del pronóstico", "El número, desarmado y puesto a prueba"), anatomia, bitacora]


# ---------------------------------------------------------------------- modelo
def _estados(c):
    m, X = E.model(c), E.features(c)
    ok = X.notna().all(axis=1)
    Z = m._state(X[ok].values)
    f = E.forecast(c)
    d = pd.DataFrame({"fecha": X.index[ok][m.oof_rows_], "vol": Z[m.oof_rows_, 0], "brecha": Z[m.oof_rows_, 1], "peso": f["local"]})
    return d, m._state(X.iloc[[-1]].values)[0], f


@callback(Output("px-modelo", "children"), Input("pais", "value"))
@U.safe
def modelo(c):
    c = c or C.DEFAULT
    d, act, f = _estados(c)
    hp = E.meta(c)["hp"]
    xhi = max(np.quantile(d["vol"], 0.93), act[0]) * 1.08
    ylo = min(np.quantile(d["brecha"], 0.02), act[1]) - 0.3
    yhi = max(np.quantile(d["brecha"], 0.95), act[1]) + 0.3
    fuera = int(((d["vol"] > xhi) | (d["brecha"] < ylo) | (d["brecha"] > yhi)).sum())
    top = d.nlargest(8, "peso")
    resto = d.drop(top.index)
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, x1=xhi, y0=0, y1=0, line=dict(color=T.DIM, width=0.8, dash="dot"))
    fig.add_trace(go.Scatter(x=resto["vol"], y=resto["brecha"], mode="markers", showlegend=False, marker=dict(size=5, color="#BFC5CD"),
                             customdata=resto["fecha"].dt.strftime("%Y-%m"), hovertemplate="%{customdata}: volatilidad %{x:.2f}, brecha %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=top["vol"], y=top["brecha"], mode="markers", showlegend=False,
                             marker=dict(size=6 + top["peso"] * 30 / top["peso"].max(), color=T.NAVY, opacity=0.7, line=dict(color="#FFFFFF", width=1)),
                             customdata=np.c_[top["fecha"].dt.strftime("%Y-%m"), top["peso"] * 100],
                             hovertemplate="%{customdata[0]}: peso %{customdata[1]:.1f} %<extra></extra>"))
    fig.add_trace(go.Scatter(x=[act[0]], y=[act[1]], mode="markers", showlegend=False,
                             marker=dict(symbol="diamond", size=14, color=T.ALERT, line=dict(color="#FFFFFF", width=1.5)), hovertemplate="Mes actual<extra></extra>"))
    fig.add_annotation(x=act[0], y=act[1], text=C.month_long(f["last_date"]).capitalize(), showarrow=True, arrowhead=0, arrowwidth=0.7,
                       arrowcolor=T.ALERT, ax=70, ay=-60, font=dict(size=12.5, color=T.ALERT), bgcolor="#FFFFFF")
    fig.update_layout(height=430, margin=dict(l=60, r=14, t=10, b=50),
                      xaxis=dict(range=[0, xhi], title="Volatilidad de los últimos 3 meses (pp)", showgrid=True),
                      yaxis=dict(range=[ylo, yhi], title="Brecha frente a la media de 12 meses (pp)"))
    recientes = int((top["fecha"] >= f["last_date"] - pd.DateOffset(months=12)).sum())
    filas = [{"a": C.month_short(r["fecha"]).capitalize(), "b": f"{N(r['vol'])}", "c": f"{N(r['brecha'], 2, True)}", "d": f"{N(r['peso'] * 100, 1)} %"}
             for _, r in top.iterrows()]
    mapa = U.panel(f"Mapa de estados: ¿a qué meses se parece {C.month_long(f['last_date'])}?", html.Div([
        U.grid(html.Div([U.graph(fig),
                         U.legend(("Mes actual", "px-act"),("Ocho meses con más peso (tamaño: peso)", "px-top"), ("Resto de los meses de entrenamiento", "none"))]),
               html.Div([U.table(filas, [("a", "Mes"), ("b", "Volatilidad"), ("c", "Brecha"), ("d", "Peso")], numeric=("b", "c", "d")),
                         html.P([f"De los ocho meses con más peso, {recientes} son de los últimos doce meses. El peso combina el parecido de estado "
                                 f"(ancho del núcleo h = {N(hp['bandwidth'])}) y la cercanía en el tiempo (vida media de {N(hp['half_life'], 1)} meses), "
                                 "como en la ecuación de los pesos de localización de la sección 3.2."], className="li-cap")])),
        html.P(f"Cada punto es un mes de entrenamiento de {C.name(c)}. "
               + (f"No se muestran {fuera} meses fuera de escala (eneros muy volátiles y choques como el de 2020)." if fuera else ""),
               className="li-cap"),
    ]), sub=NUEVO, cls="px-panel")
    return [U.block("3.4 · Localización", "Los meses que deciden los pesos de hoy"), mapa]
