"""Informe académico en PDF del país seleccionado.

Todo sale de las mismas funciones que usa la plataforma (``labor.engine``):
pronóstico, intervalo conforme, validación fuera de muestra, métricas y
pruebas guardadas. Las gráficas se dibujan con matplotlib y el documento con
fpdf2, usando las fuentes DejaVu que trae matplotlib (acentos y símbolos).
"""
import io
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from fpdf import FPDF  # noqa: E402
from fpdf.fonts import FontFace  # noqa: E402

from . import catalog as C  # noqa: E402
from . import engine as E  # noqa: E402

N = C.num
FONTS = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
NAVY, OBS, PRED, OCHRE, ALERT, MUTED, LINE = "#183B56", "#47799D", "#258477", "#C18B43", "#B85C5C", "#687585", "#E1E6E8"


def _rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# ------------------------------------------------------------------ gráficas
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color="#ECEFF1", linewidth=0.8)
    ax.set_axisbelow(True)


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def chart_forecast(c, months=120):
    f, s = E.forecast(c), E.series(c).iloc[-months:]
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    ax.plot(s.index, s.values, color=OBS, linewidth=1.5, label="Observado")
    x0, x1 = f["last_date"], f["target_date"]
    ax.fill([x0, x1, x1, x0], [f["observed"], f["hi"], f["lo"], f["observed"]], color=PRED, alpha=0.18, linewidth=0, label="Intervalo del 90 %")
    ax.plot([x0, x1], [f["observed"], f["pred"]], color=PRED, linewidth=2)
    ax.plot([x1], [f["pred"]], marker="D", color=PRED, markersize=6, label="Pronóstico de ALEE (un mes)")
    ax.axvline(x0, color=MUTED, linewidth=0.8, linestyle=":")
    ax.annotate(f"{N(f['pred'])} %", (x1, f["pred"]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=PRED, fontweight="bold")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f} %")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _style(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=3)
    return _png(fig)


def chart_track(c, n=36):
    d = E.recent_track(c, n)
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.fill_between(d["fecha"], d["ALEE"] - d["hw"], d["ALEE"] + d["hw"], color=PRED, alpha=0.15, linewidth=0, label="Intervalo del 90 % (errores anteriores)")
    ax.plot(d["fecha"], d["y_true"], color=OBS, linewidth=1.5, marker="o", markersize=2.5, label="Observado")
    ax.plot(d["fecha"], d["ALEE"], color=PRED, linewidth=1.3, linestyle="--", marker="D", markersize=2.5, label="Pronóstico de ALEE")
    out = d[~d["dentro"] & d["hw"].notna()]
    if len(out):
        ax.scatter(out["fecha"], out["y_true"], s=36, facecolors="white", edgecolors=ALERT, linewidths=1.4, zorder=5, label="Fuera del intervalo")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1f} %".replace(".", ","))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    _style(ax)
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
    return _png(fig)


def chart_models(c):
    row = E.metric_table().loc[c].sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    cols = [NAVY if m == "ALEE" else OCHRE if m == E.best_baseline(c)[0] else "#C9D6E0" for m in row.index]
    ax.barh([C.MODEL_LABELS[m] for m in row.index][::-1], row.values[::-1], color=cols[::-1], height=0.65)
    for i, v in enumerate(row.values[::-1]):
        ax.text(v, i, f" {N(v, 3)}", va="center", fontsize=7, color="#172536")
    ax.set_xlabel("RMSE medio por pliegue (pp)", fontsize=8, color=MUTED)
    _style(ax)
    ax.grid(axis="x", color="#ECEFF1")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=7.5)
    return _png(fig)


def chart_reductions(selected):
    t = E.metric_table()
    r = (1 - t["ALEE"] / t.drop(columns="ALEE").min(axis=1)).sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.barh([C.name(c) for c in r.index], 100 * r.values, color=[NAVY if c == selected else "#9FB3C4" for c in r.index], height=0.62)
    for i, v in enumerate(r.values):
        ax.text(100 * v, i, f" {N(100 * v, 0)} %", va="center", fontsize=7, color="#172536")
    ax.set_xlabel("Reducción del RMSE frente al mejor modelo de comparación (%)", fontsize=8, color=MUTED)
    ax.set_xlim(0, max(40, 100 * r.max() + 6))
    _style(ax)
    ax.grid(axis="x", color="#ECEFF1")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=7.5)
    return _png(fig)


# --------------------------------------------------------------- documento
class Report(FPDF):
    def __init__(self, country):
        super().__init__(format="A4")
        self.country = country
        self.set_auto_page_break(True, margin=18)
        self.set_margins(18, 16, 18)
        self.add_font("Sans", "", str(FONTS / "DejaVuSans.ttf"))
        self.add_font("Sans", "B", str(FONTS / "DejaVuSans-Bold.ttf"))
        self.add_font("Sans", "I", str(FONTS / "DejaVuSans-Oblique.ttf"))
        self.add_font("Serif", "B", str(FONTS / "DejaVuSerif-Bold.ttf"))
        self.add_font("Mono", "", str(FONTS / "DejaVuSansMono.ttf"))
        self.set_title(f"LABOR INSIGHTS · {C.name(country)}")
        self.set_author("Andrés Parejo, Santiago Hurtado y Juan Marín")
        self.set_creator("LABOR INSIGHTS")

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Sans", "", 7.5)
        self.set_text_color(*_rgb(MUTED))
        self.cell(0, 5, f"LABOR INSIGHTS · Informe de {C.name(self.country)}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*_rgb(LINE))
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("Sans", "", 7.5)
        self.set_text_color(*_rgb(MUTED))
        self.cell(0, 5, f"Plataforma académica. No sustituye las cifras oficiales.   ·   Página {self.page_no()}/{{nb}}", align="C")

    # bloques
    def h1(self, text):
        self.ln(2)
        self.set_font("Serif", "B", 14)
        self.set_text_color(*_rgb(NAVY))
        self.multi_cell(0, 7, text, new_x="LMARGIN", new_y="NEXT", align="L")
        self.ln(1)

    def p(self, text, size=9.5, color="#172536", style=""):
        self.set_font("Sans", style, size)
        self.set_text_color(*_rgb(color))
        self.multi_cell(0, 5, text, new_x="LMARGIN", new_y="NEXT", align="L")
        self.ln(1.5)

    def bullets(self, items, size=9.2):
        self.set_font("Sans", "", size)
        self.set_text_color(*_rgb("#172536"))
        for x in items:
            self.set_x(self.l_margin + 2)
            self.multi_cell(0, 4.8, "•  " + x, new_x="LMARGIN", new_y="NEXT", align="L")
            self.ln(0.6)
        self.ln(1)

    def figure(self, buf, caption, w=None):
        w = w or (self.w - self.l_margin - self.r_margin)
        if self.get_y() > self.h - 85:
            self.add_page()
        self.image(buf, x=self.l_margin, w=w)
        self.set_font("Sans", "I", 7.8)
        self.set_text_color(*_rgb(MUTED))
        self.multi_cell(0, 4, caption, new_x="LMARGIN", new_y="NEXT", align="L")
        self.ln(2)

    def kv(self, rows, widths=(84, 90)):
        self.set_font("Sans", "", 9)
        self.set_text_color(*_rgb("#172536"))
        self.set_fill_color(255, 255, 255)
        self.set_draw_color(*_rgb(LINE))
        with self.table(col_widths=widths, line_height=5.6, first_row_as_headings=False, borders_layout="HORIZONTAL_LINES",
                        cell_fill_color=(246, 247, 245), cell_fill_mode="ROWS") as t:
            for k, v in rows:
                r = t.row()
                r.cell(k)
                r.cell(v)
        self.ln(2)

    def grid(self, head, rows, widths, size=8):
        self.set_font("Sans", "", size)
        self.set_text_color(*_rgb("#172536"))
        self.set_fill_color(255, 255, 255)
        self.set_draw_color(*_rgb(LINE))
        with self.table(col_widths=widths, line_height=5, borders_layout="HORIZONTAL_LINES", text_align="LEFT",
                        headings_style=FontFace(emphasis="BOLD", color=_rgb(NAVY), fill_color=(238, 241, 240))) as t:
            r = t.row()
            for h in head:
                r.cell(h)
            for row in rows:
                r = t.row()
                for v in row:
                    r.cell(str(v))
        self.ln(2)


def _p(p):
    return "< 0,001" if p < 0.001 else N(p, 3)


def build_report(c) -> bytes:
    f, k = E.forecast(c), E.conformal(c)
    best, b = E.best_baseline(c)
    rmse = E.metric_table()
    r_alee = float(rmse.loc[c, "ALEE"])
    d = E.oof(c)
    tests = E.result("pruebas_por_baseline")
    tc = tests[tests["conjunto"] == c]
    abl = E.result("ablacion")
    loc = abl[(abl["variante"] == "Sin localización") & (abl["dm_p_holm"] < 0.05)]["conjunto"].map(C.name).tolist()
    red = 1 - rmse["ALEE"] / rmse.drop(columns="ALEE").min(axis=1)
    m = C.COUNTRIES[c]

    pdf = Report(c)
    pdf.alias_nb_pages()
    pdf.add_page()

    # 1. Título y metadatos
    pdf.set_fill_color(*_rgb(NAVY))
    pdf.rect(0, 0, pdf.w, 4, "F")
    pdf.ln(6)
    pdf.set_font("Sans", "B", 8.5)
    pdf.set_text_color(*_rgb(PRED))
    pdf.cell(0, 5, "LABOR INSIGHTS · INFORME ACADÉMICO", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Serif", "B", 21)
    pdf.set_text_color(*_rgb(NAVY))
    pdf.multi_cell(0, 9.5, f"Pronóstico a un mes de la tasa de desempleo: {C.name(c)}", new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.set_fill_color(255, 255, 255)
    pdf.ln(1)
    pdf.p("Andrés Parejo, Santiago Hurtado y Juan Marín · Pregrado en Ciencia de Datos, curso de Machine Learning · Universidad del Norte", 9, MUTED)
    pdf.p(f"Generado el {date.today().strftime('%d/%m/%Y')} desde la plataforma, con los datos y resultados vigentes. País seleccionado: {C.name(c)}.", 8.5, MUTED)

    # 2. Objetivo
    pdf.h1("1. Objetivo de la investigación")
    pdf.p("Pronosticar la tasa de desempleo del mes siguiente con ALEE (agregación local entrópica de expertos), un modelo original que combina tres expertos sencillos "
          "(persistencia, núcleo del cambio y cambio estacional) con pesos que dependen del estado del mercado laboral y que se obtienen de un problema convexo regularizado "
          "hacia una combinación global. ALEE se evalúa fuera de muestra frente a los ocho modelos de regresión del curso y LightGBM, en Colombia y ocho países más. "
          "El horizonte es siempre de un mes.")

    # 3-4. Indicadores
    pdf.h1(f"2. Indicadores principales: {C.name(c)}")
    pdf.p(f"Fuente: {m['source']}. {m['definition']}.", 8.5, MUTED)
    pdf.kv([("Última tasa observada (observado)", f"{N(f['observed'])} %  ·  {C.month_long(f['last_date'])}"),
            ("Mes anterior (observado)", f"{N(f['previous'])} %"),
            ("Pronóstico del mes siguiente (pronosticado)", f"{N(f['pred'])} %  ·  {C.month_long(f['target_date'])}"),
            ("Variación pronosticada (calculado)", f"{N(f['change'], 2, True)} pp"),
            ("Mismo mes, un año antes (observado)", f"{N(f['same_month'])} %")])

    # 5. Pronóstico e intervalo
    pdf.h1("3. Pronóstico e intervalo de predicción")
    pdf.kv([("Intervalo del 90 %", f"{N(f['lo'])} % a {N(f['hi'])} %  (± {N(k['hw'])} pp)"),
            ("Cobertura nominal", "90 %"),
            ("Cobertura empírica, medida hacia adelante", f"{N(100 * k['coverage'], 1)} % en {k['n_forward']} meses"),
            ("Ancho medio en esa medición", f"{N(k['width'])} pp"),
            ("Errores fuera de muestra usados", str(k["n"]))])
    pdf.p("El intervalo es conforme partido: el cuantil ajustado del 90 % del error absoluto fuera de muestra se suma y se resta al pronóstico. Como los errores de una serie de "
          "tiempo dependen entre sí, la cobertura no está garantizada y por eso se mide hacia adelante. El ancho es el mismo en meses tranquilos y agitados.", 8.8)
    pdf.figure(chart_forecast(c), f"Figura 1. {C.name(c)}: últimos diez años observados y pronóstico de ALEE para {C.month_long(f['target_date'])} con su intervalo del 90 %.")
    pdf.p(f"Descomposición exacta: último dato {N(f['last'])} % + persistencia {N(f['contrib'][0], 2, True)} pp + núcleo {N(f['contrib'][1], 2, True)} pp + "
          f"estacional {N(f['contrib'][2], 2, True)} pp = {N(f['pred'])} %. Pesos: " + ", ".join(f"{C.EXPERT_LABELS[n].lower()} {N(100 * f['weights'][i], 1)} %" for i, n in enumerate(C.EXPERT_LABELS)) + ".", 8.8)

    # 6. Validación
    pdf.h1("4. Validación del pronóstico")
    val = E.later_validation(c)
    ok = val[val["pred"].notna()] if len(val) else val
    if len(ok):
        pdf.p("Contraste con el dato oficial publicado después del corte (no usado por el modelo):", 9)
        pdf.grid(["Origen", "Mes pronosticado", "Pronóstico", "Publicado", "Error", "¿Dentro del 90 %?"],
                 [[C.month_short(r["origen"]), C.month_short(r["objetivo"]), f"{N(r['pred'])} %", f"{N(r['obs'], 1)} %", f"{N(r['error'], 2, True)} pp", "Sí" if r["dentro"] else "No"] for _, r in ok.iterrows()],
                 (26, 32, 26, 26, 26, 38))
        pdf.p("Un solo mes no valida ni invalida el modelo; la evidencia de desempeño es la evaluación fuera de muestra.", 8.5, MUTED)
    else:
        pdf.p(f"El dato oficial de {C.month_long(f['target_date'])} todavía no está en el proyecto, así que este pronóstico aún no se puede contrastar.", 9)
    tr = E.recent_track(c, 24)
    has = tr["hw"].notna()
    pdf.p(f"Historial fuera de muestra: en los últimos {len(tr)} meses de prueba el error absoluto medio fue de {N(tr['error'].abs().mean())} pp y "
          f"{int(tr.loc[has, 'dentro'].sum())} de {int(has.sum())} observaciones cayeron dentro del intervalo calculado solo con errores anteriores.", 9)
    pdf.figure(chart_track(c), "Figura 2. Últimos 36 pronósticos fuera de muestra frente a lo observado. Cada pronóstico e intervalo usa solo información anterior.")

    # 7-9. Evaluación
    pdf.h1("5. Evaluación del modelo y comparación")
    pdf.p(f"Validación anidada de origen móvil con cinco pliegues externos ({len(d)} meses de prueba, {d['fecha'].min().year} a {d['fecha'].max().year}), un mes de separación "
          f"y búsqueda bayesiana de 40 evaluaciones. En {C.name(c)}, ALEE tuvo un RMSE medio por pliegue de {N(r_alee, 3)} pp frente a {N(b, 3)} de {C.MODEL_LABELS[best]}, "
          f"el mejor modelo de comparación ({N(100 * (1 - r_alee / b), 0)} % menos error). Diebold-Mariano-HLN con Holm rechaza la igualdad frente a "
          f"{int((tc['dm_p_holm'] < 0.05).sum())} de {len(tc)} modelos.", 9)
    pdf.figure(chart_models(c), f"Figura 3. RMSE medio por pliegue en {C.name(c)}. Azul marino: ALEE. Ocre: mejor modelo de comparación.")
    pdf.grid(["Modelo", "RMSE", "Dif. de pérdida", "DM-HLN, p", "GW, p"],
             [[C.MODEL_LABELS[r["baseline"]], N(rmse.loc[c, r["baseline"]], 3), N(r["dif_perdida_media"], 3), _p(r["dm_p_holm"]), _p(r["gw_p_holm"])] for _, r in tc.iterrows()],
             (52, 26, 34, 30, 32), 7.8)

    # 8. Comparación internacional
    pdf.h1("6. Comparación internacional")
    pdf.p(f"ALEE tuvo el menor RMSE en {int((rmse.idxmin(axis=1) == 'ALEE').sum())} de 9 países, con entre {N(100 * red.min(), 0)} % y {N(100 * red.max(), 0)} % menos error que el mejor "
          f"modelo de comparación de cada uno. Diebold-Mariano rechaza la igualdad en {int((tests['dm_p_holm'] < 0.05).sum())} de {len(tests)} comparaciones país-modelo. "
          "Cada país tiene su propio ALEE: lo que se comprueba es que el método generaliza. Las tasas no son perfectamente comparables entre países.", 9)
    pdf.figure(chart_reductions(c), "Figura 4. Reducción del error por país. Resaltado, el país seleccionado.")

    # 10. Conclusiones
    pdf.h1("7. Conclusiones")
    pdf.bullets([
        "La tasa de desempleo es persistente y estacional; los choques, como el de 2020, son pocos y concentran los mayores errores de todos los modelos.",
        f"ALEE mejora de forma consistente sobre los modelos de comparación en los nueve países ({N(100 * red.min(), 0)} % a {N(100 * red.max(), 0)} % menos RMSE).",
        "La ventaja proviene sobre todo de que sus expertos incorporan la persistencia y el calendario, y solo en parte de la combinación y la localización por estado.",
        "Cada pronóstico se descompone con exactitud en el aporte de tres expertos y se acompaña de un intervalo cuya cobertura se mide.",
    ])

    # 11. Limitaciones
    pdf.h1("8. Limitaciones")
    pdf.bullets([
        f"La localización por estado, la idea central, solo aporta con significancia en {' y '.join(loc) if loc else 'ningún país'}.",
        "Solo pronostica un mes y reacciona tarde al inicio de una crisis.",
        "En Colombia se evaluaron diez iteraciones de diseño sobre los mismos 240 meses; los ocho países adicionales, evaluados una vez con el diseño congelado, son la evidencia más confiable.",
        "Nueve países dan una potencia moderada a las pruebas entre países.",
        "Se usa la versión vigente de las series y no la publicada en tiempo real; con errores dependientes, la cobertura del intervalo no está garantizada.",
    ])

    # 12. Referencias (texto exacto de la bibliografía del paper)
    pdf.h1("9. Referencias principales")
    pdf.bullets(bib(KEYS), 8.2)
    pdf.p("La bibliografía completa está en el paper del proyecto y en la sección Insights & Conclusions de la plataforma.", 8.2, MUTED)
    return bytes(pdf.output())


KEYS = ["dane_geih", "banrep2025series", "fred_unratensa", "eurostat_une", "oecd_lfs", "nadaraya1964", "watson1964", "kivinen1997", "kish1965",
        "timmermann2006", "dieboldmariano1995", "harveyetal1997", "giacominiwhite2006", "holm1979", "demsar2006", "hansen2011"]


def bib(keys):
    """Entradas de la bibliografía del paper, en texto plano y sin modificar su contenido."""
    import re

    from src import config

    tex = (config.ROOT_DIR / "paper" / "EDA_Mercado_Laboral_Colombiano_ALEE.tex").read_text(encoding="utf-8")
    body = tex.split(r"\begin{thebibliography}")[1].split(r"\end{thebibliography}")[0]
    out = {}
    for m in re.finditer(r"\\bibitem\{([^}]*)\}(.*?)(?=\\bibitem\{|\Z)", body, re.S):
        t = " ".join(m.group(2).split()).replace(r"\={a}", "ā")
        t = re.sub(r"\\textit\{([^}]*)\}", r"\1", t).replace(r"\url{", "").replace("}", "")
        t = t.replace(r"\_", "_").replace(r"\&", "&").replace("--", "–").replace(r"\,", " ").replace(r"\ ", " ")
        t = re.sub(r"\\v\{?([a-zA-Z])\}?", lambda x: {"s": "š", "c": "č"}.get(x.group(1), x.group(1)), t).replace("``", "«").replace("''", "»")
        out[m.group(1)] = t.strip()
    return [out[k] for k in keys if k in out]
