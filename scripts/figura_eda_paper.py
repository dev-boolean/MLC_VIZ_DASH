"""Figura de análisis exploratorio del paper (paper/figuras/fig_eda.pdf).

Cuatro paneles con los patrones de los datos que condicionan el diseño de ALEE:
(a) la serie con el choque de 2020, (b) el cambio mensual medio por mes del año,
(c) la autocorrelación del nivel y del cambio mensual y (d) el error de los
modelos del segundo entregable por tercil de volatilidad reciente.

Uso: python scripts/figura_eda_paper.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import dash_data as D  # noqa: E402
from src.explain import acf  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "paper" / "figuras" / "fig_eda.pdf"
AZUL, VIOLETA, AMBAR, GRIS = "#2563eb", "#7c3aed", "#d97706", "#64748b"
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


def main():
    raw = D.raw()
    y = raw.set_index("fecha")["tasa_desempleo_nacional"]
    ds = D.dataset()
    oof = D.oof("reg")
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.4))

    a = ax[0, 0]
    a.plot(y.index, y.values, color=AZUL, lw=1.2)
    a.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-30"), color=AMBAR, alpha=0.2, lw=0)
    a.set_ylabel("Tasa de desempleo (%)")
    a.set_title("(a) Tasa de desempleo nacional, 2001-2025", loc="left", fontsize=9)

    b = ax[0, 1]
    cambio = y.diff().dropna()
    g = cambio.groupby(cambio.index.month)
    media, err = g.mean(), 1.96 * g.std() / np.sqrt(g.count())
    b.bar(range(12), media.values, yerr=err.values, color=[AMBAR if v > 0 else AZUL for v in media.values], capsize=2)
    b.axhline(0, color=GRIS, lw=0.8)
    b.set_xticks(range(12)); b.set_xticklabels(MESES, rotation=0)
    b.set_ylabel("Cambio mensual medio (pp)")
    b.set_title("(b) Cambio mensual medio por mes del año (IC 95 %)", loc="left", fontsize=9)

    c = ax[1, 0]
    lags = np.arange(1, 25)
    c.bar(lags - 0.2, acf(y.values, 24)[1:], width=0.4, color=AZUL, label="Nivel")
    c.bar(lags + 0.2, acf(cambio.values, 24)[1:], width=0.4, color=VIOLETA, label="Cambio mensual")
    c.axhline(1.96 / np.sqrt(len(cambio)), color=GRIS, ls="--", lw=0.8)
    c.axhline(-1.96 / np.sqrt(len(cambio)), color=GRIS, ls="--", lw=0.8)
    c.set_xlabel("Rezago (meses)"); c.set_ylabel("Autocorrelación")
    c.set_title("(c) Autocorrelación del nivel y del cambio mensual", loc="left", fontsize=9)
    c.legend(frameon=False, fontsize=8)

    d = ax[1, 1]
    modelos = ["XGBoost", "Ridge", "SVR", "KNN", "RandomForest"]
    vol = ds.X["td_nacional_rollstd3"].values[oof["idx"]]
    tercil = pd.qcut(vol, 3, labels=["Baja", "Media", "Alta"])
    mse = pd.DataFrame({D.MODEL_LABELS[m]: (oof["y_true"] - oof[f"pred_{m}"]) ** 2 for m in modelos}).groupby(tercil, observed=True).mean()
    x = np.arange(3)
    estilos = ["", "//", "..", "xx", "\\\\"]
    for i, m in enumerate(mse.columns):
        d.bar(x + (i - 2) * 0.16, mse[m].values, width=0.16, label=m, color=[AZUL, VIOLETA, AMBAR, GRIS, "#0f172a"][i],
              hatch=estilos[i], edgecolor="white", lw=0.3)
    d.set_xticks(x); d.set_xticklabels(["Volatilidad baja", "Media", "Alta"])
    d.set_ylabel("Error cuadrático medio (pp²)")
    d.set_title("(d) Error de los modelos del segundo entregable por volatilidad", loc="left", fontsize=9)
    d.legend(frameon=False, fontsize=7, ncol=2)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Figura guardada en {OUT}")
    print("Mejor modelo por tercil:", mse.idxmin(axis=1).to_dict())
    print("Meses con cambio medio positivo:", [MESES[i] for i, v in enumerate(media.values) if v > 0])


if __name__ == "__main__":
    main()
