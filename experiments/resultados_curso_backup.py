"""Resultados del paper: ALEE frente a los ocho modelos de regresión del curso, en nueve conjuntos de datos.

Uso: python experiments/resultados_curso.py
Reutiliza las predicciones de run_confirmatorio.py, run_conjuntos_adicionales.py y resultados_finales.py
y escribe tablas y figuras en artifacts/resultados/solo_curso/. Holm, MCS, Friedman y los rangos se
calculan con los 9 modelos (ALEE + 8 del curso). Conjuntos: los 4 originales y los 5 de la sección 19
del protocolo.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import resultados_finales as rf  # noqa: E402
from scipy import stats  # noqa: E402
from src import stats_tests as ST  # noqa: E402

rf.BASELINES = list(rf.CURSO)
CURSO = rf.CURSO
CONJ = rf.CONJUNTOS_BASE + rf.CONJUNTOS_NUEVOS
OUT = rf.OUT / "solo_curso"
FIG = OUT / "figuras"
L = OUT / "latex"
LAB = {"ALEE": "ALEE", "SVR_RBF": "SVR (RBF)", "BayesianRidge": "Bayesian ridge", "DecisionTree": "Decision tree",
       "RandomForest": "Random forest"}
METRICAS = ["RMSE", "MAE", "MAPE", "R2", "MASE"]
VARIANTES = [("ALEE completo", None), ("Sin localización", "sin_localizacion"), ("Sin λ adaptativa", "sin_lambda_adaptativa"),
             ("Previo uniforme", "previo_uniforme"), ("Promedio simple", "fijos:PromedioSimple"),
             ("Solo persistencia", "P:Persistencia"), ("Solo núcleo", "fijos:NucleoSolo"), ("Solo estacional", "P:EstacionalALEE")]
VAR_EN = {"ALEE completo": "Full ALEE", "Sin localización": "No localization", "Sin λ adaptativa": "No adaptive λ",
          "Previo uniforme": "Uniform prior", "Promedio simple": "Simple average", "Solo persistencia": "Persistence only",
          "Solo núcleo": "Kernel only", "Solo estacional": "Seasonal only"}


def metricas(y, pred, y_all, sp):
    """Métricas por pliegue para un vector de predicciones indexado como y."""
    rows = []
    for k, (tr, te) in enumerate(sp):
        ytr = y_all[tr]
        escala = np.mean(np.abs(ytr[12:] - ytr[:-12]))
        yt, pt = y.loc[te], pred.loc[te]
        e = yt - pt
        rows.append({"pliegue": k, "RMSE": np.sqrt(np.mean(e ** 2)), "MAE": np.mean(np.abs(e)), "MAPE": 100 * np.mean(np.abs(e / yt)),
                     "R2": 1 - np.sum(e ** 2) / np.sum((yt - yt.mean()) ** 2), "MASE": np.mean(np.abs(e)) / escala})
    return pd.DataFrame(rows)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "pdf.fonttype": 42,
                         "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7})
    for d in (OUT, FIG, L):
        d.mkdir(parents=True, exist_ok=True)
    preds, met, pruebas, mcs_all, reg, abl = {}, [], [], [], [], []
    for conj in CONJ:
        Xdf, ydf, _, _ = rf.get_data(conj)
        y_all = ydf.to_numpy(float)
        Pfull = rf.cargar_predicciones(conj)
        P = Pfull[["y", "ALEE"] + CURSO]
        preds[conj] = P
        M = rf.metricas_por_pliegue(P, y_all, conj)
        met.append(M)
        E = P[["ALEE"] + CURSO].rsub(P["y"], axis=0)
        Lo = E ** 2
        sp = rf.outer_splits(len(ydf))
        rho = np.mean([len(te) / (len(tr) + len(te)) for tr, te in sp])
        rmse_f = M.pivot(index="pliegue", columns="modelo", values="RMSE")
        rows = []
        for b in CURSO:
            dm = ST.diebold_mariano_test(E["ALEE"], E[b])
            gw = ST.giacomini_white_test(E["ALEE"], E[b], conditional=True)
            dl = Lo["ALEE"] - Lo[b]
            _, lo, hi = ST.bootstrap_interval(dl.values, np.mean, mean_block=12, n_boot=2000, seed=42)
            pb = rf.bayes_t_correlacionado(((rmse_f["ALEE"] - rmse_f[b]) / rmse_f[b]).values, rho)
            a, c = np.abs(E["ALEE"].values), np.abs(E[b].values)
            a12 = (np.sum(a[:, None] < c[None, :]) + 0.5 * np.sum(a[:, None] == c[None, :])) / (len(a) * len(c))
            rows.append({"conjunto": conj, "baseline": b, "dif_perdida_media": dl.mean(), "ic95_inf": lo, "ic95_sup": hi,
                         "dm_estadistico": dm["dm_hln"], "dm_p": dm["p_value"], "gw_estadistico": gw["wald"], "gw_p": gw["p_value"],
                         "p_bayes_mejor": pb[0], "p_bayes_rope": pb[1], "p_bayes_peor": pb[2],
                         "d_cohen": ST.cohens_d_paired(dl.values), "a12_vargha": a12})
        R = pd.DataFrame(rows)
        R["dm_p_holm"] = ST.adjust_pvalues(R["dm_p"].values, "holm")
        R["gw_p_holm"] = ST.adjust_pvalues(R["gw_p"].values, "holm")
        pruebas.append(R)
        mcs = ST.model_confidence_set(Lo, alpha=0.10, block=12, seed=42)
        mcs.insert(0, "conjunto", conj)
        mcs_all.append(mcs)
        vol = Xdf["td_nacional_rollstd3"].to_numpy()[P.index.values]
        g = np.digitize(vol, np.quantile(vol, [1 / 3, 2 / 3]))
        for m in ["ALEE"] + CURSO:
            for k, lab in enumerate(["baja", "media", "alta"]):
                e = (P["y"] - P[m]).values[g == k]
                reg.append({"conjunto": conj, "modelo": m, "volatilidad": lab, "rmse": np.sqrt(np.mean(e ** 2))})
        # ablación con todas las métricas
        fx = pd.read_csv(rf.OUT / conj / "fijos.csv").set_index("idx")
        eA = P["y"] - P["ALEE"]
        for nom, src in VARIANTES:
            if src is None:
                pr = P["ALEE"]
            elif src.startswith("fijos:"):
                pr = fx[src[6:]]
            elif src.startswith("P:"):
                pr = Pfull[src[2:]]
            else:
                pr = pd.read_csv(rf.OUT / conj / f"alee_{src}.csv").set_index("idx")["y_pred"]
            pr = pr.loc[P.index]
            mm = metricas(P["y"], pr, y_all, sp)
            r = {"conjunto": conj, "variante": nom, **{f"{q}": mm[q].mean() for q in METRICAS}, "rmse_sd": mm["RMSE"].std(ddof=1)}
            if src is not None:
                dm = ST.diebold_mariano_test(eA, P["y"] - pr)
                r.update({"dm_p": dm["p_value"], "dif_perdida": dm["mean_loss_diff"]})
            abl.append(r)
    met = pd.concat(met, ignore_index=True)
    pruebas = pd.concat(pruebas, ignore_index=True)
    reg = pd.DataFrame(reg)
    abl = pd.DataFrame(abl)
    for conj in CONJ:
        msk = (abl.conjunto == conj) & abl.dm_p.notna()
        abl.loc[msk, "dm_p_holm"] = ST.adjust_pvalues(abl.loc[msk, "dm_p"].values, "holm")
    met.to_csv(OUT / "metricas_por_pliegue.csv", index=False)
    pruebas.to_csv(OUT / "pruebas_por_baseline.csv", index=False)
    pd.concat(mcs_all).to_csv(OUT / "mcs.csv", index=False)
    reg.to_csv(OUT / "errores_por_regimen.csv", index=False)
    abl.to_csv(OUT / "ablacion.csv", index=False)
    mean = met.groupby(["conjunto", "modelo"]).mean(numeric_only=True)
    sd = met.groupby(["conjunto", "modelo"]).std(numeric_only=True)

    # entre conjuntos
    W = mean["RMSE"].unstack().loc[CONJ, ["ALEE"] + CURSO]
    N, k = W.shape
    chi2, p_f = stats.friedmanchisquare(*[W[c].values for c in W.columns])
    ff = (N - 1) * chi2 / (N * (k - 1) - chi2)
    p_id = 1 - stats.f.cdf(ff, k - 1, (k - 1) * (N - 1))
    ranks = W.rank(axis=1).mean().sort_values()
    import scikit_posthocs as sph
    con = sph.posthoc_conover_friedman(W.reset_index(drop=True), p_adjust="holm")
    entre = []
    for b in CURSO:
        pb = rf.bayes_signed_rank(((W["ALEE"] - W[b]) / W[b]).values)
        pw = stats.wilcoxon(W["ALEE"], W[b]).pvalue
        entre.append({"baseline": b, "conjuntos_ALEE_gana": int((W["ALEE"] < W[b]).sum()), "conover_p_holm": con.loc["ALEE", b],
                      "wilcoxon_p": pw, "p_bayes_mejor": pb[0], "p_bayes_rope": pb[1], "p_bayes_peor": pb[2]})
    entre = pd.DataFrame(entre)
    entre["wilcoxon_p_holm"] = ST.adjust_pvalues(entre["wilcoxon_p"].values, "holm")
    entre.to_csv(OUT / "pruebas_entre_conjuntos.csv", index=False)
    cd = ST.critical_difference(k, N)
    fr = pd.Series({"friedman_chi2": chi2, "friedman_p": p_f, "iman_davenport_F": ff, "iman_davenport_p": p_id,
                    "nemenyi_cd": cd, **{f"rango_{m}": r for m, r in ranks.items()}})
    fr.to_csv(OUT / "friedman.csv")

    # ------------------------------------------------------------- figuras
    fig, ax = plt.subplots(figsize=(6.5, 2.2))
    ST.plot_cd_diagram(ranks.rename(index=lambda s: LAB.get(s, s)), cd, ax=ax, title=None)
    for txt in ax.texts:
        txt.set_text(txt.get_text().replace("DC", "CD"))
    fig.tight_layout()
    fig.savefig(FIG / "fig_cd.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(6.5, 6.6))
    for ax, conj in zip(axes.ravel(), CONJ):
        sub = met[met.conjunto == conj]
        order = sub.groupby("modelo")["RMSE"].mean().sort_values().index
        ax.boxplot([sub[sub.modelo == m]["RMSE"].values for m in order], vert=False, widths=0.6)
        ax.set_yticks(range(1, len(order) + 1), [LAB.get(m, m) for m in order], fontsize=6)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(rf.NAME_EN[conj])
    for ax in axes[-1]:
        ax.set_xlabel("RMSE per outer fold (pp)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_rmse_pliegues.pdf")
    plt.close(fig)
    mejor = {c: mean.loc[c].loc[CURSO]["RMSE"].idxmin() for c in CONJ}
    from src.data import load_raw
    fechas = pd.to_datetime(load_raw()["fecha"]).values
    P = preds["colombia"]
    b = mejor["colombia"]
    t = fechas[P.index.values + 12]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.5, 4.0), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(t, P["y"], color="black", lw=1.2, label="Observed")
    a1.plot(t, P["ALEE"], color="tab:blue", lw=0.9, label="ALEE")
    a1.plot(t, P[b], color="tab:green", lw=0.9, ls="--", label=f"{LAB.get(b, b)} (best course model)")
    a1.set_ylabel("Unemployment rate (%)")
    a1.legend(frameon=False, ncol=3)
    a2.plot(t, P["y"] - P["ALEE"], color="tab:blue", lw=0.8)
    a2.plot(t, P["y"] - P[b], color="tab:green", lw=0.8, ls="--")
    a2.axhline(0, color="grey", lw=0.5)
    a2.set_ylabel("Residual (pp)")
    a2.set_xlabel("Forecast month")
    fig.tight_layout()
    fig.savefig(FIG / "fig_pronostico_colombia.pdf")
    plt.close(fig)
    # predicho frente a observado (3x3) y distribución de residuos (3x3)
    for nombre, tipo in [("fig_predicho_observado.pdf", "scatter"), ("fig_residuos.pdf", "hist")]:
        fig, axes = plt.subplots(3, 3, figsize=(6.5, 6.0))
        for ax, conj in zip(axes.ravel(), CONJ):
            P, b = preds[conj], mejor[conj]
            if tipo == "scatter":
                lo, hi = P["y"].min(), P["y"].max()
                ax.plot([lo, hi], [lo, hi], color="grey", lw=0.6)
                ax.scatter(P["y"], P[b], s=2, color="tab:green", alpha=0.5, label=LAB.get(b, b))
                ax.scatter(P["y"], P["ALEE"], s=2, color="tab:blue", alpha=0.5, label="ALEE")
                ax.set_xlabel("Observed (%)", fontsize=6)
                ax.set_ylabel("Predicted (%)", fontsize=6)
            else:
                rA, rB = P["y"] - P["ALEE"], P["y"] - P[b]
                bins = np.linspace(np.quantile(np.r_[rA, rB], 0.005), np.quantile(np.r_[rA, rB], 0.995), 30)
                ax.hist(rB, bins=bins, color="tab:green", alpha=0.5, label=LAB.get(b, b))
                ax.hist(rA, bins=bins, color="tab:blue", alpha=0.5, label="ALEE")
                ax.set_xlabel("Residual (pp)", fontsize=6)
            ax.set_title(rf.NAME_EN[conj])
            ax.legend(frameon=False, fontsize=5, markerscale=2)
        fig.tight_layout()
        fig.savefig(FIG / nombre)
        plt.close(fig)
    # ablación (3x3)
    fig, axes = plt.subplots(3, 3, figsize=(6.5, 6.0), sharey=True)
    for ax, conj in zip(axes.ravel(), CONJ):
        s = abl[abl.conjunto == conj].set_index("variante").reindex([v for v, _ in VARIANTES])
        ax.barh(range(len(s)), s["RMSE"], xerr=s["rmse_sd"], color=["tab:blue"] + ["tab:grey"] * (len(s) - 1), error_kw={"lw": 0.6})
        ax.set_yticks(range(len(s)), [VAR_EN[v] for v in s.index], fontsize=6)
        ax.invert_yaxis()
        ax.set_title(rf.NAME_EN[conj])
    for ax in axes[-1]:
        ax.set_xlabel("RMSE (pp)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_ablacion.pdf")
    plt.close(fig)
    # sensibilidad: mapas de calor + curva de aprendizaje
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4))
    for ax, conj in zip(axes[:2], ["colombia", "eeuu"]):
        g = pd.read_csv(rf.OUT / conj / "sensibilidad_hiperparametros.csv").pivot(index="lam", columns="bandwidth", values="rmse_medio")
        im = ax.imshow(g.values, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(g.shape[1]), [f"{v:.2g}" for v in g.columns], rotation=90)
        ax.set_yticks(range(g.shape[0]), [f"{v:.0e}" for v in g.index])
        ax.set_xlabel("Bandwidth h")
        ax.set_ylabel("λ")
        ax.set_title(f"{rf.NAME_EN[conj]}: RMSE")
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[2]
    lcv = pd.read_csv(rf.OUT / "curva_aprendizaje_eeuu.csv")
    for m, c, ls in [("ALEE", "tab:blue", "-"), ("Ridge", "tab:green", ":")]:
        s = lcv[lcv.modelo == m].sort_values("meses_entrenamiento")
        ax.plot(s["meses_entrenamiento"], s["rmse"], marker="o", ms=3, color=c, ls=ls, label=m)
    ax.set_xscale("log")
    ticks = sorted(lcv["meses_entrenamiento"].unique())
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(v) for v in ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Training months (log scale)")
    ax.set_ylabel("Test RMSE (pp)")
    ax.set_title("Learning curve (US)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_sensibilidad.pdf")
    plt.close(fig)
    # pesos de ALEE y error por régimen (Colombia)
    w = pd.read_csv(rf.OUT / "colombia" / "alee_canonico_pesos.csv").set_index("idx")
    tw = fechas[w.index.values + 12]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 2.4), gridspec_kw={"width_ratios": [1.6, 1]})
    a1.stackplot(tw, w["w_persistencia"], w["w_nucleo_cambio"], w["w_estacional"], labels=["Persistence", "Kernel", "Seasonal"],
                 colors=["tab:red", "tab:green", "tab:blue"], alpha=0.8)
    a1.set_ylabel("Combination weight")
    a1.set_xlabel("Forecast month")
    a1.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    a1.set_title("ALEE expert weights, Colombia")
    er = reg[reg.conjunto == "colombia"].pivot(index="volatilidad", columns="modelo", values="rmse").loc[["baja", "media", "alta"]]
    tres = ["ALEE"] + list(mean.loc["colombia"].loc[CURSO]["RMSE"].sort_values().index[:3])
    xs = np.arange(3)
    for i, m in enumerate(tres):
        a2.bar(xs + (i - 1.5) * 0.2, er[m], width=0.2, label=LAB.get(m, m))
    a2.set_xticks(xs, ["Low", "Medium", "High"])
    a2.set_xlabel("Recent volatility tercile")
    a2.set_ylabel("RMSE (pp)")
    a2.legend(frameon=False, fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    fig.savefig(FIG / "fig_pesos_errores.pdf")
    plt.close(fig)
    imp = pd.read_csv(rf.OUT / "importancia_permutacion_alee.csv").head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.barh(imp["variable"], imp["aumento_rmse_medio"], xerr=imp["aumento_rmse_sd"], color="tab:blue", error_kw={"lw": 0.6})
    ax.set_xlabel("RMSE increase when permuted (pp)")
    ax.set_title("Permutation importance, Colombia")
    fig.tight_layout()
    fig.savefig(FIG / "fig_importancia.pdf")
    plt.close(fig)

    # -------------------------------------------------------------- tablas
    fp, ES, NOMBRE = rf.fp, rf.ES, rf.NOMBRE
    # Tabla RMSE: filas = modelos, columnas = conjuntos (media; la desviación está en la tabla completa)
    lines = [r"\begin{tabular}{l" + "c" * len(CONJ) + "}", r"\toprule", "Modelo & " + " & ".join(NOMBRE[c] for c in CONJ) + r" \\", r"\midrule"]
    orden = {c: mean.loc[c]["RMSE"].sort_values().index.tolist() for c in CONJ}
    for m in ["ALEE"] + CURSO:
        cells = []
        for c in CONJ:
            s = f"{mean.loc[(c, m), 'RMSE']:.3f}"
            s = r"\textbf{" + s + "}" if orden[c][0] == m else (r"\underline{" + s + "}" if orden[c][1] == m else s)
            if m != "ALEE":
                r = pruebas[(pruebas.conjunto == c) & (pruebas.baseline == m)].iloc[0]
                if r.dm_p_holm < 0.05:
                    s += r"$^{\dagger}$" if r.dif_perdida_media < 0 else r"$^{\ddagger}$"
            cells.append(s)
        lines.append(ES[m] + " & " + " & ".join(cells) + r" \\")
        if m == "ALEE":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (L / "tabla_rmse.tex").write_text("\n".join(lines), encoding="utf-8")

    def fmt(q, mu, s):
        return f"{mu:.2f} $\\pm$ {s:.2f}" if q == "MAPE" else f"{mu:.3f} $\\pm$ {s:.3f}"

    # Tabla completa de métricas (longtable): una fila por modelo y conjunto, una columna por métrica
    lines = [r"\begin{longtable}{llccccc}",
             r"\caption{Todas las métricas (media $\pm$ desviación estándar entre los cinco pliegues externos) de los nueve modelos en los nueve conjuntos. En negrita, el mejor valor de cada métrica en cada conjunto.}\label{tab:metricas}\\",
             r"\toprule", r"Conjunto & Modelo & RMSE & MAE & MAPE (\%) & $R^2$ & MASE \\", r"\midrule", r"\endfirsthead",
             r"\toprule", r"Conjunto & Modelo & RMSE & MAE & MAPE (\%) & $R^2$ & MASE \\", r"\midrule", r"\endhead",
             r"\bottomrule", r"\endfoot"]
    for c in CONJ:
        best = {q: (mean.loc[c][q].idxmax() if q == "R2" else mean.loc[c][q].idxmin()) for q in METRICAS}
        for i, m in enumerate(mean.loc[c]["RMSE"].sort_values().index):
            cells = []
            for q in METRICAS:
                s = fmt(q, mean.loc[(c, m), q], sd.loc[(c, m), q])
                cells.append(r"\textbf{" + s + "}" if best[q] == m else s)
            lines.append((NOMBRE[c] if i == 0 else "") + " & " + ES[m] + " & " + " & ".join(cells) + r" \\")
        if c != CONJ[-1]:
            lines.append(r"\midrule")
    lines.append(r"\end{longtable}")
    (L / "tabla_metricas_completa.tex").write_text("\n".join(lines), encoding="utf-8")

    # Tabla de pruebas (longtable), todos los conjuntos, con el estadístico de cada prueba
    lines = [r"\begin{longtable}{llcccccccc}",
             r"\caption{Pruebas de ALEE frente a cada modelo del curso en los nueve conjuntos. $\bar d$: diferencia media de pérdida cuadrática (ALEE menos el modelo; negativo favorece a ALEE) con IC del 95\,\% por \textit{bootstrap} estacionario. DM: estadístico de Diebold-Mariano-HLN; GW: estadístico de Wald de Giacomini-White condicional ($\chi^2$ con 2 g.l.); ambos con $p$ ajustado por Holm sobre los ocho modelos. $P$: probabilidades posteriores de la $t$ bayesiana correlacionada de que ALEE sea mejor, equivalente (ROPE $\pm1$\,\%) o peor. $d$: d de Cohen. $A_{12}$: probabilidad de que el error absoluto de ALEE sea menor.}\label{tab:pruebas}\\",
             r"\toprule", r"Conjunto & Modelo & $\bar d$ [IC 95\,\%] & DM & $p_{\mathrm{Holm}}$ & GW & $p_{\mathrm{Holm}}$ & $P$(mejor/ROPE/peor) & $d$ & $A_{12}$ \\", r"\midrule", r"\endfirsthead",
             r"\toprule", r"Conjunto & Modelo & $\bar d$ [IC 95\,\%] & DM & $p_{\mathrm{Holm}}$ & GW & $p_{\mathrm{Holm}}$ & $P$(mejor/ROPE/peor) & $d$ & $A_{12}$ \\", r"\midrule", r"\endhead",
             r"\bottomrule", r"\endfoot"]
    for c in CONJ:
        for i, (_, r) in enumerate(pruebas[pruebas.conjunto == c].iterrows()):
            lines.append(f"{NOMBRE[c] if i == 0 else ''} & {ES[r.baseline]} & {r.dif_perdida_media:.3f} [{r.ic95_inf:.3f}, {r.ic95_sup:.3f}] & "
                         f"{r.dm_estadistico:.2f} & {fp(r.dm_p_holm)} & {r.gw_estadistico:.1f} & {fp(r.gw_p_holm)} & "
                         f"{r.p_bayes_mejor:.2f}/{r.p_bayes_rope:.2f}/{r.p_bayes_peor:.2f} & {r.d_cohen:.2f} & {r.a12_vargha:.2f} \\\\")
        if c != CONJ[-1]:
            lines.append(r"\midrule")
    lines.append(r"\end{longtable}")
    (L / "tabla_pruebas.tex").write_text("\n".join(lines), encoding="utf-8")

    # Tabla entre conjuntos
    out = [r"\begin{tabular}{lccccc}", r"\toprule",
           r"Modelo & Rango medio & Conjuntos en que ALEE gana & Wilcoxon $p_{\mathrm{Holm}}$ & Conover $p_{\mathrm{Holm}}$ & $P$(mejor/ROPE/peor) \\", r"\midrule",
           f"ALEE & {fr['rango_ALEE']:.2f} & --- & --- & --- & --- \\\\", r"\midrule"]
    for _, r in entre.sort_values("baseline", key=lambda s: s.map(lambda b: fr[f"rango_{b}"])).iterrows():
        out.append(f"{ES[r.baseline]} & {fr['rango_' + r.baseline]:.2f} & {int(r.conjuntos_ALEE_gana)}/{N} & {fp(r.wilcoxon_p_holm)} & {fp(r.conover_p_holm)} & "
                   f"{r.p_bayes_mejor:.3f}/{r.p_bayes_rope:.3f}/{r.p_bayes_peor:.3f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (L / "tabla_entre_conjuntos.tex").write_text("\n".join(out), encoding="utf-8")

    # Tabla de ablación: RMSE y p por conjunto (resumen) + todas las métricas (longtable)
    out = [r"\begin{tabular}{l" + "c" * len(CONJ) + "}", r"\toprule", "Variante & " + " & ".join(NOMBRE[c] for c in CONJ) + r" \\", r"\midrule"]
    for v, _ in VARIANTES:
        cells = []
        for c in CONJ:
            r = abl[(abl.conjunto == c) & (abl.variante == v)].iloc[0]
            s = f"{r.RMSE:.3f}"
            if v != "ALEE completo" and r.dm_p_holm < 0.05:
                # dif_perdida = pérdida de ALEE menos la de la variante: negativa => la variante es peor
                s += r"$^{\dagger}$" if r.dif_perdida < 0 else r"$^{\ddagger}$"
            cells.append(s)
        out.append(v.replace("λ", r"$\lambda$") + " & " + " & ".join(cells) + r" \\")
        if v == "ALEE completo":
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}"]
    (L / "tabla_ablacion.tex").write_text("\n".join(out), encoding="utf-8")
    lines = [r"\begin{longtable}{llcccccc}",
             r"\caption{Ablación con todas las métricas (media entre pliegues) y $p$ de Diebold-Mariano-HLN frente a ALEE completo, ajustado por Holm dentro de cada conjunto.}\label{tab:ablacion_completa}\\",
             r"\toprule", r"Conjunto & Variante & RMSE & MAE & MAPE (\%) & $R^2$ & MASE & DM $p_{\mathrm{Holm}}$ \\", r"\midrule", r"\endfirsthead",
             r"\toprule", r"Conjunto & Variante & RMSE & MAE & MAPE (\%) & $R^2$ & MASE & DM $p_{\mathrm{Holm}}$ \\", r"\midrule", r"\endhead",
             r"\bottomrule", r"\endfoot"]
    for c in CONJ:
        for i, (v, _) in enumerate(VARIANTES):
            r = abl[(abl.conjunto == c) & (abl.variante == v)].iloc[0]
            p = "---" if v == "ALEE completo" else fp(r.dm_p_holm)
            lines.append(f"{NOMBRE[c] if i == 0 else ''} & {v.replace('λ', '$\\lambda$')} & {r.RMSE:.3f} & {r.MAE:.3f} & {r.MAPE:.2f} & {r.R2:.3f} & {r.MASE:.3f} & {p} \\\\")
        if c != CONJ[-1]:
            lines.append(r"\midrule")
    lines.append(r"\end{longtable}")
    (L / "tabla_ablacion_completa.tex").write_text("\n".join(lines), encoding="utf-8")

    co = pd.read_csv(rf.OUT / "costo_computacional.csv")
    co = co[co.modelo.isin(["ALEE"] + CURSO)]
    out = [r"\begin{tabular}{lcccc}", r"\toprule", r"Modelo & Entrenamiento (s) & Inferencia (s) & Memoria (MB) & Parámetros \\", r"\midrule"]
    for _, r in co.iterrows():
        out.append(f"{ES[r.modelo]} & {r.entrenamiento_s:.3f} & {r.inferencia_s:.4f} & {r.memoria_pico_MB:.2f} & {int(r.parametros)} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (L / "tabla_costo.tex").write_text("\n".join(out), encoding="utf-8")
    sv = pd.read_csv(rf.OUT / "semillas_baselines_colombia.csv")
    sv = sv[sv.modelo.isin(CURSO)].groupby("modelo")["rmse_medio"].agg(["mean", "std", "min", "max"])
    sa = pd.read_csv(rf.OUT / "sensibilidad_semillas.csv")
    sa = sa[sa.semilla.astype(str).str.fullmatch(r"\d+")]
    out = [r"\begin{tabular}{llcccc}", r"\toprule", r"Conjunto & Modelo (semillas) & Media & Desv. & Mín. & Máx. \\", r"\midrule"]
    for m, r in sv.iterrows():
        out.append(f"Colombia & {ES[m]} (42--51) & {r['mean']:.3f} & {r['std']:.3f} & {r['min']:.3f} & {r['max']:.3f} \\\\")
    for c in rf.CONJUNTOS_BASE:
        s = sa[sa.conjunto == c]["rmse_medio"]
        out.append(f"{NOMBRE[c]} & ALEE (42--46) & {s.mean():.3f} & {s.std():.3f} & {s.min():.3f} & {s.max():.3f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (L / "tabla_semillas.tex").write_text("\n".join(out), encoding="utf-8")

    # resumen numérico para la redacción
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(mean["RMSE"].unstack().T[CONJ].round(3))
    best_course = {c: (mejor[c], mean.loc[(c, mejor[c]), "RMSE"], mean.loc[(c, "ALEE"), "RMSE"],
                       1 - mean.loc[(c, "ALEE"), "RMSE"] / mean.loc[(c, mejor[c]), "RMSE"]) for c in CONJ}
    print(pd.DataFrame(best_course, index=["mejor_curso", "rmse_mejor", "rmse_alee", "mejora"]).T)
    print("DM max p_holm:", pruebas.dm_p_holm.max(), pruebas.loc[pruebas.dm_p_holm.idxmax(), ["conjunto", "baseline"]].tolist(),
          "| n sig:", int((pruebas.dm_p_holm < 0.05).sum()), "de", len(pruebas))
    print("GW max p_holm:", pruebas.gw_p_holm.max(), pruebas.loc[pruebas.gw_p_holm.idxmax(), ["conjunto", "baseline"]].tolist(),
          "| n sig:", int((pruebas.gw_p_holm < 0.05).sum()))
    print("d:", pruebas.d_cohen.min(), pruebas.d_cohen.max(), "A12:", pruebas.a12_vargha.min(), pruebas.a12_vargha.max(),
          "Pbayes mejor:", pruebas.p_bayes_mejor.min(), pruebas.p_bayes_mejor.max(), "ROPE max:", pruebas.p_bayes_rope.max())
    print(entre.round(5).to_string(index=False))
    print(fr.round(6).to_string())
    m2 = pd.concat(mcs_all)
    for c, g in m2[m2.en_conjunto].groupby("conjunto"):
        print("MCS", c, list(g.modelo))
    print(abl.round(4).to_string(index=False))
    w = met.loc[met.groupby(["conjunto", "pliegue"])["RMSE"].idxmin()]
    print("pliegues ganados por ALEE:", int((w.modelo == "ALEE").sum()), "de", len(w))
    rg = reg.loc[reg.groupby(["conjunto", "volatilidad"])["rmse"].idxmin()]
    print("terciles ganados por ALEE:", int((rg.modelo == "ALEE").sum()), "de", len(rg), rg[rg.modelo != "ALEE"].to_string())
    for q in METRICAS:
        bq = mean.groupby(level=0)[q].idxmax() if q == "R2" else mean.groupby(level=0)[q].idxmin()
        print(q, "ALEE mejor en", sum(v[1] == "ALEE" for v in bq.values), "de", len(bq))


if __name__ == "__main__":
    main()
