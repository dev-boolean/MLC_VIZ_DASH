"""Pruebas pre-registradas (protocolo, secciones 6 y 14) sobre los conjuntos adicionales.

Uso: python experiments/analizar_conjuntos_adicionales.py
Lee artifacts/adicionales/<pais>/ y artifacts/confirmatorio/ (Colombia) y escribe
artifacts/adicionales/resumen_*.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402
from scipy import stats  # noqa: E402

from src.stats_tests import (  # noqa: E402
    model_confidence_set, giacomini_white_test, diebold_mariano_test, spa_test,
    adjust_pvalues, cliffs_delta, cohens_d_paired,
)

BASE = ROOT / "artifacts" / "adicionales"
PAISES = ["eeuu", "espana", "mexico"]
ML = ["Ridge", "Lasso", "KNN", "DecisionTree", "RandomForest", "XGBoost", "SVR", "LightGBM"]


def load(folder, colombia=False):
    preds = {}
    names = ["ALEE", "EstacionalALEE"] + ML
    for n in names:
        f = folder / (f"oof_{n}.csv" if not (colombia and n == "ALEE") else "iter4/oof_ALEE_K3_sin_ridge.csv")
        if f.exists():
            preds[n] = pd.read_csv(f).sort_values("idx").reset_index(drop=True)
    bl = pd.read_csv(folder / "baselines_series_tiempo.csv").sort_values("idx").reset_index(drop=True)
    idx = preds["ALEE"]["idx"].values
    y = preds["ALEE"]["y_true"].values
    E = {}
    for n, d in preds.items():
        assert (d["idx"].values == idx).all(), n
        E[n] = d["y_true"].values - d["y_pred"].values
    for c in ["Persistencia", "Estacional", "SARIMA"]:
        assert (bl["idx"].values == idx).all()
        E[c] = bl["y_true"].values - bl[c].values
    return pd.DataFrame(E), idx, y


def fold_rmse(E, idx, n_rows):
    folds = list(TimeSeriesSplit(n_splits=5, gap=1).split(np.arange(n_rows)))
    out = {}
    for k, (_, te) in enumerate(folds):
        m = np.isin(idx, te)
        out[k] = np.sqrt((E[m] ** 2).mean())
    return pd.DataFrame(out).T  # filas = pliegues


def analyse(name, E, idx, n_rows):
    F = fold_rmse(E, idx, n_rows)
    tabla = pd.DataFrame({"rmse_medio_pliegues": F.mean(), "rmse_sd_pliegues": F.std(),
                          "rmse_agrupado": np.sqrt((E ** 2).mean()), "mae": E.abs().mean()}).sort_values("rmse_medio_pliegues")
    best = tabla.drop(index="ALEE").index[0]
    losses = E ** 2
    mcs = model_confidence_set(losses, alpha=0.10, block=12, seed=42)
    fin = mcs[mcs["en_conjunto"]]["modelo"].tolist()
    gw = [{"comparacion": f"ALEE vs {m}", **giacomini_white_test(E["ALEE"], E[m], conditional=True)} for m in fin if m != "ALEE"]
    gw = pd.DataFrame(gw)
    if len(gw):
        gw["p_holm"] = adjust_pvalues(gw["p_value"].values, method="holm")
    dm_best = diebold_mariano_test(E["ALEE"], E[best])
    dm_sar = diebold_mariano_test(E["ALEE"], E["SARIMA"])
    dm_est = diebold_mariano_test(E["ALEE"], E["EstacionalALEE"])
    spa = spa_test(losses[best], losses.drop(columns=[best]), seed=42)
    diffs = E["ALEE"].abs() - E[best].abs()
    res = {
        "conjunto": name, "n_meses_prueba": len(E), "mejor_referencia": best,
        "rmse_ALEE": tabla.loc["ALEE", "rmse_medio_pliegues"], "rmse_mejor_ref": tabla.loc[best, "rmse_medio_pliegues"],
        "rmse_SARIMA": tabla.loc["SARIMA", "rmse_medio_pliegues"], "rmse_EstacionalALEE": tabla.loc["EstacionalALEE", "rmse_medio_pliegues"],
        "posicion_ALEE": int(list(tabla.index).index("ALEE")) + 1, "n_modelos": len(tabla),
        "ALEE_mejor_que_mejor_ref": bool(tabla.loc["ALEE", "rmse_medio_pliegues"] < tabla.loc[best, "rmse_medio_pliegues"]),
        "ALEE_en_MCS": "ALEE" in fin, "finalistas_MCS": ";".join(fin),
        "DM_vs_mejor_p": dm_best["p_value"], "DM_vs_mejor_dif": dm_best["mean_loss_diff"],
        "DM_vs_SARIMA_p": dm_sar["p_value"], "DM_vs_SARIMA_dif": dm_sar["mean_loss_diff"],
        "DM_vs_EstALEE_p": dm_est["p_value"], "DM_vs_EstALEE_dif": dm_est["mean_loss_diff"],
        "GW_min_p_holm": float(gw["p_holm"].min()) if len(gw) else np.nan,
        "SPA_p": spa["p_consistent"], "SPA_mejor": spa["best_candidate"],
        "d_cohen": cohens_d_paired(diffs), "delta_cliff": cliffs_delta(E["ALEE"].abs(), E[best].abs()),
    }
    return tabla, mcs, gw, res, F


def main():
    resumen, folds_all = [], {}
    sets = [("colombia", ROOT / "artifacts" / "confirmatorio", True, 288)] + \
           [(p, BASE / p, False, None) for p in PAISES]
    for name, folder, col, n_rows in sets:
        E, idx, y = load(folder, colombia=col)
        if col:
            E = E.drop(columns=[c for c in ["LightGBM"] if c not in E.columns], errors="ignore")
        n_rows = n_rows or len(pd.read_csv(folder / "fechas.csv"))
        tabla, mcs, gw, res, F = analyse(name, E, idx, n_rows)
        out = BASE / f"tabla_{name}.csv"
        tabla.to_csv(out)
        mcs.to_csv(BASE / f"mcs_{name}.csv", index=False)
        gw.to_csv(BASE / f"gw_{name}.csv", index=False)
        resumen.append(res)
        folds_all[name] = tabla["rmse_medio_pliegues"]
        print(f"\n=== {name} ({len(E)} meses de prueba) ===")
        print(tabla.round(4).to_string())
    R = pd.DataFrame(resumen)
    adic = R[R.conjunto != "colombia"].copy()
    adic["DM_vs_mejor_p_holm"] = adjust_pvalues(adic["DM_vs_mejor_p"].values, method="holm")
    adic["significativamente_peor"] = (adic["DM_vs_mejor_p_holm"] < 0.05) & (adic["DM_vs_mejor_dif"] > 0)
    n_mejor = int(adic["ALEE_mejor_que_mejor_ref"].sum())
    cond_d = (n_mejor >= 2) and not adic["significativamente_peor"].any()
    R = R.merge(adic[["conjunto", "DM_vs_mejor_p_holm", "significativamente_peor"]], on="conjunto", how="left")
    R.to_csv(BASE / "resumen_conjuntos.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print("\n=== Resumen ===")
    print(R.round(4).T.to_string())
    print(f"\nCondición (d): ALEE mejor que el mejor de referencia en {n_mejor}/3 conjuntos adicionales; "
          f"significativamente peor en {int(adic['significativamente_peor'].sum())}. -> {'SE CUMPLE' if cond_d else 'NO SE CUMPLE'}")

    # Entre conjuntos: Friedman / Iman-Davenport y Wilcoxon-Holm (ALEE vs cada uno)
    W = pd.DataFrame(folds_all).T  # filas = conjuntos, columnas = modelos
    W = W.dropna(axis=1)
    k, N = W.shape[1], W.shape[0]
    ranks = W.rank(axis=1)
    chi2, p_f = stats.friedmanchisquare(*[W[c].values for c in W.columns])
    ff = (N - 1) * chi2 / (N * (k - 1) - chi2) if N * (k - 1) != chi2 else np.inf
    p_id = 1 - stats.f.cdf(ff, k - 1, (k - 1) * (N - 1))
    print(f"\nFriedman sobre {N} conjuntos y {k} modelos: chi2={chi2:.3f}, p={p_f:.4f}; Iman-Davenport F={ff:.3f}, p={p_id:.4f}")
    print("Rango medio (1 = mejor):")
    print(ranks.mean().sort_values().round(2).to_string())
    wil = []
    for c in W.columns:
        if c == "ALEE":
            continue
        d = W["ALEE"] - W[c]
        try:
            p = stats.wilcoxon(W["ALEE"], W[c]).pvalue
        except ValueError:
            p = 1.0
        wil.append({"comparacion": f"ALEE vs {c}", "conjuntos_ALEE_mejor": int((d < 0).sum()), "p": p})
    wil = pd.DataFrame(wil)
    wil["p_holm"] = adjust_pvalues(wil["p"].values, method="holm")
    print(wil.round(4).to_string(index=False))
    ranks.to_csv(BASE / "rangos_entre_conjuntos.csv")
    wil.to_csv(BASE / "wilcoxon_holm_entre_conjuntos.csv", index=False)
    pd.Series({"friedman_chi2": chi2, "friedman_p": p_f, "iman_davenport_F": ff, "iman_davenport_p": p_id,
               "condicion_d": cond_d, "n_conjuntos_ALEE_mejor": n_mejor}).to_csv(BASE / "friedman_y_condicion_d.csv")


if __name__ == "__main__":
    main()
