"""Precalcula todo lo que el dashboard muestra y que sería costoso calcular en línea.

El dashboard nunca entrena modelos ni corre pruebas pesadas: solo lee los
archivos que deja este script en ``data/dashboard/`` y los artefactos de
``artifacts/``. Se vuelve a correr cada vez que cambian los artefactos o los
registros de experimentos::

    python scripts/build_dashboard_data.py

Archivos que genera:

* ``oof_reg.csv`` y ``oof_clf.csv``: predicciones fuera de muestra (240 meses)
  de la mejor combinación de cada modelo, del modelo original (ALEE de tres
  expertos) y de los modelos que el paper agregó (regresión bayesiana, SVR con
  kernel RBF y LightGBM).
* ``mcs.csv``: Model Confidence Set con todos los modelos de regresión anteriores.
* ``friedman_clf.json``: rangos promedio, estadístico y diferencia crítica de
  Friedman-Nemenyi con la mejor variante de cada clasificador.
* ``conformal.json``: semiancho del intervalo conforme al 90 % y su cobertura
  hacia adelante, por modelo de regresión.
* ``shap/<artefacto>.npz``: valores de Shapley de cada fila (interpretabilidad global).
* ``background.npy``: filas de referencia para los valores de Shapley en línea.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import analysis as A  # noqa: E402
from src import config  # noqa: E402
from src import stats_tests as ST  # noqa: E402
from src.artifacts import ARTIFACTS_DIR, SELECTION_PATH  # noqa: E402
from src.data import assemble_dataset  # noqa: E402
from src.explain import conformal_halfwidth, sequential_coverage, shapley_values  # noqa: E402

OUT = config.ROOT_DIR / "data" / "dashboard"
N_BACKGROUND = 20
# Predicciones fuera de muestra en Colombia de los modelos del paper que no están
# entre las 140 combinaciones: ALEE de tres expertos (iteración 4, la versión del
# paper), la regresión bayesiana, el SVR con kernel RBF y LightGBM.
_RES = config.ROOT_DIR / "artifacts"
PAPER_OOF = {
    "ALEE": _RES / "confirmatorio" / "iter4" / "oof_ALEE_K3_sin_ridge.csv",
    "BayesianRidge": _RES / "resultados" / "colombia" / "oof_BayesianRidge.csv",
    "SVR_RBF": _RES / "resultados" / "colombia" / "oof_SVR_RBF.csv",
    "LightGBM": _RES / "confirmatorio" / "oof_LightGBM.csv",
}
ALTO = config.CLASS_LABELS.index("Alto")


def target_dates(ds, idx):
    """Mes pronosticado (t+1) de cada fila."""
    return (ds.fechas.iloc[idx] + pd.offsets.MonthEnd(1)).dt.strftime("%Y-%m-%d").values


def build_oof(ds, selection):
    """Predicciones fuera de muestra apiladas de los cinco pliegues externos."""
    reg_rows, clf_rows = None, None
    for task in ["reg", "clf"]:
        frame = None
        for _, row in selection[selection["task"] == task].iterrows():
            p = A.pooled_oof(row["run_id"])
            order = np.argsort(p["idx"])
            idx = p["idx"][order]
            if frame is None:
                frame = pd.DataFrame({"idx": idx, "fecha": target_dates(ds, idx), "fold": p["fold"][order],
                                      "y_true": p["y_true"][order]})
            frame[f"pred_{row['model']}"] = p["y_pred"][order]
            if task == "clf":
                for k, label in enumerate(config.CLASS_LABELS):
                    frame[f"proba_{row['model']}_{label}"] = p["y_proba"][order][:, k]
        if task == "reg":
            for name, path in PAPER_OOF.items():
                p = pd.read_csv(path).set_index("idx").loc[frame["idx"]]
                if not np.allclose(p["y_true"].values, frame["y_true"].values):
                    raise ValueError(f"Las predicciones de {name} no corresponden a los mismos meses")
                frame[f"pred_{name}"] = p["y_pred"].values
            reg_rows = frame
        else:
            clf_rows = frame
    return reg_rows, clf_rows


def build_mcs(oof_reg):
    cols = [c for c in oof_reg.columns if c.startswith("pred_")]
    losses = pd.DataFrame({c[5:]: (oof_reg["y_true"] - oof_reg[c]) ** 2 for c in cols})
    return ST.model_confidence_set(losses, alpha=0.10, n_boot=5000, block=12)


def build_friedman(selection):
    folds = A.load_folds()
    f1 = folds[folds["metric"] == "f1_macro"]
    wide = pd.DataFrame({
        row["model"]: f1[f1["run_id"] == row["run_id"]].sort_values("fold")["value"].values
        for _, row in selection[selection["task"] == "clf"].iterrows()
    })
    fr = ST.friedman_test(wide)
    return {"statistic": fr.statistic, "p_value": fr.p_value, "df": fr.df, "n_blocks": int(len(wide)),
            "avg_ranks": fr.avg_ranks.to_dict(), "cd": ST.critical_difference(wide.shape[1], wide.shape[0])}


def build_conformal(oof_reg):
    out = {}
    for c in [c for c in oof_reg.columns if c.startswith("pred_")]:
        res = (oof_reg["y_true"] - oof_reg[c]).values
        out[c[5:]] = {"halfwidth_90": conformal_halfwidth(res, 0.9), "nominal": 0.9,
                      "forward": sequential_coverage(res, 0.9, warmup=48)}
    return out


def output_fn(pipe, task):
    """Salida que se explica: el pronóstico en regresión y P(Alto) en clasificación."""
    if task == "reg":
        return lambda M: pipe.predict(M)
    return lambda M: pipe.predict_proba(M)[:, ALTO]


def build_shap(ds, background):
    (OUT / "shap").mkdir(parents=True, exist_ok=True)
    X = ds.X.values
    for art in sorted(p for p in ARTIFACTS_DIR.glob("*_*") if (p / "model.joblib").exists()):
        t0 = time.perf_counter()
        task = art.name.split("_")[0]
        pipe = joblib.load(art / "model.joblib")
        f = output_fn(pipe, task)
        fdf = lambda M, f=f: f(pd.DataFrame(M, columns=ds.feature_names))
        # El modelo original es más lento: se explican uno de cada tres meses con menos permutaciones.
        rows = np.arange(0, len(X), 3) if art.name == "reg_ALEE" else np.arange(len(X))
        bg, n_perm = (background[:5], 4) if art.name == "reg_ALEE" else (background, 10)
        phi = np.array([shapley_values(fdf, X[i], bg, n_permutations=n_perm, seed=config.SEED)["values"] for i in rows])
        np.savez_compressed(OUT / "shap" / f"{art.name}.npz", rows=rows, phi=phi, columns=np.array(ds.feature_names))
        print(f"  SHAP {art.name:24s} {len(rows)} filas  {time.perf_counter() - t0:6.1f} s", flush=True)


def main():
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    ds = assemble_dataset()
    selection = pd.read_csv(SELECTION_PATH)
    rng = np.random.default_rng(config.SEED)
    background = ds.X.values[np.sort(rng.choice(len(ds.X), N_BACKGROUND, replace=False))]
    np.save(OUT / "background.npy", background)

    oof_reg, oof_clf = build_oof(ds, selection)
    oof_reg.to_csv(OUT / "oof_reg.csv", index=False)
    oof_clf.to_csv(OUT / "oof_clf.csv", index=False)
    build_mcs(oof_reg).to_csv(OUT / "mcs.csv", index=False)
    with open(OUT / "friedman_clf.json", "w", encoding="utf-8") as fh:
        json.dump(build_friedman(selection), fh, indent=2)
    with open(OUT / "conformal.json", "w", encoding="utf-8") as fh:
        json.dump(build_conformal(oof_reg), fh, indent=2)
    print("Predicciones, MCS, Friedman e intervalos listos", flush=True)
    build_shap(ds, background)


if __name__ == "__main__":
    main()
