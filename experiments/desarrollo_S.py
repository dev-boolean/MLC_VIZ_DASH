"""Ronda S, fase de desarrollo (protocolo §16.2): variantes de ALEE-S sobre los 4 conjuntos ya vistos.

Uso:
    python experiments/desarrollo_S.py prep <conjunto>             # historial de expertos y SARIMA
    python experiments/desarrollo_S.py run <conjunto> <variante>   # validación anidada de una variante
    python experiments/desarrollo_S.py resumen                      # rangos frente a los 13 modelos
conjunto en {colombia, eeuu, espana, mexico}.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import optuna  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402

from src import config  # noqa: E402
from src.alee_s import ALEES, expert_history, sarima_walkforward  # noqa: E402
from src.data import load_raw  # noqa: E402
from src.models import ORIGINAL_MODEL  # noqa: E402
from src.search_space import to_optuna  # noqa: E402
from run_conjuntos_adicionales import FEATURES, FIRST_ROW, build_dataset, load_series  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
OUT = ROOT / "artifacts" / "desarrollo_S"

VARIANTES = {
    "S1": dict(experts=("sarima", "persistencia", "nucleo_cambio", "estacional"), prior="ancla", anchor=0.94),
    "S2": dict(experts=("sarima", "persistencia", "nucleo_cambio", "estacional"), prior="global"),
    "S3": dict(experts=("sarima", "persistencia"), prior="ancla", anchor=0.94),
    "S4": dict(experts=("sarima", "persistencia", "estacional"), prior="ancla", anchor=0.94),
    # S5-S7: el peso del ancla se elige en los pliegues internos, en [0.5, 0.995]
    "S5": dict(experts=("sarima", "persistencia", "estacional"), prior="ancla", anchor="tuned"),
    "S6": dict(experts=("sarima", "persistencia", "nucleo_cambio", "estacional"), prior="ancla", anchor="tuned"),
    "S7": dict(experts=("sarima", "persistencia"), prior="ancla", anchor="tuned"),
}
ANCHOR_RANGE = (0.5, 0.995)


def get_data(conj):
    if conj == "colombia":
        d = load_raw()
        s = pd.Series(d["tasa_desempleo_nacional"].values, index=pd.to_datetime(d["fecha"]))
        frozen = ROOT / "artifacts" / "confirmatorio" / "baselines_series_tiempo.csv"
    else:
        s = load_series(conj)
        frozen = ROOT / "artifacts" / "adicionales" / conj / "baselines_series_tiempo.csv"
    X, y, _ = build_dataset(s)
    return s, X.to_numpy(float), y.to_numpy(float), pd.read_csv(frozen).sort_values("idx")


def prep(conj):
    s, X, y, _ = get_data(conj)
    cols = {"lag1": FEATURES.index("td_nacional_lag1"), "mes_sin": FEATURES.index("mes_sin"), "mes_cos": FEATURES.index("mes_cos")}
    t0 = time.time()
    sp = sarima_walkforward(s.values, FIRST_ROW, len(y))
    F, E = expert_history(X, y, cols, sp)
    (OUT / conj).mkdir(parents=True, exist_ok=True)
    np.savez(OUT / conj / "hist.npz", F=F, E=E)
    print(f"{conj}: historial listo en {time.time() - t0:.0f}s, filas {len(y)}", flush=True)


def run(conj, var):
    s, X, y, frozen = get_data(conj)
    h = np.load(OUT / conj / "hist.npz")
    F, E = h["F"], h["E"]
    cfg = dict(VARIANTES[var])
    tuned = cfg.get("anchor") == "tuned"
    if tuned:
        cfg.pop("anchor")
    sar_frozen = pd.Series(frozen["SARIMA"].values, index=frozen["idx"].values)
    rows, hp = [], []
    t0 = time.time()
    for k, (tr, te) in enumerate(TimeSeriesSplit(n_splits=config.N_SPLITS_OUTER, gap=config.CV_GAP).split(X)):
        n_tr = len(tr)
        inner = list(TimeSeriesSplit(n_splits=config.N_SPLITS_INNER, gap=config.CV_GAP).split(X[:n_tr]))

        def objective(trial):
            p = to_optuna(ORIGINAL_MODEL.space, trial)
            if tuned:
                p["anchor"] = trial.suggest_float("anchor", *ANCHOR_RANGE)
            sc = []
            for itr, iva in inner:
                try:
                    m = ALEES(feature_names=FEATURES, **cfg, **p).fit(X, y, len(itr), E)
                except ValueError:
                    continue
                Fv = m.own_expert_predictions(X[iva], F[iva, 0])
                sc.append(-float(np.sqrt(np.mean((y[iva] - m.predict(X[iva], Fv)) ** 2))))
            return float(np.mean(sc)) if sc else -1e9

        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.SEED, n_startup_trials=10))
        st.optimize(objective, n_trials=config.FULL_BUDGETS["bayes_n_trials"], show_progress_bar=False)
        m = ALEES(feature_names=FEATURES, **cfg, **st.best_params).fit(X, y, n_tr, E)
        Ft = m.own_expert_predictions(X[te], sar_frozen.loc[te].values)
        W = m.weights(X[te])
        rows.append(pd.DataFrame({"idx": te, "y_true": y[te], "y_pred": (W * Ft[:, m.sel]).sum(1),
                                  "peso_sarima_medio": W[:, 0] if "sarima" in cfg["experts"] else np.nan}))
        hp.append({"pliegue": k, **st.best_params})
    d = OUT / conj
    pd.concat(rows).to_csv(d / f"oof_{var}.csv", index=False)
    pd.DataFrame(hp).to_csv(d / f"hp_{var}.csv", index=False)
    print(f"{conj} {var}: {time.time() - t0:.0f}s", flush=True)


def resumen():
    folder = {"colombia": ROOT / "artifacts" / "confirmatorio"}
    for p in ["eeuu", "espana", "mexico"]:
        folder[p] = ROOT / "artifacts" / "adicionales" / p
    sys.path.insert(0, str(ROOT / "experiments"))
    from analizar_conjuntos_adicionales import load, fold_rmse
    rows, ranks = [], []
    for conj, f in folder.items():
        Eref, idx, yv = load(f, colombia=(conj == "colombia"))
        n_rows = 288 if conj == "colombia" else len(pd.read_csv(f / "fechas.csv"))
        h = np.load(OUT / conj / "hist.npz")
        Eref["SARIMA_reestimado12"] = h["E"][idx, 0]
        for v in VARIANTES:
            p = OUT / conj / f"oof_{v}.csv"
            if p.exists():
                o = pd.read_csv(p).sort_values("idx")
                assert (o["idx"].values == idx).all()
                Eref[v] = o["y_true"].values - o["y_pred"].values
        R = fold_rmse(Eref, idx, n_rows).mean()
        base = [c for c in R.index if c not in VARIANTES and c != "SARIMA_reestimado12"]
        for v in [c for c in R.index if c in VARIANTES]:
            rk = R[base + [v]].rank()
            ranks.append({"conjunto": conj, "variante": v, "rango": rk[v], "rango_SARIMA": rk["SARIMA"]})
        rows.append(R.rename(conj))
    T = pd.DataFrame(rows).T
    pd.set_option("display.width", 200)
    print("RMSE medio de pliegues:")
    print(T.loc[["SARIMA", "SARIMA_reestimado12", "ALEE", "EstacionalALEE"] + [v for v in VARIANTES if v in T.index]].round(4).to_string())
    K = pd.DataFrame(ranks)
    print("\nRango entre los 13 modelos + la variante (1 = mejor):")
    print(K.pivot(index="variante", columns="conjunto", values="rango").assign(medio=lambda d: d.mean(axis=1)).round(2).to_string())
    print("Rango medio de SARIMA en esos mismos rankings:", K.groupby("variante")["rango_SARIMA"].mean().round(2).to_dict())
    T.to_csv(OUT / "rmse_desarrollo.csv")
    K.to_csv(OUT / "rangos_desarrollo.csv", index=False)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        prep(sys.argv[2])
    elif cmd == "run":
        run(sys.argv[2], sys.argv[3])
    else:
        resumen()
