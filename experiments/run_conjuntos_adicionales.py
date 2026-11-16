"""Conjuntos adicionales (protocolo, secciones 5, 6 y 14): EE.UU., España y México.

Uso (desde la raíz del proyecto, con el entorno .venv):
    python experiments/run_conjuntos_adicionales.py <pais>     # pais en {eeuu, espana, mexico}

Evalúa ALEE-3 congelado, las siete combinaciones, LightGBM y las referencias
(persistencia, estacional ingenuo, SARIMA, experto estacional de ALEE) con la
misma validación anidada del problema principal. Guarda las predicciones fuera
de muestra en artifacts/adicionales/<pais>/. No cambia ningún diseño.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import optuna  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402

from src import config  # noqa: E402
from src.models import REGRESSOR_BY_NAME, LIGHTGBM_MODEL, ORIGINAL_MODEL  # noqa: E402
from src.nested_cv import run_nested_cv  # noqa: E402
from src.original_model import ModeloOriginal, _Experts, EXPERTS  # noqa: E402
from src.search_space import to_optuna  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
config.RUN_MODE = "full"
config.set_global_seed(config.SEED)
DATA = ROOT / "data" / "adicionales"
ML_MODELS = ["Ridge", "Lasso", "KNN", "DecisionTree", "RandomForest", "XGBoost", "SVR"]
FEATURES = (
    [f"td_nacional_lag{k}" for k in [1, 2, 3, 6, 12]]
    + [f"td_nacional_roll{s}{w}" for w in [3, 6, 12] for s in ("mean", "std")]
    + ["mes_sin", "mes_cos", "trend_idx"]
)
FIRST_ROW = 11  # la fila r del conjunto corresponde al mes r + 11 de la serie (igual que src/data.py)
OECD_ISO = {"canada": "CAN", "australia": "AUS", "japon": "JPN", "alemania": "DEU", "corea": "KOR"}


# ------------------------------------------------------------------ datos
def load_series(pais):
    if pais == "eeuu":
        d = pd.read_csv(DATA / "fred_UNRATENSA.csv")
        s = pd.Series(pd.to_numeric(d["UNRATENSA"], errors="coerce").values, index=pd.to_datetime(d["observation_date"]))
        # 2025-10 no se publicó (cierre del gobierno federal de EE.UU.): interpolación lineal, declarada en protocolo §14.
        assert s.isna().sum() == 1 and s.index[s.isna()][0] == pd.Timestamp("2025-10-01")
        s = s.interpolate(method="linear")
    elif pais == "espana":
        d = pd.read_csv(DATA / "es_une_rt_m.csv")
        s = pd.Series(d["OBS_VALUE"].astype(float).values, index=pd.to_datetime(d["TIME_PERIOD"]))
    elif pais == "mexico":
        d = pd.read_csv(DATA / "mex_nsa.csv", header=None)
        s = pd.Series(d[11].astype(float).values, index=pd.to_datetime(d[10]))
    elif pais in OECD_ISO:
        # protocolo §19: series de la OCDE preparadas con scripts/preparar_oecd.py
        d = pd.read_csv(DATA / f"oecd_{OECD_ISO[pais]}.csv")
        s = pd.Series(d["valor"].astype(float).values, index=pd.to_datetime(d["fecha"]))
    else:
        raise ValueError(pais)
    s = s.sort_index()
    full = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    assert len(full) == len(s), f"{pais}: la serie tiene meses faltantes"
    return s


def build_dataset(s):
    """Mismas variables que src/data.build_features, sin las tres de área."""
    t = pd.Series(s.values)
    f = pd.DataFrame(index=t.index)
    for lag in [1, 2, 3, 6, 12]:
        f[f"td_nacional_lag{lag}"] = t.shift(lag - 1)
    for w in [3, 6, 12]:
        f[f"td_nacional_rollmean{w}"] = t.rolling(w).mean()
        f[f"td_nacional_rollstd{w}"] = t.rolling(w).std()
    month = s.index.month.values
    f["mes_sin"] = np.sin(2 * np.pi * month / 12)
    f["mes_cos"] = np.cos(2 * np.pi * month / 12)
    f["trend_idx"] = np.arange(len(t))
    y = t.shift(-1)
    ok = f.notna().all(axis=1) & y.notna()
    assert int(np.flatnonzero(ok.values)[0]) == FIRST_ROW
    X = f.loc[ok, FEATURES].reset_index(drop=True)
    return X, y[ok].reset_index(drop=True), s.index[ok.values]


# ------------------------------------------------------------------ ALEE
class ExpertErrors:
    """Errores fuera de muestra de los expertos de toda la serie (ventana expansiva).

    El error del mes s solo usa los meses 0..s-1, así que el ajuste de
    ModeloOriginal sobre el prefijo 0..n'-1 usa exactamente las primeras filas
    de esta tabla (con m0 = 36, que aplica cuando n' >= 72).
    """

    def __init__(self, X, y, cols, m0=36):
        self.m0 = m0
        self.E = np.array([y[s] - _Experts(X[:s], y[:s], cols).predict(X[s:s + 1])[0] for s in range(m0, len(y))])


def alee_fit(params, X, y, n_train, pre):
    m = ModeloOriginal(feature_names=tuple(FEATURES), random_state=config.SEED, **params)
    m0 = int(min(m.min_train, max(2, n_train // 2)))
    if m0 != pre.m0:  # prefijos cortos: ajuste estándar
        return m.fit(X[:n_train], y[:n_train])
    m._validate_params()
    m.cols_ = {"lag1": m._col(m.lag_col), "mes_sin": m._col("mes_sin"), "mes_cos": m._col("mes_cos")}
    Xt = X[:n_train]
    m.errors_ = pre.E[: n_train - m0]
    m.oof_rows_ = np.arange(m0, n_train)
    st = m._state(Xt)
    m.state_mu_, m.state_sd_ = st.mean(0), st.std(0)
    m.state_sd_[m.state_sd_ < 1e-12] = 1.0
    m.state_ = (st[m.oof_rows_] - m.state_mu_) / m.state_sd_
    g = 0.0 if np.isinf(m.half_life) else np.log(0.5) / m.half_life
    m.log_decay_ = g * (n_train - 1 - m.oof_rows_)
    m.experts_ = _Experts(Xt, y[:n_train], m.cols_)
    m.n_features_in_ = X.shape[1]
    m.expert_names_ = EXPERTS
    m.prior_ = m._global_prior()
    return m


def run_alee(X, y, log):
    cols = {"lag1": FEATURES.index("td_nacional_lag1"), "mes_sin": FEATURES.index("mes_sin"), "mes_cos": FEATURES.index("mes_cos")}
    t0 = time.time()
    pre = ExpertErrors(X, y, cols)
    log(f"  ALEE: errores de expertos precalculados en {time.time() - t0:.0f}s")
    # verificación contra .fit()
    p0 = dict(bandwidth=0.7, half_life=40.0, lam=0.05)
    n_chk = min(200, len(y) - 30)
    a = alee_fit(p0, X, y, n_chk, pre).predict(X[n_chk:n_chk + 24])
    b = ModeloOriginal(feature_names=tuple(FEATURES), **p0).fit(X[:n_chk], y[:n_chk]).predict(X[n_chk:n_chk + 24])
    diff = float(np.abs(a - b).max())
    log(f"  ALEE: diferencia máxima vs ModeloOriginal.fit() = {diff:.2e}")
    assert diff < 1e-8
    outer = TimeSeriesSplit(n_splits=config.N_SPLITS_OUTER, gap=config.CV_GAP)
    oof, seas, hp = [], [], []
    for k, (tr, te) in enumerate(outer.split(X)):
        n_tr = len(tr)
        inner = list(TimeSeriesSplit(n_splits=config.N_SPLITS_INNER, gap=config.CV_GAP).split(X[:n_tr]))

        def objective(trial):
            params = to_optuna(ORIGINAL_MODEL.space, trial)
            sc = []
            for itr, iva in inner:
                try:
                    pred = alee_fit(params, X, y, len(itr), pre).predict(X[iva])
                    sc.append(-float(np.sqrt(np.mean((y[iva] - pred) ** 2))))
                except Exception:
                    continue
            return float(np.mean(sc)) if sc else -1e9

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.SEED, n_startup_trials=10))
        study.optimize(objective, n_trials=config.FULL_BUDGETS["bayes_n_trials"], show_progress_bar=False)
        best = study.best_params
        m = alee_fit(best, X, y, n_tr, pre)
        oof.append(pd.DataFrame({"idx": te, "y_true": y[te], "y_pred": m.predict(X[te])}))
        ex = m.expert_predictions(X[te])
        seas.append(pd.DataFrame({"idx": te, "y_true": y[te], "y_pred": ex[:, EXPERTS.index("estacional")]}))
        hp.append({"pliegue": k, **best, "inner_score": study.best_value})
        log(f"  ALEE pliegue {k}: {best}")
    return pd.concat(oof, ignore_index=True), pd.concat(seas, ignore_index=True), pd.DataFrame(hp), diff


# ------------------------------------------------------------- referencias
def baselines(s, X, y):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    ys = s.values
    rows = []
    for tr, te in TimeSeriesSplit(n_splits=config.N_SPLITS_OUTER, gap=config.CV_GAP).split(X):
        a, last = len(tr), int(te[-1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = SARIMAX(ys[: a + FIRST_ROW + 1], order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False, maxiter=200)
        ext = fit.append(ys[a + FIRST_ROW + 1: last + FIRST_ROW + 2], refit=False)
        sar = ext.get_prediction(start=int(te[0]) + FIRST_ROW + 1, end=last + FIRST_ROW + 1, dynamic=False).predicted_mean
        rows.append(pd.DataFrame({"idx": te, "y_true": y[te], "Persistencia": X[te, FEATURES.index("td_nacional_lag1")],
                                  "Estacional": X[te, FEATURES.index("td_nacional_lag12")], "SARIMA": np.asarray(sar)}))
    return pd.concat(rows, ignore_index=True)


def main():
    pais = sys.argv[1]
    out = ROOT / "artifacts" / "adicionales" / pais
    out.mkdir(parents=True, exist_ok=True)
    logf = out / "log.txt"

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(logf, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    s = load_series(pais)
    Xdf, ydf, fechas = build_dataset(s)
    X, y = Xdf.to_numpy(float), ydf.to_numpy(float)
    log(f"{pais}: serie {s.index.min():%Y-%m} a {s.index.max():%Y-%m} ({len(s)} meses), {len(y)} filas, {X.shape[1]} variables")
    pd.DataFrame({"idx": np.arange(len(y)), "fecha_objetivo": (fechas + pd.offsets.MonthBegin(1)).strftime("%Y-%m"), "y": y}).to_csv(out / "fechas.csv", index=False)

    if not (out / "baselines_series_tiempo.csv").exists():
        baselines(s, X, y).to_csv(out / "baselines_series_tiempo.csv", index=False)
        log("referencias listas")

    if not (out / "oof_ALEE.csv").exists():
        oof, seas, hp, diff = run_alee(X, y, log)
        oof.to_csv(out / "oof_ALEE.csv", index=False)
        seas.to_csv(out / "oof_EstacionalALEE.csv", index=False)
        hp.assign(dif_vs_fit=diff).to_csv(out / "hiperparametros_ALEE.csv", index=False)

    for name in ML_MODELS + ["LightGBM"]:
        path = out / f"oof_{name}.csv"
        if path.exists():
            continue
        spec = LIGHTGBM_MODEL if name == "LightGBM" else REGRESSOR_BY_NAME[name]
        t0 = time.time()
        res = run_nested_cv(model_spec=spec, X=Xdf, y=ydf, task="reg", balancing_method="none",
                            optimizer_name="bayesian", seed=config.SEED, n_jobs=5, budgets=config.FULL_BUDGETS)
        if res.error:
            log(f"  ERROR {name}: {res.error}")
        rows = [pd.DataFrame({"idx": d["test_idx"], "y_true": d["y_true"], "y_pred": d["y_pred"]}) for d in res.oof.values()]
        pd.concat(rows, ignore_index=True).sort_values("idx").to_csv(path, index=False)
        log(f"  {name}: {time.time() - t0:.0f}s")
    log("terminado")


if __name__ == "__main__":
    main()
