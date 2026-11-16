"""Sección de resultados del paper: un único script que reproduce cada tabla, prueba y figura.

Etapas (cada una guarda su salida en artifacts/resultados/ y se puede volver a correr):
    python experiments/resultados_finales.py modelos <conjunto>     # BayesianRidge y SVR-RBF (modelos del curso que faltaban)
    python experiments/resultados_finales.py alee <conjunto> <variante> [semilla]
        variantes: completo, sin_localizacion, sin_lambda_adaptativa, previo_uniforme
    python experiments/resultados_finales.py fijos <conjunto>       # promedio simple, núcleo solo, mapa de sensibilidad
    python experiments/resultados_finales.py costo                  # costo computacional (Colombia)
    python experiments/resultados_finales.py analisis               # tablas, pruebas y figuras
conjunto en {colombia, eeuu, espana, mexico}. Ver experiments/protocolo.md §17.
"""
import json
import sys
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import optuna  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402

from src import config  # noqa: E402
from src.data import assemble_dataset  # noqa: E402
from src.models import BAYESIAN_RIDGE_MODEL, SVR_RBF_MODEL, ORIGINAL_MODEL, REGRESSOR_BY_NAME, LIGHTGBM_MODEL  # noqa: E402
from src.nested_cv import run_nested_cv  # noqa: E402
from src.original_model import ModeloOriginal, _Experts, EXPERTS, DEFAULT_FEATURES  # noqa: E402
from src.search_space import to_optuna  # noqa: E402
from src import stats_tests as ST  # noqa: E402
from run_conjuntos_adicionales import FEATURES as F14, build_dataset, load_series  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
config.RUN_MODE = "full"
OUT = ROOT / "artifacts" / "resultados"
FIG = OUT / "figuras"
CONJUNTOS_BASE = ["colombia", "eeuu", "espana", "mexico"]
CONJUNTOS_NUEVOS = ["canada", "australia", "japon", "alemania", "corea"]  # protocolo §19
CONJUNTOS = CONJUNTOS_BASE  # el análisis completo (con referencias) usa los 4 originales
NOMBRE = {"colombia": "Colombia", "eeuu": "EE.UU.", "espana": "España", "mexico": "México", "canada": "Canadá",
          "australia": "Australia", "japon": "Japón", "alemania": "Alemania", "corea": "Corea del Sur"}
NAME_EN = {"colombia": "Colombia", "eeuu": "United States", "espana": "Spain", "mexico": "Mexico", "canada": "Canada",
           "australia": "Australia", "japon": "Japan", "alemania": "Germany", "corea": "South Korea"}
CURSO = ["KNN", "Ridge", "Lasso", "BayesianRidge", "DecisionTree", "RandomForest", "XGBoost", "SVR_RBF"]
EXTRA = ["LightGBM", "SARIMA", "Persistencia", "Estacional", "EstacionalALEE"]
BASELINES = CURSO + EXTRA
ROPE = 0.01  # equivalencia práctica: |diferencia relativa de RMSE| <= 1 %, fijado antes de calcular la prueba


# ----------------------------------------------------------------- datos
def get_data(conj):
    """X (DataFrame), y, nombres de variables y carpeta con las predicciones previas."""
    if conj == "colombia":
        ds = assemble_dataset()
        return ds.X, ds.y_reg, list(ds.X.columns), ROOT / "artifacts" / "confirmatorio"
    Xdf, ydf, _ = build_dataset(load_series(conj))
    return Xdf, ydf, list(F14), ROOT / "artifacts" / "adicionales" / conj


def outer_splits(n):
    return list(TimeSeriesSplit(n_splits=config.N_SPLITS_OUTER, gap=config.CV_GAP).split(np.arange(n)))


# ------------------------------------------------- etapa 1: modelos que faltaban
def etapa_modelos(conj):
    Xdf, ydf, _, _ = get_data(conj)
    d = OUT / conj
    d.mkdir(parents=True, exist_ok=True)
    for spec in [BAYESIAN_RIDGE_MODEL, SVR_RBF_MODEL]:
        p = d / f"oof_{spec.name}.csv"
        if p.exists():
            continue
        t0 = time.time()
        res = run_nested_cv(model_spec=spec, X=Xdf, y=ydf, task="reg", balancing_method="none",
                            optimizer_name="bayesian", seed=config.SEED, n_jobs=5, budgets=config.FULL_BUDGETS)
        rows = [pd.DataFrame({"idx": v["test_idx"], "y_true": v["y_true"], "y_pred": v["y_pred"]}) for v in res.oof.values()]
        pd.concat(rows).sort_values("idx").to_csv(p, index=False)
        print(f"{conj} {spec.name}: {time.time() - t0:.0f}s {res.error or ''}", flush=True)


# ------------------------------------------------------ ALEE con historial común
class Historial:
    """Errores fuera de muestra de los 3 expertos (ventana expansiva, desde la fila 36)."""

    def __init__(self, X, y, names, m0=36):
        self.m0 = m0
        self.cols = {"lag1": names.index("td_nacional_lag1"), "mes_sin": names.index("mes_sin"), "mes_cos": names.index("mes_cos")}
        cache = OUT / f"_hist_{len(y)}_{len(names)}.npy"
        if cache.exists():
            self.E = np.load(cache)
        else:
            self.E = np.array([y[s] - _Experts(X[:s], y[:s], self.cols).predict(X[s:s + 1])[0] for s in range(m0, len(y))])
            np.save(cache, self.E)


def alee_fit(params, X, y, n_train, H, names, **variant):
    m = ModeloOriginal(feature_names=tuple(names), random_state=config.SEED, **params, **variant)
    m0 = int(min(m.min_train, max(2, n_train // 2)))
    if m0 != H.m0:
        return m.fit(X[:n_train], y[:n_train])
    m._validate_params()
    m.cols_ = H.cols
    Xt = X[:n_train]
    m.errors_ = H.E[: n_train - m0]
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


VARIANTES = {
    "completo": {},
    "sin_localizacion": {"localize": False},
    "sin_lambda_adaptativa": {"adaptive_lam": False},
    "previo_uniforme": {"shrink_to": "uniforme"},
}


def etapa_alee(conj, variante, seed=42):
    tag0 = variante if seed == 42 else f"{variante}_semilla{seed}"
    if (OUT / conj / f"alee_{tag0}.csv").exists():
        print(f"{conj} ALEE {tag0}: ya calculado, se reutiliza")
        return
    Xdf, ydf, names, _ = get_data(conj)
    X, y = Xdf.to_numpy(float), ydf.to_numpy(float)
    H = Historial(X, y, names)
    kw = VARIANTES[variante]
    rows, hp = [], []
    t0 = time.time()
    for k, (tr, te) in enumerate(outer_splits(len(y))):
        n_tr = len(tr)
        inner = list(TimeSeriesSplit(n_splits=config.N_SPLITS_INNER, gap=config.CV_GAP).split(X[:n_tr]))

        def objective(trial):
            p = to_optuna(ORIGINAL_MODEL.space, trial)
            sc = []
            for itr, iva in inner:
                try:
                    pr = alee_fit(p, X, y, len(itr), H, names, **kw).predict(X[iva])
                    sc.append(-float(np.sqrt(np.mean((y[iva] - pr) ** 2))))
                except Exception:
                    continue
            return float(np.mean(sc)) if sc else -1e9

        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=10))
        st.optimize(objective, n_trials=config.FULL_BUDGETS["bayes_n_trials"], show_progress_bar=False)
        m = alee_fit(st.best_params, X, y, n_tr, H, names, **kw)
        W = m.combination_weights(X[te])
        rows.append(pd.DataFrame({"idx": te, "y_true": y[te], "y_pred": m.predict(X[te]),
                                  **{f"w_{e}": W[:, i] for i, e in enumerate(EXPERTS)}}))
        hp.append({"pliegue": k, **st.best_params})
    d = OUT / conj
    d.mkdir(parents=True, exist_ok=True)
    tag = variante if seed == 42 else f"{variante}_semilla{seed}"
    pd.concat(rows).to_csv(d / f"alee_{tag}.csv", index=False)
    pd.DataFrame(hp).to_csv(d / f"alee_{tag}_hp.csv", index=False)
    print(f"{conj} ALEE {tag}: {time.time() - t0:.0f}s", flush=True)


def etapa_fijos(conj):
    """Promedio simple, núcleo solo y mapa de sensibilidad (hiperparámetros fijos, sin ajuste)."""
    if (OUT / conj / "sensibilidad_hiperparametros.csv").exists():
        print(f"{conj} fijos: ya calculado, se reutiliza")
        return
    Xdf, ydf, names, _ = get_data(conj)
    X, y = Xdf.to_numpy(float), ydf.to_numpy(float)
    H = Historial(X, y, names)
    d = OUT / conj
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for tr, te in outer_splits(len(y)):
        F = _Experts(X[:len(tr)], y[:len(tr)], H.cols).predict(X[te])
        rows.append(pd.DataFrame({"idx": te, "y_true": y[te], "PromedioSimple": F.mean(1), "NucleoSolo": F[:, 1]}))
    pd.concat(rows).to_csv(d / "fijos.csv", index=False)
    # mapa de sensibilidad: RMSE medio de pliegues con (bandwidth, lam) fijos y half_life = 60
    bws = np.geomspace(0.05, 3.0, 7)
    lams = np.geomspace(1e-3, 100.0, 7)
    grid = []
    for b in bws:
        for l in lams:
            rm = []
            for tr, te in outer_splits(len(y)):
                pr = alee_fit(dict(bandwidth=b, half_life=60.0, lam=l), X, y, len(tr), H, names).predict(X[te])
                rm.append(np.sqrt(np.mean((y[te] - pr) ** 2)))
            grid.append({"bandwidth": b, "lam": l, "rmse_medio": np.mean(rm)})
    pd.DataFrame(grid).to_csv(d / "sensibilidad_hiperparametros.csv", index=False)
    print(f"{conj} fijos listos", flush=True)


# ------------------------------------------------------------- costo computacional
def n_parametros(name, est, n, p):
    try:
        if name in ("Ridge", "Lasso", "BayesianRidge"):
            return p + 1
        if name == "KNN":
            return n * (p + 1)
        if name == "DecisionTree":
            return int(est.tree_.node_count)
        if name == "RandomForest":
            return int(sum(t.tree_.node_count for t in est.estimators_))
        if name == "XGBoost":
            return int(est.get_booster().trees_to_dataframe().shape[0])
        if name == "LightGBM":
            return int(est.booster_.trees_to_dataframe().shape[0])
        if name == "SVR_RBF":
            return int(est.support_vectors_.size + est.dual_coef_.size + 1)
    except Exception:
        return np.nan
    return np.nan


def etapa_costo():
    from sklearn.preprocessing import StandardScaler
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from src.data import load_raw
    Xdf, ydf, names, _ = get_data("colombia")
    X, y = Xdf.to_numpy(float), ydf.to_numpy(float)
    tr, te = outer_splits(len(y))[-1]
    specs = {n: REGRESSOR_BY_NAME[n] for n in ["KNN", "Ridge", "Lasso", "DecisionTree", "RandomForest", "XGBoost"]}
    specs.update({"BayesianRidge": BAYESIAN_RIDGE_MODEL, "SVR_RBF": SVR_RBF_MODEL, "LightGBM": LIGHTGBM_MODEL})
    rows = []

    def medir(fit_fn, pred_fn, reps=3):
        tf, tp, mem = [], [], 0
        obj = None
        for r in range(reps):
            tracemalloc.start()
            t0 = time.perf_counter()
            obj = fit_fn()
            tf.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            pred_fn(obj)
            tp.append(time.perf_counter() - t0)
            mem = max(mem, tracemalloc.get_traced_memory()[1])
            tracemalloc.stop()
        return np.median(tf), np.median(tp), mem / 2 ** 20, obj

    for name, spec in specs.items():
        def fit_fn(spec=spec):
            est = spec.estimator()
            sc = StandardScaler().fit(X[tr]) if spec.needs_scaling else None
            est.fit(sc.transform(X[tr]) if sc else X[tr], y[tr])
            return est, sc
        tf, tp, mem, (est, sc) = medir(fit_fn, lambda o: o[0].predict(o[1].transform(X[te]) if o[1] else X[te]))
        rows.append({"modelo": name, "entrenamiento_s": tf, "inferencia_s": tp, "memoria_pico_MB": mem,
                     "parametros": n_parametros(name, est, len(tr), X.shape[1])})
    ys = load_raw()["tasa_desempleo_nacional"].values
    a = len(tr)

    def fit_sar():
        return SARIMAX(ys[: a + 12], order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False, maxiter=200)
    tf, tp, mem, _ = medir(fit_sar, lambda f: f.append(ys[a + 12: te[-1] + 12], refit=False).get_prediction(start=te[0] + 12, end=te[-1] + 12).predicted_mean)
    rows.append({"modelo": "SARIMA", "entrenamiento_s": tf, "inferencia_s": tp, "memoria_pico_MB": mem, "parametros": 3})
    tf, tp, mem, m = medir(lambda: ModeloOriginal(feature_names=tuple(names)).fit(X[tr], y[tr]), lambda o: o.predict(X[te]))
    rows.append({"modelo": "ALEE", "entrenamiento_s": tf, "inferencia_s": tp, "memoria_pico_MB": mem,
                 "parametros": int(m.errors_.size + m.state_.size + 3 + 3)})
    for nm, npar in [("Persistencia", 0), ("Estacional", 0), ("EstacionalALEE", 12)]:
        rows.append({"modelo": nm, "entrenamiento_s": 0.0, "inferencia_s": 0.0, "memoria_pico_MB": 0.0, "parametros": npar})
    import platform
    import os
    info = {"procesador": platform.processor(), "nucleos": os.cpu_count(), "sistema": platform.platform(), "python": platform.python_version()}
    pd.DataFrame(rows).to_csv(OUT / "costo_computacional.csv", index=False)
    (OUT / "hardware.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(pd.DataFrame(rows).round(4).to_string(index=False))


# ---------------------------------------------------------------- análisis
def cargar_predicciones(conj):
    """DataFrame con una columna de predicción por modelo, alineado por idx."""
    _, ydf, _, prev = get_data(conj)
    if conj == "colombia":
        base = pd.read_csv(prev / "iter4" / "oof_ALEE_K3_sin_ridge.csv")  # ALEE-3 canónico (iteración 4)
    else:
        base = pd.read_csv(prev / "oof_ALEE.csv")
    P = base.sort_values("idx").set_index("idx")[["y_true"]].rename(columns={"y_true": "y"})
    P["ALEE"] = base.set_index("idx")["y_pred"]
    for n in ["KNN", "Ridge", "Lasso", "DecisionTree", "RandomForest", "XGBoost", "LightGBM", "EstacionalALEE"]:
        P[n] = pd.read_csv(prev / f"oof_{n}.csv").set_index("idx")["y_pred"]
    for n in ["BayesianRidge", "SVR_RBF"]:
        P[n] = pd.read_csv(OUT / conj / f"oof_{n}.csv").set_index("idx")["y_pred"]
    bl = pd.read_csv(prev / "baselines_series_tiempo.csv").set_index("idx")
    for n in ["SARIMA", "Persistencia", "Estacional"]:
        P[n] = bl[n]
    assert P.notna().all().all(), conj
    assert np.allclose(P["y"].values, ydf.values[P.index.values])
    return P


def metricas_por_pliegue(P, y_all, conj):
    rows = []
    for k, (tr, te) in enumerate(outer_splits(len(y_all))):
        ytr = y_all[tr]
        escala = np.mean(np.abs(ytr[12:] - ytr[:-12]))  # MASE: error del ingenuo estacional dentro del entrenamiento
        sub = P.loc[te]
        for m in ["ALEE"] + BASELINES:
            e = sub["y"] - sub[m]
            rows.append({"conjunto": conj, "pliegue": k, "modelo": m, "n_train": len(tr),
                         "RMSE": np.sqrt(np.mean(e ** 2)), "MAE": np.mean(np.abs(e)),
                         "MAPE": 100 * np.mean(np.abs(e / sub["y"])),
                         "R2": 1 - np.sum(e ** 2) / np.sum((sub["y"] - sub["y"].mean()) ** 2),
                         "MASE": np.mean(np.abs(e)) / escala})
    return pd.DataFrame(rows)


def bayes_t_correlacionado(d, rho, rope=ROPE):
    """t bayesiana correlacionada (Corani y Benavoli, 2015) sobre diferencias por pliegue."""
    k = len(d)
    s2 = np.var(d, ddof=1)
    scale = np.sqrt((1.0 / k + rho / (1 - rho)) * s2) if s2 > 0 else 1e-12
    dist = stats.t(df=k - 1, loc=np.mean(d), scale=scale)
    p_mejor = dist.cdf(-rope)  # d < 0: ALEE tiene menor RMSE
    p_peor = 1 - dist.cdf(rope)
    return p_mejor, 1 - p_mejor - p_peor, p_peor


def bayes_signed_rank(z, rope=ROPE, n_samples=50000, seed=42):
    """Signed-rank bayesiano (Benavoli et al., 2014) con observación previa en 0 de peso 0.5."""
    rng = np.random.default_rng(seed)
    z = np.concatenate([[0.0], np.asarray(z)])
    n = len(z)
    W = rng.dirichlet(np.r_[0.5, np.ones(n - 1)], size=n_samples)
    S = (z[:, None] + z[None, :])  # promedios de Walsh (sin dividir; el ROPE se escala igual)
    left = (S < -2 * rope).astype(float)
    right = (S > 2 * rope).astype(float)
    mid = 1 - left - right
    res = np.zeros(3)
    for w in W:
        ww = np.outer(w, w)
        v = np.array([(ww * left).sum(), (ww * mid).sum(), (ww * right).sum()])
        res[np.argmax(v)] += 1
    return res / n_samples  # (P ALEE mejor, P equivalente, P ALEE peor)


def etapa_analisis():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "pdf.fonttype": 42,
                         "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7})
    FIG.mkdir(parents=True, exist_ok=True)
    preds, met, pruebas, mcs_all, abl_rows, errores_reg = {}, [], [], [], [], []
    for conj in CONJUNTOS:
        Xdf, ydf, names, _ = get_data(conj)
        y_all = ydf.to_numpy(float)
        P = cargar_predicciones(conj)
        preds[conj] = P
        M = metricas_por_pliegue(P, y_all, conj)
        met.append(M)
        E = P[["ALEE"] + BASELINES].rsub(P["y"], axis=0)
        L = E ** 2
        sp = outer_splits(len(y_all))
        rho = np.mean([len(te) / (len(tr) + len(te)) for tr, te in sp])
        rmse_f = M.pivot(index="pliegue", columns="modelo", values="RMSE")
        rows = []
        for b in BASELINES:
            dm = ST.diebold_mariano_test(E["ALEE"], E[b])
            gw = ST.giacomini_white_test(E["ALEE"], E[b], conditional=True)
            dloss = L["ALEE"] - L[b]
            est, lo, hi = ST.bootstrap_interval(dloss.values, np.mean, mean_block=12, n_boot=2000, seed=42)
            rel = (rmse_f["ALEE"] - rmse_f[b]) / rmse_f[b]
            pb = bayes_t_correlacionado(rel.values, rho)
            rows.append({"conjunto": conj, "baseline": b, "dif_perdida_media": dloss.mean(), "ic95_inf": lo, "ic95_sup": hi,
                         "dm_estadistico": dm["dm_hln"], "dm_p": dm["p_value"], "gw_estadistico": gw.get("statistic", np.nan),
                         "gw_p": gw["p_value"], "p_bayes_mejor": pb[0], "p_bayes_rope": pb[1], "p_bayes_peor": pb[2],
                         "d_cohen": ST.cohens_d_paired(dloss.values), "a12_vargha": None})
        R = pd.DataFrame(rows)
        R["dm_p_holm"] = ST.adjust_pvalues(R["dm_p"].values, "holm")
        R["gw_p_holm"] = ST.adjust_pvalues(R["gw_p"].values, "holm")
        # A12 de Vargha-Delaney sobre errores absolutos: P(|e_ALEE| < |e_b|)
        for i, b in enumerate(BASELINES):
            a, c = np.abs(E["ALEE"].values), np.abs(E[b].values)
            R.loc[i, "a12_vargha"] = (np.sum(a[:, None] < c[None, :]) + 0.5 * np.sum(a[:, None] == c[None, :])) / (len(a) * len(c))
        pruebas.append(R)
        mcs = ST.model_confidence_set(L, alpha=0.10, block=12, seed=42)
        mcs.insert(0, "conjunto", conj)
        mcs_all.append(mcs)
        # ablación
        A = {"ALEE completo": P["ALEE"]}
        for v, lab in [("sin_localizacion", "Sin localización"), ("sin_lambda_adaptativa", "Sin λ adaptativa"), ("previo_uniforme", "Previo uniforme")]:
            f = OUT / conj / f"alee_{v}.csv"
            if f.exists():
                A[lab] = pd.read_csv(f).set_index("idx")["y_pred"]
        fx = pd.read_csv(OUT / conj / "fijos.csv").set_index("idx")
        A["Promedio simple"] = fx["PromedioSimple"]
        A["Solo persistencia"] = P["Persistencia"]
        A["Solo núcleo"] = fx["NucleoSolo"]
        A["Solo estacional"] = P["EstacionalALEE"]
        eA = P["y"] - A["ALEE completo"]
        pv = []
        for lab, pr in A.items():
            e = P["y"] - pr.loc[P.index]
            fr = [np.sqrt(np.mean(e.loc[te] ** 2)) for _, te in sp]
            r = {"conjunto": conj, "variante": lab, "rmse_medio": np.mean(fr), "rmse_sd": np.std(fr, ddof=1)}
            if lab != "ALEE completo":
                dm = ST.diebold_mariano_test(eA, e)
                r.update({"dm_p": dm["p_value"], "dif_perdida": dm["mean_loss_diff"]})
                pv.append(dm["p_value"])
            abl_rows.append(r)
        # errores por régimen de volatilidad (terciles de la desviación móvil de 3 meses)
        vol = Xdf["td_nacional_rollstd3"].to_numpy()[P.index.values]
        cuts = np.quantile(vol, [1 / 3, 2 / 3])
        reg = np.digitize(vol, cuts)
        for m in ["ALEE", "SARIMA", "EstacionalALEE", "Persistencia", "Ridge"]:
            for g, lab in enumerate(["baja", "media", "alta"]):
                e = (P["y"] - P[m]).values[reg == g]
                errores_reg.append({"conjunto": conj, "modelo": m, "volatilidad": lab, "rmse": np.sqrt(np.mean(e ** 2)), "n": len(e)})
    met = pd.concat(met)
    pruebas = pd.concat(pruebas)
    abl = pd.DataFrame(abl_rows)
    for conj in CONJUNTOS:
        msk = (abl.conjunto == conj) & abl.dm_p.notna()
        abl.loc[msk, "dm_p_holm"] = ST.adjust_pvalues(abl.loc[msk, "dm_p"].values, "holm")
    met.to_csv(OUT / "metricas_por_pliegue.csv", index=False)
    pruebas.to_csv(OUT / "pruebas_por_baseline.csv", index=False)
    pd.concat(mcs_all).to_csv(OUT / "mcs.csv", index=False)
    abl.to_csv(OUT / "ablacion.csv", index=False)
    pd.DataFrame(errores_reg).to_csv(OUT / "errores_por_regimen.csv", index=False)

    # sensibilidad a la semilla del optimizador (42 = predicciones de la tabla principal)
    sem = []
    for conj in CONJUNTOS:
        P = preds[conj]
        sp = outer_splits(len(get_data(conj)[1]))
        for s in [42, 43, 44, 45, 46]:
            if s == 42:
                pr = P["ALEE"]
            else:
                f = OUT / conj / f"alee_completo_semilla{s}.csv"
                if not f.exists():
                    continue
                pr = pd.read_csv(f).set_index("idx")["y_pred"].loc[P.index]
            e = P["y"] - pr
            sem.append({"conjunto": conj, "semilla": s, "rmse_medio": np.mean([np.sqrt(np.mean(e.loc[te] ** 2)) for _, te in sp])})
        if conj == "colombia" and (OUT / conj / "alee_completo.csv").exists():
            pr = pd.read_csv(OUT / conj / "alee_completo.csv").set_index("idx")["y_pred"].loc[P.index]
            e = P["y"] - pr
            sem.append({"conjunto": conj, "semilla": "42_recalculo_3hp",
                        "rmse_medio": np.mean([np.sqrt(np.mean(e.loc[te] ** 2)) for _, te in sp])})
    pd.DataFrame(sem).to_csv(OUT / "sensibilidad_semillas.csv", index=False)

    # importancia por permutación de ALEE (Colombia, último pliegue, hiperparámetros elegidos en ese pliegue)
    Xdf, ydf, names, _ = get_data("colombia")
    X, y = Xdf.to_numpy(float), ydf.to_numpy(float)
    hpf = ITER4_HP  # hiperparámetros del ALEE canónico (iteración 4)
    if hpf.exists():
        H = Historial(X, y, names)
        tr, te = outer_splits(len(y))[-1]
        hp = pd.read_csv(hpf).iloc[-1][["bandwidth", "half_life", "lam"]].to_dict()
        m = alee_fit(hp, X, y, len(tr), H, names)
        base_rmse = np.sqrt(np.mean((y[te] - m.predict(X[te])) ** 2))
        rng = np.random.default_rng(42)
        imp = []
        for j, nm in enumerate(names):
            deltas = []
            for _ in range(20):
                Xp = X[te].copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                deltas.append(np.sqrt(np.mean((y[te] - m.predict(Xp)) ** 2)) - base_rmse)
            imp.append({"variable": nm, "aumento_rmse_medio": np.mean(deltas), "aumento_rmse_sd": np.std(deltas, ddof=1)})
        imp = pd.DataFrame(imp).sort_values("aumento_rmse_medio", ascending=False)
        imp.to_csv(OUT / "importancia_permutacion_alee.csv", index=False)
        top = imp.head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(3.2, 2.6))
        ax.barh(top["variable"], top["aumento_rmse_medio"], xerr=top["aumento_rmse_sd"], color="tab:blue", error_kw={"lw": 0.6})
        ax.set_xlabel("RMSE increase when permuted (pp)")
        ax.set_title("ALEE permutation importance (Colombia)")
        fig.tight_layout()
        fig.savefig(FIG / "fig_importancia.pdf")
        plt.close(fig)

    # tabla principal: media ± desviación por pliegue
    resumen = met.groupby(["conjunto", "modelo"])[["RMSE", "MAE", "MAPE", "R2", "MASE"]].agg(["mean", "std"])
    resumen.to_csv(OUT / "tabla_principal.csv")

    # entre conjuntos: Friedman + Iman-Davenport, Conover-Holm frente a ALEE, signed-rank bayesiano
    W = met.groupby(["conjunto", "modelo"])["RMSE"].mean().unstack()[["ALEE"] + BASELINES]
    N, k = W.shape
    chi2, p_f = stats.friedmanchisquare(*[W[c].values for c in W.columns])
    ff = (N - 1) * chi2 / (N * (k - 1) - chi2)
    p_id = 1 - stats.f.cdf(ff, k - 1, (k - 1) * (N - 1))
    ranks = W.rank(axis=1).mean().sort_values()
    import scikit_posthocs as sph
    con = sph.posthoc_conover_friedman(W.reset_index(drop=True), p_adjust="holm")
    entre = []
    for b in BASELINES:
        z = ((W["ALEE"] - W[b]) / W[b]).values
        pb = bayes_signed_rank(z)
        try:
            pw = stats.wilcoxon(W["ALEE"], W[b]).pvalue
        except ValueError:
            pw = 1.0
        entre.append({"baseline": b, "conjuntos_ALEE_gana": int((W["ALEE"] < W[b]).sum()), "conover_p_holm": con.loc["ALEE", b],
                      "wilcoxon_p": pw, "p_bayes_mejor": pb[0], "p_bayes_rope": pb[1], "p_bayes_peor": pb[2]})
    entre = pd.DataFrame(entre)
    entre["wilcoxon_p_holm"] = ST.adjust_pvalues(entre["wilcoxon_p"].values, "holm")
    entre.to_csv(OUT / "pruebas_entre_conjuntos.csv", index=False)
    cd = ST.critical_difference(k, N)
    pd.Series({"friedman_chi2": chi2, "friedman_p": p_f, "iman_davenport_F": ff, "iman_davenport_p": p_id,
               "nemenyi_cd": cd, **{f"rango_{m}": r for m, r in ranks.items()}}).to_csv(OUT / "friedman.csv")

    # ------------------------------------------------------------ figuras
    lab = {"ALEE": "ALEE", "EstacionalALEE": "Seasonal expert", "Estacional": "Seasonal naive", "Persistencia": "Persistence",
           "SVR_RBF": "SVR (RBF)", "BayesianRidge": "Bayesian ridge", "DecisionTree": "Decision tree", "RandomForest": "Random forest"}
    # F1: diagrama de diferencia crítica
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    ST.plot_cd_diagram(ranks.rename(index=lambda s: lab.get(s, s)), cd, ax=ax, title=None)
    for txt in ax.texts:
        txt.set_text(txt.get_text().replace("DC", "CD"))
    fig.tight_layout()
    fig.savefig(FIG / "fig_cd.pdf")
    plt.close(fig)
    # F2: cajas de RMSE por pliegue
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.2))
    for ax, conj in zip(axes.ravel(), CONJUNTOS):
        sub = met[met.conjunto == conj]
        order = sub.groupby("modelo")["RMSE"].mean().sort_values().index
        ax.boxplot([sub[sub.modelo == m]["RMSE"].values for m in order], vert=False, widths=0.6)
        ax.set_yticks(range(1, len(order) + 1), [lab.get(m, m) for m in order])
        ax.invert_yaxis()
        ax.set_xscale("log")
        from matplotlib.ticker import NullFormatter
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("RMSE per outer fold (percentage points)")
        ax.set_title(NAME_EN[conj])
    fig.tight_layout()
    fig.savefig(FIG / "fig_rmse_pliegues.pdf")
    plt.close(fig)
    # F3: serie pronosticada frente a observada y residuos (Colombia)
    from src.data import load_raw
    fechas = pd.to_datetime(load_raw()["fecha"]).values
    P = preds["colombia"]
    t = fechas[P.index.values + 12]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.5, 4.0), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(t, P["y"], color="black", lw=1.2, label="Observed")
    a1.plot(t, P["ALEE"], color="tab:blue", lw=0.9, label="ALEE")
    a1.plot(t, P["SARIMA"], color="tab:orange", lw=0.9, ls="--", label="SARIMA")
    a1.set_ylabel("Unemployment rate (%)")
    a1.legend(frameon=False, ncol=3)
    a2.plot(t, P["y"] - P["ALEE"], color="tab:blue", lw=0.8, label="ALEE")
    a2.plot(t, P["y"] - P["SARIMA"], color="tab:orange", lw=0.8, ls="--", label="SARIMA")
    a2.axhline(0, color="grey", lw=0.5)
    a2.set_ylabel("Residual (pp)")
    a2.set_xlabel("Forecast month")
    fig.tight_layout()
    fig.savefig(FIG / "fig_pronostico_colombia.pdf")
    plt.close(fig)
    # F4: ablación
    names_en = {"ALEE completo": "Full ALEE", "Sin localización": "No localization", "Sin λ adaptativa": "No adaptive λ",
                "Previo uniforme": "Uniform prior", "Promedio simple": "Simple average", "Solo persistencia": "Persistence only",
                "Solo núcleo": "Kernel only", "Solo estacional": "Seasonal only"}
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.6), sharey=True)
    for ax, conj in zip(axes, CONJUNTOS):
        s = abl[abl.conjunto == conj].set_index("variante").reindex(list(names_en))
        ax.barh(range(len(s)), s["rmse_medio"], xerr=s["rmse_sd"], color=["tab:blue"] + ["tab:grey"] * (len(s) - 1), error_kw={"lw": 0.6})
        ax.set_yticks(range(len(s)), [names_en[v] for v in s.index])
        ax.invert_yaxis()
        ax.set_title(NAME_EN[conj])
        ax.set_xlabel("RMSE (pp)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_ablacion.pdf")
    plt.close(fig)
    # F5: sensibilidad (mapa de calor) y curva de aprendizaje
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4))
    for ax, conj in zip(axes[:2], ["colombia", "eeuu"]):
        g = pd.read_csv(OUT / conj / "sensibilidad_hiperparametros.csv").pivot(index="lam", columns="bandwidth", values="rmse_medio")
        im = ax.imshow(g.values, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(g.shape[1]), [f"{v:.2g}" for v in g.columns], rotation=90)
        ax.set_yticks(range(g.shape[0]), [f"{v:.0e}" for v in g.index])
        ax.set_xlabel("Bandwidth h")
        ax.set_ylabel("λ")
        ax.set_title(f"{NAME_EN[conj]}: RMSE")
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[2]
    lcv = pd.read_csv(OUT / "curva_aprendizaje_eeuu.csv")  # periodo de prueba fijo (etapa extras)
    for m, c, ls in [("ALEE", "tab:blue", "-"), ("SARIMA", "tab:orange", "--"), ("EstacionalALEE", "tab:purple", "-."), ("Ridge", "tab:green", ":")]:
        s = lcv[lcv.modelo == m].sort_values("meses_entrenamiento")
        ax.plot(s["meses_entrenamiento"], s["rmse"], marker="o", ms=3, color=c, ls=ls, label=lab.get(m, m))
    ax.set_xscale("log")
    from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
    ticks = sorted(lcv["meses_entrenamiento"].unique())
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Training months (log scale)")
    ax.set_ylabel("Test RMSE (pp)")
    ax.set_title("Learning curve (US)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_sensibilidad.pdf")
    plt.close(fig)
    # F6: pesos de ALEE en el tiempo (Colombia, recálculo con el mismo protocolo) y error por régimen
    f = OUT / "colombia" / "alee_canonico_pesos.csv"  # ALEE canónico reconstruido (etapa extras)
    if f.exists():
        w = pd.read_csv(f).set_index("idx")
        tw = fechas[w.index.values + 12]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 2.4), gridspec_kw={"width_ratios": [1.6, 1]})
        a1.stackplot(tw, w["w_persistencia"], w["w_nucleo_cambio"], w["w_estacional"], labels=["Persistence", "Kernel", "Seasonal"],
                     colors=["tab:red", "tab:green", "tab:blue"], alpha=0.8)
        a1.set_ylabel("Combination weight")
        a1.set_xlabel("Forecast month")
        a1.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))
        a1.set_title("ALEE weights, Colombia")
        er = pd.DataFrame(errores_reg)
        er = er[er.conjunto == "colombia"].pivot(index="volatilidad", columns="modelo", values="rmse").loc[["baja", "media", "alta"]]
        xs = np.arange(3)
        for i, m in enumerate(["ALEE", "SARIMA", "EstacionalALEE", "Persistencia"]):
            a2.bar(xs + (i - 1.5) * 0.2, er[m], width=0.2, label=lab.get(m, m))
        a2.set_xticks(xs, ["Low", "Medium", "High"])
        a2.set_xlabel("Recent volatility tercile")
        a2.set_ylabel("RMSE (pp)")
        a2.legend(frameon=False, fontsize=6)
        fig.tight_layout()
        fig.savefig(FIG / "fig_pesos_errores.pdf")
        plt.close(fig)
    print("análisis listo:", OUT)


ES = {"ALEE": "ALEE", "KNN": "KNN", "Ridge": "Ridge", "Lasso": "Lasso", "BayesianRidge": "Regresión bayesiana",
      "DecisionTree": "Árbol de decisión", "RandomForest": "Random Forest", "XGBoost": "XGBoost", "SVR_RBF": "SVR (RBF)",
      "LightGBM": "LightGBM", "SARIMA": "SARIMA", "Persistencia": "Persistencia", "Estacional": "Estacional ingenuo",
      "EstacionalALEE": "Experto estacional"}


def fnum(x, d=3):
    return f"{x:.{d}f}"


def fp(p):
    """Valor p exacto: tres decimales, o notación científica si es menor que 0.001."""
    if p >= 0.001:
        return f"{p:.3f}"
    if p <= 0:
        return r"$<10^{-15}$"  # bajo la precisión de máquina de la distribución t
    m, e = f"{p:.1e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


# ------------------------------------------------------------ etapa de extras
ITER4_HP = ROOT / "artifacts" / "confirmatorio" / "iter4" / "rmse_K3_sin_ridge.csv"


def etapa_extras():
    """ALEE canónico de Colombia reconstruido, residuos, curva de aprendizaje, semillas de los baselines y apéndice."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "pdf.fonttype": 42,
                         "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7})
    FIG.mkdir(parents=True, exist_ok=True)
    # 1) ALEE canónico (iteración 4) reconstruido con sus hiperparámetros guardados
    Xdf, ydf, names, prev = get_data("colombia")
    X, y = Xdf.to_numpy(float), ydf.to_numpy(float)
    H = Historial(X, y, names)
    hp = pd.read_csv(ITER4_HP)
    canon = pd.read_csv(prev / "iter4" / "oof_ALEE_K3_sin_ridge.csv").set_index("idx")["y_pred"]
    rows = []
    for k, (tr, te) in enumerate(outer_splits(len(y))):
        p = hp.iloc[k][["bandwidth", "half_life", "lam"]].to_dict()
        m = alee_fit(p, X, y, len(tr), H, names)
        W = m.combination_weights(X[te])
        rows.append(pd.DataFrame({"idx": te, "y_pred": m.predict(X[te]), **{f"w_{e}": W[:, i] for i, e in enumerate(EXPERTS)}}))
    rec = pd.concat(rows).set_index("idx")
    dif = float(np.abs(rec["y_pred"] - canon.loc[rec.index]).max())
    rec.to_csv(OUT / "colombia" / "alee_canonico_pesos.csv")
    print(f"ALEE canónico reconstruido: diferencia máxima con las predicciones guardadas = {dif:.2e}")
    json.dump({"dif_max_reconstruccion": dif}, open(OUT / "colombia" / "alee_canonico_verificacion.json", "w"))
    # 2) residuos: predicho frente a observado y distribución, ALEE y SARIMA
    fig, axes = plt.subplots(2, 4, figsize=(6.5, 3.6))
    for j, conj in enumerate(CONJUNTOS):
        P = cargar_predicciones(conj)
        ax = axes[0, j]
        lo, hi = P["y"].min(), P["y"].max()
        ax.plot([lo, hi], [lo, hi], color="grey", lw=0.6)
        ax.scatter(P["y"], P["SARIMA"], s=3, color="tab:orange", alpha=0.5, label="SARIMA")
        ax.scatter(P["y"], P["ALEE"], s=3, color="tab:blue", alpha=0.5, label="ALEE")
        ax.set_title(NAME_EN[conj])
        ax.set_xlabel("Observed (%)")
        if j == 0:
            ax.set_ylabel("Predicted (%)")
            ax.legend(frameon=False, markerscale=2)
        ax = axes[1, j]
        rA, rS = P["y"] - P["ALEE"], P["y"] - P["SARIMA"]
        bins = np.linspace(np.quantile(np.r_[rA, rS], 0.005), np.quantile(np.r_[rA, rS], 0.995), 30)
        ax.hist(rS, bins=bins, color="tab:orange", alpha=0.5, label="SARIMA")
        ax.hist(rA, bins=bins, color="tab:blue", alpha=0.5, label="ALEE")
        ax.set_xlabel("Residual (pp)")
        if j == 0:
            ax.set_ylabel("Months")
    fig.tight_layout()
    fig.savefig(FIG / "fig_residuos.pdf")
    plt.close(fig)
    # 3) curva de aprendizaje: periodo de prueba fijo (últimos 156 meses de EE.UU.), entrenamiento creciente
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    s = load_series("eeuu")
    Xdf, ydf, names, prev = get_data("eeuu")
    X, y = Xdf.to_numpy(float), ydf.to_numpy(float)
    n = len(y)
    t0 = n - 156
    test = np.arange(t0, n)
    b = t0 - 1  # gap de un mes
    hpu = pd.read_csv(prev / "hiperparametros_ALEE.csv")[["bandwidth", "half_life", "lam"]].median().to_dict()
    lc = []
    for L in [60, 120, 240, 480, b]:
        a = b - L
        m = ModeloOriginal(feature_names=tuple(names), **hpu).fit(X[a:b], y[a:b])
        pa = m.predict(X[test])
        pe = m.expert_predictions(X[test])[:, EXPERTS.index("estacional")]
        sc = StandardScaler().fit(X[a:b])
        pr = Ridge(alpha=1.0).fit(sc.transform(X[a:b]), y[a:b]).predict(sc.transform(X[test]))
        ys = s.values
        start = a + 11
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = SARIMAX(ys[start: b + 12], order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False, maxiter=200)
            ext = fit.append(ys[b + 12: test[-1] + 12], refit=False)
            ps = np.asarray(ext.get_prediction(start=test[0] + 12 - start, end=test[-1] + 12 - start).predicted_mean)
        for nm, pred in [("ALEE", pa), ("EstacionalALEE", pe), ("Ridge", pr), ("SARIMA", ps)]:
            lc.append({"meses_entrenamiento": L, "modelo": nm, "rmse": float(np.sqrt(np.mean((y[test] - pred) ** 2)))})
    lc = pd.DataFrame(lc)
    lc.to_csv(OUT / "curva_aprendizaje_eeuu.csv", index=False)
    json.dump({"hiperparametros_ALEE": hpu, "meses_prueba": 156}, open(OUT / "curva_aprendizaje_config.json", "w"))
    # 4) variabilidad entre semillas de los baselines estocásticos (Colombia, semillas 42 a 51)
    cprev = ROOT / "artifacts" / "confirmatorio"
    sp = outer_splits(288)
    sv = []
    for mdl in ["RandomForest", "XGBoost", "LightGBM"]:
        for sd in range(42, 52):
            f = cprev / (f"oof_{mdl}.csv" if sd == 42 else f"oof_{mdl}_seed{sd}.csv")
            d = pd.read_csv(f).set_index("idx")
            e = d["y_true"] - d["y_pred"]
            sv.append({"modelo": mdl, "semilla": sd, "rmse_medio": np.mean([np.sqrt(np.mean(e.loc[te] ** 2)) for _, te in sp])})
    sv = pd.DataFrame(sv)
    sv.to_csv(OUT / "semillas_baselines_colombia.csv", index=False)
    print(sv.groupby("modelo")["rmse_medio"].agg(["mean", "std", "min", "max"]).round(4))


def etapa_costo_ajustado():
    """Costo con los hiperparámetros elegidos en el último pliegue de Colombia, cada modelo en un proceso aparte."""
    import subprocess
    from src.search import run_bayesian_search
    Xdf, ydf, names, _ = get_data("colombia")
    tr, te = outer_splits(len(ydf))[-1]
    Xtr, ytr = Xdf.iloc[tr], ydf.iloc[tr]
    inner = list(TimeSeriesSplit(n_splits=config.N_SPLITS_INNER, gap=config.CV_GAP).split(Xtr))
    specs = {n: REGRESSOR_BY_NAME[n] for n in ["KNN", "Ridge", "Lasso", "DecisionTree", "RandomForest", "XGBoost"]}
    specs.update({"BayesianRidge": BAYESIAN_RIDGE_MODEL, "SVR_RBF": SVR_RBF_MODEL, "LightGBM": LIGHTGBM_MODEL})
    params = {}
    for nm, spec in specs.items():
        r = run_bayesian_search(spec, Xtr, ytr, inner, "reg", "none", config.FULL_BUDGETS, config.SEED)
        params[nm] = r.best_params
    params["ALEE"] = pd.read_csv(ITER4_HP).iloc[-1][["bandwidth", "half_life", "lam"]].to_dict()
    params["SARIMA"] = {}
    (OUT / "hiperparametros_costo.json").write_text(json.dumps(params, default=str, indent=1), encoding="utf-8")
    rows = []
    for nm in params:
        out = subprocess.run([sys.executable, __file__, "costo_uno", nm], capture_output=True, text=True)
        rows.append(json.loads(out.stdout.strip().splitlines()[-1]))
    for nm, npar in [("Persistencia", 0), ("Estacional", 0), ("EstacionalALEE", 12)]:
        rows.append({"modelo": nm, "entrenamiento_s": 0.0, "inferencia_s": 0.0, "memoria_pico_MB": 0.0, "parametros": npar})
    pd.DataFrame(rows).to_csv(OUT / "costo_computacional.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


def etapa_costo_uno(nm):
    """Mide un modelo en un proceso limpio: mediana de 3 repeticiones y memoria pico del proceso (incluye memoria nativa)."""
    import psutil
    from sklearn.preprocessing import StandardScaler
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from src.data import load_raw
    params = json.loads((OUT / "hiperparametros_costo.json").read_text(encoding="utf-8"))[nm]
    Xdf, ydf, names, _ = get_data("colombia")
    X, y = Xdf.to_numpy(float), ydf.to_numpy(float)
    tr, te = outer_splits(len(y))[-1]
    proc = psutil.Process()
    base = proc.memory_info().rss
    tf, tp = [], []
    for _ in range(3):
        if nm == "SARIMA":
            ys = load_raw()["tasa_desempleo_nacional"].values
            a = len(tr)
            t = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est = SARIMAX(ys[: a + 12], order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False, maxiter=200)
            tf.append(time.perf_counter() - t)
            t = time.perf_counter()
            est.append(ys[a + 12: te[-1] + 12], refit=False).get_prediction(start=te[0] + 12, end=te[-1] + 12)
            tp.append(time.perf_counter() - t)
            npar = 3
        elif nm == "ALEE":
            t = time.perf_counter()
            est = ModeloOriginal(feature_names=tuple(names), **params).fit(X[tr], y[tr])
            tf.append(time.perf_counter() - t)
            t = time.perf_counter()
            est.predict(X[te])
            tp.append(time.perf_counter() - t)
            npar = int(est.errors_.size + est.state_.size + 3 + 3)
        else:
            specs = {n: REGRESSOR_BY_NAME[n] for n in ["KNN", "Ridge", "Lasso", "DecisionTree", "RandomForest", "XGBoost"]}
            specs.update({"BayesianRidge": BAYESIAN_RIDGE_MODEL, "SVR_RBF": SVR_RBF_MODEL, "LightGBM": LIGHTGBM_MODEL})
            spec = specs[nm]
            t = time.perf_counter()
            est = spec.estimator()
            est.set_params(**params)
            sc = StandardScaler().fit(X[tr]) if spec.needs_scaling else None
            est.fit(sc.transform(X[tr]) if sc else X[tr], y[tr])
            tf.append(time.perf_counter() - t)
            t = time.perf_counter()
            est.predict(sc.transform(X[te]) if sc else X[te])
            tp.append(time.perf_counter() - t)
            npar = n_parametros(nm, est, len(tr), X.shape[1])
    mi = proc.memory_info()
    pico = getattr(mi, "peak_wset", mi.rss)
    print(json.dumps({"modelo": nm, "entrenamiento_s": float(np.median(tf)), "inferencia_s": float(np.median(tp)),
                      "memoria_pico_MB": max(pico - base, 0) / 2 ** 20, "parametros": npar}))


def etapa_latex():
    """Tablas del paper en LaTeX (booktabs), generadas desde los CSV de artifacts/resultados/."""
    L = OUT / "latex"
    L.mkdir(exist_ok=True)
    met = pd.read_csv(OUT / "metricas_por_pliegue.csv")
    pr = pd.read_csv(OUT / "pruebas_por_baseline.csv")
    g = met.groupby(["conjunto", "modelo"])
    mean, sd = g.mean(numeric_only=True), g.std(numeric_only=True)
    modelos = ["ALEE"] + BASELINES
    # Tabla RMSE: media ± desviación, mejor en negrita, segundo subrayado, símbolos de significancia
    lines = [r"\begin{tabular}{l" + "c" * 4 + "}", r"\toprule", "Modelo & " + " & ".join(NOMBRE[c] for c in CONJUNTOS) + r" \\", r"\midrule"]
    orden = {c: mean.loc[c]["RMSE"].sort_values().index.tolist() for c in CONJUNTOS}
    for m in modelos:
        cells = []
        for c in CONJUNTOS:
            s = f"{mean.loc[(c, m), 'RMSE']:.3f} $\\pm$ {sd.loc[(c, m), 'RMSE']:.3f}"
            if orden[c][0] == m:
                s = r"\textbf{" + s + "}"
            elif orden[c][1] == m:
                s = r"\underline{" + s + "}"
            if m != "ALEE":
                r = pr[(pr.conjunto == c) & (pr.baseline == m)].iloc[0]
                if r.dm_p_holm < 0.05:
                    s += r"$^{\dagger}$" if r.dif_perdida_media < 0 else r"$^{\ddagger}$"
            cells.append(s)
        lines.append(ES[m] + " & " + " & ".join(cells) + r" \\")
        if m == "ALEE" or m == "SVR_RBF":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (L / "tabla_rmse.tex").write_text("\n".join(lines), encoding="utf-8")
    # Tabla de métricas secundarias: ALEE, SARIMA, experto estacional y mejor modelo del curso
    lines = [r"\begin{tabular}{llccccc}", r"\toprule", r"Conjunto & Modelo & RMSE & MAE & MAPE (\%) & $R^2$ & MASE \\", r"\midrule"]
    for c in CONJUNTOS:
        mejor_curso = mean.loc[c].loc[CURSO]["RMSE"].idxmin()
        for i, m in enumerate(["ALEE", "SARIMA", "EstacionalALEE", mejor_curso]):
            cells = [f"{mean.loc[(c, m), k]:.3f} $\\pm$ {sd.loc[(c, m), k]:.3f}" if k != "MAPE" else f"{mean.loc[(c, m), k]:.2f} $\\pm$ {sd.loc[(c, m), k]:.2f}"
                     for k in ["RMSE", "MAE", "MAPE", "R2", "MASE"]]
            nm = ES[m] + (" (mejor del curso)" if m == mejor_curso else "")
            lines.append((NOMBRE[c] if i == 0 else "") + " & " + nm + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule" if c != CONJUNTOS[-1] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    (L / "tabla_secundarias.tex").write_text("\n".join(lines), encoding="utf-8")
    # Tabla de pruebas, Colombia (detalle completo)
    def tabla_pruebas(c):
        out = [r"\begin{tabular}{lccccccc}", r"\toprule",
               r"Baseline & $\bar d$ [IC 95\,\%] & DM & $p_{\mathrm{Holm}}$ & GW $p_{\mathrm{Holm}}$ & $P$(mejor/ROPE/peor) & $d$ & $A_{12}$ \\", r"\midrule"]
        for _, r in pr[pr.conjunto == c].iterrows():
            out.append(f"{ES[r.baseline]} & {r.dif_perdida_media:.3f} [{r.ic95_inf:.3f}, {r.ic95_sup:.3f}] & {r.dm_estadistico:.2f} & {fp(r.dm_p_holm)} & "
                       f"{fp(r.gw_p_holm)} & {r.p_bayes_mejor:.2f}/{r.p_bayes_rope:.2f}/{r.p_bayes_peor:.2f} & {r.d_cohen:.2f} & {float(r.a12_vargha):.2f} \\\\")
        out += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(out)
    for c in CONJUNTOS:
        (L / f"tabla_pruebas_{c}.tex").write_text(tabla_pruebas(c), encoding="utf-8")
    # semillas de los baselines estocásticos y de ALEE
    sv = pd.read_csv(OUT / "semillas_baselines_colombia.csv").groupby("modelo")["rmse_medio"].agg(["mean", "std", "min", "max"])
    sa = pd.read_csv(OUT / "sensibilidad_semillas.csv")
    sa = sa[sa.semilla.astype(str).str.fullmatch(r"\d+")]
    out = [r"\begin{tabular}{llcccc}", r"\toprule", r"Conjunto & Modelo (semillas) & Media & Desv. & Mín. & Máx. \\", r"\midrule"]
    for m, r in sv.iterrows():
        out.append(f"Colombia & {ES[m]} (42--51) & {r['mean']:.3f} & {r['std']:.3f} & {r['min']:.3f} & {r['max']:.3f} \\\\")
    for c in CONJUNTOS:
        s = sa[sa.conjunto == c]["rmse_medio"]
        out.append(f"{NOMBRE[c]} & ALEE (42--46) & {s.mean():.3f} & {s.std():.3f} & {s.min():.3f} & {s.max():.3f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (L / "tabla_semillas.tex").write_text("\n".join(out), encoding="utf-8")
    # Tabla de pruebas resumida, conjuntos adicionales
    out = [r"\begin{tabular}{l" + "ccc" * 3 + "}", r"\toprule",
           " & " + " & ".join(r"\multicolumn{3}{c}{" + NOMBRE[c] + "}" for c in CONJUNTOS[1:]) + r" \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
           "Baseline" + r" & DM & GW & $P_{\mathrm{mejor}}$" * 3 + r" \\", r"\midrule"]
    for b in BASELINES:
        cells = []
        for c in CONJUNTOS[1:]:
            r = pr[(pr.conjunto == c) & (pr.baseline == b)].iloc[0]
            sign = "+" if r.dif_perdida_media < 0 else "$-$"
            cells += [f"{fp(r.dm_p_holm)}{sign}", fp(r.gw_p_holm), f"{r.p_bayes_mejor:.2f}"]
        out.append(ES[b] + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (L / "tabla_pruebas_adicionales.tex").write_text("\n".join(out), encoding="utf-8")
    # Tabla entre conjuntos
    en = pd.read_csv(OUT / "pruebas_entre_conjuntos.csv")
    fr = pd.read_csv(OUT / "friedman.csv", index_col=0).iloc[:, 0]
    out = [r"\begin{tabular}{lcccc}", r"\toprule", r"Baseline & Rango medio & Conjuntos en que ALEE gana & Conover $p_{\mathrm{Holm}}$ & $P$(mejor/ROPE/peor) \\", r"\midrule",
           f"ALEE & {fr['rango_ALEE']:.2f} & --- & --- & --- \\\\", r"\midrule"]
    for _, r in en.sort_values("baseline", key=lambda s: s.map(lambda b: fr[f"rango_{b}"])).iterrows():
        out.append(f"{ES[r.baseline]} & {fr['rango_' + r.baseline]:.2f} & {int(r.conjuntos_ALEE_gana)}/4 & {fp(r.conover_p_holm)} & "
                   f"{r.p_bayes_mejor:.2f}/{r.p_bayes_rope:.2f}/{r.p_bayes_peor:.2f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (L / "tabla_entre_conjuntos.tex").write_text("\n".join(out), encoding="utf-8")
    # Tabla de ablación
    ab = pd.read_csv(OUT / "ablacion.csv")
    vs = ["ALEE completo", "Sin localización", "Sin λ adaptativa", "Previo uniforme", "Promedio simple", "Solo persistencia", "Solo núcleo", "Solo estacional"]
    out = [r"\begin{tabular}{l" + "cc" * 4 + "}", r"\toprule",
           " & " + " & ".join(r"\multicolumn{2}{c}{" + NOMBRE[c] + "}" for c in CONJUNTOS) + r" \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
           "Variante" + r" & RMSE & $p_{\mathrm{Holm}}$" * 4 + r" \\", r"\midrule"]
    for v in vs:
        cells = []
        for c in CONJUNTOS:
            r = ab[(ab.conjunto == c) & (ab.variante == v)].iloc[0]
            cells += [f"{r.rmse_medio:.3f}", "---" if v == "ALEE completo" else fp(r.dm_p_holm)]
        out.append(v.replace("λ", r"$\lambda$") + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (L / "tabla_ablacion.tex").write_text("\n".join(out), encoding="utf-8")
    # Tabla de costo
    co = pd.read_csv(OUT / "costo_computacional.csv")
    out = [r"\begin{tabular}{lcccc}", r"\toprule", r"Modelo & Entrenamiento (s) & Inferencia (s) & Memoria pico (MB) & Parámetros \\", r"\midrule"]
    for _, r in co.iterrows():
        out.append(f"{ES[r.modelo]} & {r.entrenamiento_s:.3f} & {r.inferencia_s:.4f} & {r.memoria_pico_MB:.2f} & {int(r.parametros) if pd.notna(r.parametros) else '---'} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (L / "tabla_costo.tex").write_text("\n".join(out), encoding="utf-8")
    print("tablas LaTeX en", L)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "latex":
        etapa_latex()
        sys.exit()
    if cmd == "extras":
        etapa_extras()
        sys.exit()
    if cmd == "costo_ajustado":
        etapa_costo_ajustado()
        sys.exit()
    if cmd == "costo_uno":
        etapa_costo_uno(sys.argv[2])
        sys.exit()
    if cmd == "modelos":
        etapa_modelos(sys.argv[2])
    elif cmd == "alee":
        etapa_alee(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 42)
    elif cmd == "fijos":
        etapa_fijos(sys.argv[2])
    elif cmd == "costo":
        etapa_costo()
    elif cmd == "analisis":
        etapa_analisis()
