"""Iteración 5a (exploratoria, posterior a la validación confirmatoria y a la
iteración 4 ya adoptada): combinar SARIMA y el experto estacional aislado
-los dos únicos modelos que ALEE no ha logrado vencer- dentro del mecanismo
de ALEE (localización + regularización entrópica adaptativa), y probar
también una variante robusta (mediana) del experto estacional.

No modifica src/original_model.py, src/models.py, el libro ni el paper.
Reutiliza el solucionador `solve_weights` tal cual está, sin tocarlo.

Diseño para que sea computacionalmente viable
----------------------------------------------
El error de un experto en el mes s (entrenado solo con las filas 0..s-1) no
depende de los hiperparámetros de ALEE ni del pliegue en el que se esté
evaluando: es el mismo valor siempre que s sea el mismo, porque las filas de
todos los pliegues (internos y externos) son siempre un PREFIJO cronológico
de la serie completa. Por eso los errores de cada experto se precalculan UNA
SOLA VEZ, globalmente, y luego se recortan (slice) según el tamaño de
entrenamiento de cada pliegue -- exactamente lo que ya hace
`_Experts.fit()` para los expertos NumPy, solo que aquí se cachea en vez de
recomputar en cada evaluación bayesiana.

SARIMA necesita una ventana mínima real para estimar un término de media
móvil estacional (con 6-13 meses, como ocurre en los pliegues internos más
pequeños de este problema, no converge de forma confiable). Por eso se fija
un piso de entrenamiento MIN_TRAIN=24 para las CUATRO variantes de este
experimento (no solo para las que incluyen SARIMA), de modo que la
comparación sea consistente. Esto es una limitación de diseño declarada, no
un ajuste hecho mirando el pliegue de prueba.
"""
import os
import sys
import time
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.data import assemble_dataset, load_raw
from src.original_model import _Experts, _month_of, solve_weights
from src.stats_tests import diebold_mariano_test

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT = config.ROOT_DIR / "artifacts" / "confirmatorio" / "iter5"
OUT.mkdir(parents=True, exist_ok=True)

MIN_TRAIN = 24          # piso de entrenamiento para las 4 variantes (ver docstring)
FIRST_ROW = 11           # fila r del dataset <-> mes r + FIRST_ROW de la serie cruda
N_TRIALS = 40
SEED = config.SEED

ds = assemble_dataset()
N = len(ds.X)
X_ALL = ds.X.values
Y_ALL = ds.y_reg.values
Y_SERIES = load_raw()["tasa_desempleo_nacional"].values
COLS = {"lag1": list(ds.feature_names).index("td_nacional_lag1"),
        "mes_sin": list(ds.feature_names).index("mes_sin"),
        "mes_cos": list(ds.feature_names).index("mes_cos")}

print(f"N={N}  MIN_TRAIN={MIN_TRAIN}  len(Y_SERIES)={len(Y_SERIES)}")


# --------------------------------------------------------------------- SARIMA
def sarima_walkforward_errors(min_train=MIN_TRAIN, refit_every=24):
    """Errores e_s = y_s - f_SARIMA^(s)(x_s) para s = min_train..N-1.

    Ajusta SARIMAX una vez cada `refit_every` meses (más los puntos de corte
    de los 5 pliegues externos, para que la comparación con el baseline
    oficial sea lo más fiel posible) y extiende con `.append(refit=False)`
    entre ajustes, igual que `src.analysis.regression_baselines`.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    outer = list(TimeSeriesSplit(config.N_SPLITS_OUTER, gap=config.CV_GAP).split(ds.X))
    outer_train_ends = sorted({len(tr) for tr, _ in outer})
    checkpoints = sorted(set([min_train] + list(range(min_train, N, refit_every)) + outer_train_ends))
    checkpoints = [c for c in checkpoints if min_train <= c < N]
    if checkpoints[0] != min_train:
        checkpoints = [min_train] + checkpoints

    preds = np.full(N, np.nan)
    t0 = time.time()
    for i, cp in enumerate(checkpoints):
        next_cp = checkpoints[i + 1] if i + 1 < len(checkpoints) else N
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = SARIMAX(Y_SERIES[: cp + FIRST_ROW + 1], order=(0, 1, 1),
                          seasonal_order=(0, 1, 1, 12)).fit(disp=False, maxiter=200)
            extended = fit.append(Y_SERIES[cp + FIRST_ROW + 1: next_cp + FIRST_ROW + 1], refit=False)
        pred = extended.get_prediction(start=cp + FIRST_ROW + 1, end=next_cp + FIRST_ROW, dynamic=False).predicted_mean
        preds[cp:next_cp] = np.asarray(pred)
    print(f"  SARIMA walk-forward: {len(checkpoints)} ajustes, {time.time()-t0:.1f}s")
    errors = Y_ALL - preds
    return preds, errors  # NaN antes de min_train


def validate_sarima_against_baseline(preds):
    """Compara contra el baseline oficial de SARIMA sobre los 240 meses de prueba."""
    outer = list(TimeSeriesSplit(config.N_SPLITS_OUTER, gap=config.CV_GAP).split(ds.X))
    fold_rmse = []
    all_err = []
    for tr, te in outer:
        err = Y_ALL[te] - preds[te]
        fold_rmse.append(np.sqrt(np.mean(err ** 2)))
        all_err.append(err)
    all_err = np.concatenate(all_err)
    print(f"  Validación SARIMA iter5a: RMSE medio pliegues = {np.mean(fold_rmse):.4f} +/- {np.std(fold_rmse):.4f}"
          f"  (oficial: 0.818).  RMSE agrupado = {np.sqrt(np.mean(all_err**2)):.4f}  (oficial: 0.9954)")
    return np.mean(fold_rmse), np.sqrt(np.mean(all_err ** 2))


# ---------------------------------------------------- expertos NumPy (caché global)
def global_numpy_errors(min_train=MIN_TRAIN):
    """Errores de persistencia, nucleo_cambio, estacional y estacional_robusto.

    Mismo cálculo expansivo que `_Experts`, pero una sola vez para toda la
    serie (s = min_train..N-1), reutilizable por cualquier pliegue mediante
    slicing (ver docstring del módulo).
    """
    names = ["persistencia", "nucleo_cambio", "estacional", "estacional_robusto"]
    out = {n: np.full(N, np.nan) for n in names}
    t0 = time.time()
    for s in range(min_train, N):
        Xs, ys = X_ALL[:s], Y_ALL[:s]
        exp = _Experts(Xs, ys, COLS)
        pred3 = exp.predict(X_ALL[s:s + 1])[0]  # persistencia, nucleo_cambio, estacional
        out["persistencia"][s] = pred3[0]
        out["nucleo_cambio"][s] = pred3[1]
        out["estacional"][s] = pred3[2]
        # variante robusta: mediana del cambio por mes en vez de la media encogida
        lag1 = Xs[:, COLS["lag1"]]
        dy = ys - lag1
        months = _month_of(Xs[:, COLS["mes_sin"]], Xs[:, COLS["mes_cos"]])
        m_target = _month_of(X_ALL[s:s + 1, COLS["mes_sin"]], X_ALL[s:s + 1, COLS["mes_cos"]])[0]
        sel = months == m_target
        med_change = np.median(dy[sel]) if sel.sum() > 0 else 0.0
        out["estacional_robusto"][s] = Y_ALL[s - 1] if s == 0 else X_ALL[s, COLS["lag1"]] + med_change
    print(f"  Expertos NumPy (persistencia/nucleo/estacional/robusto): {time.time()-t0:.1f}s")
    errors = {n: Y_ALL - out[n] for n in names}
    return out, errors


# ------------------------------------------------------------------- combinador
STATE_COLS = ("td_nacional_rollstd3",)
GAP_COLS = ("td_nacional_lag1", "td_nacional_rollmean12")


def state_matrix(X):
    c1 = list(ds.feature_names).index(STATE_COLS[0])
    c2a, c2b = list(ds.feature_names).index(GAP_COLS[0]), list(ds.feature_names).index(GAP_COLS[1])
    return np.column_stack([X[:, c1], X[:, c2a] - X[:, c2b]])


def run_variant(name, expert_names, errors_dict, min_train=MIN_TRAIN, n_trials=N_TRIALS):
    """Validación anidada honesta (solo pliegues internos) para un conjunto de expertos."""
    print(f"\n=== Variante {name}: {expert_names} ===")
    E_global = np.column_stack([errors_dict[e] for e in expert_names])  # (N, K), NaN antes de min_train
    state_all = state_matrix(X_ALL)

    def fit_predict(train_rows, test_rows, bandwidth, half_life, lam):
        """train_rows/test_rows: índices GLOBALES (prefijos), igual semántica que ModeloOriginal."""
        n_tr = train_rows[-1] + 1  # tamaño de entrenamiento (prefijo 0..n_tr-1)
        s_idx = np.arange(min_train, n_tr)
        if len(s_idx) < 2:
            return np.full(len(test_rows), np.nan)
        E = E_global[s_idx]
        if np.isnan(E).any():
            return np.full(len(test_rows), np.nan)
        state_mu, state_sd = state_all[:n_tr].mean(0), state_all[:n_tr].std(0)
        state_sd[state_sd < 1e-12] = 1.0
        state_hist = (state_all[s_idx] - state_mu) / state_sd
        gamma = 0.0 if np.isinf(half_life) else np.log(0.5) / half_life
        log_decay = gamma * (n_tr - 1 - s_idx)
        # previo global (igual que ModeloOriginal._global_prior, shrink_to="global")
        a_ref = np.exp(log_decay - log_decay.max()); a_ref /= a_ref.sum()
        n_eff_ref = 1.0 / np.sum(a_ref ** 2)
        Qg = (E * a_ref[:, None]).T @ E
        prior = solve_weights(Qg, 1e-2, max_iter=5000)
        prior = np.maximum(prior, 1e-6); prior /= prior.sum()

        preds = np.empty(len(test_rows))
        for i, t in enumerate(test_rows):
            z = (state_all[t] - state_mu) / state_sd
            d2 = ((z[None, :] - state_hist) ** 2).sum(-1)
            logk = -d2 / (2.0 * bandwidth ** 2) + log_decay
            logk -= logk.max()
            a = np.exp(logk); a /= a.sum()
            n_eff = 1.0 / np.sum(a ** 2)
            lam_eff = lam * n_eff_ref / n_eff
            Q = (E * a[:, None]).T @ E
            w = solve_weights(Q, lam_eff, w0=prior, prior=prior)
            # pronóstico final de cada experto en t, entrenado con 0..n_tr-1 (igual que ModeloOriginal)
            f_t = np.array([_final_expert_pred(e, t, n_tr) for e in expert_names])
            preds[i] = w @ f_t
        return preds

    fold_rmses, all_y, all_p, hp_rows = [], [], [], []
    outer = list(TimeSeriesSplit(config.N_SPLITS_OUTER, gap=config.CV_GAP).split(ds.X))
    t0 = time.time()
    for f, (tr, te) in enumerate(outer):
        inner = list(TimeSeriesSplit(config.N_SPLITS_INNER, gap=config.CV_GAP).split(tr))

        def objective(trial):
            bw = trial.suggest_float("bandwidth", 0.05, 3.0, log=True)
            hl = trial.suggest_float("half_life", 6.0, 120.0, log=True)
            lam = trial.suggest_float("lam", 1e-3, 100.0, log=True)
            errs = []
            for itr, ite in inner:
                if itr[-1] < min_train:
                    continue
                p = fit_predict(itr, ite, bw, hl, lam)
                if np.isnan(p).any():
                    return 1e6
                errs.append(np.sqrt(np.mean((Y_ALL[ite] - p) ** 2)))
            return float(np.mean(errs)) if errs else 1e6

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = study.best_params
        p_test = fit_predict(tr, te, best["bandwidth"], best["half_life"], best["lam"])
        rmse = np.sqrt(np.mean((Y_ALL[te] - p_test) ** 2))
        fold_rmses.append(rmse); all_y.append(Y_ALL[te]); all_p.append(p_test)
        hp_rows.append({"pliegue": f, "rmse": rmse, **best})
        print(f"  pliegue {f}: RMSE={rmse:.4f}  hp={best}")
    all_y, all_p = np.concatenate(all_y), np.concatenate(all_p)
    pooled = np.sqrt(np.mean((all_y - all_p) ** 2))
    print(f"  [{name}] RMSE medio pliegues = {np.mean(fold_rmses):.4f} +/- {np.std(fold_rmses):.4f}"
          f"   RMSE agrupado = {pooled:.4f}   tiempo = {time.time()-t0:.1f}s")
    pd.DataFrame(hp_rows).to_csv(OUT / f"rmse_iter5a_{name}.csv", index=False)
    pd.DataFrame({"idx": np.concatenate([te for _, te in outer]), "y_true": all_y, "y_pred": all_p}).to_csv(
        OUT / f"oof_iter5a_{name}.csv", index=False)
    return np.mean(fold_rmses), np.std(fold_rmses), pooled, all_y - all_p


def _final_expert_pred(expert_name, t, n_tr):
    """Pronóstico del experto `expert_name` en la fila t, entrenado con filas 0..n_tr-1.

    Para los expertos NumPy, reutiliza el objeto `_Experts` entrenado con el
    tramo correspondiente (barato). Para SARIMA, usa la predicción ya
    precalculada globalmente en `sarima_preds` (ver nota: estrictamente el
    ajuste de SARIMA usado puede provenir de un punto de corte ligeramente
    distinto a n_tr exacto, ver limitación declarada en el reporte).
    """
    global _FINAL_CACHE
    if expert_name == "sarima":
        return _SARIMA_PREDS[t]
    key = n_tr
    if key not in _FINAL_CACHE:
        _FINAL_CACHE[key] = _Experts(X_ALL[:n_tr], Y_ALL[:n_tr], COLS)
    exp = _FINAL_CACHE[key]
    pred3 = exp.predict(X_ALL[t:t + 1])[0]
    return {"persistencia": pred3[0], "nucleo_cambio": pred3[1], "estacional": pred3[2]}[expert_name] \
        if expert_name != "estacional_robusto" else _robust_pred(n_tr, t)


def _robust_pred(n_tr, t):
    Xs, ys = X_ALL[:n_tr], Y_ALL[:n_tr]
    lag1 = Xs[:, COLS["lag1"]]
    dy = ys - lag1
    months = _month_of(Xs[:, COLS["mes_sin"]], Xs[:, COLS["mes_cos"]])
    m_target = _month_of(X_ALL[t:t + 1, COLS["mes_sin"]], X_ALL[t:t + 1, COLS["mes_cos"]])[0]
    sel = months == m_target
    med_change = np.median(dy[sel]) if sel.sum() > 0 else 0.0
    return X_ALL[t, COLS["lag1"]] + med_change


_FINAL_CACHE = {}
_SARIMA_PREDS = None

if __name__ == "__main__":
    print("Paso 1: walk-forward de SARIMA + validación contra el baseline oficial")
    sarima_preds, sarima_errors = sarima_walkforward_errors()
    _SARIMA_PREDS = sarima_preds
    validate_sarima_against_baseline(sarima_preds)

    print("\nPaso 2: expertos NumPy (persistencia, núcleo, estacional, estacional robusto)")
    numpy_preds, numpy_errors = global_numpy_errors()

    errors_dict = {**numpy_errors, "sarima": sarima_errors}

    print("\nPaso 3: combinaciones vía ALEE (validación anidada honesta)")
    results = {}
    variants = {
        "SARIMA_estacional": ("sarima", "estacional"),
        "SARIMA_estacional_persistencia": ("sarima", "estacional", "persistencia"),
        "SARIMA_estacional_robusto": ("sarima", "estacional", "estacional_robusto"),
        "SARIMA_robusto": ("sarima", "estacional_robusto"),
    }
    for name, experts in variants.items():
        results[name] = run_variant(name, experts, errors_dict)

    print("\n=== RESUMEN ===")
    summary = pd.DataFrame([
        {"variante": name, "rmse_medio_pliegues": r[0], "std": r[1], "rmse_agrupado": r[2]}
        for name, r in results.items()
    ])
    print(summary.to_string(index=False))
    summary.to_csv(OUT / "resumen_iter5a.csv", index=False)

    print("\nPaso 4: Diebold-Mariano-HLN contra estacional aislado y SARIMA, para variantes competitivas")
    est_oof = pd.read_csv(config.ROOT_DIR / "artifacts" / "confirmatorio" / "oof_EstacionalALEE.csv").set_index("idx")
    base = pd.read_csv(config.ROOT_DIR / "artifacts" / "confirmatorio" / "baselines_series_tiempo.csv").set_index("idx")
    outer = list(TimeSeriesSplit(config.N_SPLITS_OUTER, gap=config.CV_GAP).split(ds.X))
    test_idx = np.concatenate([te for _, te in outer])
    err_est = (base.loc[test_idx, "y_true"] - est_oof.loc[test_idx, "y_pred"]).values
    err_sarima_off = (base.loc[test_idx, "y_true"] - base.loc[test_idx, "SARIMA"]).values
    for name, r in results.items():
        err_variant = r[3]
        if r[0] <= 0.80:  # solo si empata o le gana al estacional (0.780) en el punto estimado
            dm_est = diebold_mariano_test(err_variant, err_est)
            dm_sar = diebold_mariano_test(err_variant, err_sarima_off)
            print(f"{name}: DM vs estacional p={dm_est['p_value']:.4f} (diff={dm_est['mean_loss_diff']:.4f}); "
                  f"DM vs SARIMA p={dm_sar['p_value']:.4f} (diff={dm_sar['mean_loss_diff']:.4f})")
        else:
            print(f"{name}: RMSE medio pliegues {r[0]:.4f} > 0.780, no se corre DM (no compite con el estacional)")
