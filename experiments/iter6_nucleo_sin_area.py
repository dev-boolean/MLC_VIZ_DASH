"""Iteracion 6 (exploratoria, posterior a la iteracion 4 adoptada): el experto
``nucleo_cambio`` de ALEE hace Nadaraya-Watson sobre la distancia euclidiana
estandarizada de las 17 columnas completas de DEFAULT_FEATURES, incluidas las
3 columnas de AREA_COLS (indicadores de las 13 areas metropolitanas). Esas
columnas tienen multicolinealidad severa con las versiones nacionales (ya
documentado desde el segundo entregable), asi que podrian estar diluyendo el
kernel con dimensiones redundantes/ruidosas en vez de ayudar a encontrar
meses parecidos. Ademas, al evaluar ALEE contra datos de 2026 sin columnas de
area reales, rellenarlas con un valor sintetico distorsiona esa distancia de
forma severa (hallazgo que motivo esta iteracion).

Se prueba una variante de ALEE donde el kernel de ``nucleo_cambio`` excluye
AREA_COLS al estandarizar y calcular distancias. Todo lo demas (persistencia,
estacional, localizacion, regularizacion entropica adaptativa) queda igual.
Validacion honesta: mismo protocolo del proyecto (TimeSeriesSplit 5/3 gap=1,
40 evaluaciones bayesianas, semilla 42), optimizando SOLO con pliegues
internos.
"""
import sys, os, time
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from src import config
from src.data import assemble_dataset, AREA_COLS
from src.original_model import ModeloOriginal, EXPERTS, _month_of, _standardize_fit, solve_weights
from src.stats_tests import diebold_mariano_test

optuna.logging.set_verbosity(optuna.logging.WARNING)
ds = assemble_dataset()


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


class _ExpertsSinArea:
    """Igual a ``_Experts`` pero el kernel de nucleo_cambio excluye AREA_COLS."""

    def __init__(self, X, y, cols, kernel_mask):
        self.cols = cols
        self.kernel_mask = kernel_mask
        lag1 = X[:, cols["lag1"]]
        Xk = X[:, kernel_mask]
        self.mu, self.sd = _standardize_fit(Xk)
        Z = (Xk - self.mu) / self.sd
        dy = y - lag1
        self.Z, self.dy = Z, dy
        if len(Z) > 1:
            d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
            med = np.median(d2[np.triu_indices(len(Z), 1)])
            self.bw2 = med if med > 0 else 1.0
        else:
            self.bw2 = 1.0
        months = _month_of(X[:, cols["mes_sin"]], X[:, cols["mes_cos"]])
        self.month_change = np.zeros(13)
        for m in range(1, 13):
            sel = months == m
            self.month_change[m] = dy[sel].sum() / (sel.sum() + 1.0)

    def predict(self, X):
        lag1 = X[:, self.cols["lag1"]]
        Xk = X[:, self.kernel_mask]
        Z = (Xk - self.mu) / self.sd
        d2 = ((Z[:, None, :] - self.Z[None, :, :]) ** 2).sum(-1)
        k = np.exp(-(d2 - d2.min(axis=1, keepdims=True)) / (2 * self.bw2))
        nw = (k * self.dy).sum(axis=1) / k.sum(axis=1)
        months = _month_of(X[:, self.cols["mes_sin"]], X[:, self.cols["mes_cos"]])
        return np.column_stack([
            lag1,
            lag1 + nw,
            lag1 + self.month_change[months],
        ])


class ModeloNucleoSinArea(ModeloOriginal):
    """ModeloOriginal con el kernel de nucleo_cambio sin columnas de area."""

    def fit(self, X, y):
        self._validate_params()
        X, y = check_X_y(X, y, dtype=float, y_numeric=True)
        if X.shape[1] != len(self.feature_names):
            raise ValueError(f"X tiene {X.shape[1]} columnas y feature_names {len(self.feature_names)}")
        n = len(y)
        self.cols_ = {"lag1": self._col(self.lag_col), "mes_sin": self._col(self.month_cols[0]),
                      "mes_cos": self._col(self.month_cols[1])}
        self.kernel_mask_ = np.array([c not in AREA_COLS for c in self.feature_names])
        m0 = int(min(self.min_train, max(2, n // 2)))
        errors, rows = [], []
        for s in range(m0, n):
            experts = _ExpertsSinArea(X[:s], y[:s], self.cols_, self.kernel_mask_)
            errors.append(y[s] - experts.predict(X[s:s + 1])[0])
            rows.append(s)
        self.errors_ = np.asarray(errors)
        self.oof_rows_ = np.asarray(rows)
        state = self._state(X)
        self.state_mu_, self.state_sd_ = _standardize_fit(state)
        self.state_ = (state[self.oof_rows_] - self.state_mu_) / self.state_sd_
        gamma = 0.0 if np.isinf(self.half_life) else np.log(0.5) / self.half_life
        self.log_decay_ = gamma * (n - 1 - self.oof_rows_)
        self.experts_ = _ExpertsSinArea(X, y, self.cols_, self.kernel_mask_)
        self.n_features_in_ = X.shape[1]
        self.expert_names_ = EXPERTS
        self.prior_ = self._global_prior()
        return self


def make_objective(Xtr, ytr):
    inner = list(TimeSeriesSplit(config.N_SPLITS_INNER, gap=config.CV_GAP).split(Xtr))

    def objective(trial):
        params = dict(
            bandwidth=trial.suggest_float("bandwidth", 0.05, 3.0, log=True),
            half_life=trial.suggest_float("half_life", 6.0, 120.0, log=True),
            lam=trial.suggest_float("lam", 1e-3, 100.0, log=True),
        )
        errs = []
        for itr, ite in inner:
            m = ModeloNucleoSinArea(**params).fit(Xtr[itr], ytr[itr])
            p = m.predict(Xtr[ite])
            errs.append(rmse(ytr[ite], p))
        return float(np.mean(errs))
    return objective


def main():
    os.makedirs("artifacts/confirmatorio/iter6", exist_ok=True)
    rows_fold = []
    all_y, all_p, all_idx = [], [], []
    t0 = time.time()
    for f, (tr, te) in enumerate(TimeSeriesSplit(config.N_SPLITS_OUTER, gap=config.CV_GAP).split(ds.X)):
        Xtr, ytr = ds.X.values[tr], ds.y_reg.values[tr]
        Xte, yte = ds.X.values[te], ds.y_reg.values[te]
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(make_objective(Xtr, ytr), n_trials=40, show_progress_bar=False)
        best = study.best_params
        m = ModeloNucleoSinArea(**best).fit(Xtr, ytr)
        p = m.predict(Xte)
        f_rmse = rmse(yte, p)
        all_y.append(yte); all_p.append(p); all_idx.append(te)
        rows_fold.append({"pliegue": f, "rmse": f_rmse, **best})
        print(f"  pliegue {f}: rmse={f_rmse:.4f} params={best}", flush=True)
    all_y = np.concatenate(all_y); all_p = np.concatenate(all_p); all_idx = np.concatenate(all_idx)
    fold_rmses = [r["rmse"] for r in rows_fold]
    print(f"\nRMSE medio de pliegues: {np.mean(fold_rmses):.4f} +/- {np.std(fold_rmses):.4f}")
    print(f"RMSE agrupado: {rmse(all_y, all_p):.4f}")
    print(f"tiempo total: {time.time()-t0:.1f}s")
    pd.DataFrame(rows_fold).to_csv("artifacts/confirmatorio/iter6/rmse_nucleo_sin_area.csv", index=False)
    oof = pd.DataFrame({"idx": all_idx, "y_true": all_y, "y_pred": all_p}).sort_values("idx")
    oof.to_csv("artifacts/confirmatorio/iter6/oof_nucleo_sin_area.csv", index=False)

    # -------------------- comparacion con referencias --------------------
    est = pd.read_csv("artifacts/confirmatorio/oof_EstacionalALEE.csv").set_index("idx")
    base = pd.read_csv("artifacts/confirmatorio/baselines_series_tiempo.csv").set_index("idx")
    oof_i = oof.set_index("idx")
    common = oof_i.index
    y_true = oof_i["y_true"].values
    p_new = oof_i["y_pred"].values
    p_est = est.loc[common, "y_pred"].values
    p_sarima = base.loc[common, "SARIMA"].values

    print("\n--- Comparacion (240 meses) ---")
    print(f"Nucleo sin area:  medio pliegues={np.mean(fold_rmses):.4f}  agrupado={rmse(all_y, all_p):.4f}")
    print(f"Estacional aislado: agrupado={rmse(y_true, p_est):.4f}")
    print(f"SARIMA: agrupado={rmse(y_true, p_sarima):.4f}")

    if np.mean(fold_rmses) < 0.90:  # solo correr DM si se ve competitivo
        dm_est = diebold_mariano_test(y_true - p_new, y_true - p_est)
        dm_sarima = diebold_mariano_test(y_true - p_new, y_true - p_sarima)
        print(f"\nDM-HLN vs estacional aislado: {dm_est}")
        print(f"DM-HLN vs SARIMA: {dm_sarima}")


if __name__ == "__main__":
    main()
