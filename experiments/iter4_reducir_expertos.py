"""Iteracion 4 (exploratoria): probar si reducir el numero de expertos (quitar
los mas debiles: ridge_nivel y ridge_cambio, que individualmente tienen RMSE
~2x peor que el experto estacional) mejora la combinacion, bajo el mismo
presupuesto del protocolo congelado (40 evaluaciones, nested CV 5/3, gap=1).
"""
import sys, os, time
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
import optuna
from sklearn.model_selection import TimeSeriesSplit
from src.data import assemble_dataset
from src import config
from src.original_model import ModeloOriginal, EXPERTS

optuna.logging.set_verbosity(optuna.logging.WARNING)
ds = assemble_dataset()

def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))

class ModeloReducido(ModeloOriginal):
    """Igual a ModeloOriginal pero restringido a un subconjunto de expertos."""
    def __init__(self, expert_names=EXPERTS, **kw):
        self.expert_names = expert_names
        super().__init__(**kw)

    def fit(self, X, y):
        super().fit(X, y)
        idx = [EXPERTS.index(n) for n in self.expert_names]
        self.errors_ = self.errors_[:, idx]
        self._idx = idx
        self.prior_ = self._global_prior()
        return self

    def expert_predictions(self, X):
        F = super().expert_predictions(X)
        return F[:, self._idx]

def make_objective(Xtr, ytr, expert_names):
    inner = list(TimeSeriesSplit(config.N_SPLITS_INNER, gap=config.CV_GAP).split(Xtr))
    def objective(trial):
        params = dict(
            bandwidth=trial.suggest_float("bandwidth", 0.05, 3.0, log=True),
            half_life=trial.suggest_float("half_life", 6.0, 120.0, log=True),
            lam=trial.suggest_float("lam", 1e-3, 100.0, log=True),
            ridge_alpha=trial.suggest_float("ridge_alpha", 0.1, 300.0, log=True),
        )
        errs = []
        for itr, ite in inner:
            m = ModeloReducido(expert_names=expert_names, **params).fit(Xtr[itr], ytr[itr])
            p = m.predict(Xtr[ite])
            errs.append(rmse(ytr[ite], p))
        return float(np.mean(errs))
    return objective

def run(expert_names, tag):
    rows_fold = []
    all_y, all_p = [], []
    t0 = time.time()
    for f, (tr, te) in enumerate(TimeSeriesSplit(config.N_SPLITS_OUTER, gap=config.CV_GAP).split(ds.X)):
        Xtr, ytr = ds.X.values[tr], ds.y_reg.values[tr]
        Xte, yte = ds.X.values[te], ds.y_reg.values[te]
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(make_objective(Xtr, ytr, expert_names), n_trials=40, show_progress_bar=False)
        best = study.best_params
        m = ModeloReducido(expert_names=expert_names, **best).fit(Xtr, ytr)
        p = m.predict(Xte)
        f_rmse = rmse(yte, p)
        all_y.append(yte); all_p.append(p)
        rows_fold.append({"pliegue": f, "rmse": f_rmse, **best})
    all_y = np.concatenate(all_y); all_p = np.concatenate(all_p)
    fold_rmses = [r["rmse"] for r in rows_fold]
    print(f"[{tag}] expertos={expert_names}")
    print(f"  RMSE medio de pliegues: {np.mean(fold_rmses):.4f} +/- {np.std(fold_rmses):.4f}   "
          f"RMSE agrupado: {rmse(all_y, all_p):.4f}   tiempo: {time.time()-t0:.1f}s")
    pd.DataFrame(rows_fold).to_csv(f"artifacts/confirmatorio/iter4/rmse_{tag}.csv", index=False)
    return np.mean(fold_rmses), rmse(all_y, all_p)

os.makedirs("artifacts/confirmatorio/iter4", exist_ok=True)
run(("estacional", "nucleo_cambio", "persistencia"), "K3_sin_ridge")
run(("estacional", "nucleo_cambio"), "K2_estacional_nucleo")
run(("estacional", "persistencia"), "K2_estacional_persistencia")
