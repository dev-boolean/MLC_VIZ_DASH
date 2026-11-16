"""Iteración 4 (exploratoria, post-hoc a la validación confirmatoria 2026-11-05).

Hipótesis a probar: el espacio de búsqueda de `lam` en las iteraciones previas
pudo haber sido demasiado angosto, llevando a una regularización insuficiente
hacia el previo global pi (que ya favorece fuertemente al experto estacional).
Se reoptimiza con un rango de lam mucho más amplio, con el mismo presupuesto
(40 evaluaciones bayesianas, validacion anidada 5 externos / 3 internos,
TimeSeriesSplit gap=1) que exige el protocolo congelado.
"""
import sys, os, time
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
import optuna
from sklearn.model_selection import TimeSeriesSplit
from src.data import assemble_dataset
from src import config
from src.original_model import ModeloOriginal

optuna.logging.set_verbosity(optuna.logging.WARNING)
ds = assemble_dataset()

def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))

def make_objective(Xtr, ytr):
    inner = list(TimeSeriesSplit(config.N_SPLITS_INNER, gap=config.CV_GAP).split(Xtr))
    def objective(trial):
        params = dict(
            bandwidth=trial.suggest_float("bandwidth", 0.05, 3.0, log=True),
            half_life=trial.suggest_float("half_life", 6.0, 120.0, log=True),
            lam=trial.suggest_float("lam", 1e-3, 100.0, log=True),  # <- rango ampliado (antes ~[1e-3, 1])
            ridge_alpha=trial.suggest_float("ridge_alpha", 0.1, 300.0, log=True),
        )
        errs = []
        for itr, ite in inner:
            m = ModeloOriginal(**params).fit(Xtr[itr], ytr[itr])
            p = m.predict(Xtr[ite])
            errs.append(rmse(ytr[ite], p))
        return float(np.mean(errs))
    return objective

rows_hp, rows_fold = [], []
t0 = time.time()
all_y, all_p = [], []
for f, (tr, te) in enumerate(TimeSeriesSplit(config.N_SPLITS_OUTER, gap=config.CV_GAP).split(ds.X)):
    Xtr, ytr = ds.X.values[tr], ds.y_reg.values[tr]
    Xte, yte = ds.X.values[te], ds.y_reg.values[te]
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(make_objective(Xtr, ytr), n_trials=40, show_progress_bar=False)
    best = study.best_params
    m = ModeloOriginal(**best).fit(Xtr, ytr)
    p = m.predict(Xte)
    f_rmse = rmse(yte, p)
    all_y.append(yte); all_p.append(p)
    rows_hp.append({**best, "pliegue": f})
    rows_fold.append({"pliegue": f, "rmse": f_rmse, "n_test": len(te)})
    print(f"pliegue {f}: lam*={best['lam']:.4f}  bandwidth*={best['bandwidth']:.3f}  "
          f"half_life*={best['half_life']:.2f}  ridge_alpha*={best['ridge_alpha']:.2f}  RMSE_fold={f_rmse:.4f}")

all_y = np.concatenate(all_y); all_p = np.concatenate(all_p)
fold_rmses = [r["rmse"] for r in rows_fold]
print()
print(f"RMSE medio de los pliegues (primario, protocolo): {np.mean(fold_rmses):.4f} +/- {np.std(fold_rmses):.4f}")
print(f"RMSE agrupado (secundario): {rmse(all_y, all_p):.4f}")
print(f"tiempo total: {time.time()-t0:.1f}s")

os.makedirs("artifacts/confirmatorio/iter4", exist_ok=True)
pd.DataFrame(rows_hp).to_csv("artifacts/confirmatorio/iter4/hiperparametros_lam_amplio.csv", index=False)
pd.DataFrame(rows_fold).to_csv("artifacts/confirmatorio/iter4/rmse_por_pliegue.csv", index=False)
pd.DataFrame({"idx": np.concatenate([te for _, te in TimeSeriesSplit(config.N_SPLITS_OUTER, gap=config.CV_GAP).split(ds.X)]),
              "y_true": all_y, "y_pred": all_p}).to_csv("artifacts/confirmatorio/iter4/oof_ALEE_lam_amplio.csv", index=False)
