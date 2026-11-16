"""Iteracion 7.2 (confirmatoria, protocolo pre-registrado en
experiments/protocolo.md seccion 13): ALEE con un piso minimo de peso para el
experto estacional dentro del simplex (``w_estacional >= floor``), motivado
por que el estacional ha sido, en las seis iteraciones anteriores,
consistentemente el componente individual mas fuerte. Hipotesis, variante y
rango de busqueda de ``floor`` fueron declarados en la seccion 13 del
protocolo ANTES de correr esto; no se cambia nada aqui despues de ver
resultados.

Mismo protocolo de validacion anidada honesta que iter4_reducir_expertos.py:
TimeSeriesSplit(5, gap=1) externo, TimeSeriesSplit(3, gap=1) interno,
optimizacion bayesiana (Optuna, TPESampler semilla 42, 40 evaluaciones por
pliegue externo), hiperparametros elegidos solo con los pliegues internos de
Xtr. Mismos 3 expertos de la iteracion 4. ``bandwidth``, ``half_life`` y
``lam`` en los mismos rangos de siempre; ``floor`` es el unico hiperparametro
nuevo, log-uniforme en [0.01, 0.5].
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

OUT_DIR = "artifacts/confirmatorio/iter7"

IDX_ESTACIONAL = list(EXPERTS).index("estacional")
IDX_OTROS = [i for i in range(len(EXPERTS)) if i != IDX_ESTACIONAL]


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


class ModeloPisoEstacional(ModeloOriginal):
    """ALEE con un piso minimo ``floor`` para el peso del experto estacional.

    Se resuelve el simplex normalmente con ``solve_weights`` (via
    ``ModeloOriginal.combination_weights``, sin tocar ``src/original_model.py``)
    y, si el peso resultante del estacional queda por debajo de ``floor``, se
    fija en ``floor`` y se renormaliza proporcionalmente la masa restante
    (``1 - floor``) entre los otros dos expertos, manteniendo sus proporciones
    relativas entre si.
    """

    def __init__(self, floor=0.1, **kw):
        self.floor = floor
        super().__init__(**kw)

    def combination_weights(self, X):
        W = super().combination_weights(X)
        floor = self.floor
        for i in range(len(W)):
            w = W[i]
            if w[IDX_ESTACIONAL] < floor:
                remaining = 1.0 - floor
                otros_sum = w[IDX_OTROS].sum()
                if otros_sum > 1e-12:
                    w[IDX_OTROS] = w[IDX_OTROS] * (remaining / otros_sum)
                else:
                    w[IDX_OTROS] = remaining / len(IDX_OTROS)
                w[IDX_ESTACIONAL] = floor
        return W


def make_objective(Xtr, ytr):
    inner = list(TimeSeriesSplit(config.N_SPLITS_INNER, gap=config.CV_GAP).split(Xtr))

    def objective(trial):
        params = dict(
            bandwidth=trial.suggest_float("bandwidth", 0.05, 3.0, log=True),
            half_life=trial.suggest_float("half_life", 6.0, 120.0, log=True),
            lam=trial.suggest_float("lam", 1e-3, 100.0, log=True),
            floor=trial.suggest_float("floor", 0.01, 0.5, log=True),
        )
        errs = []
        for itr, ite in inner:
            m = ModeloPisoEstacional(**params).fit(Xtr[itr], ytr[itr])
            p = m.predict(Xtr[ite])
            errs.append(rmse(ytr[ite], p))
        return float(np.mean(errs))

    return objective


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows_fold = []
    all_idx, all_y, all_p = [], [], []
    t0 = time.time()
    outer_splits = list(TimeSeriesSplit(config.N_SPLITS_OUTER, gap=config.CV_GAP).split(ds.X))
    for f, (tr, te) in enumerate(outer_splits):
        Xtr, ytr = ds.X.values[tr], ds.y_reg.values[tr]
        Xte, yte = ds.X.values[te], ds.y_reg.values[te]
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(make_objective(Xtr, ytr), n_trials=40, show_progress_bar=False)
        best = study.best_params
        m = ModeloPisoEstacional(**best).fit(Xtr, ytr)
        p = m.predict(Xte)
        f_rmse = rmse(yte, p)
        all_idx.append(te)
        all_y.append(yte)
        all_p.append(p)
        rows_fold.append({"pliegue": f, "rmse": f_rmse, **best})
        print(f"  [iter7.2] pliegue {f}: rmse={f_rmse:.4f} params={best}", flush=True)
    all_idx = np.concatenate(all_idx)
    all_y = np.concatenate(all_y)
    all_p = np.concatenate(all_p)
    fold_rmses = [r["rmse"] for r in rows_fold]
    print(
        f"[iter7_2_piso_estacional] RMSE medio de pliegues: {np.mean(fold_rmses):.4f} +/- {np.std(fold_rmses):.4f}   "
        f"RMSE agrupado: {rmse(all_y, all_p):.4f}   tiempo: {time.time() - t0:.1f}s"
    )
    pd.DataFrame(rows_fold).to_csv(f"{OUT_DIR}/rmse_iter7_2_piso_estacional.csv", index=False)
    pd.DataFrame({"idx": all_idx, "y_true": all_y, "y_pred": all_p}).to_csv(
        f"{OUT_DIR}/oof_iter7_2_piso_estacional.csv", index=False
    )
    return np.mean(fold_rmses), np.std(fold_rmses), rmse(all_y, all_p)


if __name__ == "__main__":
    run()
