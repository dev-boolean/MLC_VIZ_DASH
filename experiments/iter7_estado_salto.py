"""Iteracion 7.1 (confirmatoria, protocolo pre-registrado en
experiments/protocolo.md seccion 13): ALEE con una tercera variable de estado
de localizacion, |y_t - y_{t-1}|, para distinguir un salto puntual reciente de
una volatilidad sostenida (que ya captura rollstd3). Hipotesis, variante y
rangos de busqueda fueron declarados en la seccion 13 del protocolo y en
docs/registro_desarrollo.md ANTES de correr esto; no se cambia nada aqui
despues de ver resultados.

Mismo protocolo de validacion anidada honesta que iter4_reducir_expertos.py:
TimeSeriesSplit(5, gap=1) externo, TimeSeriesSplit(3, gap=1) interno,
optimizacion bayesiana (Optuna, TPESampler semilla 42, 40 evaluaciones por
pliegue externo), hiperparametros elegidos solo con los pliegues internos de
Xtr (nunca se mira Xte al elegir hiperparametros). Mismos 3 expertos de la
iteracion 4 (persistencia, nucleo_cambio, estacional) -- son ya los unicos que
implementa ``_Experts`` en ``src/original_model.py``. Mismos rangos de
busqueda de bandwidth/half_life/lam que el resto del proyecto. Sin
hiperparametros nuevos.
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

OUT_DIR = "artifacts/confirmatorio/iter7"


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


class ModeloEstadoSalto(ModeloOriginal):
    """ALEE con el estado de localizacion ampliado con |lag1 - lag2|.

    ``ModeloOriginal._state`` ya devuelve ``rollstd3`` y la brecha
    ``lag1 - rollmean12``. Aqui se agrega una tercera coordenada: la magnitud
    del ultimo cambio mensual observado ANTES del mes que se pronostica
    (``|y_{t-1} - y_{t-2}|`` = ``|lag1 - lag2|`` en la fila que se usa para
    predecir ``y_t``), sin usar ninguna informacion del mes ``t``. ``lag2``
    ya existe como columna de ``X`` (``LAGS = [1, 2, 3, 6, 12]`` en
    ``src/data.py``), asi que no hace falta anadir una columna nueva a ``X``:
    basta con calcular la diferencia dentro de ``_state``.
    """

    def _state(self, X):
        base = super()._state(X)
        salto = np.abs(
            X[:, self._col("td_nacional_lag1")] - X[:, self._col("td_nacional_lag2")]
        )
        return np.column_stack([base, salto])


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
            m = ModeloEstadoSalto(**params).fit(Xtr[itr], ytr[itr])
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
        m = ModeloEstadoSalto(**best).fit(Xtr, ytr)
        p = m.predict(Xte)
        f_rmse = rmse(yte, p)
        all_idx.append(te)
        all_y.append(yte)
        all_p.append(p)
        rows_fold.append({"pliegue": f, "rmse": f_rmse, **best})
        print(f"  [iter7.1] pliegue {f}: rmse={f_rmse:.4f} params={best}", flush=True)
    all_idx = np.concatenate(all_idx)
    all_y = np.concatenate(all_y)
    all_p = np.concatenate(all_p)
    fold_rmses = [r["rmse"] for r in rows_fold]
    print(
        f"[iter7_1_estado_salto] RMSE medio de pliegues: {np.mean(fold_rmses):.4f} +/- {np.std(fold_rmses):.4f}   "
        f"RMSE agrupado: {rmse(all_y, all_p):.4f}   tiempo: {time.time() - t0:.1f}s"
    )
    pd.DataFrame(rows_fold).to_csv(f"{OUT_DIR}/rmse_iter7_1_estado_salto.csv", index=False)
    pd.DataFrame({"idx": all_idx, "y_true": all_y, "y_pred": all_p}).to_csv(
        f"{OUT_DIR}/oof_iter7_1_estado_salto.csv", index=False
    )
    return np.mean(fold_rmses), np.std(fold_rmses), rmse(all_y, all_p)


if __name__ == "__main__":
    run()
