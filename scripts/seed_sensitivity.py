"""Sensibilidad a la semilla (Sección 5.4).

Parte A: aleatoriedad del modelo. Para el mejor modelo de cada familia, se
repite el ajuste final de cada pliegue externo con 10 semillas distintas
(la del modelo y la de SMOTE/ADASYN), conservando los hiperparámetros
elegidos en la corrida real, y se mide cuánto varía la métrica agrupada de
los cinco pliegues. Los modelos deterministas (KNN, Naive Bayes, Ridge, Lasso)
sirven de control: su variación debe ser cero.

Parte B: aleatoriedad del optimizador. Con cinco semillas distintas se repite
la búsqueda completa (validación anidada) de modelos baratos, y se mide la
variación de la métrica externa y de los hiperparámetros elegidos para Random
Search, la optimización bayesiana y el algoritmo genético. Grid Search es
determinista.

Salidas (en experiments/): seed_sensitivity_models.csv y
seed_sensitivity_optimizers.csv.

Uso::

    python scripts/seed_sensitivity.py        # partes A y B
    python scripts/seed_sensitivity.py A      # solo la parte A (la B es la que más tarda)
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from src import analysis as A
from src import config
from src.balancing import apply_balancing
from src.data import assemble_dataset
from src.models import CLASSIFIER_BY_NAME, REGRESSOR_BY_NAME
from src.nested_cv import _fit_final, run_nested_cv

if os.environ.get("EXP_DIR"):  # permite redirigir la salida (por ejemplo, para pruebas)
    config.set_experiments_dir(os.environ["EXP_DIR"])

QUICK = os.environ.get("BENCH_QUICK") == "1"
SEEDS_MODEL = [42, 0, 1] if QUICK else [42, 0, 1, 2, 3, 4, 5, 6, 7, 8]
SEEDS_OPT = [1, 2] if QUICK else [1, 2, 3, 4, 5]
BUDGETS = config.SMOKE_BUDGETS if QUICK else config.FULL_BUDGETS


def refit_pooled_metric(task, model, balancing, optimizer, seed, ds):
    """Métrica agrupada de los cinco pliegues con otra semilla para el modelo y el balanceo."""
    registry = CLASSIFIER_BY_NAME if task == "clf" else REGRESSOR_BY_NAME
    spec = registry[model]
    history = A.load_history(f"{task}_{model}_{balancing}_{optimizer}")
    y = ds.y_clf if task == "clf" else ds.y_reg
    has_seed = "random_state" in spec.estimator().get_params()
    truth, pred = [], []
    for fold, (train, test) in enumerate(TimeSeriesSplit(n_splits=config.N_SPLITS_OUTER, gap=config.CV_GAP).split(ds.X)):
        params = dict(history["fold_results"][fold]["best_params"])
        if has_seed:
            params["random_state"] = seed
        X_fit, y_fit, sw = ds.X.iloc[train], y.iloc[train], None
        if task == "clf" and balancing != "none":
            X_fit, y_fit, sw = apply_balancing(balancing, X_fit, y_fit, seed=seed)
        est, X_test_t = _fit_final(spec, params, X_fit, y_fit, sw, ds.X.iloc[test])
        truth.append(y.iloc[test].values)
        pred.append(est.predict(X_test_t))
    truth, pred = np.concatenate(truth), np.concatenate(pred)
    if task == "clf":
        return float(f1_score(truth, pred, average="macro", zero_division=0))
    return float(np.sqrt(np.mean((truth - pred) ** 2)))


def part_a(ds, master):
    rows = []
    for task in ["clf", "reg"]:
        best = A.best_per_model(master, task)
        for model, row in best.iterrows():
            values = [refit_pooled_metric(task, model, row["balancing"], row["optimizer"], s, ds) for s in SEEDS_MODEL]
            rows.append({
                "tarea": task, "modelo": model, "balanceo": row["balancing"], "optimizador": row["optimizer"],
                "metrica": "F1-macro" if task == "clf" else "RMSE", "n_semillas": len(values),
                "media": np.mean(values), "desviacion": np.std(values, ddof=1), "minimo": np.min(values),
                "maximo": np.max(values), "valor_semilla_42": values[0],
            })
            print(f"A {task} {model}: {np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}", flush=True)
    return pd.DataFrame(rows)


def part_b(ds):
    cases = [("clf", "KNN", "smote"), ("clf", "DecisionTree", "none"), ("reg", "Ridge", "none"), ("reg", "KNN", "none")]
    rows = []
    for task, model, balancing in cases:
        registry = CLASSIFIER_BY_NAME if task == "clf" else REGRESSOR_BY_NAME
        y = ds.y_clf if task == "clf" else ds.y_reg
        key = "f1_macro" if task == "clf" else "rmse"
        for optimizer in ["grid", "random", "bayesian", "genetic"]:
            seeds = [config.SEED] if optimizer == "grid" else SEEDS_OPT
            for s in seeds:
                result = run_nested_cv(registry[model], ds.X, y, task, balancing, optimizer, seed=s, budgets=BUDGETS)
                metric = np.mean([fr.metrics[key] for fr in result.fold_results if fr.metrics])
                last = result.fold_results[-1].best_params
                rows.append({
                    "tarea": task, "modelo": model, "balanceo": balancing, "optimizador": optimizer, "semilla": s,
                    "metrica": key, "valor": metric, "mejores_hiperparametros_ultimo_pliegue": str(last),
                })
            print(f"B {task} {model} {optimizer}: listo", flush=True)
    return pd.DataFrame(rows)


def main(parts="AB"):
    ds = assemble_dataset()
    master = A.load_master()
    config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    if "A" in parts:
        part_a(ds, master).to_csv(config.EXPERIMENTS_DIR / "seed_sensitivity_models.csv", index=False)
    if "B" in parts:
        part_b(ds).to_csv(config.EXPERIMENTS_DIR / "seed_sensitivity_optimizers.csv", index=False)


if __name__ == "__main__":
    main(parts=(sys.argv[1].upper() if len(sys.argv) > 1 else "AB"))
