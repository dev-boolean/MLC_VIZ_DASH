"""Optimización multi-fidelidad (Sección 3.3): Successive Halving.

Compara, para Random Forest y XGBoost en clasificación y regresión, tres
formas de buscar hiperparámetros:

* Successive Halving: 27 configuraciones evaluadas con 11 árboles, las 9
  mejores con 33, las 3 mejores con 100 y la mejor con 300 (factor de
  reducción 3, 40 evaluaciones).
* Random Search con el mismo costo en árboles que Successive Halving.
* Random Search estándar de 40 evaluaciones (número de árboles libre entre 50
  y 300).

Cada método se repite con tres semillas y se evalúa con la validación anidada
completa.

Salida: experiments/multifidelity.csv

Uso::

    python scripts/multifidelity_experiment.py                 # los cuatro casos
    python scripts/multifidelity_experiment.py clf:XGBoost     # solo ese caso; se fusiona con el CSV existente
"""
import functools
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from src import config
from src.data import assemble_dataset
from src.models import CLASSIFIER_BY_NAME, REGRESSOR_BY_NAME
from src.multifidelity import run_random_cost_matched, run_successive_halving
from src.nested_cv import run_nested_cv

if os.environ.get("EXP_DIR"):  # permite redirigir la salida (por ejemplo, para pruebas)
    config.set_experiments_dir(os.environ["EXP_DIR"])

QUICK = os.environ.get("BENCH_QUICK") == "1"
SEEDS = [1] if QUICK else [1, 2, 3]
BUDGETS = config.SMOKE_BUDGETS if QUICK else config.FULL_BUDGETS


def summarize(result, key):
    values = [fr.metrics[key] for fr in result.fold_results if fr.metrics]
    return {
        "metrica_externa": float(np.mean(values)),
        "tiempo_busqueda_s": float(sum(fr.search_wall_time_s for fr in result.fold_results)),
        "evaluaciones": int(sum(fr.n_evaluations for fr in result.fold_results)),
    }


CASES = [("clf", "RandomForest"), ("clf", "XGBoost"), ("reg", "RandomForest"), ("reg", "XGBoost")]


def main(only=None):
    """Corre los casos pedidos (todos si ``only`` es ``None``) y actualiza ``multifidelity.csv``."""
    ds = assemble_dataset()
    cases = [c for c in CASES if only is None or f"{c[0]}:{c[1]}" in only]
    rows = []
    for task, model in cases:
        registry = CLASSIFIER_BY_NAME if task == "clf" else REGRESSOR_BY_NAME
        spec = registry[model]
        y = ds.y_clf if task == "clf" else ds.y_reg
        key = "f1_macro" if task == "clf" else "rmse"
        for seed in SEEDS:
            halving = run_nested_cv(spec, ds.X, y, task, "none", "halving", seed=seed,
                                    optimizer_fn=run_successive_halving, budgets=BUDGETS)
            cost = int(halving.extras[0]["cost_tree_units"])
            matched = run_nested_cv(spec, ds.X, y, task, "none", "random_igual_costo", seed=seed,
                                    optimizer_fn=functools.partial(run_random_cost_matched, cost_tree_units=cost),
                                    budgets=BUDGETS)
            standard = run_nested_cv(spec, ds.X, y, task, "none", "random", seed=seed, budgets=BUDGETS)
            for label, result, tree_cost in [
                ("Successive Halving", halving, cost),
                ("Random (mismo costo)", matched, int(matched.extras[0]["cost_tree_units"])),
                ("Random (40 evaluaciones)", standard, np.nan),
            ]:
                rows.append({"tarea": task, "modelo": model, "semilla": seed, "metodo": label, "metrica": key,
                             "costo_arboles_por_pliegue": tree_cost, **summarize(result, key)})
            print(f"{task} {model} semilla {seed}: listo", flush=True)
    config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.EXPERIMENTS_DIR / "multifidelity.csv"
    table = pd.DataFrame(rows)
    if only is not None and path.exists():  # se reemplazan solo las filas de los casos repetidos
        previous = pd.read_csv(path)
        keep = ~(previous["tarea"] + ":" + previous["modelo"]).isin(only)
        table = pd.concat([previous[keep], table], ignore_index=True)
        order = {f"{t}:{m}": i for i, (t, m) in enumerate(CASES)}
        table = table.assign(_orden=(table["tarea"] + ":" + table["modelo"]).map(order))
        table = table.sort_values(["_orden", "semilla"], kind="stable").drop(columns="_orden")
    table.to_csv(path, index=False)


if __name__ == "__main__":
    main(only=sys.argv[1:] or None)
