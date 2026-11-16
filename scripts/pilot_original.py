"""Piloto de desarrollo del modelo original (no es un experimento confirmatorio).

Corre la validación anidada del segundo entregable (mismas particiones
externas, misma validación interna, misma semilla y el mismo presupuesto de 40
evaluaciones) con el optimizador bayesiano, y compara contra las mejores
combinaciones y los modelos de referencia. Cada ejecución se registra como una
iteración de diseño en ``docs/registro_desarrollo.md``: la guía permite
iterar el algoritmo antes de congelar el protocolo, pero exige informar
cuántas iteraciones se evaluaron. Los resultados quedan en
``experiments/desarrollo/`` y no deben usarse para sustentar R3.

Uso::

    python scripts/pilot_original.py <número de iteración>
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import analysis as A  # noqa: E402
from src import config  # noqa: E402
from src.data import assemble_dataset  # noqa: E402
from src.models import ORIGINAL_MODEL  # noqa: E402
from src.nested_cv import run_nested_cv  # noqa: E402
from src.original_model import EXPERTS  # noqa: E402

OUT = config.EXPERIMENTS_DIR / "desarrollo"


def main(iteration: int):
    warnings.filterwarnings("ignore")
    config.RUN_MODE = "full"
    config.set_global_seed(config.SEED)
    ds = assemble_dataset()
    res = run_nested_cv(ORIGINAL_MODEL, ds.X, ds.y_reg, "reg", "none", "bayesian", seed=config.SEED)
    if res.error:
        raise SystemExit(res.error)

    idx = np.concatenate([res.oof[k]["test_idx"] for k in sorted(res.oof)])
    pred = np.concatenate([res.oof[k]["y_pred"] for k in sorted(res.oof)])
    y = ds.y_reg.values[idx]
    folds = np.concatenate([np.full(len(res.oof[k]["test_idx"]), k) for k in sorted(res.oof)])
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"idx": idx, "fold": folds, "y_true": y, "y_pred": pred}).to_csv(OUT / f"iter{iteration}_oof.csv", index=False)

    rows = []

    def add(name, y_hat):
        e = y - y_hat
        per_fold = [np.sqrt(np.mean(e[folds == k] ** 2)) for k in np.unique(folds)]
        rows.append({"modelo": name, "rmse_medio_pliegues": np.mean(per_fold), "rmse_std": np.std(per_fold),
                     "rmse_agrupado": np.sqrt(np.mean(e ** 2)), "mae_agrupado": np.mean(np.abs(e))})

    add("ALEE (original)", pred)
    master = A.load_master()
    for model, row in A.best_per_model(master, "reg").iterrows():
        p = A.pooled_oof(row["run_id"])
        order = np.argsort(p["idx"])
        assert np.array_equal(p["idx"][order], idx)
        add(f"{model} ({row['optimizer']})", p["y_pred"][order])
    base = A.regression_baselines(ds).set_index("idx").loc[idx]
    for col in ["Persistencia", "Estacional", "SARIMA"]:
        add(col, base[col].values)
    table = pd.DataFrame(rows).sort_values("rmse_medio_pliegues").reset_index(drop=True)
    table.to_csv(OUT / f"iter{iteration}_resumen.csv", index=False)

    params = pd.DataFrame([fr.best_params for fr in res.fold_results])
    params.to_csv(OUT / f"iter{iteration}_hiperparametros.csv", index=False)
    print(table.round(3).to_string())
    print("\nHiperparámetros por pliegue:\n", params.round(3).to_string())
    print(f"\nTiempo total: {res.total_wall_time_s:.1f} s; expertos: {', '.join(EXPERTS)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
