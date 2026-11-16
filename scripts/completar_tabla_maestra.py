"""Agrega a ``results_master.csv`` la columna de hiperparámetros elegidos en cada pliegue.

Las corridas anteriores a esta columna dejaron los hiperparámetros solo en los
historiales (``experiments/histories/<run_id>.pkl``). Este script los lee y
completa la tabla maestra sin volver a entrenar nada. Las corridas nuevas de
``src.runner`` ya escriben la columna directamente.

Uso::

    python scripts/completar_tabla_maestra.py
"""
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.runner import params_to_json


def main():
    master = pd.read_csv(config.RESULTS_MASTER_PATH)
    columna = []
    for run_id in master["run_id"]:
        with open(config.HISTORIES_DIR / f"{run_id}.pkl", "rb") as f:
            historial = pickle.load(f)
        columna.append(params_to_json([fr["best_params"] for fr in historial["fold_results"]]))
    master["hiperparametros_por_pliegue"] = columna
    master.to_csv(config.RESULTS_MASTER_PATH, index=False)
    print(f"Columna agregada a {len(master)} combinaciones.")


if __name__ == "__main__":
    main()
