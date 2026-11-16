"""Perfilamiento de CPU y memoria del pipeline (Sección 4.4).

Perfila con ``cProfile`` una combinación representativa (Random Forest con
SMOTE y búsqueda aleatoria, la familia que más tiempo consume) para localizar
los cuellos de botella reales antes de optimizar, y mide la memoria máxima de
una corrida completa de cuatro familias de modelos.

Salidas (en experiments/): profile_top.csv y profile_memory.csv.

Uso::

    python scripts/profile_pipeline.py
"""
import cProfile
import os
import pstats
import sys
import threading
import time
import warnings
from pathlib import Path

# Los tiempos de este script deben medir el algoritmo y no la contención entre hilos de BLAS/OpenMP (con 12 hilos de
# OpenBLAS compitiendo con otros procesos, un Lasso pasó de 0.01 s a 1 s): se fijan a un hilo antes de importar NumPy.
# El único paralelismo que se mide es el que se pide de forma explícita (n_jobs, joblib).
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from src import config
from src.data import assemble_dataset
from src.models import CLASSIFIER_BY_NAME, REGRESSOR_BY_NAME
from src.nested_cv import run_nested_cv

if os.environ.get("EXP_DIR"):  # permite redirigir la salida (por ejemplo, para pruebas)
    config.set_experiments_dir(os.environ["EXP_DIR"])

QUICK = os.environ.get("BENCH_QUICK") == "1"
BUDGETS = config.SMOKE_BUDGETS if QUICK else config.FULL_BUDGETS


def peak_memory_mb(fn):
    """Máximo de memoria residente adicional (MB) mientras corre ``fn``."""
    process = psutil.Process()
    baseline, peak, stop = process.memory_info().rss, [0], threading.Event()
    peak[0] = baseline

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], process.memory_info().rss)
            time.sleep(0.005)

    thread = threading.Thread(target=sample)
    thread.start()
    try:
        result = fn()
    finally:
        stop.set()
        thread.join()
    return (peak[0] - baseline) / 2 ** 20, result


def main():
    ds = assemble_dataset()
    spec = CLASSIFIER_BY_NAME["RandomForest"]

    profiler = cProfile.Profile()
    profiler.enable()
    run_nested_cv(spec, ds.X, ds.y_clf, "clf", "smote", "random", budgets=BUDGETS)
    profiler.disable()

    stats = pstats.Stats(profiler)
    total = stats.total_tt
    rows = []
    for (filename, line, name), (cc, nc, tt, ct, _) in stats.stats.items():
        rows.append({
            "funcion": f"{Path(filename).name}:{name}", "llamadas": nc, "tiempo_propio_s": tt,
            "tiempo_acumulado_s": ct, "pct_propio": 100 * tt / total, "pct_acumulado": 100 * ct / total,
        })
    table = pd.DataFrame(rows)
    top = pd.concat([
        table.sort_values("tiempo_propio_s", ascending=False).head(12).assign(orden="tiempo propio"),
        table.sort_values("tiempo_acumulado_s", ascending=False).head(12).assign(orden="tiempo acumulado"),
    ])
    top["tiempo_total_perfil_s"] = total
    config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    top.to_csv(config.EXPERIMENTS_DIR / "profile_top.csv", index=False)

    memory_rows = []
    for task, name, balancing in [("clf", "KNN", "smote"), ("clf", "RandomForest", "smote"),
                                  ("clf", "XGBoost", "smote"), ("reg", "Ridge", "none")]:
        registry = CLASSIFIER_BY_NAME if task == "clf" else REGRESSOR_BY_NAME
        y = ds.y_clf if task == "clf" else ds.y_reg
        started = time.perf_counter()
        memory, _ = peak_memory_mb(lambda: run_nested_cv(registry[name], ds.X, y, task, balancing, "random", budgets=BUDGETS))
        memory_rows.append({"modelo": name, "tarea": task, "memoria_pico_mb": memory, "tiempo_s": time.perf_counter() - started})
        print(f"{name}: {memory:.0f} MB", flush=True)
    pd.DataFrame(memory_rows).to_csv(config.EXPERIMENTS_DIR / "profile_memory.csv", index=False)


if __name__ == "__main__":
    main()
