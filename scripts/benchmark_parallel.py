"""Paralelización (Sección 4.3): aceleración al repartir el trabajo con joblib.

Parte A: entre combinaciones. Se corren 24 combinaciones baratas (KNN, Ridge y
Lasso, con sus cuatro métodos de optimización) en serie y repartidas entre 2,
4, 8 y 12 procesos con el backend ``loky``, y entre 4 y 8 hilos con el backend
``threading``.

Parte B: entre pliegues. Una combinación pesada (Random Forest con búsqueda
aleatoria) se corre en serie y con sus cinco pliegues externos repartidos en
cinco procesos (``n_jobs`` de ``run_nested_cv``).

En ambos casos se comprueba que el resultado paralelo es idéntico al serial:
la paralelización no puede cambiar las métricas.

Salida: experiments/parallel_speedup.csv

Uso::

    python scripts/benchmark_parallel.py
"""
import os
import sys
import time
import warnings
from pathlib import Path

# Los tiempos de este script deben medir el algoritmo y no la contención entre hilos de BLAS/OpenMP (con 12 hilos de
# OpenBLAS compitiendo con otros procesos, un Lasso pasó de 0.01 s a 1 s): se fijan a un hilo antes de importar NumPy.
# El único paralelismo que se mide es el que se pide de forma explícita (n_jobs, joblib).
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

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


def run_combo(spec, X, y_clf, y_reg):
    """Corre una combinación completa y devuelve su métrica principal."""
    task, model, balancing, optimizer = spec
    registry = CLASSIFIER_BY_NAME if task == "clf" else REGRESSOR_BY_NAME
    result = run_nested_cv(
        registry[model], X, y_clf if task == "clf" else y_reg, task, balancing, optimizer, budgets=BUDGETS
    )
    key = "f1_macro" if task == "clf" else "rmse"
    return float(np.mean([fr.metrics[key] for fr in result.fold_results if fr.metrics]))


def combos():
    out = [("clf", "KNN", b, o) for b in ["none", "smote", "adasyn", "class_weight"]
           for o in ["grid", "random", "bayesian", "genetic"]]
    out += [("reg", m, "none", o) for m in ["Ridge", "Lasso"] for o in ["grid", "random", "bayesian", "genetic"]]
    return out


def main():
    ds = assemble_dataset()
    rows = []

    jobs = combos()
    t0 = time.perf_counter()
    serial = [run_combo(c, ds.X, ds.y_clf, ds.y_reg) for c in jobs]
    serial_s = time.perf_counter() - t0
    rows.append({"parte": "A: entre combinaciones", "backend": "serie", "trabajadores": 1, "tiempo_s": serial_s,
                 "aceleracion": 1.0, "eficiencia": 1.0, "resultado_identico": True})
    print(f"A serie: {serial_s:.1f} s", flush=True)

    settings = [("loky", k) for k in ([2] if QUICK else [2, 4, 8, 12])] + [("threading", k) for k in ([4] if QUICK else [4, 8])]
    for backend, k in settings:
        t0 = time.perf_counter()
        parallel = Parallel(n_jobs=k, backend=backend)(delayed(run_combo)(c, ds.X, ds.y_clf, ds.y_reg) for c in jobs)
        elapsed = time.perf_counter() - t0
        rows.append({"parte": "A: entre combinaciones", "backend": backend, "trabajadores": k, "tiempo_s": elapsed,
                     "aceleracion": serial_s / elapsed, "eficiencia": serial_s / elapsed / k,
                     "resultado_identico": bool(np.allclose(serial, parallel))})
        print(f"A {backend} x{k}: {elapsed:.1f} s (x{serial_s / elapsed:.2f})", flush=True)

    spec = CLASSIFIER_BY_NAME["RandomForest"]
    t0 = time.perf_counter()
    base = run_nested_cv(spec, ds.X, ds.y_clf, "clf", "none", "random", budgets=BUDGETS, n_jobs=1)
    base_s = time.perf_counter() - t0
    rows.append({"parte": "B: entre pliegues (Random Forest)", "backend": "serie", "trabajadores": 1, "tiempo_s": base_s,
                 "aceleracion": 1.0, "eficiencia": 1.0, "resultado_identico": True})
    print(f"B serie: {base_s:.1f} s", flush=True)
    t0 = time.perf_counter()
    par = run_nested_cv(spec, ds.X, ds.y_clf, "clf", "none", "random", budgets=BUDGETS, n_jobs=5, backend="loky")
    par_s = time.perf_counter() - t0
    same = all(
        np.isclose(a.metrics["f1_macro"], b.metrics["f1_macro"]) for a, b in zip(base.fold_results, par.fold_results)
    )
    rows.append({"parte": "B: entre pliegues (Random Forest)", "backend": "loky", "trabajadores": 5, "tiempo_s": par_s,
                 "aceleracion": base_s / par_s, "eficiencia": base_s / par_s / 5, "resultado_identico": bool(same)})
    print(f"B loky x5: {par_s:.1f} s (x{base_s / par_s:.2f})", flush=True)

    config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(config.EXPERIMENTS_DIR / "parallel_speedup.csv", index=False)


if __name__ == "__main__":
    main()
