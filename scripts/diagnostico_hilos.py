"""Diagnóstico de la paralelización con hilos (Sección 4.3).

En ``benchmark_parallel.py`` el backend ``threading`` no reproduce los resultados
del código serial. Este script averigua qué combinaciones cambian: corre 16
combinaciones baratas (KNN sin balanceo y con SMOTE, Ridge y Lasso, con los
cuatro optimizadores) en serie, con hilos y con procesos (``loky``) y compara
cada resultado con el serial.

Salida: experiments/diagnostico_hilos.txt

Uso::

    python scripts/diagnostico_hilos.py
"""
import os
import sys
import warnings
from pathlib import Path

# Un hilo de BLAS/OpenMP, igual que en los demás benchmarks, para que solo se paralelice lo que se pide con joblib
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

import numpy as np
from joblib import Parallel, delayed

from benchmark_parallel import combos, run_combo
from src import config
from src.data import assemble_dataset


def main():
    ds = assemble_dataset()
    jobs = [c for c in combos() if c[2] in ("none", "smote") or c[0] == "reg"]
    serial = [run_combo(c, ds.X, ds.y_clf, ds.y_reg) for c in jobs]
    lines = [f"Se comparan {len(jobs)} combinaciones frente a su resultado serial."]
    for backend, workers in [("threading", 4), ("loky", 4)]:
        parallel = Parallel(n_jobs=workers, backend=backend)(delayed(run_combo)(c, ds.X, ds.y_clf, ds.y_reg) for c in jobs)
        changed = [(c, abs(a - b)) for c, a, b in zip(jobs, serial, parallel) if not np.isclose(a, b)]
        lines.append(f"{backend} con {workers} trabajadores: {len(changed)} de {len(jobs)} combinaciones cambian")
        lines += [f"    {c}: diferencia absoluta {d:.5f}" for c, d in changed]
    config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.EXPERIMENTS_DIR / "diagnostico_hilos.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
