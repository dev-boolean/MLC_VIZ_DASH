"""Orquesta las 140 combinaciones del proyecto.

Clasificación: 7 modelos x 4 técnicas de balanceo x 4 optimizadores = 112.
Regresión: 7 modelos x 4 optimizadores (sin balanceo) = 28.

Cada combinación corre su validación anidada y deja tres registros en
``experiments/``:

* ``results_master.csv``: una fila por combinación (métricas promedio y
  desviación entre pliegues, tiempos, número de evaluaciones y los
  hiperparámetros elegidos en cada pliegue externo, en JSON).
* ``results_folds.csv``: una fila por combinación, pliegue y métrica.
* ``histories/<run_id>.pkl`` y ``oof/<run_id>.npz``: historial de cada
  búsqueda (puntaje y tiempo de cada evaluación, diversidad genética por
  generación) y predicciones fuera de muestra de los pliegues externos.

Uso::

    python -m src.runner full
    python -m src.runner smoke --only-models KNN,Ridge --out-dir /tmp/prueba
"""
import argparse
import json
import pickle
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .balancing import BALANCING_METHODS
from .data import assemble_dataset
from .models import CLASSIFIERS, REGRESSORS
from .nested_cv import run_nested_cv
from .search import OPTIMIZERS

OPTIMIZER_NAMES = list(OPTIMIZERS.keys())
CLF_METRIC_KEYS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "auc_ovr"]
REG_METRIC_KEYS = ["rmse", "mae"]


def build_combinations(models=None, optimizers=None, tasks=None):
    """Lista las combinaciones a correr, con filtros opcionales.

    Parameters
    ----------
    models : iterable of str, optional
        Nombres de modelos a incluir (todos si es ``None``).
    optimizers : iterable of str, optional
        Optimizadores a incluir.
    tasks : iterable of {"clf", "reg"}, optional
        Tareas a incluir.

    Returns
    -------
    list of dict
        Cada elemento tiene ``task``, ``model`` (ModelSpec), ``balancing`` y
        ``optimizer``.
    """
    models = set(models) if models else None
    optimizers = list(optimizers) if optimizers else OPTIMIZER_NAMES
    tasks = set(tasks) if tasks else {"clf", "reg"}
    combos = []
    if "clf" in tasks:
        for m in CLASSIFIERS:
            if models and m.name not in models:
                continue
            for b in BALANCING_METHODS:
                for o in optimizers:
                    combos.append({"task": "clf", "model": m, "balancing": b, "optimizer": o})
    if "reg" in tasks:
        for m in REGRESSORS:
            if models and m.name not in models:
                continue
            for o in optimizers:
                combos.append({"task": "reg", "model": m, "balancing": "none", "optimizer": o})
    return combos


def run_id_of(combo) -> str:
    """Identificador único de una combinación."""
    return f"{combo['task']}_{combo['model'].name}_{combo['balancing']}_{combo['optimizer']}"


def params_to_json(params_per_fold) -> str:
    """Serializa los hiperparámetros elegidos en cada pliegue externo (lista de diccionarios) como texto JSON."""
    return json.dumps([dict(p) if p else None for p in params_per_fold], default=str, ensure_ascii=False, sort_keys=True)


def _aggregate_metrics(fold_results, metric_keys):
    """Promedia cada métrica entre pliegues externos, ignorando None y NaN."""
    agg = {}
    for k in metric_keys:
        vals = [fr.metrics.get(k) for fr in fold_results if fr.metrics.get(k) is not None]
        vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        agg[f"{k}_mean"] = float(np.mean(vals)) if vals else None
        agg[f"{k}_std"] = float(np.std(vals)) if vals else None
        agg[f"{k}_n_folds"] = len(vals)
    return agg


def _save_run_artifacts(run_id, result, save_histories=True):
    """Guarda historial de búsqueda y predicciones fuera de muestra de una corrida."""
    config.HISTORIES_DIR.mkdir(parents=True, exist_ok=True)
    config.OOF_DIR.mkdir(parents=True, exist_ok=True)
    if save_histories:
        payload = {
            "fold_results": [
                {
                    "fold": fr.fold,
                    "metrics": fr.metrics,
                    "best_params": fr.best_params,
                    "n_evaluations": fr.n_evaluations,
                    "search_wall_time_s": fr.search_wall_time_s,
                    "n_train": fr.n_train,
                    "n_test": fr.n_test,
                    "fit_time_s": fr.fit_time_s,
                    "predict_time_s": fr.predict_time_s,
                }
                for fr in result.fold_results
            ],
            "histories": result.histories,
            "extras": result.extras,
        }
        with open(config.HISTORIES_DIR / f"{run_id}.pkl", "wb") as f:
            pickle.dump(payload, f)

    arrays = {}
    for fold, data in result.oof.items():
        arrays[f"test_idx_{fold}"] = data["test_idx"]
        arrays[f"y_true_{fold}"] = data["y_true"]
        arrays[f"y_pred_{fold}"] = data["y_pred"]
        if data["y_proba"] is not None:
            arrays[f"y_proba_{fold}"] = data["y_proba"]
    if arrays:
        np.savez_compressed(config.OOF_DIR / f"{run_id}.npz", **arrays)


def run_all(
    run_mode=None,
    save_histories=True,
    verbose=True,
    resume=True,
    models=None,
    optimizers=None,
    tasks=None,
):
    """Corre todas las combinaciones y devuelve la tabla maestra.

    Parameters
    ----------
    run_mode : {"smoke", "full"}, optional
        Presupuestos a usar (ver ``config``).
    save_histories : bool
        Si se guarda el historial de cada búsqueda.
    verbose : bool
        Si se imprime el avance.
    resume : bool
        Si ya existe una tabla maestra del mismo modo, se omiten las
        combinaciones que terminaron sin error y se continúa con las que
        faltan. Permite cortar y reanudar una corrida larga.
    models, optimizers, tasks : iterable, optional
        Filtros sobre las combinaciones (ver ``build_combinations``).

    Returns
    -------
    pandas.DataFrame
        Una fila por combinación.
    """
    warnings.filterwarnings("ignore")
    if run_mode is not None:
        config.RUN_MODE = run_mode
    config.set_global_seed(config.SEED)
    config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    ds = assemble_dataset()
    combos = build_combinations(models, optimizers, tasks)
    n_total = len(combos)
    rows, fold_rows, done_ids = [], [], set()

    if resume and config.RESULTS_MASTER_PATH.exists():
        prev = pd.read_csv(config.RESULTS_MASTER_PATH)
        prev_ok = prev[(prev["run_mode"] == config.RUN_MODE) & (prev["error"].isna())]
        wanted = {run_id_of(c) for c in combos}
        prev_ok = prev_ok[prev_ok["run_id"].isin(wanted)]
        if len(prev_ok):
            rows = prev_ok.to_dict("records")
            done_ids = set(prev_ok["run_id"])
            if config.RESULTS_FOLDS_PATH.exists():
                prev_folds = pd.read_csv(config.RESULTS_FOLDS_PATH)
                fold_rows = prev_folds[prev_folds["run_id"].isin(done_ids)].to_dict("records")
            if verbose:
                print(f"Reanudando: {len(done_ids)}/{n_total} combinaciones ya completas.", flush=True)

    t0 = time.perf_counter()
    processed = 0
    for combo in combos:
        run_id = run_id_of(combo)
        if run_id in done_ids:
            continue
        row = {
            "run_id": run_id,
            "task": combo["task"],
            "model": combo["model"].name,
            "balancing": combo["balancing"],
            "optimizer": combo["optimizer"],
            "run_mode": config.RUN_MODE,
            "seed": config.SEED,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            y = ds.y_clf if combo["task"] == "clf" else ds.y_reg
            result = run_nested_cv(
                model_spec=combo["model"], X=ds.X, y=y, task=combo["task"],
                balancing_method=combo["balancing"], optimizer_name=combo["optimizer"], seed=config.SEED,
            )
            metric_keys = CLF_METRIC_KEYS if combo["task"] == "clf" else REG_METRIC_KEYS
            ok = [fr for fr in result.fold_results if fr.metrics]
            row.update(_aggregate_metrics(result.fold_results, metric_keys))
            row["total_wall_time_s"] = result.total_wall_time_s
            row["search_time_s"] = sum(fr.search_wall_time_s for fr in result.fold_results)
            row["total_evaluations"] = sum(fr.n_evaluations for fr in result.fold_results)
            row["time_per_eval_s"] = (
                row["search_time_s"] / row["total_evaluations"] if row["total_evaluations"] else None
            )
            row["final_fit_time_s"] = float(np.mean([fr.fit_time_s for fr in ok])) if ok else None
            row["predict_time_s"] = float(np.mean([fr.predict_time_s for fr in ok])) if ok else None
            row["hiperparametros_por_pliegue"] = params_to_json([fr.best_params for fr in result.fold_results])
            row["n_folds_ok"] = len(ok)
            row["n_folds"] = len(result.fold_results)
            row["error"] = result.error

            for fr in result.fold_results:
                for metric_name, value in fr.metrics.items():
                    fold_rows.append(
                        {
                            "run_id": run_id, "task": combo["task"], "model": combo["model"].name,
                            "balancing": combo["balancing"], "optimizer": combo["optimizer"],
                            "fold": fr.fold, "metric": metric_name, "value": value,
                        }
                    )
            _save_run_artifacts(run_id, result, save_histories)
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            if verbose:
                print(f"  [ERROR] {run_id}: {row['error']}", flush=True)
                traceback.print_exc()

        rows.append(row)
        processed += 1
        if verbose:
            elapsed = time.perf_counter() - t0
            eta = elapsed / processed * (n_total - len(done_ids) - processed)
            print(
                f"[{len(done_ids) + processed:3d}/{n_total}] {run_id:58s} "
                f"({elapsed/60:6.1f} min transcurridos, quedan ~{eta/60:5.1f} min)",
                flush=True,
            )

        pd.DataFrame(rows).to_csv(config.RESULTS_MASTER_PATH, index=False)
        if fold_rows:
            pd.DataFrame(fold_rows).to_csv(config.RESULTS_FOLDS_PATH, index=False)

    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Corre la grilla de modelos del proyecto.")
    parser.add_argument("mode", nargs="?", default="smoke", choices=["smoke", "full"])
    parser.add_argument("--only-models", default=None, help="nombres separados por coma")
    parser.add_argument("--only-optimizers", default=None, help="grid,random,bayesian,genetic")
    parser.add_argument("--only-tasks", default=None, help="clf,reg")
    parser.add_argument("--out-dir", default=None, help="carpeta de salida (por defecto experiments/)")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    if args.out_dir:
        config.set_experiments_dir(args.out_dir)
    split = lambda s: s.split(",") if s else None
    df = run_all(
        run_mode=args.mode,
        resume=not args.no_resume,
        models=split(args.only_models),
        optimizers=split(args.only_optimizers),
        tasks=split(args.only_tasks),
    )
    print(f"Terminado: {len(df)} combinaciones, {int(df['error'].notna().sum())} con error.", flush=True)


if __name__ == "__main__":
    main()
