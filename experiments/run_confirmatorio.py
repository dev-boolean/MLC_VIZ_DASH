"""Validación confirmatoria del protocolo congelado (experiments/protocolo.md).

Corre, con optimizador bayesiano (40 evaluaciones, tope de 15 min por modelo
por pliegue) y validacion anidada TimeSeriesSplit(5,gap=1)/TimeSeriesSplit(3,gap=1):

* Las 7 combinaciones del entregable anterior + ALEE + LightGBM (TabPFN quedo
  fuera: ver limitaciones declaradas).
* Las referencias de series de tiempo (persistencia, estacional, SARIMA) via
  src.analysis.regression_baselines, sobre los mismos 240 meses.
* El experto estacional de ALEE aislado, extraido con los hiperparametros que
  gano cada pliegue externo.

Guarda resultados parciales en artifacts/confirmatorio/ para poder cortar y
retomar sin perder lo ya corrido.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config
from src.data import assemble_dataset
from src.models import REGRESSOR_BY_NAME, ORIGINAL_MODEL, LIGHTGBM_MODEL
from src.nested_cv import run_nested_cv
from src.original_model import ModeloOriginal, EXPERTS
from src.analysis import regression_baselines
from sklearn.model_selection import TimeSeriesSplit

OUT_DIR = ROOT / "artifacts" / "confirmatorio"
OUT_DIR.mkdir(parents=True, exist_ok=True)

config.RUN_MODE = "full"
config.set_global_seed(config.SEED)

MODEL_NAMES = ["Ridge", "Lasso", "KNN", "DecisionTree", "RandomForest", "XGBoost", "SVR"]
STOCHASTIC = {"RandomForest", "XGBoost", "LightGBM"}
SEEDS_STOCH = list(range(42, 52))  # 42..51
TIME_CAP_S = 15 * 60  # tope de 15 min por modelo por pliegue (salvaguarda, no se corta a mitad de pliegue)

log_path = OUT_DIR / "log_ejecucion.txt"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")


def run_one(model_spec, ds, seed):
    t0 = time.perf_counter()
    result = run_nested_cv(
        model_spec=model_spec, X=ds.X, y=ds.y_reg, task="reg",
        balancing_method="none", optimizer_name="bayesian", seed=seed,
    )
    elapsed = time.perf_counter() - t0
    per_fold_avg = elapsed / max(len(result.fold_results), 1)
    if per_fold_avg > TIME_CAP_S:
        log(f"  AVISO: {model_spec.name} (seed {seed}) promedio {per_fold_avg/60:.1f} min/pliegue, excede el tope de 15 min declarado como limite de referencia (no se interrumpio a mitad de pliegue).")
    return result


def pooled_oof_from_result(result, ds):
    """Arma (idx, y_true, y_pred) apilado de los pliegues externos de un NestedCVResult."""
    rows = []
    for fold_i, data in result.oof.items():
        rows.append(pd.DataFrame({"idx": data["test_idx"], "y_true": data["y_true"], "y_pred": data["y_pred"]}))
    return pd.concat(rows, ignore_index=True).sort_values("idx").reset_index(drop=True)


def main():
    log("=== Validacion confirmatoria ALEE: arranque ===")
    ds = assemble_dataset()
    log(f"Dataset: {ds.X.shape[0]} filas, {ds.X.shape[1]} variables")

    all_results = {}
    pooled_preds = {}
    budget_report = []

    # 1) Modelos existentes (7) + ALEE + LightGBM, con semilla 42 (y multi-semilla si son estocasticos)
    specs = [REGRESSOR_BY_NAME[n] for n in MODEL_NAMES] + [ORIGINAL_MODEL, LIGHTGBM_MODEL]
    for spec in specs:
        cache_path = OUT_DIR / f"oof_{spec.name}.csv"
        if cache_path.exists():
            log(f"{spec.name}: ya corrido, se reusa cache")
            pooled_preds[spec.name] = pd.read_csv(cache_path)
            continue
        log(f"Corriendo {spec.name} (seed 42)...")
        result = run_one(spec, ds, seed=42)
        all_results[spec.name] = result
        if result.error:
            log(f"  ERROR en {spec.name}: {result.error}")
        pooled = pooled_oof_from_result(result, ds)
        pooled.to_csv(cache_path, index=False)
        pooled_preds[spec.name] = pooled
        budget_report.append({
            "modelo": spec.name, "seed": 42,
            "evaluaciones_totales": sum(fr.n_evaluations for fr in result.fold_results),
            "tiempo_busqueda_s": sum(fr.search_wall_time_s for fr in result.fold_results),
            "tiempo_total_s": result.total_wall_time_s,
        })
        pd.DataFrame(budget_report).to_csv(OUT_DIR / "presupuesto_consumido.csv", index=False)

        if spec.name in STOCHASTIC:
            seed_rmses = []
            for s in SEEDS_STOCH[1:]:  # ya tenemos 42, faltan 43..51
                cache_s = OUT_DIR / f"oof_{spec.name}_seed{s}.csv"
                if cache_s.exists():
                    p = pd.read_csv(cache_s)
                else:
                    log(f"  {spec.name} seed {s}...")
                    r = run_one(spec, ds, seed=s)
                    p = pooled_oof_from_result(r, ds)
                    p.to_csv(cache_s, index=False)
                rmse = float(np.sqrt(np.mean((p["y_true"] - p["y_pred"]) ** 2)))
                seed_rmses.append(rmse)
            if seed_rmses:
                pd.Series(seed_rmses, index=SEEDS_STOCH[1:]).to_csv(OUT_DIR / f"variabilidad_semillas_{spec.name}.csv")
                log(f"  {spec.name}: RMSE variabilidad entre semillas 43-51: media={np.mean(seed_rmses):.4f} sd={np.std(seed_rmses):.4f}")

    # 2) Referencias de series de tiempo: persistencia, estacional, SARIMA (sobre los mismos 240 meses)
    bl_path = OUT_DIR / "baselines_series_tiempo.csv"
    if bl_path.exists():
        baselines = pd.read_csv(bl_path)
    else:
        log("Corriendo referencias de series de tiempo (Persistencia, Estacional, SARIMA)...")
        baselines = regression_baselines(ds)
        baselines.to_csv(bl_path, index=False)
    log("Referencias de series de tiempo listas.")

    # 3) Experto estacional de ALEE aislado, con los hiperparametros que gano cada pliegue externo de ALEE
    seasonal_path = OUT_DIR / "oof_EstacionalALEE.csv"
    if seasonal_path.exists():
        seasonal_alee = pd.read_csv(seasonal_path)
    else:
        log("Extrayendo experto estacional de ALEE aislado por pliegue...")
        if "ALEE" not in all_results:
            raise RuntimeError("Se necesita el NestedCVResult de ALEE recien corrido para extraer el experto estacional; borra el cache oof_ALEE.csv para forzar recorrerlo junto con esto.")
        alee_result = all_results["ALEE"]
        outer = TimeSeriesSplit(n_splits=config.N_SPLITS_OUTER, gap=config.CV_GAP)
        splits = list(outer.split(ds.X))
        rows = []
        for fold_i, (train_idx, test_idx) in enumerate(splits):
            fr = alee_result.fold_results[fold_i]
            if not fr.metrics:
                continue
            m = ModeloOriginal(random_state=config.SEED, **fr.best_params)
            m.fit(ds.X.values[train_idx], ds.y_reg.values[train_idx])
            F = m.expert_predictions(ds.X.values[test_idx])
            idx_estacional = list(EXPERTS).index("estacional")
            rows.append(pd.DataFrame({"idx": test_idx, "y_true": ds.y_reg.values[test_idx], "y_pred": F[:, idx_estacional]}))
        seasonal_alee = pd.concat(rows, ignore_index=True)
        seasonal_alee.to_csv(seasonal_path, index=False)
    log("Experto estacional de ALEE aislado listo.")

    log("=== Corrida principal terminada. Ver artifacts/confirmatorio/ ===")


if __name__ == "__main__":
    main()
