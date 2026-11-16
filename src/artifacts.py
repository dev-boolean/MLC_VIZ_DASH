"""Mejores combinaciones, reentrenamiento final y serialización (bloque B del
tercer entregable).

De las 140 combinaciones del segundo entregable el dashboard solo usa la mejor
de cada modelo base: la variante (técnica de balanceo y optimizador) con mejor
métrica media del bucle externo, F1-macro en clasificación y RMSE en
regresión. Para cada una:

1. Se repite la búsqueda de hiperparámetros con el mismo optimizador, el mismo
   presupuesto y la misma validación interna (ventana expansiva), ahora sobre
   las 288 observaciones. En la validación anidada cada pliegue externo eligió
   sus propios hiperparámetros, así que no hay un único conjunto "óptimo" que
   reutilizar; repetir la búsqueda con todos los datos es la forma estándar de
   obtener el modelo final sin volver a mirar las métricas del bucle externo.
2. Se ajusta el pipeline completo (balanceo, escalado y modelo) con todas las
   observaciones, exactamente como en ``nested_cv._fit_final``.
3. Se guarda con ``joblib`` en ``artifacts/<task>_<modelo>/`` junto con
   ``metadata.json`` y las predicciones de referencia de la prueba de ida y
   vuelta (``roundtrip.npz``).

El balanceo (SMOTE, ADASYN o pesos de clase) solo actúa al entrenar, así que
el pipeline que se guarda contiene únicamente los pasos que se aplican al
predecir: el escalado (si el modelo lo usa) y el modelo. De este modo el
dashboard no necesita ``imbalanced-learn``.

Uso::

    python -m src.artifacts
"""
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from . import analysis as A
from . import config
from . import stats_tests as ST
from .balancing import apply_balancing
from .data import assemble_dataset
from .fit_utils import fit_estimator, to_numpy
from .models import CLASSIFIER_BY_NAME, REGRESSOR_BY_NAME
from .search import OPTIMIZERS

ARTIFACTS_DIR = config.ROOT_DIR / "artifacts"
SELECTION_PATH = ARTIFACTS_DIR / "seleccion.csv"

# Métrica principal de cada tarea: (nombre en la tabla maestra, mayor es mejor)
PRIMARY_METRIC = {"clf": ("f1_macro", True), "reg": ("rmse", False)}
METRIC_KEYS = {"clf": ["f1_macro", "accuracy", "precision_macro", "recall_macro", "auc_ovr"], "reg": ["rmse", "mae"]}

# El MCS se corre igual que en el segundo entregable: pérdida cuadrática, alfa
# de 0.10 y bootstrap por bloques de 12 meses.
MCS_ALPHA, MCS_BLOCK, MCS_N_BOOT = 0.10, 12, 5000

# Tamaño máximo razonable de un artefacto (Render gratuito y GitHub sin LFS).
MAX_ARTIFACT_MB = 50

LIBRARIES = ["numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "imbalanced-learn", "joblib", "optuna", "deap", "scipy"]


# ------------------------------------------------------------------ selección
def classification_finalists(master: pd.DataFrame, best: pd.DataFrame) -> dict:
    """Finalistas de clasificación según Friedman y Nemenyi.

    Se usa la misma versión estricta del segundo entregable: la mejor variante
    de cada modelo en los cinco pliegues externos. Si Friedman no rechaza la
    igualdad, ningún modelo se distingue del mejor y todos son finalistas; si
    la rechaza, son finalistas los que están a menos de una diferencia crítica
    de Nemenyi del primer lugar.

    Returns
    -------
    dict
        ``finalists`` (nombres de modelo), ``friedman`` (resultado) y ``cd``.
    """
    folds = A.load_folds()
    f1 = folds[folds["metric"] == "f1_macro"]
    wide = pd.DataFrame({
        model: f1[f1["run_id"] == row["run_id"]].sort_values("fold")["value"].values
        for model, row in best.iterrows()
    })
    fr = ST.friedman_test(wide, higher_is_better=True)
    cd = ST.critical_difference(wide.shape[1], wide.shape[0])
    if fr.significant:
        top = fr.avg_ranks.min()
        finalists = [m for m, r in fr.avg_ranks.items() if r - top < cd]
    else:
        finalists = list(wide.columns)
    return {"finalists": finalists, "friedman": fr, "cd": cd}


def regression_finalists(master: pd.DataFrame) -> dict:
    """Finalistas de regresión: el Model Confidence Set sobre los 28 modelos.

    Las pérdidas son los errores cuadráticos de las 240 predicciones fuera de
    muestra apiladas de los cinco pliegues externos.

    Returns
    -------
    dict
        ``finalists`` (``run_id`` en el conjunto de confianza) y ``mcs`` (tabla).
    """
    reg = master[master["task"] == "reg"]
    losses = {}
    for run_id in reg["run_id"]:
        pooled = A.pooled_oof(run_id)
        losses[run_id] = (pooled["y_true"] - pooled["y_pred"]) ** 2
    mcs = ST.model_confidence_set(pd.DataFrame(losses), alpha=MCS_ALPHA, n_boot=MCS_N_BOOT, block=MCS_BLOCK)
    return {"finalists": set(mcs.loc[mcs["en_conjunto"], "modelo"]), "mcs": mcs}


def select_best(master: pd.DataFrame = None) -> pd.DataFrame:
    """Tabla maestra de selección: la mejor combinación de cada modelo base.

    Returns
    -------
    pandas.DataFrame
        Una fila por tarea y modelo, con la métrica principal como
        media ± desviación, si es la mejor global de su tarea, si pertenece a
        los finalistas de la comparación estadística y el motivo de la selección.
    """
    master = master if master is not None else A.load_master()
    rows = []
    for task in ["clf", "reg"]:
        metric, higher = PRIMARY_METRIC[task]
        best = A.best_per_model(master, task)
        if task == "clf":
            fin = classification_finalists(master, best)
            in_finalists = {m: m in fin["finalists"] for m in best.index}
            fr = fin["friedman"]
            criterio = (
                f"Friedman sobre la mejor variante de cada modelo (5 pliegues): chi2 = {fr.statistic:.2f}, "
                f"p = {fr.p_value:.4f}; "
                + ("no se rechaza la igualdad, todos son finalistas" if not fr.significant
                   else f"finalistas a menos de CD = {fin['cd']:.2f} del primer rango")
            )
        else:
            fin = regression_finalists(master)
            in_finalists = {m: best.loc[m, "run_id"] in fin["finalists"] for m in best.index}
            criterio = (
                f"MCS sobre los 28 modelos (alfa = {MCS_ALPHA}, bloques de {MCS_BLOCK} meses): "
                f"{len(fin['finalists'])} de 28 en el conjunto de confianza"
            )
        variants = master[master["task"] == task].groupby("model").size()
        for rank, (model, row) in enumerate(best.iterrows(), start=1):
            rows.append({
                "task": task,
                "model": model,
                "run_id": row["run_id"],
                "balancing": row["balancing"],
                "optimizer": row["optimizer"],
                "metric": metric,
                "metric_mean": row[f"{metric}_mean"],
                "metric_std": row[f"{metric}_std"],
                "metric_mean_std": A.mean_std(row, metric),
                "rank_in_task": rank,
                "is_global_best": rank == 1,
                "in_statistical_finalists": in_finalists[model],
                "finalist_criterion": criterio,
                "reason": (
                    f"{'Mayor' if higher else 'Menor'} {metric} medio del bucle externo entre las "
                    f"{variants[model]} variantes de {model}"
                ),
                "predict_time_s": row["predict_time_s"],
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------- entrenamiento
def final_search(task: str, model_name: str, balancing: str, optimizer: str, ds, seed: int = config.SEED):
    """Repite la búsqueda de hiperparámetros con todas las observaciones.

    Usa el mismo optimizador, el mismo presupuesto (modo ``full``) y la misma
    validación interna (``TimeSeriesSplit`` de ``N_SPLITS_INNER`` pliegues con
    ``CV_GAP``) que el bucle interno de la validación anidada.

    Returns
    -------
    SearchResult
        Mejores hiperparámetros, puntaje interno, número de evaluaciones y tiempo.
    """
    spec = (CLASSIFIER_BY_NAME if task == "clf" else REGRESSOR_BY_NAME)[model_name]
    y = ds.y_clf if task == "clf" else ds.y_reg
    inner = list(TimeSeriesSplit(n_splits=config.N_SPLITS_INNER, gap=config.CV_GAP).split(ds.X))
    config.set_global_seed(seed)
    return OPTIMIZERS[optimizer](spec, ds.X, y, inner, task, balancing, config.FULL_BUDGETS, seed)


def fit_final_pipeline(task: str, model_name: str, balancing: str, params: dict, ds, seed: int = config.SEED) -> Pipeline:
    """Ajusta el pipeline final con todas las observaciones.

    Reproduce ``nested_cv._fit_final``: balanceo sobre los datos de
    entrenamiento, escalador ajustado con los datos (ya balanceados) y modelo
    ajustado con arreglos NumPy.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pasos ``prep`` (escalado o conversión a arreglo) y ``model``, ya ajustados.
    """
    spec = (CLASSIFIER_BY_NAME if task == "clf" else REGRESSOR_BY_NAME)[model_name]
    y = ds.y_clf if task == "clf" else ds.y_reg
    X_fit, y_fit, sw = ds.X, y, None
    if task == "clf" and balancing != "none":
        X_fit, y_fit, sw = apply_balancing(balancing, ds.X, y, seed=seed)
    prep = StandardScaler() if spec.needs_scaling else FunctionTransformer(to_numpy)
    prep.fit(X_fit)
    est = spec.estimator()
    est.set_params(**params)
    fit_estimator(est, to_numpy(prep.transform(X_fit)), y_fit, sw)
    return Pipeline([("prep", prep), ("model", est)])


def predictions(pipe: Pipeline, X: pd.DataFrame, task: str) -> dict:
    """Predicciones (y probabilidades, en clasificación) de un pipeline."""
    out = {"y_pred": np.asarray(pipe.predict(X))}
    if task == "clf":
        out["y_proba"] = np.asarray(pipe.predict_proba(X))
    return out


def input_schema(X: pd.DataFrame) -> list:
    """Esquema de las variables de entrada: nombre, tipo y rango observado."""
    return [
        {"name": c, "dtype": str(X[c].dtype), "min": float(X[c].min()), "max": float(X[c].max())}
        for c in X.columns
    ]


def library_versions() -> dict:
    """Versiones de Python y de las librerías que intervienen en el artefacto."""
    from importlib.metadata import version

    out = {"python": platform.python_version()}
    for lib in LIBRARIES:
        out[lib] = version(lib)
    return out


def sha256(path) -> str:
    """Huella SHA-256 de un archivo."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_artifact(sel_row: pd.Series, master: pd.DataFrame, ds, verbose: bool = True) -> dict:
    """Busca, reentrena, serializa y verifica una de las mejores combinaciones.

    Returns
    -------
    dict
        Metadatos guardados en ``metadata.json``.
    """
    task, model_name = sel_row["task"], sel_row["model"]
    balancing, optimizer = sel_row["balancing"], sel_row["optimizer"]
    out_dir = ARTIFACTS_DIR / f"{task}_{model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    search = final_search(task, model_name, balancing, optimizer, ds)
    search_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    pipe = fit_final_pipeline(task, model_name, balancing, search.best_params, ds)
    fit_time = time.perf_counter() - t0

    # Prueba de ida y vuelta: el artefacto cargado debe predecir exactamente lo
    # mismo que el modelo en memoria sobre un conjunto fijo (las 288 observaciones).
    reference = predictions(pipe, ds.X, task)
    model_path = out_dir / "model.joblib"
    joblib.dump(pipe, model_path, compress=3)
    loaded = predictions(joblib.load(model_path), ds.X, task)
    roundtrip_ok = all(np.array_equal(reference[k], loaded[k]) for k in reference)
    if not roundtrip_ok:
        raise RuntimeError(f"La prueba de ida y vuelta falló para {task}_{model_name}")
    np.savez_compressed(out_dir / "roundtrip.npz", X=ds.X.values, columns=np.array(ds.feature_names), **reference)

    size_mb = model_path.stat().st_size / 2**20
    if size_mb > MAX_ARTIFACT_MB:
        raise RuntimeError(f"{model_path} pesa {size_mb:.1f} MB; hay que reducirlo antes de desplegar")

    row = master.set_index("run_id").loc[sel_row["run_id"]]
    metrics = {k: {"mean": float(row[f"{k}_mean"]), "std": float(row[f"{k}_std"])} for k in METRIC_KEYS[task]}
    meta = {
        "artifact": f"{task}_{model_name}",
        "task": "clasificacion" if task == "clf" else "regresion",
        "model": model_name,
        "balancing": balancing,
        "optimizer": optimizer,
        "source_run_id": sel_row["run_id"],
        "target": (
            "Categoría de la tasa de desempleo nacional del mes siguiente (0 = Bajo, 1 = Medio, 2 = Alto)"
            if task == "clf" else "Tasa de desempleo nacional del mes siguiente (%)"
        ),
        "classes": {str(i): label for i, label in enumerate(config.CLASS_LABELS)} if task == "clf" else None,
        "seed": config.SEED,
        "hyperparameters": search.best_params,
        "final_search": {
            "optimizer": optimizer,
            "inner_cv": f"TimeSeriesSplit(n_splits={config.N_SPLITS_INNER}, gap={config.CV_GAP})",
            "budget": config.FULL_BUDGETS,
            "n_evaluations": search.n_evaluations,
            "best_inner_score": float(search.best_score),
            "wall_time_s": search_time,
        },
        "training": {
            "n_observations": int(len(ds.X)),
            # Cada fila es el mes t (información disponible) y su objetivo es el mes t+1.
            "first_feature_month": str(ds.fechas.min().date()),
            "last_feature_month": str(ds.fechas.max().date()),
            "fit_time_s": fit_time,
        },
        "input_schema": input_schema(ds.X),
        "outer_cv_metrics": metrics,
        "selection": {
            "primary_metric": sel_row["metric"],
            "rank_in_task": int(sel_row["rank_in_task"]),
            "is_global_best": bool(sel_row["is_global_best"]),
            "in_statistical_finalists": bool(sel_row["in_statistical_finalists"]),
            "finalist_criterion": sel_row["finalist_criterion"],
            "reason": sel_row["reason"],
        },
        "roundtrip_test": {"passed": roundtrip_ok, "n_rows": int(len(ds.X)), "exact": True},
        "file": {"name": model_path.name, "size_mb": round(size_mb, 3), "sha256": sha256(model_path), "compress": 3},
        "libraries": library_versions(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    if verbose:
        print(
            f"{meta['artifact']:24s} {balancing:12s} {optimizer:9s} {search.n_evaluations:3d} evaluaciones "
            f"{search_time:6.1f} s  {size_mb:6.2f} MB  ida y vuelta OK",
            flush=True,
        )
    return meta


def paper_metrics(model: str, dataset: str = "colombia") -> dict:
    """RMSE medio ± desviación de los pliegues externos según la tabla del paper.

    Lee ``artifacts/resultados/tabla_principal.csv``, que genera
    ``experiments/resultados_finales.py``.
    """
    tabla = pd.read_csv(ARTIFACTS_DIR / "resultados" / "tabla_principal.csv", header=[0, 1], index_col=[0, 1])
    fila = tabla.loc[(dataset, model)]
    return {"rmse": {"mean": float(fila[("RMSE", "mean")]), "std": float(fila[("RMSE", "std")])},
            "mae": {"mean": float(fila[("MAE", "mean")]), "std": float(fila[("MAE", "std")])}}


def build_paper_artifact(spec, artifact: str, metrics_name: str, status: str, reason: str,
                         ds=None, verbose: bool = True) -> dict:
    """Artefacto de un modelo del paper que no está entre las 140 combinaciones.

    Mismo procedimiento que las mejores combinaciones: búsqueda bayesiana con
    todas las observaciones y el mismo presupuesto, ajuste final y prueba de
    ida y vuelta. Las métricas del bucle externo son las del paper
    (:func:`paper_metrics`).
    """
    ds = ds or assemble_dataset()
    out_dir = ARTIFACTS_DIR / artifact
    out_dir.mkdir(parents=True, exist_ok=True)
    inner = list(TimeSeriesSplit(n_splits=config.N_SPLITS_INNER, gap=config.CV_GAP).split(ds.X))
    config.set_global_seed(config.SEED)
    t0 = time.perf_counter()
    search = OPTIMIZERS["bayesian"](spec, ds.X, ds.y_reg, inner, "reg", "none", config.FULL_BUDGETS, config.SEED)
    search_time = time.perf_counter() - t0
    est = spec.estimator()
    est.set_params(**search.best_params)
    prep = StandardScaler() if spec.needs_scaling else FunctionTransformer(to_numpy)
    prep.fit(ds.X)
    t0 = time.perf_counter()
    est.fit(to_numpy(prep.transform(ds.X)), ds.y_reg.values)
    fit_time = time.perf_counter() - t0
    pipe = Pipeline([("prep", prep), ("model", est)])

    reference = predictions(pipe, ds.X, "reg")
    model_path = out_dir / "model.joblib"
    joblib.dump(pipe, model_path, compress=3)
    loaded = predictions(joblib.load(model_path), ds.X, "reg")
    if not np.array_equal(reference["y_pred"], loaded["y_pred"]):
        raise RuntimeError(f"La prueba de ida y vuelta falló para {artifact}")
    np.savez_compressed(out_dir / "roundtrip.npz", X=ds.X.values, columns=np.array(ds.feature_names), **reference)

    meta = {
        "artifact": artifact, "task": "regresion", "model": spec.name, "balancing": "none", "optimizer": "bayesian",
        "status": status,
        "target": "Tasa de desempleo nacional del mes siguiente (%)", "classes": None, "seed": config.SEED,
        "hyperparameters": search.best_params,
        "final_search": {"optimizer": "bayesian", "inner_cv": f"TimeSeriesSplit(n_splits={config.N_SPLITS_INNER}, gap={config.CV_GAP})",
                         "budget": config.FULL_BUDGETS, "n_evaluations": search.n_evaluations,
                         "best_inner_score": float(search.best_score), "wall_time_s": search_time},
        "training": {"n_observations": int(len(ds.X)), "first_feature_month": str(ds.fechas.min().date()),
                     "last_feature_month": str(ds.fechas.max().date()), "fit_time_s": fit_time},
        "input_schema": input_schema(ds.X),
        "outer_cv_metrics": paper_metrics(metrics_name),
        "selection": {"primary_metric": "rmse", "reason": reason},
        "roundtrip_test": {"passed": True, "n_rows": int(len(ds.X)), "exact": True},
        "file": {"name": model_path.name, "size_mb": round(model_path.stat().st_size / 2**20, 3),
                 "sha256": sha256(model_path), "compress": 3},
        "libraries": library_versions(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    if verbose:
        print(f"{artifact}  {search.n_evaluations} evaluaciones {search_time:.1f} s  ida y vuelta OK", flush=True)
    return meta


def build_original_artifact(ds=None, verbose: bool = True) -> dict:
    """Artefacto del modelo original (ALEE de tres expertos, iteración 4)."""
    from .models import ORIGINAL_MODEL

    return build_paper_artifact(
        ORIGINAL_MODEL, "reg_ALEE", "ALEE",
        status="congelado: ALEE de tres expertos (iteración 4), el modelo del paper",
        reason="Modelo original del tercer entregable", ds=ds, verbose=verbose)


def build_course_artifacts(ds=None, verbose: bool = True) -> list:
    """Artefactos de los modelos del paper que no estaban entre las 140 combinaciones.

    La regresión bayesiana y el SVR con kernel RBF los exige la guía del curso;
    LightGBM es el método reciente fijado en el protocolo.
    """
    from .models import BAYESIAN_RIDGE_MODEL, LIGHTGBM_MODEL, SVR_RBF_MODEL

    ds = ds or assemble_dataset()
    specs = [(BAYESIAN_RIDGE_MODEL, "reg_BayesianRidge", "BayesianRidge", "Modelo del curso agregado en el tercer entregable"),
             (SVR_RBF_MODEL, "reg_SVR_RBF", "SVR_RBF", "SVR con kernel RBF, el que pide la guía del curso"),
             (LIGHTGBM_MODEL, "reg_LightGBM", "LightGBM", "Método reciente fijado en el protocolo")]
    return [build_paper_artifact(spec, artifact, name, status="modelo de comparación del paper", reason=reason,
                                 ds=ds, verbose=verbose)
            for spec, artifact, name, reason in specs]


def build_all(verbose: bool = True) -> pd.DataFrame:
    """Selecciona las mejores combinaciones y genera todos los artefactos.

    Returns
    -------
    pandas.DataFrame
        La tabla de selección, que también se guarda en ``artifacts/seleccion.csv``.
    """
    import warnings

    warnings.filterwarnings("ignore")
    config.RUN_MODE = "full"
    master = A.load_master()
    ds = assemble_dataset()
    selection = select_best(master)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    selection.to_csv(SELECTION_PATH, index=False)
    for _, row in selection.iterrows():
        meta = build_artifact(row, master, ds, verbose=verbose)
        selection.loc[(selection["task"] == row["task"]) & (selection["model"] == row["model"]), "artifact"] = meta["artifact"]
    selection.to_csv(SELECTION_PATH, index=False)
    return selection


if __name__ == "__main__":
    if "--original" in sys.argv or "--curso" in sys.argv:
        import warnings

        warnings.filterwarnings("ignore")
        config.RUN_MODE = "full"
        if "--original" in sys.argv:
            build_original_artifact()
        if "--curso" in sys.argv:
            build_course_artifacts()
    else:
        sel = build_all()
        print(f"\n{len(sel)} artefactos en {ARTIFACTS_DIR}", file=sys.stderr)
