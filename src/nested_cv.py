"""Validación cruzada anidada con ventana expansiva (walk-forward).

* Bucle externo: ``TimeSeriesSplit`` sobre todo el conjunto. Estima el
  desempeño de generalización del modelo ya optimizado y nunca interviene en
  la búsqueda de hiperparámetros.
* Bucle interno: ``TimeSeriesSplit`` solo sobre el tramo de entrenamiento de
  cada pliegue externo. Es el que usa el optimizador (Grid, Random, bayesiano
  o genético) para elegir hiperparámetros.

La métrica que se reporta es siempre la del bucle externo, nunca el mejor
valor encontrado dentro del bucle interno.

Se usa ventana expansiva en vez de K-fold aleatorio porque las observaciones
son una serie mensual y las variables incluyen rezagos de la propia serie:
con folds aleatorios, meses casi idénticos terminarían repartidos entre
entrenamiento y prueba.
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from . import config
from .balancing import apply_balancing
from .fit_utils import fit_estimator
from .metrics import expand_proba, full_classification_metrics, full_regression_metrics
from .models import ModelSpec
from .search import OPTIMIZERS


@dataclass
class FoldResult:
    """Resultado de un pliegue externo."""

    fold: int
    metrics: dict
    best_params: dict
    n_evaluations: int
    search_wall_time_s: float
    n_train: int
    n_test: int
    fit_time_s: float = 0.0
    predict_time_s: float = 0.0


@dataclass
class NestedCVResult:
    """Resultado completo de la validación anidada de una combinación."""

    model_name: str
    task: str
    balancing_method: str
    optimizer_name: str
    fold_results: List[FoldResult] = field(default_factory=list)
    histories: List[list] = field(default_factory=list)
    extras: List[dict] = field(default_factory=list)
    oof: Dict[int, dict] = field(default_factory=dict)
    total_wall_time_s: float = 0.0
    error: Optional[str] = None


def _fit_final(model_spec: ModelSpec, params: dict, X_fit, y_fit, sample_weight, X_eval):
    """Ajusta el modelo final de un pliegue externo y transforma el tramo de prueba."""
    est = model_spec.estimator()
    est.set_params(**params)
    if model_spec.needs_scaling:
        scaler = StandardScaler().fit(X_fit)
        X_fit_t, X_eval_t = scaler.transform(X_fit), scaler.transform(X_eval)
    else:
        X_fit_t = X_fit.values if hasattr(X_fit, "values") else X_fit
        X_eval_t = X_eval.values if hasattr(X_eval, "values") else X_eval
    fit_estimator(est, X_fit_t, y_fit, sample_weight)
    return est, X_eval_t


def run_outer_fold(
    model_spec: ModelSpec,
    X,
    y,
    task: str,
    balancing_method: str,
    optimizer_fn,
    fold_i: int,
    train_idx,
    test_idx,
    seed: int = config.SEED,
    budgets: Optional[dict] = None,
) -> dict:
    """Procesa un pliegue externo: optimiza, ajusta el modelo final y lo evalúa.

    Es la unidad que se puede repartir entre procesos (``n_jobs`` en
    ``run_nested_cv``): cada pliegue externo es independiente de los demás.

    Parameters
    ----------
    model_spec : ModelSpec
        Modelo y espacio de búsqueda.
    X, y : pandas.DataFrame, pandas.Series
        Datos completos, ordenados cronológicamente.
    task : {"clf", "reg"}
        Tipo de problema.
    balancing_method : str
        Técnica de balanceo.
    optimizer_fn : callable
        Función de búsqueda (ver ``src.search``).
    fold_i : int
        Número del pliegue.
    train_idx, test_idx : ndarray
        Posiciones de entrenamiento y prueba del pliegue externo.
    seed : int
        Semilla.
    budgets : dict, optional
        Presupuestos de búsqueda (por defecto los del modo activo).

    Returns
    -------
    dict
        ``fold_result`` (FoldResult), ``history``, ``extra`` y ``oof``.
    """
    budgets = budgets or config.get_budgets()
    X_tr_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
    y_tr_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]

    inner_cv = TimeSeriesSplit(n_splits=config.N_SPLITS_INNER, gap=config.CV_GAP)
    inner_splits = list(inner_cv.split(X_tr_outer))

    search_result = optimizer_fn(
        model_spec, X_tr_outer, y_tr_outer, inner_splits, task, balancing_method, budgets, seed
    )

    X_fit, y_fit, sample_weight = X_tr_outer, y_tr_outer, None
    if task == "clf" and balancing_method != "none":
        X_fit, y_fit, sample_weight = apply_balancing(balancing_method, X_tr_outer, y_tr_outer, seed=seed)

    t0 = time.perf_counter()
    est, X_test_t = _fit_final(model_spec, search_result.best_params, X_fit, y_fit, sample_weight, X_test_outer)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = est.predict(X_test_t)
    y_proba = est.predict_proba(X_test_t) if task == "clf" and hasattr(est, "predict_proba") else None
    predict_time = time.perf_counter() - t0

    if task == "clf":
        m = full_classification_metrics(y_test_outer, y_pred, y_proba, classes=getattr(est, "classes_", None))
        proba_full = expand_proba(y_proba, est.classes_) if y_proba is not None else None
    else:
        m = full_regression_metrics(y_test_outer, y_pred)
        proba_full = None

    return {
        "fold_result": FoldResult(
            fold=fold_i,
            metrics=m.as_dict(),
            best_params=search_result.best_params,
            n_evaluations=search_result.n_evaluations,
            search_wall_time_s=search_result.wall_time_s,
            n_train=len(train_idx),
            n_test=len(test_idx),
            fit_time_s=fit_time,
            predict_time_s=predict_time,
        ),
        "history": search_result.history,
        "extra": search_result.extra,
        "oof": {
            "test_idx": np.asarray(test_idx, dtype=int),
            "y_true": np.asarray(y_test_outer),
            "y_pred": np.asarray(y_pred),
            "y_proba": proba_full,
        },
    }


def _safe_outer_fold(*args, **kwargs):
    """Ejecuta un pliegue externo y devuelve su excepción en vez de lanzarla."""
    try:
        return run_outer_fold(*args, **kwargs)
    except Exception as e:  # un pliegue degenerado no debe tumbar toda la combinación
        return {"error": f"{type(e).__name__}: {e}"}


def run_nested_cv(
    model_spec: ModelSpec,
    X,
    y,
    task: str,
    balancing_method: str,
    optimizer_name: str,
    seed: int = config.SEED,
    n_jobs: int = 1,
    backend: str = "loky",
    optimizer_fn=None,
    budgets: Optional[dict] = None,
) -> NestedCVResult:
    """Corre la validación anidada de una combinación modelo-balanceo-optimizador.

    Para cada pliegue externo: (1) el optimizador elige hiperparámetros con la
    validación interna sobre el tramo de entrenamiento, (2) se aplica la
    técnica de balanceo a ese tramo completo, (3) se ajusta el modelo final y
    (4) se evalúa sobre el tramo de prueba, que no intervino en nada de lo
    anterior.

    Un pliegue que falle por alguna razón inesperada se registra sin métricas y
    no interrumpe a los demás.

    Parameters
    ----------
    model_spec : ModelSpec
        Modelo y espacio de búsqueda.
    X : pandas.DataFrame
        Variables predictoras, ordenadas cronológicamente.
    y : pandas.Series
        Objetivo (etiquetas enteras en clasificación, tasa en regresión).
    task : {"clf", "reg"}
        Tipo de problema.
    balancing_method : str
        Técnica de balanceo (solo clasificación).
    optimizer_name : {"grid", "random", "bayesian", "genetic"}
        Método de optimización de hiperparámetros.
    seed : int
        Semilla de los métodos estocásticos.
    n_jobs : int
        Procesos en los que se reparten los pliegues externos (1 = en serie).
    backend : {"loky", "threading"}
        Backend de ``joblib`` cuando ``n_jobs`` es distinto de 1.
    optimizer_fn : callable, optional
        Función de búsqueda propia (por ejemplo, ``Successive Halving``); si se
        entrega, ``optimizer_name`` solo sirve de etiqueta.
    budgets : dict, optional
        Presupuestos de búsqueda (por defecto los del modo activo).

    Returns
    -------
    NestedCVResult
        Métricas por pliegue, historial de cada búsqueda y predicciones fuera
        de muestra de los pliegues externos (``oof``).
    """
    result = NestedCVResult(
        model_name=model_spec.name, task=task, balancing_method=balancing_method, optimizer_name=optimizer_name
    )
    optimizer_fn = optimizer_fn or OPTIMIZERS[optimizer_name]
    # Se fijan aquí, en el proceso principal: los procesos de joblib vuelven a
    # importar `config` y no heredan el modo de ejecución elegido en tiempo de corrida.
    budgets = budgets or config.get_budgets()
    outer_cv = TimeSeriesSplit(n_splits=config.N_SPLITS_OUTER, gap=config.CV_GAP)
    splits = list(outer_cv.split(X))

    t_total0 = time.perf_counter()
    jobs = [
        (model_spec, X, y, task, balancing_method, optimizer_fn, fold_i, train_idx, test_idx, seed, budgets)
        for fold_i, (train_idx, test_idx) in enumerate(splits)
    ]
    if n_jobs == 1:
        outputs = [_safe_outer_fold(*job) for job in jobs]
    else:
        from joblib import Parallel, delayed

        outputs = Parallel(n_jobs=n_jobs, backend=backend)(delayed(_safe_outer_fold)(*job) for job in jobs)

    for fold_i, ((train_idx, test_idx), out) in enumerate(zip(splits, outputs)):
        if "error" in out:
            result.fold_results.append(
                FoldResult(
                    fold=fold_i, metrics={}, best_params={}, n_evaluations=0,
                    search_wall_time_s=0.0, n_train=len(train_idx), n_test=len(test_idx),
                )
            )
            result.histories.append([])
            result.extras.append({})
            if result.error is None:
                result.error = f"fold {fold_i}: {out['error']}"
            continue
        result.fold_results.append(out["fold_result"])
        result.histories.append(out["history"])
        result.extras.append(out["extra"])
        result.oof[fold_i] = out["oof"]

    result.total_wall_time_s = time.perf_counter() - t_total0
    return result
