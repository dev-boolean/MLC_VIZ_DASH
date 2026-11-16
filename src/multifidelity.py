"""Optimización multi-fidelidad: Successive Halving.

En vez de evaluar cada configuración con todo el presupuesto, se evalúan muchas
con un presupuesto pequeño, se descarta a la mayoría y se reparte el
presupuesto ahorrado entre las más prometedoras. Aquí la fidelidad es el
número de árboles (``n_estimators``) de Random Forest y XGBoost, que son los
modelos más caros del proyecto: con pocos árboles una configuración ya
muestra si es prometedora, y entrenarla con 300 cuesta unas nueve veces más.

El factor de reducción entre rondas es ``eta = 3``: en cada ronda sobrevive la
tercera parte de las configuraciones y su presupuesto se triplica.
"""
import time
from typing import List

import numpy as np

from . import config
from .search import SearchResult, _make_evaluator, _Tracker
from .search_space import sample_random

FIDELITY_PARAM = "n_estimators"
MAX_FIDELITY = 300
ETA = 3


def run_successive_halving(
    model_spec, X_tr, y_tr, inner_splits, task, balancing_method, budgets, seed,
    n_configs: int = 27, eta: int = ETA, max_fidelity: int = MAX_FIDELITY,
) -> SearchResult:
    """Successive Halving sobre el número de árboles.

    Se muestrean ``n_configs`` configuraciones al azar, se evalúan con
    ``max_fidelity / eta**(rondas - 1)`` árboles y en cada ronda se conserva
    la mejor fracción ``1 / eta``, multiplicando por ``eta`` los árboles, hasta
    llegar a ``max_fidelity``.

    Parameters
    ----------
    model_spec : ModelSpec
        Modelo con un hiperparámetro ``n_estimators`` (Random Forest o XGBoost).
    X_tr, y_tr, inner_splits, task, balancing_method, budgets, seed
        Ver ``src.search.run_grid_search``.
    n_configs : int
        Configuraciones iniciales (potencia de ``eta`` para rondas completas).
    eta : int
        Factor de reducción entre rondas.
    max_fidelity : int
        Número de árboles de la ronda final.

    Returns
    -------
    SearchResult
        ``history`` incluye, por evaluación, la fidelidad (``fidelity``) y la
        ronda; ``extra["cost_tree_units"]`` es el costo total en árboles
        entrenados (cada árbol evaluado cuenta una unidad por pliegue interno).
    """
    space = [p for p in model_spec.space if p.name != FIDELITY_PARAM]
    rounds = int(round(np.log(n_configs) / np.log(eta))) + 1
    fidelities = [max(int(round(max_fidelity / eta ** (rounds - 1 - r))), 1) for r in range(rounds)]
    rng = np.random.RandomState(seed)
    evaluate = _make_evaluator(model_spec, X_tr, y_tr, inner_splits, task, balancing_method, seed)

    candidates: List[dict] = [sample_random(space, rng) for _ in range(n_configs)]
    history, cost, best_params, best_score = [], 0, None, -np.inf
    t0 = time.perf_counter()
    cumulative = 0.0
    for r, fidelity in enumerate(fidelities):
        scored = []
        for params in candidates:
            started = time.perf_counter()
            score = evaluate({**params, FIDELITY_PARAM: fidelity})
            elapsed = time.perf_counter() - started
            cumulative += elapsed
            cost += fidelity * len(inner_splits)
            scored.append((score, params))
            history.append(
                {
                    "iteration": len(history), "score": score, "time_s": elapsed, "cum_time_s": cumulative,
                    "fidelity": fidelity, "round": r,
                    "best_so_far": max([h["score"] for h in history if h["fidelity"] == max_fidelity] + [-np.inf]),
                }
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        if fidelity == max_fidelity:
            best_score, best_params = scored[0][0], {**scored[0][1], FIDELITY_PARAM: max_fidelity}
        keep = max(len(candidates) // eta, 1)
        candidates = [params for _, params in scored[:keep]]
    wall = time.perf_counter() - t0
    return SearchResult(
        best_params=best_params, best_score=float(best_score), history=history,
        n_evaluations=len(history), wall_time_s=wall,
        extra={"fidelities": fidelities, "cost_tree_units": cost, "eta": eta, "n_configs": n_configs},
    )


def run_random_cost_matched(
    model_spec, X_tr, y_tr, inner_splits, task, balancing_method, budgets, seed, cost_tree_units: int,
    max_fidelity: int = MAX_FIDELITY,
) -> SearchResult:
    """Random Search con el mismo costo en árboles que Successive Halving.

    Todas las configuraciones se evalúan con ``max_fidelity`` árboles, así que
    el número de configuraciones es el entero más cercano a
    ``cost / (max_fidelity * pliegues internos)``; el costo resultante queda
    a menos de la mitad de una configuración del de Successive Halving.
    """
    space = [p for p in model_spec.space if p.name != FIDELITY_PARAM]
    n = max(int(round(cost_tree_units / (max_fidelity * len(inner_splits)))), 1)
    rng = np.random.RandomState(seed)
    tracker = _Tracker(_make_evaluator(model_spec, X_tr, y_tr, inner_splits, task, balancing_method, seed))
    t0 = time.perf_counter()
    for _ in range(n):
        tracker.evaluate({**sample_random(space, rng), FIDELITY_PARAM: max_fidelity})
    return tracker.result(time.perf_counter() - t0, {"cost_tree_units": n * max_fidelity * len(inner_splits), "n_configs": n})
