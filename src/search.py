"""Los cuatro métodos de optimización de hiperparámetros: Grid Search, Random
Search, optimización bayesiana (Optuna, estimador TPE) y algoritmo genético
(DEAP).

Los cuatro se apoyan en la misma función, ``evaluate_params``, que reconstruye
a mano el pipeline completo (balanceo, escalado y modelo) dentro de cada
pliegue interno y lo ajusta solo con el tramo de entrenamiento, nunca con el
de validación. Optuna y DEAP no integran por sí solos un esquema de validación
cruzada, así que esa construcción manual es la que evita la fuga de
información; se aplica igual a Grid y Random para que los cuatro métodos
compartan exactamente el mismo protocolo de evaluación.
"""
import itertools
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np
import optuna
from deap import base, creator, tools
from sklearn.preprocessing import StandardScaler

from . import config
from .balancing import apply_balancing
from .fit_utils import fit_estimator
from .metrics import classification_inner_score, regression_inner_score
from .models import ModelSpec
from .search_space import decode_genotype, sample_random, to_grid, to_optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Puntaje que se asigna a una evaluación que no se pudo completar.
FAILED_SCORE = -1e6

# Parámetros del algoritmo genético (ver run_genetic_search).
GA_TOURNAMENT_SIZE = 3
GA_P_CROSSOVER = 0.6
GA_P_MUTATION = 0.3
GA_BLEND_ALPHA = 0.5
GA_MUTATION_SIGMA = 0.15
GA_GENE_MUTATION_PROB = 0.3
GA_MAX_GENERATIONS = 100


@dataclass
class SearchResult:
    """Resultado de una búsqueda de hiperparámetros.

    Attributes
    ----------
    best_params : dict
        Mejor combinación encontrada.
    best_score : float
        Puntaje interno (validación cruzada interna) de esa combinación.
    history : list of dict
        Una entrada por evaluación con ``iteration``, ``score``,
        ``best_so_far``, ``time_s`` y ``cum_time_s``. Sirve para construir las
        curvas de desempeño en cualquier momento (anytime performance).
    n_evaluations : int
        Número de evaluaciones realizadas.
    wall_time_s : float
        Tiempo total de la búsqueda, incluido el costo del propio optimizador.
    extra : dict
        Información propia de cada método (tamaño de la grilla completa,
        diversidad genética por generación, tiempo de overhead, etc.).
    """

    best_params: dict
    best_score: float
    history: List[dict] = field(default_factory=list)
    n_evaluations: int = 0
    wall_time_s: float = 0.0
    extra: dict = field(default_factory=dict)


def _fit_predict_fold(model_spec: ModelSpec, params: dict, X_tr, y_tr, X_val, sample_weight=None):
    """Ajusta el pipeline (escalado + modelo) en un tramo y predice otro.

    Dos ajustes propios de la búsqueda: durante la búsqueda solo se necesitan
    etiquetas, así que SVC no calibra probabilidades (esa calibración interna
    cuesta cinco ajustes por modelo y solo hace falta en el ajuste final); y
    KNN no puede pedir más vecinos que observaciones de entrenamiento, lo que
    ocurre en los pliegues internos más pequeños.
    """
    est = model_spec.estimator()
    est.set_params(**params)
    estimator_params = est.get_params()
    if "probability" in estimator_params:
        est.set_params(probability=False)
    if "n_neighbors" in estimator_params:
        est.set_params(n_neighbors=min(int(estimator_params["n_neighbors"]), len(X_tr)))
    if model_spec.needs_scaling:
        scaler = StandardScaler().fit(X_tr)
        X_tr_t, X_val_t = scaler.transform(X_tr), scaler.transform(X_val)
    else:
        X_tr_t = X_tr.values if hasattr(X_tr, "values") else X_tr
        X_val_t = X_val.values if hasattr(X_val, "values") else X_val
    fit_estimator(est, X_tr_t, y_tr, sample_weight)
    return est.predict(X_val_t)


def evaluate_params(
    params: dict,
    model_spec: ModelSpec,
    X_outer_train,
    y_outer_train,
    inner_splits,
    task: str,
    balancing_method: str = "none",
    seed: int = config.SEED,
) -> float:
    """Puntaje de validación cruzada interna de una combinación de hiperparámetros.

    Dentro de cada pliegue interno se aplica la técnica de balanceo, se ajusta
    el escalador y el modelo solo con el tramo de entrenamiento y se puntúa
    sobre el tramo de validación. Los pliegues internos en los que no se puede
    entrenar (en clasificación, porque solo aparece una clase) se omiten; si
    ningún pliegue es utilizable se devuelve ``FAILED_SCORE``.

    Parameters
    ----------
    params : dict
        Hiperparámetros del estimador.
    model_spec : ModelSpec
        Modelo a evaluar.
    X_outer_train, y_outer_train : pandas.DataFrame, pandas.Series
        Tramo de entrenamiento del pliegue externo.
    inner_splits : list of (ndarray, ndarray)
        Índices (entrenamiento, validación) de la validación interna.
    task : {"clf", "reg"}
        Tipo de problema.
    balancing_method : str
        Técnica de balanceo (solo clasificación).
    seed : int
        Semilla de SMOTE/ADASYN.

    Returns
    -------
    float
        Promedio del puntaje a lo largo de los pliegues internos (F1-macro en
        clasificación, menos el RMSE en regresión). Mayor es mejor.
    """
    scores = []
    for tr_idx, val_idx in inner_splits:
        X_tr, X_val = X_outer_train.iloc[tr_idx], X_outer_train.iloc[val_idx]
        y_tr, y_val = y_outer_train.iloc[tr_idx], y_outer_train.iloc[val_idx]

        if task == "clf" and y_tr.nunique() < 2:
            continue

        try:
            sample_weight = None
            if task == "clf" and balancing_method != "none":
                X_tr, y_tr, sample_weight = apply_balancing(balancing_method, X_tr, y_tr, seed=seed)
            y_pred = _fit_predict_fold(model_spec, params, X_tr, y_tr, X_val, sample_weight)
        except Exception:
            continue

        if task == "clf":
            scores.append(classification_inner_score(y_val, y_pred))
        else:
            scores.append(regression_inner_score(y_val, y_pred))

    if not scores:
        return FAILED_SCORE
    return float(np.mean(scores))


class _Tracker:
    """Lleva el historial de una búsqueda: puntaje, mejor hasta el momento y tiempo."""

    def __init__(self, evaluator: Callable[[dict], float]):
        self.evaluator = evaluator
        self.history: List[dict] = []
        self.best_score = -np.inf
        self.best_params = None
        self.cum_time = 0.0

    def evaluate(self, params: dict) -> float:
        t0 = time.perf_counter()
        score = self.evaluator(params)
        elapsed = time.perf_counter() - t0
        if score > self.best_score:
            self.best_score, self.best_params = score, dict(params)
        self.cum_time += elapsed
        self.history.append(
            {
                "iteration": len(self.history),
                "score": score,
                "best_so_far": self.best_score,
                "time_s": elapsed,
                "cum_time_s": self.cum_time,
            }
        )
        return score

    def result(self, wall_time_s: float, extra: dict = None) -> SearchResult:
        extra = dict(extra or {})
        extra["overhead_s"] = max(wall_time_s - self.cum_time, 0.0)
        return SearchResult(
            best_params=self.best_params,
            best_score=float(self.best_score),
            history=self.history,
            n_evaluations=len(self.history),
            wall_time_s=wall_time_s,
            extra=extra,
        )


def _make_evaluator(model_spec, X_tr, y_tr, inner_splits, task, balancing_method, seed):
    return lambda params: evaluate_params(
        params, model_spec, X_tr, y_tr, inner_splits, task, balancing_method, seed
    )


# ------------------------------------------------------------------ Grid Search
def run_grid_search(model_spec, X_tr, y_tr, inner_splits, task, balancing_method, budgets, seed) -> SearchResult:
    """Grid Search sobre la grilla discretizada del espacio de búsqueda.

    Si la grilla completa tiene más combinaciones que el presupuesto de
    evaluaciones, se toma una submuestra aleatoria de la grilla (recorrida en
    su orden natural). Si tiene menos, se recorre completa.

    Parameters
    ----------
    model_spec : ModelSpec
        Modelo y espacio de búsqueda.
    X_tr, y_tr : pandas.DataFrame, pandas.Series
        Entrenamiento del pliegue externo.
    inner_splits : list of (ndarray, ndarray)
        Pliegues de la validación interna.
    task : {"clf", "reg"}
        Tipo de problema.
    balancing_method : str
        Técnica de balanceo.
    budgets : dict
        Presupuestos (ver ``config.get_budgets``).
    seed : int
        Semilla.

    Returns
    -------
    SearchResult
    """
    grid = to_grid(model_spec.space)
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    n_full = len(combos)
    max_combos = budgets.get("grid_max_combinations")
    if max_combos is not None and n_full > max_combos:
        rng = np.random.RandomState(seed)
        idx = np.sort(rng.choice(n_full, size=max_combos, replace=False))
        combos = [combos[i] for i in idx]

    tracker = _Tracker(_make_evaluator(model_spec, X_tr, y_tr, inner_splits, task, balancing_method, seed))
    t0 = time.perf_counter()
    for combo in combos:
        tracker.evaluate(dict(zip(keys, combo)))
    return tracker.result(time.perf_counter() - t0, {"n_grid_full": n_full})


# ---------------------------------------------------------------- Random Search
def run_random_search(model_spec, X_tr, y_tr, inner_splits, task, balancing_method, budgets, seed) -> SearchResult:
    """Random Search: ``random_n_iter`` combinaciones muestreadas del espacio.

    Parameters, Returns
    -------------------
    Ver ``run_grid_search``.
    """
    rng = np.random.RandomState(seed)
    tracker = _Tracker(_make_evaluator(model_spec, X_tr, y_tr, inner_splits, task, balancing_method, seed))
    t0 = time.perf_counter()
    for _ in range(budgets["random_n_iter"]):
        tracker.evaluate(sample_random(model_spec.space, rng))
    return tracker.result(time.perf_counter() - t0)


# ------------------------------------------------------ Optimización bayesiana
def run_bayesian_search(model_spec, X_tr, y_tr, inner_splits, task, balancing_method, budgets, seed) -> SearchResult:
    """Optimización bayesiana con Optuna.

    El modelo sustituto es el estimador de Parzen de estructura de árbol
    (TPE): separa las evaluaciones en un grupo "bueno" (el mejor 10 %) y uno
    "malo", ajusta una densidad a cada grupo, l(x) y g(x), y propone el punto
    que maximiza l(x)/g(x), equivalente a maximizar la mejora esperada (EI).
    Las 10 primeras evaluaciones son aleatorias (arranque) y el resto
    concentra la búsqueda donde l(x) es alta, con la exploración que deja la
    incertidumbre de ambas densidades.

    Parameters, Returns
    -------------------
    Ver ``run_grid_search``.
    """
    tracker = _Tracker(_make_evaluator(model_spec, X_tr, y_tr, inner_splits, task, balancing_method, seed))
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=10)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        return tracker.evaluate(to_optuna(model_spec.space, trial))

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=budgets["bayes_n_trials"], show_progress_bar=False)
    return tracker.result(time.perf_counter() - t0, {"n_startup_trials": 10})


# ------------------------------------------------------------ Algoritmo genético
def genotype_diversity(population_matrix: np.ndarray):
    """Medidas de diversidad genética de una población.

    Parameters
    ----------
    population_matrix : ndarray de forma (individuos, genes)
        Genotipos, con cada gen en [0, 1].

    Returns
    -------
    pairwise : float
        Distancia euclidiana media entre pares de individuos, dividida por la
        raíz del número de genes (máximo teórico), por lo que queda en [0, 1].
    gene_std : float
        Desviación estándar de cada gen entre los individuos, promediada.
    """
    n, d = population_matrix.shape
    if n < 2:
        return 0.0, 0.0
    diffs = population_matrix[:, None, :] - population_matrix[None, :, :]
    distances = np.sqrt((diffs ** 2).sum(axis=-1))
    pairwise = distances[np.triu_indices(n, k=1)].mean() / np.sqrt(d)
    return float(pairwise), float(population_matrix.std(axis=0).mean())


def run_genetic_search(model_spec, X_tr, y_tr, inner_splits, task, balancing_method, budgets, seed) -> SearchResult:
    """Algoritmo genético con DEAP, con presupuesto medido en evaluaciones.

    Cada hiperparámetro es un gen en [0, 1] que se decodifica al rango real.
    La población se inicializa al azar y evoluciona con selección por torneo
    (tamaño 3), cruce por mezcla (``cxBlend``, alfa 0.5, probabilidad 0.6),
    mutación gaussiana (sigma 0.15, probabilidad 0.3 por individuo y 0.3 por
    gen) y elitismo de un individuo: el mejor de cada generación pasa intacto
    a la siguiente. Solo se evalúan los individuos que cambiaron, y la
    búsqueda se detiene al agotar el presupuesto de evaluaciones.

    En cada generación se registra la diversidad genética de la población
    (distancia media entre pares y dispersión por gen) y el mejor y el
    promedio de la aptitud, para detectar convergencia prematura.

    Parameters, Returns
    -------------------
    Ver ``run_grid_search``. La información por generación queda en
    ``SearchResult.extra["generations"]``.
    """
    pop_size = budgets["genetic_population"]
    budget = budgets["genetic_budget"]
    space = model_spec.space
    n_genes = len(space)

    # Los operadores de DEAP usan el generador global de `random`.
    config.set_global_seed(seed)

    key = f"{model_spec.name}_{task}_{n_genes}"
    fitness_name, individual_name = f"FitnessMax_{key}", f"Individual_{key}"
    if not hasattr(creator, fitness_name):
        creator.create(fitness_name, base.Fitness, weights=(1.0,))
    if not hasattr(creator, individual_name):
        creator.create(individual_name, list, fitness=getattr(creator, fitness_name))

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, getattr(creator, individual_name), toolbox.attr_float, n=n_genes)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=GA_BLEND_ALPHA)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=GA_MUTATION_SIGMA, indpb=GA_GENE_MUTATION_PROB)
    toolbox.register("select", tools.selTournament, tournsize=GA_TOURNAMENT_SIZE)

    tracker = _Tracker(_make_evaluator(model_spec, X_tr, y_tr, inner_splits, task, balancing_method, seed))
    generations: List[Dict] = []

    def evaluate_individual(individual):
        score = tracker.evaluate(decode_genotype(space, individual))
        individual.fitness.values = (score,)

    def log_generation(number, population):
        fitness = np.array([ind.fitness.values[0] for ind in population])
        pairwise, gene_std = genotype_diversity(np.array([list(ind) for ind in population]))
        unique = {tuple(sorted(decode_genotype(space, ind).items())) for ind in population}
        generations.append(
            {
                "generation": number,
                "evaluations": len(tracker.history),
                "best_fitness": float(fitness.max()),
                "mean_fitness": float(fitness.mean()),
                "diversity_pairwise": pairwise,
                "diversity_gene_std": gene_std,
                "n_unique_configs": len(unique),
            }
        )

    t0 = time.perf_counter()
    population = toolbox.population(n=pop_size)
    for individual in population:
        if len(tracker.history) >= budget:
            break
        evaluate_individual(individual)
    population = [ind for ind in population if ind.fitness.valid]
    log_generation(0, population)

    generation = 0
    while len(tracker.history) < budget and generation < GA_MAX_GENERATIONS:
        generation += 1
        elite = toolbox.clone(tools.selBest(population, 1)[0])
        offspring = [toolbox.clone(ind) for ind in toolbox.select(population, pop_size - 1)]

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < GA_P_CROSSOVER:
                toolbox.mate(child1, child2)
                del child1.fitness.values, child2.fitness.values
        for mutant in offspring:
            if random.random() < GA_P_MUTATION:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        for ind in offspring:
            for i in range(n_genes):
                ind[i] = min(max(ind[i], 0.0), 1.0)

        for ind in offspring:
            if not ind.fitness.valid:
                if len(tracker.history) >= budget:
                    break
                evaluate_individual(ind)
        population = [elite] + [ind for ind in offspring if ind.fitness.valid]
        log_generation(generation, population)

    wall = time.perf_counter() - t0
    return tracker.result(wall, {"generations": generations, "n_generations": generation})


OPTIMIZERS: Dict[str, Callable] = {
    "grid": run_grid_search,
    "random": run_random_search,
    "bayesian": run_bayesian_search,
    "genetic": run_genetic_search,
}
