"""Comparación cuantitativa de los cuatro métodos de optimización.

Trabaja con los historiales que guarda ``src.runner``: para cada búsqueda, el
puntaje de cada evaluación, el tiempo que tomó y, en el algoritmo genético, la
diversidad de la población en cada generación.

Para comparar "calidad de solución por unidad de presupuesto" se usa el
*arrepentimiento normalizado*: en cada problema (un modelo, una técnica de
balanceo y un pliegue externo) se toma como cero el mejor puntaje que
encontró cualquiera de los cuatro métodos y como uno el puntaje mediano de
todas las evaluaciones; el arrepentimiento de un método tras t evaluaciones
mide qué tan lejos queda de ese mejor puntaje. Así se pueden promediar
problemas con escalas distintas (F1 en clasificación, RMSE en regresión).
"""
from typing import Tuple

import numpy as np
import pandas as pd

from . import config
from .analysis import load_history
from .search import FAILED_SCORE

OPTIMIZER_LABELS = {"grid": "Grid", "random": "Random", "bayesian": "Bayesiano", "genetic": "Genético"}
OPTIMIZER_ORDER = ["grid", "random", "bayesian", "genetic"]


def load_search_records(master: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga los historiales de todas las combinaciones.

    Returns
    -------
    evaluations : DataFrame
        Una fila por evaluación (``score``, ``best_so_far``, ``time_s``,
        ``cum_time_s``) con las columnas que identifican la combinación y el
        pliegue.
    generations : DataFrame
        Una fila por generación del algoritmo genético (aptitud y diversidad).
    searches : DataFrame
        Una fila por búsqueda (pliegue de una combinación) con su número de
        evaluaciones, tiempo total y tiempo del propio optimizador.
    """
    evals, gens, searches = [], [], []
    for _, row in master.iterrows():
        h = load_history(row["run_id"])
        ident = {k: row[k] for k in ["run_id", "task", "model", "balancing", "optimizer"]}
        for fold, (history, extra) in enumerate(zip(h["histories"], h["extras"])):
            for rec in history:
                evals.append({**ident, "fold": fold, **rec})
            for rec in extra.get("generations", []):
                gens.append({**ident, "fold": fold, **rec})
            fr = h["fold_results"][fold]
            searches.append(
                {
                    **ident, "fold": fold, "n_evaluations": fr["n_evaluations"],
                    "search_time_s": fr["search_wall_time_s"], "overhead_s": extra.get("overhead_s", np.nan),
                }
            )
    return pd.DataFrame(evals), pd.DataFrame(gens), pd.DataFrame(searches)


def normalized_regret(evaluations: pd.DataFrame, budget: int = 40) -> pd.DataFrame:
    """Arrepentimiento normalizado de cada método tras t evaluaciones (t = 1..budget).

    Parameters
    ----------
    evaluations : DataFrame
        Salida de ``load_search_records``.
    budget : int
        Número máximo de evaluaciones consideradas.

    Returns
    -------
    DataFrame
        Una fila por problema, método y t con ``regret`` (0 = alcanzó el mejor
        puntaje conocido) y ``cum_time_s``. Un método que agota su presupuesto
        antes de t (Grid con una grilla pequeña) conserva su último valor.
    """
    df = evaluations[evaluations["score"] > FAILED_SCORE / 10].copy()
    keys = ["task", "model", "balancing", "fold"]
    rows = []
    for _, block in df.groupby(keys):
        best = block["score"].max()
        typical = block["score"].median()
        scale = best - typical
        if scale < 1e-9:
            continue
        for optimizer, sub in block.groupby("optimizer"):
            sub = sub.sort_values("iteration")
            best_so_far = sub["score"].cummax().values
            cum_time = sub["time_s"].cumsum().values
            for t in range(1, budget + 1):
                i = min(t, len(sub)) - 1
                rows.append(
                    {
                        **dict(zip(keys, block[keys].iloc[0])), "optimizer": optimizer, "t": t,
                        "regret": max((best - best_so_far[i]) / scale, 0.0), "cum_time_s": cum_time[i],
                    }
                )
    return pd.DataFrame(rows)


def regret_checkpoints(regret: pd.DataFrame, checkpoints=(5, 10, 20, 30, 40)) -> pd.DataFrame:
    """Arrepentimiento medio de cada método en varios puntos del presupuesto."""
    sub = regret[regret["t"].isin(checkpoints)]
    table = sub.pivot_table(index="optimizer", columns="t", values="regret", aggfunc="mean")
    table["área bajo la curva"] = regret.groupby("optimizer")["regret"].mean()
    return table.reindex(OPTIMIZER_ORDER)


def evaluations_to_threshold(regret: pd.DataFrame, threshold: float = 0.10) -> pd.DataFrame:
    """Evaluaciones y tiempo necesarios para quedar a menos de ``threshold`` del mejor puntaje conocido."""
    out = []
    for (optimizer, *_), group in regret.groupby(["optimizer", "task", "model", "balancing", "fold"]):
        reached = group[group["regret"] <= threshold].sort_values("t")
        out.append(
            {
                "optimizer": optimizer,
                "evaluations": reached["t"].iloc[0] if len(reached) else np.inf,
                "time_s": reached["cum_time_s"].iloc[0] if len(reached) else np.inf,
            }
        )
    table = pd.DataFrame(out).groupby("optimizer").agg(
        mediana_evaluaciones=("evaluations", "median"),
        mediana_tiempo_s=("time_s", "median"),
        porcentaje_que_lo_alcanza=("evaluations", lambda s: 100 * np.isfinite(s).mean()),
    )
    return table.reindex(OPTIMIZER_ORDER)


def cost_table(searches: pd.DataFrame) -> pd.DataFrame:
    """Evaluaciones, tiempo total, tiempo por evaluación y overhead de cada método."""
    g = searches.groupby("optimizer")
    table = pd.DataFrame(
        {
            "evaluaciones_por_busqueda": g["n_evaluations"].mean(),
            "tiempo_por_busqueda_s": g["search_time_s"].mean(),
            "tiempo_total_min": g["search_time_s"].sum() / 60,
            "overhead_pct": 100 * g["overhead_s"].sum() / g["search_time_s"].sum(),
        }
    )
    table["tiempo_por_evaluacion_s"] = table["tiempo_por_busqueda_s"] / table["evaluaciones_por_busqueda"]
    return table.reindex(OPTIMIZER_ORDER)


def genetic_diversity_summary(generations: pd.DataFrame, collapse_threshold: float = 0.05) -> dict:
    """Resumen de la diversidad genética del algoritmo genético.

    Una población "colapsa" cuando la distancia media normalizada entre sus
    individuos baja de ``collapse_threshold``: ya casi no hay variedad que
    cruzar y la búsqueda depende de la mutación.

    Returns
    -------
    dict
        ``by_generation`` (diversidad y aptitud medias por generación),
        ``share_collapsed`` (proporción de búsquedas que colapsan antes de
        agotar el presupuesto), ``mean_collapse_generation`` y
        ``gain_after_collapse`` (mejora media de la mejor aptitud desde que
        colapsa hasta el final).
    """
    keys = ["run_id", "fold"]
    by_gen = generations.groupby("generation").agg(
        diversidad_pares=("diversity_pairwise", "mean"),
        diversidad_por_gen=("diversity_gene_std", "mean"),
        configuraciones_distintas=("n_unique_configs", "mean"),
        busquedas=("run_id", "size"),
    )
    collapsed, gains, collapse_gen = 0, [], []
    total = 0
    for _, g in generations.groupby(keys):
        g = g.sort_values("generation")
        total += 1
        below = g[g["diversity_pairwise"] < collapse_threshold]
        if len(below) and below["generation"].iloc[0] < g["generation"].max():
            collapsed += 1
            first = below.iloc[0]
            collapse_gen.append(int(first["generation"]))
            gains.append(float(g["best_fitness"].iloc[-1] - first["best_fitness"]))
    return {
        "by_generation": by_gen,
        "share_collapsed": collapsed / total if total else np.nan,
        "mean_collapse_generation": float(np.mean(collapse_gen)) if collapse_gen else np.nan,
        "gain_after_collapse": float(np.mean(gains)) if gains else np.nan,
        "n_searches": total,
    }
