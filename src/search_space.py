"""Espacio de búsqueda de hiperparámetros, declarado una sola vez por modelo y
adaptado a cada uno de los cuatro métodos de optimización.

Declarar cada hiperparámetro una única vez (tipo, rango y escala) y generar a
partir de ahí la grilla (Grid), el muestreo (Random), el espacio de Optuna
(Bayesiana) y el genotipo (Genético) evita que los cuatro métodos terminen
buscando en espacios distintos, lo cual invalidaría la comparación de
eficiencia entre ellos.
"""
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np


@dataclass
class HParam:
    """Un hiperparámetro y su rango de búsqueda.

    Attributes
    ----------
    name : str
        Nombre del parámetro en el estimador.
    kind : {"int", "float", "cat"}
        Tipo del hiperparámetro.
    low, high : float, optional
        Límites del rango (tipos numéricos).
    log : bool
        Si el rango se explora en escala logarítmica.
    choices : sequence, optional
        Valores posibles (tipo categórico).
    grid_points : int
        Número de puntos que se toman al discretizar el rango para Grid Search.
    """

    name: str
    kind: str
    low: Optional[float] = None
    high: Optional[float] = None
    log: bool = False
    choices: Optional[Sequence[Any]] = None
    grid_points: int = 4

    def grid_values(self) -> List[Any]:
        """Valores del hiperparámetro para la grilla de Grid Search."""
        if self.kind == "cat":
            return list(self.choices)
        if self.log:
            pts = np.logspace(np.log10(self.low), np.log10(self.high), self.grid_points)
        else:
            pts = np.linspace(self.low, self.high, self.grid_points)
        if self.kind == "int":
            return sorted(set(int(round(p)) for p in pts))
        return [float(p) for p in pts]


def to_grid(space: List[HParam], prefix: str = "") -> dict:
    """Convierte el espacio de búsqueda en una grilla de valores discretos.

    Parameters
    ----------
    space : list of HParam
        Hiperparámetros del modelo.
    prefix : str
        Prefijo opcional para los nombres (útil con ``Pipeline``).

    Returns
    -------
    dict
        Diccionario ``{nombre: lista de valores}``.
    """
    return {f"{prefix}{p.name}": p.grid_values() for p in space}


def sample_random(space: List[HParam], rng: np.random.RandomState) -> dict:
    """Muestrea un punto del espacio de búsqueda (usado por Random Search).

    Los rangos logarítmicos se muestrean de forma uniforme en el exponente.

    Parameters
    ----------
    space : list of HParam
        Hiperparámetros del modelo.
    rng : numpy.random.RandomState
        Generador con la semilla del experimento.

    Returns
    -------
    dict
        Un valor para cada hiperparámetro.
    """
    out = {}
    for p in space:
        if p.kind == "cat":
            out[p.name] = p.choices[rng.randint(len(p.choices))]
        elif p.kind == "int":
            out[p.name] = int(rng.randint(p.low, p.high + 1))
        elif p.log:
            out[p.name] = float(10 ** rng.uniform(np.log10(p.low), np.log10(p.high)))
        else:
            out[p.name] = float(rng.uniform(p.low, p.high))
    return out


def to_optuna(space: List[HParam], trial) -> dict:
    """Pide a un ``trial`` de Optuna un valor para cada hiperparámetro.

    Parameters
    ----------
    space : list of HParam
        Hiperparámetros del modelo.
    trial : optuna.trial.Trial
        Ensayo en curso.

    Returns
    -------
    dict
        Valores sugeridos por el muestreador.
    """
    out = {}
    for p in space:
        if p.kind == "cat":
            out[p.name] = trial.suggest_categorical(p.name, list(p.choices))
        elif p.kind == "int":
            out[p.name] = trial.suggest_int(p.name, int(p.low), int(p.high), log=p.log)
        else:
            out[p.name] = trial.suggest_float(p.name, p.low, p.high, log=p.log)
    return out


def decode_genotype(space: List[HParam], genotype: Sequence[float]) -> dict:
    """Decodifica un genotipo del algoritmo genético a valores reales.

    Cada gen vive en el intervalo [0, 1] y se traduce al rango real del
    hiperparámetro correspondiente (linealmente, o en escala logarítmica si el
    parámetro lo pide; para categóricos se divide [0, 1] en tramos iguales).

    Parameters
    ----------
    space : list of HParam
        Hiperparámetros del modelo, en el mismo orden que los genes.
    genotype : sequence of float
        Un gen por hiperparámetro.

    Returns
    -------
    dict
        Valores de los hiperparámetros.
    """
    out = {}
    for gene, p in zip(genotype, space):
        u = min(max(gene, 0.0), 1.0)
        if p.kind == "cat":
            idx = min(int(u * len(p.choices)), len(p.choices) - 1)
            out[p.name] = p.choices[idx]
        elif p.kind == "int":
            out[p.name] = int(round(p.low + u * (p.high - p.low)))
        elif p.log:
            out[p.name] = float(10 ** (np.log10(p.low) + u * (np.log10(p.high) - np.log10(p.low))))
        else:
            out[p.name] = float(p.low + u * (p.high - p.low))
    return out
