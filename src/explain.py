"""Explicaciones e incertidumbre para el dashboard, sin dependencias pesadas.

* ``shapley_values``: valores de Shapley de una predicción, estimados por
  muestreo de permutaciones (Štrumbelj y Kononenko, 2014) con distribución de
  referencia intervencional, la misma definición que usa ``shap`` con
  ``KernelExplainer``/``PermutationExplainer``. Funciona con cualquier modelo
  (incluido el original) y cumple de forma exacta la propiedad de eficiencia:
  la suma de los valores es ``f(x) - promedio de f`` sobre las referencias
  usadas.
* ``conformal_halfwidth`` y ``sequential_coverage``: intervalos de predicción
  por predicción conforme partida sobre los errores fuera de muestra, y su
  cobertura empírica evaluada hacia adelante en el tiempo.
"""
from typing import Callable, Optional

import numpy as np


def shapley_values(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    background: np.ndarray,
    n_permutations: int = 20,
    seed: int = 42,
) -> dict:
    """Valores de Shapley de ``f`` en ``x`` por muestreo de permutaciones.

    Para cada permutación ``π`` de las variables y cada fila de referencia
    ``z`` se parte de ``z`` y se van reemplazando, en el orden de ``π``, sus
    valores por los de ``x``; la contribución de la variable ``j`` es el
    cambio de ``f`` al reemplazarla. El promedio sobre permutaciones y
    referencias es un estimador insesgado del valor de Shapley.

    Parameters
    ----------
    f : callable
        Recibe una matriz ``(m, p)`` y devuelve un vector de ``m`` salidas.
    x : ndarray (p,)
        Punto a explicar.
    background : ndarray (b, p)
        Filas de referencia.
    n_permutations : int
        Permutaciones por fila de referencia.
    seed : int
        Semilla del muestreo.

    Returns
    -------
    dict
        ``values`` (p,), ``base_value`` (promedio de ``f`` en las referencias)
        y ``prediction`` (``f(x)``), con ``base_value + values.sum() == prediction``.
    """
    x = np.asarray(x, dtype=float).ravel()
    background = np.atleast_2d(np.asarray(background, dtype=float))
    p = len(x)
    rng = np.random.default_rng(seed)
    # Todas las evaluaciones se arman primero y se predicen en un solo lote.
    blocks, orders = [], []
    for z in background:
        for _ in range(n_permutations):
            order = rng.permutation(p)
            path = np.repeat(z[None, :], p + 1, axis=0)
            for step, j in enumerate(order, start=1):
                path[step:, j] = x[j]
            blocks.append(path)
            orders.append(order)
    out = np.asarray(f(np.vstack(blocks)), dtype=float).reshape(len(blocks), p + 1)
    values = np.zeros(p)
    for order, y_path in zip(orders, out):
        values[order] += np.diff(y_path)
    values /= len(blocks)
    base = float(out[:, 0].mean())
    return {"values": values, "base_value": base, "prediction": float(out[0, -1])}


def conformal_halfwidth(residuals: np.ndarray, coverage: float = 0.9) -> float:
    """Semiancho del intervalo conforme partido: cuantil ajustado de ``|residuo|``.

    Usa el cuantil ``ceil((n + 1) * coverage) / n`` de los errores absolutos,
    que garantiza la cobertura marginal si los errores son intercambiables.
    """
    r = np.sort(np.abs(np.asarray(residuals, dtype=float)))
    n = len(r)
    k = int(np.ceil((n + 1) * coverage)) - 1
    return float(r[min(max(k, 0), n - 1)])


def sequential_coverage(residuals: np.ndarray, coverage: float = 0.9, warmup: int = 48) -> Optional[dict]:
    """Cobertura empírica del intervalo conforme evaluada hacia adelante.

    En cada mes ``t > warmup`` el intervalo se calcula solo con los errores de
    los meses anteriores, como se haría en tiempo real; con errores
    dependientes en el tiempo la cobertura no está garantizada, así que se
    mide.

    Returns
    -------
    dict or None
        ``coverage`` empírica, ``mean_width`` y ``n`` meses evaluados.
    """
    r = np.abs(np.asarray(residuals, dtype=float))
    if len(r) <= warmup:
        return None
    hits, widths = [], []
    for t in range(warmup, len(r)):
        h = conformal_halfwidth(r[:t], coverage)
        hits.append(r[t] <= h)
        widths.append(2 * h)
    return {"coverage": float(np.mean(hits)), "mean_width": float(np.mean(widths)), "n": len(hits)}


def acf(x: np.ndarray, nlags: int = 24) -> np.ndarray:
    """Autocorrelación muestral de los rezagos 0..nlags."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    denom = np.sum(x ** 2)
    return np.array([1.0] + [np.sum(x[k:] * x[:-k]) / denom for k in range(1, nlags + 1)])


def pacf(x: np.ndarray, nlags: int = 24) -> np.ndarray:
    """Autocorrelación parcial por la recursión de Durbin-Levinson."""
    r = acf(x, nlags)
    phi = np.zeros((nlags + 1, nlags + 1))
    out = [1.0]
    for k in range(1, nlags + 1):
        num = r[k] - np.sum(phi[k - 1, 1:k] * r[1:k][::-1])
        den = 1.0 - np.sum(phi[k - 1, 1:k] * r[1:k])
        phi[k, k] = num / den
        phi[k, 1:k] = phi[k - 1, 1:k] - phi[k, k] * phi[k - 1, 1:k][::-1]
        out.append(phi[k, k])
    return np.array(out)
