"""Comparación estadística entre modelos.

Con 140 modelos hay 9730 pares posibles, así que aplicar DeLong o
Diebold-Mariano a todos ellos sin control produciría unos 487 falsos
positivos solo por azar. Se sigue una estrategia jerárquica:

* Clasificación: Friedman, Nemenyi con diagrama de diferencia crítica,
  DeLong sobre un subconjunto reducido (con corrección por comparaciones
  múltiples) y tamaño del efecto.
* Regresión: Model Confidence Set sobre todos los modelos, Giacomini-White
  entre los finalistas, Diebold-Mariano con corrección de
  Harvey-Leybourne-Newbold y bootstrap estacionario como confirmación,
  prueba SPA frente a un modelo de referencia y tamaño del efecto.

Las pruebas sobre series de errores (Diebold-Mariano, Giacomini-White, MCS,
SPA) trabajan con los errores mensuales fuera de muestra, no con las
métricas agregadas por pliegue: con solo cinco pliegues no habría potencia.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import studentized_range
from statsmodels.stats.multitest import multipletests

try:
    import scikit_posthocs as sp
except ImportError:  # pragma: no cover
    sp = None


# =====================================================================
# Friedman, Nemenyi y diagrama de diferencia crítica
# =====================================================================
@dataclass
class FriedmanResult:
    statistic: float
    p_value: float
    df: int
    significant: bool
    avg_ranks: pd.Series


def friedman_test(scores_wide: pd.DataFrame, alpha: float = 0.05, higher_is_better: bool = True) -> FriedmanResult:
    """Prueba de Friedman, equivalente no paramétrico del ANOVA de medidas repetidas.

    Parameters
    ----------
    scores_wide : pandas.DataFrame
        Una fila por bloque (pliegue) y una columna por modelo.
    alpha : float
        Nivel de significancia.
    higher_is_better : bool
        Si un valor mayor de la métrica es mejor (F1, AUC) o peor (RMSE).

    Returns
    -------
    FriedmanResult
        Estadístico chi-cuadrado, p-valor, grados de libertad y rango
        promedio de cada modelo (rango 1 = mejor).
    """
    arrays = [scores_wide[c].values for c in scores_wide.columns]
    stat, p = stats.friedmanchisquare(*arrays)
    ranks = scores_wide.rank(axis=1, ascending=not higher_is_better)
    return FriedmanResult(
        statistic=float(stat),
        p_value=float(p),
        df=scores_wide.shape[1] - 1,
        significant=bool(p < alpha),
        avg_ranks=ranks.mean(axis=0).sort_values(),
    )


def nemenyi_posthoc(scores_wide: pd.DataFrame, higher_is_better: bool = True) -> pd.DataFrame:
    """Matriz de p-valores de la prueba post-hoc de Nemenyi entre todos los pares."""
    if sp is None:
        raise ImportError("scikit-posthocs no está instalado")
    data = scores_wide if higher_is_better else -scores_wide
    result = sp.posthoc_nemenyi_friedman(data.values)
    result.index = result.columns = scores_wide.columns
    return result


def critical_difference(n_models: int, n_blocks: int, alpha: float = 0.05) -> float:
    """Diferencia crítica de Nemenyi: ``q_alpha * sqrt(k (k + 1) / (6 N))``.

    ``q_alpha`` es el cuantil del rango estudentizado (infinitos grados de
    libertad) dividido por raíz de 2.

    Parameters
    ----------
    n_models : int
        Número de modelos comparados (k).
    n_blocks : int
        Número de bloques (N).
    alpha : float
        Nivel de significancia.
    """
    q_alpha = studentized_range.ppf(1 - alpha, n_models, np.inf) / np.sqrt(2)
    return float(q_alpha * np.sqrt(n_models * (n_models + 1) / (6 * n_blocks)))


def cd_groups(avg_ranks: pd.Series, cd: float) -> List[List[str]]:
    """Grupos de modelos que Nemenyi no distingue (diferencia de rango < CD)."""
    ordered = avg_ranks.sort_values()
    names = list(ordered.index)
    groups = []
    for i, start in enumerate(names):
        members = [start]
        for other in names[i + 1:]:
            if ordered[other] - ordered[start] < cd:
                members.append(other)
            else:
                break
        if len(members) > 1:
            groups.append(members)
    return [g for g in groups if not any(set(g) < set(h) for h in groups)]


def plot_cd_diagram(avg_ranks: pd.Series, cd: float, ax=None, title: str = None):
    """Dibuja el diagrama de diferencia crítica de Demšar.

    Los modelos se ubican sobre un eje según su rango promedio (el mejor, a
    la izquierda) y una barra roja une a los que la prueba de Nemenyi no
    distingue entre sí (diferencia de rangos menor que la diferencia crítica).

    Parameters
    ----------
    avg_ranks : pandas.Series
        Rango promedio de cada modelo.
    cd : float
        Diferencia crítica.
    ax : matplotlib.axes.Axes, optional
        Eje donde dibujar.
    title : str, optional
        Título del gráfico.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    ordered = avg_ranks.sort_values()
    names, ranks = list(ordered.index), ordered.values
    k = len(ordered)
    groups = cd_groups(avg_ranks, cd)
    half = int(np.ceil(k / 2))
    n_levels = max(half, k - half)
    y0 = -0.25 - 0.12 * max(len(groups), 1)
    step = 0.3
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 1.8 + 0.3 * n_levels))

    ax.hlines(0, 1, k, color="black", lw=1)
    for r in range(1, k + 1):
        ax.plot([r, r], [0, 0.08], color="black", lw=1)
        ax.text(r, 0.14, str(r), ha="center", va="bottom", fontsize=9)

    ax.plot([1, 1 + cd], [0.6, 0.6], color="black", lw=2.5)
    for x in (1, 1 + cd):
        ax.plot([x, x], [0.54, 0.66], color="black", lw=1)
    ax.text(1 + cd / 2, 0.72, f"DC = {cd:.2f}", ha="center", fontsize=9)

    for i, (name, rank) in enumerate(zip(names, ranks)):
        if i < half:
            y, x_end = y0 - step * i, 0.35
            ax.plot([rank, rank, x_end], [0, y, y], color="#555555", lw=0.9)
            ax.text(x_end - 0.1, y, f"{name} ({rank:.2f})", ha="right", va="center", fontsize=9)
        else:
            y, x_end = y0 - step * (i - half), k + 0.65
            ax.plot([rank, rank, x_end], [0, y, y], color="#555555", lw=0.9)
            ax.text(x_end + 0.1, y, f"{name} ({rank:.2f})", ha="left", va="center", fontsize=9)

    for g, members in enumerate(groups):
        member_ranks = [ordered[m] for m in members]
        y = -0.1 - 0.12 * g
        ax.plot([min(member_ranks) - 0.05, max(member_ranks) + 0.05], [y, y],
                color="#d03a3a", lw=4, solid_capstyle="round")

    ax.set_xlim(0.5 - 3.0, k + 0.5 + 3.0)
    ax.set_ylim(y0 - step * n_levels, 1.0)
    ax.axis("off")
    if title:
        ax.set_title(title)
    return ax


# =====================================================================
# DeLong: comparación de AUC
# =====================================================================
def _delong_components(y_true_bin: np.ndarray, scores: np.ndarray):
    """AUC y componentes de estructura de DeLong (algoritmo de rangos medios)."""
    pos = scores[y_true_bin == 1]
    neg = scores[y_true_bin == 0]
    n_pos, n_neg = len(pos), len(neg)

    def midrank(x):
        order = np.argsort(x)
        sorted_x = x[order]
        ranks = np.empty(len(x))
        i = 0
        while i < len(x):
            j = i
            while j < len(x) - 1 and sorted_x[j + 1] == sorted_x[i]:
                j += 1
            ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
            i = j + 1
        return ranks

    tx, ty, tz = midrank(pos), midrank(neg), midrank(np.concatenate([pos, neg]))
    v01 = (tz[:n_pos] - tx) / n_neg
    v10 = 1.0 - (tz[n_pos:] - ty) / n_pos
    return v01.mean(), v01, v10


def delong_test(y_true, proba_a, proba_b, positive_class) -> dict:
    """Compara dos AUC calculados sobre las mismas observaciones (DeLong, 1988).

    Binariza uno-contra-resto alrededor de ``positive_class``.

    Parameters
    ----------
    y_true : array-like
        Etiquetas observadas.
    proba_a, proba_b : array-like
        Probabilidad predicha de la clase de interés por cada modelo.
    positive_class : int
        Etiqueta de la clase de interés (por ejemplo, 2 = "Alto").

    Returns
    -------
    dict
        ``auc_a``, ``auc_b``, ``diff``, ``z``, ``p_value`` y ``ci95`` de la
        diferencia de AUC.
    """
    y_bin = (np.asarray(y_true) == positive_class).astype(int)
    auc_a, v01_a, v10_a = _delong_components(y_bin, np.asarray(proba_a))
    auc_b, v01_b, v10_b = _delong_components(y_bin, np.asarray(proba_b))
    n_pos, n_neg = int(y_bin.sum()), int((1 - y_bin).sum())
    s10 = np.cov(np.vstack([v01_a, v01_b]))
    s01 = np.cov(np.vstack([v10_a, v10_b]))
    var = s10 / n_pos + s01 / n_neg
    diff = auc_a - auc_b
    var_diff = max(var[0, 0] + var[1, 1] - 2 * var[0, 1], 1e-12)
    z = diff / np.sqrt(var_diff)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    half_width = 1.96 * np.sqrt(var_diff)
    return {
        "auc_a": float(auc_a), "auc_b": float(auc_b), "diff": float(diff), "z": float(z),
        "p_value": float(p), "ci95": (float(diff - half_width), float(diff + half_width)),
    }


# =====================================================================
# Bootstrap por bloques
# =====================================================================
def stationary_bootstrap_indices(n: int, mean_block: float, n_boot: int, rng: np.random.RandomState) -> np.ndarray:
    """Índices del bootstrap estacionario de Politis y Romano (1994).

    Cada remuestra concatena bloques de longitud geométrica (media
    ``mean_block``) que arrancan en posiciones aleatorias y se enrollan al
    final de la serie, lo que conserva la dependencia temporal local.

    Returns
    -------
    ndarray de forma (n_boot, n)
    """
    p = 1.0 / mean_block
    idx = np.empty((n_boot, n), dtype=int)
    starts = rng.randint(0, n, size=(n_boot, n))
    new_block = rng.uniform(size=(n_boot, n)) < p
    new_block[:, 0] = True
    for t in range(n):
        if t == 0:
            idx[:, 0] = starts[:, 0]
        else:
            idx[:, t] = np.where(new_block[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def moving_block_indices(n: int, block: int, n_boot: int, rng: np.random.RandomState) -> np.ndarray:
    """Índices del bootstrap por bloques móviles circulares de longitud fija."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.randint(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(n_boot, -1)[:, :n]


def bootstrap_means(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Media de cada remuestra bootstrap, calculada con una multiplicación de matrices.

    La media de una remuestra es la suma ponderada de las observaciones, con
    peso igual al número de veces que cada una fue elegida dividido por n. Es
    idéntica a promediar cada remuestra por separado, pero mucho más rápida.

    Parameters
    ----------
    values : ndarray de forma (n, m)
        Observaciones (por ejemplo, pérdidas de m modelos en n fechas).
    indices : ndarray de forma (B, n)
        Índices de las B remuestras.

    Returns
    -------
    ndarray de forma (B, m)
    """
    n_boot, n = indices.shape
    counts = np.zeros((n_boot, n))
    np.add.at(counts, (np.repeat(np.arange(n_boot), n), indices.ravel()), 1.0)
    return counts @ values / n


def bootstrap_interval(
    series: np.ndarray, statistic, mean_block: float = 12, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42
) -> Tuple[float, float, float]:
    """Intervalo de confianza percentil con bootstrap estacionario.

    Parameters
    ----------
    series : ndarray de forma (n,) o (n, m)
        Observaciones ordenadas en el tiempo (las filas se remuestrean juntas).
    statistic : callable
        Función que recibe una remuestra y devuelve un número.

    Returns
    -------
    (estimador puntual, límite inferior, límite superior)
    """
    series = np.asarray(series)
    rng = np.random.RandomState(seed)
    idx = stationary_bootstrap_indices(len(series), mean_block, n_boot, rng)
    boot = np.array([statistic(series[i]) for i in idx], dtype=float)
    return (
        float(statistic(series)),
        float(np.nanpercentile(boot, 100 * alpha / 2)),
        float(np.nanpercentile(boot, 100 * (1 - alpha / 2))),
    )


def bootstrap_ci(values, statistic=np.mean, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42, method: str = "bca"):
    """Intervalo de confianza bootstrap i.i.d. (percentil o BCa) de una muestra pequeña.

    Pensado para métricas por pliegue. Con datos dependientes en el tiempo
    se prefiere ``bootstrap_interval``.

    Returns
    -------
    (estimador puntual, límite inferior, límite superior)
    """
    values = np.asarray(values)
    rng = np.random.RandomState(seed)
    n = len(values)
    boot = np.array([statistic(values[rng.randint(0, n, n)]) for _ in range(n_boot)])
    point = statistic(values)
    if method == "percentile" or n < 4:
        return point, np.percentile(boot, 100 * alpha / 2), np.percentile(boot, 100 * (1 - alpha / 2))
    z0 = stats.norm.ppf(np.clip((boot < point).mean(), 1e-6, 1 - 1e-6))
    jack = np.array([statistic(np.delete(values, i)) for i in range(n)])
    num = np.sum((jack.mean() - jack) ** 3)
    den = 6.0 * (np.sum((jack.mean() - jack) ** 2) ** 1.5)
    a_hat = num / den if den != 0 else 0.0

    def bound(p):
        z_p = stats.norm.ppf(p)
        return stats.norm.cdf(z0 + (z0 + z_p) / (1 - a_hat * (z0 + z_p)))

    lo = np.percentile(boot, 100 * np.clip(bound(alpha / 2), 0, 1))
    hi = np.percentile(boot, 100 * np.clip(bound(1 - alpha / 2), 0, 1))
    return point, lo, hi


# =====================================================================
# Pruebas de capacidad predictiva (regresión)
# =====================================================================
def diebold_mariano_test(errors_a, errors_b, h: int = 1, power: int = 2) -> dict:
    """Prueba de Diebold-Mariano con la corrección de Harvey-Leybourne-Newbold.

    H0: los dos modelos tienen la misma precisión predictiva. Con ``power=2``
    la pérdida es el error cuadrático.

    Parameters
    ----------
    errors_a, errors_b : array-like
        Errores de pronóstico (observado menos predicho) de cada modelo sobre
        las mismas fechas.
    h : int
        Horizonte de pronóstico (aquí 1, un mes).
    power : int
        Potencia de la función de pérdida.

    Returns
    -------
    dict
        Estadístico DM, estadístico con corrección HLN, p-valor (distribución
        t con n - 1 grados de libertad), n y diferencia media de pérdida.
    """
    d = np.abs(np.asarray(errors_a)) ** power - np.abs(np.asarray(errors_b)) ** power
    n = len(d)
    d_bar = d.mean()
    long_run = np.var(d, ddof=0)
    for lag in range(1, h):
        long_run += 2 * np.cov(d[lag:], d[:-lag], bias=True)[0, 1]
    var_d_bar = max(long_run, 1e-12) / n
    dm = d_bar / np.sqrt(var_d_bar)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * hln
    p = 2 * (1 - stats.t.cdf(abs(dm_hln), df=n - 1))
    return {"dm": float(dm), "dm_hln": float(dm_hln), "p_value": float(p), "n": n, "mean_loss_diff": float(d_bar)}


def giacomini_white_test(errors_a, errors_b, conditional: bool = True, power: int = 2) -> dict:
    """Prueba de habilidad predictiva condicional de Giacomini y White (2006), horizonte 1.

    Con ``conditional=True`` los instrumentos son una constante y la
    diferencia de pérdida del periodo anterior: se contrasta si uno de los
    modelos es mejor *dado* cómo le fue al otro en el periodo previo. Con
    ``conditional=False`` solo se usa la constante (habilidad incondicional).

    Parameters
    ----------
    errors_a, errors_b : array-like
        Errores de pronóstico de cada modelo.
    conditional : bool
        Si se incluye el instrumento rezagado.
    power : int
        Potencia de la función de pérdida.

    Returns
    -------
    dict
        Estadístico de Wald, grados de libertad y p-valor (chi-cuadrado).
    """
    d = np.abs(np.asarray(errors_a)) ** power - np.abs(np.asarray(errors_b)) ** power
    if conditional:
        instruments = np.column_stack([np.ones(len(d) - 1), d[:-1]])
        target = d[1:]
    else:
        instruments = np.ones((len(d), 1))
        target = d
    z = instruments * target[:, None]
    n, q = z.shape
    z_bar = z.mean(axis=0)
    omega = (z.T @ z) / n
    wald = float(n * z_bar @ np.linalg.pinv(omega) @ z_bar)
    return {"wald": wald, "df": q, "p_value": float(1 - stats.chi2.cdf(wald, df=q)), "n": n}


def clark_west_test(errors_restricted, errors_unrestricted, pred_restricted, pred_unrestricted) -> dict:
    """Prueba de Clark y West (2007) para modelos anidados.

    Corrige el sesgo del estadístico de Diebold-Mariano cuando el modelo
    grande contiene al pequeño. Se contrasta, con una cola, que el modelo
    grande pronostica mejor.
    """
    e1, e2 = np.asarray(errors_restricted), np.asarray(errors_unrestricted)
    adj = e1 ** 2 - (e2 ** 2 - (np.asarray(pred_restricted) - np.asarray(pred_unrestricted)) ** 2)
    n = len(adj)
    t_stat = adj.mean() / (adj.std(ddof=1) / np.sqrt(n))
    return {"t": float(t_stat), "p_value": float(1 - stats.t.cdf(t_stat, df=n - 1)), "n": n}


def model_confidence_set(
    losses: pd.DataFrame, alpha: float = 0.10, n_boot: int = 5000, block: int = 12, seed: int = 42
) -> pd.DataFrame:
    """Model Confidence Set de Hansen, Lunde y Nason (2011), estadístico T_max.

    Parte de todos los modelos y, mientras se rechace la hipótesis de igual
    capacidad predictiva entre los que quedan, elimina el que peor desempeño
    relativo tiene. Los modelos que sobreviven forman el conjunto de
    confianza: no se pueden distinguir del mejor al nivel ``alpha``. La
    distribución del estadístico se obtiene con bootstrap por bloques móviles,
    que conserva la dependencia temporal de las pérdidas.

    Parameters
    ----------
    losses : pandas.DataFrame
        Pérdida de cada modelo (columnas) en cada fecha (filas).
    alpha : float
        Nivel de significancia del conjunto.
    n_boot : int
        Número de remuestras bootstrap.
    block : int
        Longitud del bloque (meses).
    seed : int
        Semilla.

    Returns
    -------
    pandas.DataFrame
        Pérdida media, p-valor MCS y pertenencia al conjunto, por modelo.
    """
    rng = np.random.RandomState(seed)
    L = losses.values
    n, m = L.shape
    idx = moving_block_indices(n, block, n_boot, rng)
    names = list(losses.columns)
    alive = list(range(m))
    p_values = np.zeros(m)
    running_max_p = 0.0
    order_eliminated = []

    while len(alive) > 1:
        sub = L[:, alive]
        k = len(alive)
        d_bar_i = sub.mean(axis=0) - sub.mean()  # d_i. = media sobre j de (L_i - L_j)
        boot_means = bootstrap_means(sub, idx)  # (B, k)
        boot_d_i = boot_means - boot_means.mean(axis=1, keepdims=True)
        var_i = ((boot_d_i - d_bar_i) ** 2).mean(axis=0)
        se_i = np.sqrt(np.maximum(var_i, 1e-18))
        t_i = d_bar_i / se_i
        t_max = t_i.max()
        t_boot = ((boot_d_i - d_bar_i) / se_i).max(axis=1)
        p = float((t_boot >= t_max).mean())
        running_max_p = max(running_max_p, p)
        worst = int(np.argmax(t_i))
        p_values[alive[worst]] = running_max_p
        order_eliminated.append(alive[worst])
        if p >= alpha:
            for j in alive:
                if j != alive[worst]:
                    p_values[j] = 1.0
            p_values[alive[worst]] = running_max_p
            break
        alive.pop(worst)
    else:
        p_values[alive[0]] = 1.0

    mean_loss = pd.Series(L.mean(axis=0), index=names)
    out = pd.DataFrame(
        {
            "modelo": names,
            "perdida_media": mean_loss.values,
            "rango": mean_loss.rank().values,
            "p_valor_mcs": p_values,
            "en_conjunto": p_values >= alpha,
        }
    )
    return out.sort_values("rango").reset_index(drop=True)


def spa_test(
    loss_benchmark, loss_candidates: pd.DataFrame, n_boot: int = 5000, mean_block: float = 12, seed: int = 42
) -> dict:
    """Prueba de habilidad predictiva superior (SPA) de Hansen (2005).

    Responde si algún modelo candidato supera de verdad al modelo de
    referencia o si la aparente superioridad es producto de haber probado
    muchos candidatos (control de *data snooping*). El valor ``p_upper``
    corresponde al Reality Check de White (2000), más conservador.

    Parameters
    ----------
    loss_benchmark : array-like de forma (n,)
        Pérdida del modelo de referencia.
    loss_candidates : pandas.DataFrame de forma (n, m)
        Pérdida de cada candidato.
    n_boot : int
        Remuestras de bootstrap estacionario.
    mean_block : float
        Longitud media de bloque.
    seed : int
        Semilla.

    Returns
    -------
    dict
        Estadístico SPA, p-valores (consistente, inferior y superior), el
        candidato con mejor desempeño relativo y su diferencia media de pérdida.
    """
    bench = np.asarray(loss_benchmark)
    cand = loss_candidates.values
    d = bench[:, None] - cand  # positivo = el candidato es mejor que la referencia
    n, m = d.shape
    d_bar = d.mean(axis=0)
    rng = np.random.RandomState(seed)
    idx = stationary_bootstrap_indices(n, mean_block, n_boot, rng)
    boot_means = bootstrap_means(d, idx)  # (B, m)
    omega = np.sqrt(n * ((boot_means - d_bar) ** 2).mean(axis=0))
    omega = np.maximum(omega, 1e-12)
    t_stat = max(float((np.sqrt(n) * d_bar / omega).max()), 0.0)

    threshold = -np.sqrt(2 * np.log(np.log(max(n, 3))))
    mu = {
        "consistent": np.where(np.sqrt(n) * d_bar / omega <= threshold, d_bar, 0.0),
        "lower": np.minimum(d_bar, 0.0),
        "upper": np.zeros(m),
    }
    p = {}
    for key, centre in mu.items():
        z = np.sqrt(n) * (boot_means - d_bar + centre) / omega
        t_boot = np.maximum(z.max(axis=1), 0.0)
        p[key] = float((t_boot > t_stat).mean())
    best = int(np.argmax(d_bar))
    return {
        "t_spa": t_stat,
        "p_consistent": p["consistent"],
        "p_lower": p["lower"],
        "p_upper": p["upper"],
        "best_candidate": loss_candidates.columns[best],
        "best_mean_loss_gain": float(d_bar[best]),
    }


# =====================================================================
# Corrección por comparaciones múltiples y tamaño del efecto
# =====================================================================
def adjust_pvalues(pvalues: Sequence[float], method: str = "holm") -> np.ndarray:
    """Ajusta p-valores por comparaciones múltiples.

    Parameters
    ----------
    pvalues : sequence of float
        p-valores crudos.
    method : {"holm", "fdr_bh"}
        Holm-Bonferroni o Benjamini-Hochberg.
    """
    return multipletests(pvalues, method=method)[1]


def cliffs_delta(a, b) -> float:
    """Delta de Cliff: dominancia estocástica de ``a`` sobre ``b``, entre -1 y 1."""
    a, b = np.asarray(a), np.asarray(b)
    greater = (a[:, None] > b[None, :]).sum()
    less = (a[:, None] < b[None, :]).sum()
    return float((greater - less) / (len(a) * len(b)))


def cohens_d_paired(differences) -> float:
    """d de Cohen de una diferencia pareada: media sobre desviación estándar."""
    d = np.asarray(differences)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0


def effect_label(value: float, kind: str = "cliff") -> str:
    """Etiqueta cualitativa del tamaño del efecto (umbrales usuales)."""
    v = abs(value)
    cuts = (0.147, 0.33, 0.474) if kind == "cliff" else (0.2, 0.5, 0.8)
    if v < cuts[0]:
        return "insignificante" if kind == "cliff" else "muy pequeño"
    if v < cuts[1]:
        return "pequeño"
    if v < cuts[2]:
        return "mediano"
    return "grande"
