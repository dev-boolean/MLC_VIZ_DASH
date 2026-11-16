"""Gráficos del capítulo de modelado con machine learning.

Usan la misma paleta del análisis exploratorio (azul y rojo) y rótulos en
español. Cada función devuelve la figura para que el notebook decida cómo
mostrarla.
"""
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import auc, confusion_matrix, roc_curve

from . import config

BLUE, RED, GREEN, GREY = "#3876c4", "#d03a3a", "#2e8b57", "#7f7f7f"
CLASS_COLORS = {"Bajo": GREEN, "Medio": BLUE, "Alto": RED}
OPTIMIZER_COLORS = {"grid": GREY, "random": GREEN, "bayesian": BLUE, "genetic": RED}
_OPTIMIZER_NAMES = {"grid": "Grid", "random": "Random", "bayesian": "Bayesiano", "genetic": "Genético"}
CLASS_NAMES = config.CLASS_LABELS


def _grid(n_panels: int, ncols: int = 4, panel=(3.4, 3.2)):
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel[0] * ncols, panel[1] * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n_panels:]:
        ax.axis("off")
    return fig, axes


def plot_target_classes(dates, rate, thresholds: dict):
    """Serie de la tasa de desempleo coloreada por categoría, con los umbrales."""
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(dates, rate, color="#bbbbbb", linewidth=0.8, zorder=1)
    low, high = thresholds["bajo_max"], thresholds["alto_min"]
    labels = np.where(rate < low, "Bajo", np.where(rate >= high, "Alto", "Medio"))
    for name in CLASS_NAMES:
        mask = labels == name
        ax.scatter(np.asarray(dates)[mask], np.asarray(rate)[mask], s=14, color=CLASS_COLORS[name], label=name, zorder=2)
    ax.axhline(low, color=GREY, linestyle="--", linewidth=0.8)
    ax.axhline(high, color=GREY, linestyle="--", linewidth=0.8)
    ax.set_title("Tasa de desempleo nacional del mes siguiente y su categoría")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Tasa (%)")
    ax.legend(title="Categoría", loc="upper right")
    fig.tight_layout()
    return fig


def plot_walk_forward(dates, splits):
    """Tramos de entrenamiento y prueba de cada pliegue externo."""
    fig, ax = plt.subplots(figsize=(12, 3.6))
    dates = np.asarray(dates)
    for i, (train, test) in enumerate(splits):
        ax.scatter(dates[train], [i] * len(train), c=BLUE, s=5, label="Entrenamiento" if i == 0 else None)
        ax.scatter(dates[test], [i] * len(test), c=RED, s=5, label="Prueba" if i == 0 else None)
    ax.set_yticks(range(len(splits)))
    ax.set_yticklabels([f"Pliegue {i + 1}" for i in range(len(splits))])
    ax.invert_yaxis()
    ax.set_title("Validación anidada con ventana expansiva (bucle externo)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)  # debajo del gráfico para no tapar el pliegue 1
    fig.tight_layout()
    return fig


def confusion_grid(predictions: Dict[str, tuple]):
    """Matrices de confusión (normalizadas por fila) de varios modelos."""
    fig, axes = _grid(len(predictions))
    for ax, (name, (y_true, y_pred)) in zip(axes, predictions.items()):
        counts = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        share = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{counts[i, j]}", ha="center", va="center",
                        color="white" if share[i, j] > 0.55 else "black", fontsize=10)
        ax.set_xticks(range(3), CLASS_NAMES, fontsize=8)
        ax.set_yticks(range(3), CLASS_NAMES, fontsize=8)
        ax.set_xlabel("Predicha", fontsize=8)
        ax.set_ylabel("Observada", fontsize=8)
        ax.set_title(name, fontsize=10)
    fig.suptitle("Matrices de confusión (predicciones fuera de muestra de los cinco pliegues)", y=1.0)
    fig.tight_layout()
    return fig


def roc_grid(probabilities: Dict[str, tuple]):
    """Curvas ROC uno-contra-resto de varios modelos."""
    fig, axes = _grid(len(probabilities))
    for ax, (name, (y_true, proba)) in zip(axes, probabilities.items()):
        for k, cls in enumerate(CLASS_NAMES):
            binary = (np.asarray(y_true) == k).astype(int)
            if binary.min() == binary.max():
                continue
            fpr, tpr, _ = roc_curve(binary, proba[:, k])
            ax.plot(fpr, tpr, color=CLASS_COLORS[cls], label=f"{cls} ({auc(fpr, tpr):.2f})")
        ax.plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=0.8)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Tasa de falsos positivos", fontsize=8)
        ax.set_ylabel("Tasa de verdaderos positivos", fontsize=8)
        ax.legend(fontsize=7, loc="lower right", title="AUC", title_fontsize=7)
    fig.suptitle("Curvas ROC uno contra resto", y=1.0)
    fig.tight_layout()
    return fig


def reliability_grid(curves: Dict[str, tuple]):
    """Diagramas de confiabilidad de la clase "Alto".

    ``curves`` asigna a cada modelo una tupla (tabla de intervalos, Brier, ECE).
    """
    fig, axes = _grid(len(curves))
    for ax, (name, (table, brier, ece)) in zip(axes, curves.items()):
        ok = table["n"] > 0
        ax.plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=0.8)
        ax.plot(table.loc[ok, "prob_media"], table.loc[ok, "frecuencia"], color=BLUE, linewidth=1)
        ax.scatter(table.loc[ok, "prob_media"], table.loc[ok, "frecuencia"], s=10 + 4 * table.loc[ok, "n"], color=BLUE, alpha=0.7)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"{name}\nBrier {brier:.3f} · ECE {ece:.3f}", fontsize=9)
        ax.set_xlabel("Probabilidad predicha de Alto", fontsize=8)
        ax.set_ylabel("Frecuencia observada", fontsize=8)
    fig.suptitle("Diagramas de confiabilidad de la clase Alto", y=1.0)
    fig.tight_layout()
    return fig


def plot_regret(regret: pd.DataFrame):
    """Arrepentimiento normalizado frente al número de evaluaciones y al tiempo."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for optimizer in ["grid", "random", "bayesian", "genetic"]:
        sub = regret[regret["optimizer"] == optimizer]
        label = {"grid": "Grid", "random": "Random", "bayesian": "Bayesiano", "genetic": "Genético"}[optimizer]
        by_t = sub.groupby("t")["regret"]
        mean, sem = by_t.mean(), by_t.sem()
        axes[0].plot(mean.index, mean.values, color=OPTIMIZER_COLORS[optimizer], label=label)
        axes[0].fill_between(mean.index, mean - 1.96 * sem, mean + 1.96 * sem, color=OPTIMIZER_COLORS[optimizer], alpha=0.15)
        median_time = sub.groupby("t")["cum_time_s"].median()
        axes[1].plot(median_time.values, mean.values, color=OPTIMIZER_COLORS[optimizer], label=label)
    axes[0].set_xlabel("Evaluaciones realizadas (t)")
    axes[0].set_ylabel("Arrepentimiento normalizado")
    axes[0].set_title("Desempeño en cualquier momento")
    axes[1].set_xlabel("Tiempo acumulado de evaluación, mediana (s)")
    axes[1].set_xscale("log")
    axes[1].set_title("El mismo desempeño frente al tiempo")
    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.legend()
    fig.tight_layout()
    return fig


def plot_best_so_far(evaluations: pd.DataFrame, problems: list, budget: int = 40):
    """Mejor puntaje de validación interna hallado hasta la evaluación t, en las unidades de la métrica.

    Es el desempeño en cualquier momento (*anytime performance*) sin normalizar: para cada problema se
    promedia entre los cinco pliegues externos el mejor puntaje encontrado hasta la evaluación t.

    Parameters
    ----------
    evaluations : DataFrame
        Salida de ``optim_analysis.load_search_records`` (columnas ``task``, ``model``, ``balancing``,
        ``optimizer``, ``fold``, ``iteration``, ``score`` y ``best_so_far``).
    problems : list of (str, str, str, str)
        Cada elemento es ``(título, tarea, modelo, balanceo)``.
    budget : int
        Número de evaluaciones del eje horizontal.

    Returns
    -------
    matplotlib.figure.Figure
        Una gráfica por problema. En regresión el puntaje interno es el RMSE con signo negativo y aquí se
        grafica como RMSE (menor es mejor); en clasificación es el F1-macro (mayor es mejor).
    """
    fig, axes = plt.subplots(1, len(problems), figsize=(6.4 * len(problems), 4.4), squeeze=False)
    for ax, (title, task, model, balancing) in zip(axes[0], problems):
        sub = evaluations[(evaluations["task"] == task) & (evaluations["model"] == model)
                          & (evaluations["balancing"] == balancing) & (evaluations["score"] > -1e5)]
        for optimizer, color in OPTIMIZER_COLORS.items():
            curves = sub[sub["optimizer"] == optimizer].pivot_table(index="iteration", columns="fold", values="best_so_far")
            curves = curves.reindex(range(budget)).ffill()  # Grid con grilla pequeña conserva su último valor
            mean = curves.mean(axis=1) * (-1 if task == "reg" else 1)
            ax.plot(np.arange(1, budget + 1), mean.values, color=color, linewidth=1.8, label=_OPTIMIZER_NAMES[optimizer])
        ax.set_xlabel("Evaluaciones realizadas (t)")
        ax.set_ylabel("RMSE interno (menor es mejor)" if task == "reg" else "F1-macro interno (mayor es mejor)")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    return fig


def plot_genetic_diversity(by_generation: pd.DataFrame):
    """Diversidad genética de la población por generación."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    g = by_generation[by_generation["busquedas"] >= 10]
    axes[0].plot(g.index, g["diversidad_pares"], marker="o", color=RED, label="Distancia media entre pares")
    axes[0].plot(g.index, g["diversidad_por_gen"], marker="s", color=BLUE, label="Desviación estándar por gen")
    axes[0].set_xlabel("Generación")
    axes[0].set_ylabel("Diversidad (escala 0 a 1)")
    axes[0].set_title("Diversidad genética de la población")
    axes[0].legend()
    axes[1].bar(g.index, g["configuraciones_distintas"], color=GREY)
    axes[1].set_xlabel("Generación")
    axes[1].set_ylabel("Configuraciones distintas (de 8)")
    axes[1].set_title("Individuos distintos en la población")
    fig.tight_layout()
    return fig


def plot_train_val_test(dates, observed, predicted, n_train: int, n_validation: int, title: str):
    """Observado y predicho en entrenamiento, validación y prueba."""
    dates = np.asarray(dates)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axvspan(dates[0], dates[n_train - 1], color=BLUE, alpha=0.06, label="Entrenamiento")
    ax.axvspan(dates[n_train], dates[n_train + n_validation - 1], color="#e0a030", alpha=0.12, label="Validación")
    ax.axvspan(dates[n_train + n_validation], dates[-1], color=RED, alpha=0.08, label="Prueba")
    ax.plot(dates, observed, color=BLUE, linewidth=1.1, label="Observado")
    ax.plot(dates, predicted, color="#e07b00", linewidth=1.1, linestyle="--", label="Predicho")
    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Tasa de desempleo nacional (%)")
    ax.legend(loc="upper right", ncol=5, fontsize=8)
    fig.tight_layout()
    return fig


def plot_residuals(dates, errors, fitted, ljung_p: Dict[int, float], title: str):
    """Residuos en el tiempo y frente al ajuste, histograma, QQ-plot y ACF."""
    from statsmodels.tsa.stattools import acf

    errors, fitted = np.asarray(errors), np.asarray(fitted)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    axes[0, 0].plot(dates, errors, color=BLUE, linewidth=0.9)
    axes[0, 0].axhline(0, color=GREY, linewidth=0.8)
    axes[0, 0].set_title("Residuos en el tiempo")
    axes[0, 1].scatter(fitted, errors, s=10, color=BLUE, alpha=0.7)
    axes[0, 1].axhline(0, color=GREY, linewidth=0.8)
    axes[0, 1].set_xlabel("Valor predicho")
    axes[0, 1].set_title("Residuos frente al valor predicho")
    axes[0, 2].hist(errors, bins=25, density=True, color=BLUE, alpha=0.7)
    grid = np.linspace(errors.min(), errors.max(), 200)
    axes[0, 2].plot(grid, stats.norm.pdf(grid, errors.mean(), errors.std()), color=RED)
    axes[0, 2].set_title("Histograma de los residuos")
    stats.probplot(errors, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("QQ-plot")
    lags = 24
    values = acf(errors, nlags=lags, fft=True)
    axes[1, 1].bar(range(lags + 1), values, color=BLUE, width=0.6)
    bound = 1.96 / np.sqrt(len(errors))
    axes[1, 1].axhline(bound, color=RED, linestyle="--", linewidth=0.8)
    axes[1, 1].axhline(-bound, color=RED, linestyle="--", linewidth=0.8)
    axes[1, 1].set_xlabel("Rezago (meses)")
    axes[1, 1].set_title("Autocorrelación (ACF)")
    axes[1, 1].text(0.97, 0.95, "\n".join(f"Ljung-Box {k} rezagos: p = {v:.3f}" for k, v in ljung_p.items()),
                    transform=axes[1, 1].transAxes, ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    axes[1, 2].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_scaling(df: pd.DataFrame, family: str, x: str = "n", y: str = "entrenamiento_s", ax=None, title: str = None):
    """Tiempo frente a n (o p) en escala log-log, con el exponente empírico de cada variante."""
    sub = df[df["familia"] == family]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.2))
    for variant, group in sub.groupby("variante", sort=False):
        group = group.sort_values(x)
        valid = group[(group[x] > 0) & (group[y] > 0)]
        slope = np.polyfit(np.log(valid[x]), np.log(valid[y]), 1)[0] if len(valid) >= 2 else np.nan
        ax.plot(valid[x], valid[y], marker="o", label=f"{variant} (exp. {slope:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Observaciones (n)" if x == "n" else "Variables (p)")
    ax.set_ylabel("Tiempo (s)")
    ax.set_title(title or family)
    ax.legend(fontsize=7)
    return ax
