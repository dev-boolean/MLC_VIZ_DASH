"""Métricas de evaluación y funciones de puntaje para la optimización.

Criterio principal usado dentro de la búsqueda de hiperparámetros (bucle
interno):

* Clasificación: F1-macro. Las tres clases (Bajo, Medio, Alto) no cuestan lo
  mismo cuando se confunden: "Alto" corresponde a los periodos de crisis
  (2001-2003 y 2020) y es la clase minoritaria que más importa detectar. El
  promedio macro le da a cada clase el mismo peso sin importar su frecuencia,
  de modo que el optimizador no puede obtener un buen puntaje acertando solo
  la clase mayoritaria.
* Regresión: RMSE (se minimiza), con el MAE como métrica secundaria.
"""
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

N_CLASSES = 3


def classification_inner_score(y_true, y_pred) -> float:
    """Puntaje que maximiza la búsqueda en problemas de clasificación.

    Parameters
    ----------
    y_true, y_pred : array-like
        Etiquetas observadas y predichas del pliegue de validación interno.

    Returns
    -------
    float
        F1-macro (mayor es mejor). Las clases ausentes en ``y_true`` y en
        ``y_pred`` no se cuentan.
    """
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def regression_inner_score(y_true, y_pred) -> float:
    """Puntaje que maximiza la búsqueda en problemas de regresión.

    Returns
    -------
    float
        Menos el RMSE, porque el bucle interno siempre maximiza.
    """
    return -float(np.sqrt(mean_squared_error(y_true, y_pred)))


def expand_proba(proba: np.ndarray, classes: Sequence[int], n_classes: int = N_CLASSES) -> np.ndarray:
    """Completa con ceros las columnas de las clases que el modelo no vio.

    ``predict_proba`` solo devuelve columnas para las clases presentes durante
    el entrenamiento. Para poder apilar las predicciones de todos los pliegues
    se reordenan a un arreglo de ``n_classes`` columnas.

    Parameters
    ----------
    proba : ndarray de forma (n, k)
        Probabilidades devueltas por el modelo.
    classes : sequence of int
        Etiquetas (``estimator.classes_``) de las k columnas.
    n_classes : int
        Número total de clases del problema.

    Returns
    -------
    ndarray de forma (n, n_classes)
    """
    full = np.zeros((proba.shape[0], n_classes))
    for column, label in enumerate(classes):
        full[:, int(label)] = proba[:, column]
    return full


def macro_ovr_auc(y_true, proba: np.ndarray, n_classes: int = N_CLASSES) -> float:
    """AUC macro uno-contra-resto, usando solo las clases evaluables.

    Una clase es evaluable si en ``y_true`` hay al menos un caso positivo y al
    menos un caso negativo. En los pliegues de prueba pequeños puede faltar
    una clase; en ese caso se promedia el AUC de las clases restantes en vez
    de devolver NaN.

    Parameters
    ----------
    y_true : array-like de forma (n,)
        Etiquetas enteras observadas.
    proba : ndarray de forma (n, n_classes)
        Probabilidades predichas, una columna por clase.
    n_classes : int
        Número total de clases.

    Returns
    -------
    float
        AUC macro, o NaN si ninguna clase es evaluable.
    """
    y_true = np.asarray(y_true)
    aucs = []
    for k in range(n_classes):
        positives = int((y_true == k).sum())
        if positives == 0 or positives == len(y_true):
            continue
        aucs.append(roc_auc_score((y_true == k).astype(int), proba[:, k]))
    return float(np.mean(aucs)) if aucs else float("nan")


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    auc_ovr: Optional[float]

    def as_dict(self) -> dict:
        return self.__dict__


def full_classification_metrics(y_true, y_pred, y_proba=None, classes=None) -> ClassificationMetrics:
    """Métricas de un pliegue de prueba externo en clasificación.

    Parameters
    ----------
    y_true, y_pred : array-like
        Etiquetas observadas y predichas.
    y_proba : ndarray, optional
        Probabilidades predichas. Si se entrega junto con ``classes`` se
        completan las columnas ausentes antes de calcular el AUC.
    classes : sequence of int, optional
        Etiquetas de las columnas de ``y_proba`` (``estimator.classes_``).

    Returns
    -------
    ClassificationMetrics
        Accuracy, precisión y recall macro, F1-macro y AUC macro
        uno-contra-resto (``None`` si no hay probabilidades).
    """
    auc = None
    if y_proba is not None:
        full = expand_proba(y_proba, classes) if classes is not None else y_proba
        auc = macro_ovr_auc(y_true, full)
    return ClassificationMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision_macro=precision_score(y_true, y_pred, average="macro", zero_division=0),
        recall_macro=recall_score(y_true, y_pred, average="macro", zero_division=0),
        f1_macro=f1_score(y_true, y_pred, average="macro", zero_division=0),
        auc_ovr=auc,
    )


@dataclass
class RegressionMetrics:
    rmse: float
    mae: float

    def as_dict(self) -> dict:
        return self.__dict__


def full_regression_metrics(y_true, y_pred) -> RegressionMetrics:
    """RMSE y MAE de un pliegue de prueba externo en regresión."""
    return RegressionMetrics(
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mae=float(mean_absolute_error(y_true, y_pred)),
    )


def expected_calibration_error(y_true_bin: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Error de calibración esperado (ECE) para un evento binario.

    Parameters
    ----------
    y_true_bin : ndarray de 0/1
        Ocurrencia observada del evento (p. ej. "el mes es de clase Alto").
    y_prob : ndarray
        Probabilidad predicha del evento.
    n_bins : int
        Número de intervalos de igual ancho entre 0 y 1.

    Returns
    -------
    float
        Promedio ponderado de la diferencia absoluta entre la frecuencia
        observada y la probabilidad media predicha en cada intervalo.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true_bin)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(y_true_bin[mask].mean() - y_prob[mask].mean())
    return float(ece)


def multiclass_brier(y_true, proba: np.ndarray, n_classes: int = N_CLASSES) -> float:
    """Brier score multiclase: error cuadrático medio sobre todas las clases."""
    onehot = np.eye(n_classes)[np.asarray(y_true).astype(int)]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def binary_brier(y_true_bin, y_prob) -> float:
    """Brier score de un evento binario."""
    return float(brier_score_loss(y_true_bin, y_prob))
