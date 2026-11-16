"""Ajuste de estimadores tolerante a ``sample_weight``.

Algunos modelos (por ejemplo ``KNeighbors*``) no aceptan ``sample_weight`` en
``fit``. Para que la técnica de balanceo ``class_weight='balanced'`` (que aquí
se implementa con pesos por observación, ver ``src/balancing.py``) se pueda
aplicar de forma uniforme a los siete clasificadores, el chequeo se centraliza
en este módulo: si el estimador no soporta pesos, se ajusta sin ponderar, lo
que equivale a no balancear para ese modelo en particular.
"""
import inspect

import numpy as np


def supports_sample_weight(estimator) -> bool:
    """Indica si ``estimator.fit`` acepta el argumento ``sample_weight``.

    Parameters
    ----------
    estimator : object
        Estimador con método ``fit``.

    Returns
    -------
    bool
    """
    try:
        return "sample_weight" in inspect.signature(estimator.fit).parameters
    except (TypeError, ValueError):
        return False


def to_numpy(X):
    """Convierte la entrada en arreglo, como recibe el modelo en la validación anidada.

    Es el paso de preprocesamiento de los artefactos de los modelos que no se
    escalan (ver ``src/artifacts.py``). Vive en un módulo importable, y no en
    el script que genera los artefactos, para que ``joblib`` lo encuentre al
    cargar el modelo desde otro programa (el dashboard o las pruebas).
    """
    return np.asarray(X, dtype=float)


def fit_estimator(estimator, X, y, sample_weight=None):
    """Ajusta ``estimator`` usando pesos por observación solo si los soporta.

    Parameters
    ----------
    estimator : object
        Estimador sin ajustar.
    X, y : array-like
        Datos de entrenamiento.
    sample_weight : array-like, optional
        Pesos por observación. Se ignoran si el estimador no los admite.

    Returns
    -------
    object
        El mismo estimador, ya ajustado.
    """
    if sample_weight is not None and supports_sample_weight(estimator):
        estimator.fit(X, y, sample_weight=sample_weight)
    else:
        estimator.fit(X, y)
    return estimator
