"""Técnicas de balanceo de clases (solo aplican a clasificación).

``class_weight='balanced'`` se implementa de forma uniforme con
``sample_weight`` (pesos inversamente proporcionales a la frecuencia de cada
clase) y no con el parámetro nativo ``class_weight`` de cada estimador. KNN,
Naive Bayes y XGBoost no tienen ese parámetro, y usar el mismo mecanismo en
los siete clasificadores hace que la comparación entre modelos bajo una misma
técnica sea justa. KNN tampoco admite pesos por observación, así que para ese
modelo ``class_weight`` equivale a no balancear.
"""
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.utils.class_weight import compute_sample_weight

from . import config

BALANCING_METHODS = ["none", "smote", "adasyn", "class_weight"]


def _safe_k_neighbors(y: pd.Series, requested: int = 5) -> Optional[int]:
    """Número de vecinos que SMOTE/ADASYN pueden usar con las clases presentes.

    Ambas técnicas interpolan entre una observación de la clase minoritaria y
    sus vecinos más cercanos, por lo que necesitan al menos dos clases en los
    datos y más casos en la clase minoritaria que vecinos solicitados. En los
    pliegues pequeños de la validación anidada puede faltar una clase o tener
    muy pocos casos (por ejemplo, los primeros 13 meses de la serie son todos
    de clase "Alto"); en esos casos no se remuestrea.

    Parameters
    ----------
    y : pandas.Series
        Etiquetas del conjunto de entrenamiento.
    requested : int
        Número de vecinos deseado.

    Returns
    -------
    int or None
        Vecinos a usar, o ``None`` si no es posible remuestrear.
    """
    counts = y.value_counts()
    if counts.shape[0] < 2:
        return None
    min_class_n = counts.min()
    if min_class_n < 2:
        return None
    return min(requested, int(min_class_n) - 1)


def apply_balancing(
    method: str, X_train: pd.DataFrame, y_train: pd.Series, seed: int = config.SEED
) -> Tuple[pd.DataFrame, pd.Series, Optional[np.ndarray]]:
    """Aplica una técnica de balanceo sobre un conjunto de entrenamiento.

    Debe llamarse siempre sobre el conjunto de entrenamiento de cada pliegue,
    nunca sobre el de validación o prueba, para no filtrar información.

    Parameters
    ----------
    method : {"none", "smote", "adasyn", "class_weight"}
        Técnica a aplicar.
    X_train : pandas.DataFrame
        Variables predictoras de entrenamiento.
    y_train : pandas.Series
        Etiquetas de entrenamiento.
    seed : int
        Semilla de los generadores de datos sintéticos.

    Returns
    -------
    X_bal, y_bal : conjunto de entrenamiento (remuestreado en SMOTE/ADASYN).
    sample_weight : ndarray or None
        Pesos por observación (solo con ``class_weight``).
    """
    if method == "none":
        return X_train, y_train, None

    if method == "class_weight":
        return X_train, y_train, compute_sample_weight(class_weight="balanced", y=y_train)

    if method in ("smote", "adasyn"):
        k = _safe_k_neighbors(y_train)
        if k is None:
            return X_train, y_train, None
        # SMOTE y ADASYN nombran distinto el mismo parámetro en imbalanced-learn.
        if method == "smote":
            sampler = SMOTE(random_state=seed, k_neighbors=k)
        else:
            sampler = ADASYN(random_state=seed, n_neighbors=k)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        return X_res, y_res, None

    raise ValueError(f"Técnica de balanceo desconocida: {method}")
