"""Construcción de las categorías de clasificación.

El dataset original solo tiene variables continuas, así que la variable de
clasificación se deriva de la tasa de desempleo nacional del mes siguiente:

* Bajo:  menos de 10 %
* Medio: entre 10 % y 14 %
* Alto:  14 % o más (recesión de comienzos de los 2000 y choque de 2020)

Los cortes se fijaron mirando la distribución histórica (media 11.6 %,
mediana 11.2 %, primer cuartil 9.7 %, tercer cuartil 12.9 %) para que "Alto"
recoja los periodos de crisis. Producen un desbalance real (alrededor de 14 %
de los meses son Alto) y no uno forzado, como ocurriría con terciles.
"""
import numpy as np
import pandas as pd

from . import config


def categorize(rate: pd.Series) -> pd.Series:
    """Convierte la tasa de desempleo en la categoría Bajo, Medio o Alto.

    Parameters
    ----------
    rate : pandas.Series
        Tasa de desempleo nacional en porcentaje.

    Returns
    -------
    pandas.Series
        Variable categórica ordenada con las categorías de ``config.CLASS_LABELS``.
    """
    bajo_max = config.CLASS_THRESHOLDS["bajo_max"]
    alto_min = config.CLASS_THRESHOLDS["alto_min"]
    bins = [-np.inf, bajo_max, alto_min, np.inf]
    labels = config.CLASS_LABELS
    return pd.cut(rate, bins=bins, labels=labels, right=False)


def class_balance_report(y_clf: pd.Series) -> pd.DataFrame:
    """Número y porcentaje de meses de cada categoría.

    Parameters
    ----------
    y_clf : pandas.Series
        Categorías Bajo/Medio/Alto (texto, no códigos).
    """
    counts = y_clf.value_counts().reindex(config.CLASS_LABELS)
    pct = (counts / counts.sum() * 100).round(1)
    return pd.DataFrame({"n": counts, "%": pct})


def to_codes(y_categorical: pd.Series) -> pd.Series:
    """Bajo/Medio/Alto -> 0/1/2. XGBoost exige labels enteros en [0, n_clases);
    se codifica de forma UNIFORME para los 7 clasificadores (no solo XGBoost)
    para mantener un unico tipo de target en todo el pipeline. El orden de
    las categorias (config.CLASS_LABELS) fija el mapeo, asi que 0=Bajo,
    1=Medio, 2=Alto de forma consistente en todo el proyecto."""
    return y_categorical.cat.codes.astype(int)
