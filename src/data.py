"""Carga de datos e ingeniería de variables.

El problema se plantea como un pronóstico a un mes: con la información
disponible hasta el mes t se predice la tasa de desempleo nacional del mes t+1.

La tasa de ocupación y la tasa global de participación **nacionales** no se
usan como predictoras del mismo mes porque existe una identidad contable casi
exacta, TD = 1 - TO/TGP (correlación de 0.9999993 en los datos): el modelo solo
tendría que despejar una ecuación y no aprendería nada del mercado laboral. En
su lugar se usan:

* rezagos y estadísticos móviles de la propia tasa de desempleo (legítimo en
  un pronóstico: en el mes t solo se usa información hasta t), y
* los tres indicadores de las 13 áreas metropolitanas, que corresponden a una
  población distinta de la nacional y no quedan atados a esa identidad.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

TARGET_COL = "tasa_desempleo_nacional"
AREA_COLS = ["tasa_global_participacion_area", "tasa_desempleo_area", "tasa_ocupacion_area"]
# Excluidas explicitamente por la identidad contable descrita arriba:
EXCLUDED_NACIONAL_COLS = ["tasa_global_participacion_nacional", "tasa_ocupacion_nacional"]

LAGS = [1, 2, 3, 6, 12]
ROLLING_WINDOWS = [3, 6, 12]


def load_raw() -> pd.DataFrame:
    """Lee la hoja de series mensuales de ``MLC.xlsx`` ordenada por fecha.

    Returns
    -------
    pandas.DataFrame
        300 observaciones mensuales (enero de 2001 a diciembre de 2025) con las
        seis tasas laborales y la columna ``fecha``.
    """
    df = pd.read_excel(config.DATA_PATH, sheet_name=config.DATA_SHEET)
    df = df.sort_values("fecha").reset_index(drop=True)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye las variables predictoras respetando el orden temporal.

    Todas las columnas usan únicamente información disponible hasta el mes t;
    el objetivo (mes t+1) se agrega por separado en ``assemble_dataset``.

    Parameters
    ----------
    df : pandas.DataFrame
        Serie mensual devuelta por ``load_raw``.

    Returns
    -------
    pandas.DataFrame
        Rezagos (1, 2, 3, 6 y 12 meses) de la tasa de desempleo nacional,
        medias y desviaciones móviles a 3, 6 y 12 meses, codificación cíclica
        del mes, índice de tendencia y los tres indicadores de las 13 áreas
        metropolitanas del mes t, más la columna ``fecha``.
    """
    feat = pd.DataFrame(index=df.index)
    feat["fecha"] = df["fecha"]

    target = df[TARGET_COL]
    # lag_k = valor de la tasa k-1 meses antes de la fila t (lag1 = target en t,
    # el valor mas reciente conocido al momento de pronosticar t+1).
    for lag in LAGS:
        feat[f"td_nacional_lag{lag}"] = target.shift(lag - 1)

    # Rolling inclusive de t (mismo criterio que lag1: en la fila t se conoce target[t]).
    for window in ROLLING_WINDOWS:
        feat[f"td_nacional_rollmean{window}"] = target.rolling(window).mean()
        feat[f"td_nacional_rollstd{window}"] = target.rolling(window).std()

    feat["mes_sin"] = np.sin(2 * np.pi * df["fecha"].dt.month / 12)
    feat["mes_cos"] = np.cos(2 * np.pi * df["fecha"].dt.month / 12)
    feat["trend_idx"] = np.arange(len(df))

    for col in AREA_COLS:
        feat[col] = df[col]

    return feat


@dataclass
class Dataset:
    X: pd.DataFrame
    y_reg: pd.Series
    y_clf: pd.Series          # codificado como enteros 0/1/2 (ver targets.to_codes) -- usado por el pipeline
    y_clf_labels: pd.Series   # version legible Bajo/Medio/Alto -- usada en EDA/reportes
    fechas: pd.Series
    feature_names: list


def assemble_dataset() -> Dataset:
    """Arma el conjunto de datos del problema de pronóstico a un mes.

    Cada fila contiene la información disponible en el mes t y los objetivos
    del mes t+1: la tasa de desempleo nacional (regresión) y su categoría
    Bajo/Medio/Alto codificada como 0/1/2 (clasificación). Se descartan las
    primeras filas, que no tienen rezago de 12 meses, y la última, que no
    tiene mes siguiente.

    Returns
    -------
    Dataset
        Predictoras, objetivos, fechas y nombres de las variables.
    """
    from . import targets  # import local para evitar ciclos

    df = load_raw()
    feat = build_features(df)

    y_reg_next = df[TARGET_COL].shift(-1)
    y_reg_next.name = "y_reg_next_month"

    valid = feat.drop(columns=["fecha"]).notna().all(axis=1) & y_reg_next.notna()

    X = feat.loc[valid].drop(columns=["fecha"]).reset_index(drop=True)
    y_reg = y_reg_next.loc[valid].reset_index(drop=True)
    fechas = feat.loc[valid, "fecha"].reset_index(drop=True)
    y_clf_labels = targets.categorize(y_reg).reset_index(drop=True)
    y_clf = targets.to_codes(y_clf_labels)

    return Dataset(X=X, y_reg=y_reg, y_clf=y_clf, y_clf_labels=y_clf_labels, fechas=fechas, feature_names=list(X.columns))


if __name__ == "__main__":
    ds = assemble_dataset()
    print("X shape:", ds.X.shape)
    print("Features:", ds.feature_names)
    print("Rango fechas:", ds.fechas.min(), "->", ds.fechas.max())
    print(ds.X.head())
    print(ds.y_reg.describe())
    print(ds.y_clf_labels.value_counts())
    print("codigos:", dict(zip(ds.y_clf_labels.cat.categories, range(len(ds.y_clf_labels.cat.categories)))))
