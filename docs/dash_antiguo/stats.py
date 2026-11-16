import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose


def tabla_descriptiva(df):
    df_num = df.drop(columns=["fecha"], errors="ignore")
    return df_num.describe().T.reset_index().rename(columns={"index": "variable"})


def tabla_caracteristicas(df):
    cols = [
        "tasa_global_participacion_area",
        "tasa_global_participacion_nacional",
        "tasa_desempleo_area",
        "tasa_desempleo_nacional",
        "tasa_ocupacion_area",
        "tasa_ocupacion_nacional"
    ]
    desc = df[cols].describe().T.reset_index()
    desc.rename(columns={"index": "variable"}, inplace=True)
    return desc.round(3)


def matriz_corr(df):
    cols = [
        "tasa_global_participacion_area",
        "tasa_global_participacion_nacional",
        "tasa_desempleo_area",
        "tasa_desempleo_nacional",
        "tasa_ocupacion_area",
        "tasa_ocupacion_nacional"
    ]
    return df[cols].corr().round(3)


def prueba_adf(df):
    cols = [
        "tasa_global_participacion_area",
        "tasa_global_participacion_nacional",
        "tasa_desempleo_area",
        "tasa_desempleo_nacional",
        "tasa_ocupacion_area",
        "tasa_ocupacion_nacional"
    ]
    resultados = []
    for c in cols:
        pval = adfuller(df[c].dropna())[1]
        resultados.append({
            "variable": c,
            "p_value": round(pval, 4),
            "estacionaria": "Sí" if pval < 0.05 else "No"
        })
    return pd.DataFrame(resultados)


def descomposicion_desempleo(df):
    serie = (
        df.set_index("fecha")["tasa_desempleo_nacional"]
        .sort_index()
        .resample("MS")
        .mean()
        .interpolate(method="linear")
        .dropna()
    )

    # Si hay valores <= 0, multiplicative puede fallar, así que usamos additive
    modelo = "multiplicative" if (serie > 0).all() else "additive"

    return seasonal_decompose(serie, model=modelo, period=12)


def acf_pacf_data(df, nlags=36):
    serie = df["tasa_desempleo_nacional"].dropna().values
    return acf(serie, nlags=nlags), pacf(serie, nlags=nlags)