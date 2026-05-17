"""
model.py — Lógica SARIMA(2,1,2)(1,1,1)[12]
Portado fielmente del server.R del Shiny original.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats


# ── Constantes ────────────────────────────────────────────────────────────────
FECHA_INICIO_TS = "2001-01-01"
N_TOTAL = 300          # observaciones completas 2001-2025
N_TRAIN = 270          # hasta jun 2023 (igual que Shiny)


def _build_ts(df: pd.DataFrame) -> pd.Series:
    """Construye la serie temporal mensual de tasa_desempleo_nacional."""
    serie = (
        df.set_index("fecha")["tasa_desempleo_nacional"]
        .sort_index()
        .resample("MS")
        .mean()
        .interpolate(method="linear")
        .dropna()
    )
    return serie


def fit_modelo_final(df: pd.DataFrame):
    """
    Ajusta SARIMA(2,1,2)(1,1,1)[12] sobre la serie completa (igual que Shiny).
    Retorna el modelo ajustado de statsmodels.
    """
    serie = _build_ts(df)
    modelo = ARIMA(
        serie,
        order=(2, 1, 2),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return modelo.fit()


def comparar_modelos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replica la tabla de comparación del Shiny: 5 candidatos,
    split train/test en obs 270/30, métricas AIC, BIC, RMSE.
    """
    serie = _build_ts(df)

    # Usar los primeros N_TRAIN puntos como entrenamiento, el resto como test
    n = len(serie)
    n_train = min(N_TRAIN, n - 6)   # al menos 6 pasos de test
    train = serie.iloc[:n_train]
    test  = serie.iloc[n_train:]

    candidatos = {
        "ARIMA(1,1,1)": dict(order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)),
        "SARIMA(0,1,1)(0,1,1)[12]": dict(order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)),
        "SARIMA(1,1,1)(0,1,1)[12]": dict(order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)),
        "SARIMA(1,1,1)(1,1,1)[12]": dict(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)),
        "SARIMA(2,1,2)(1,1,1)[12]": dict(order=(2, 1, 2), seasonal_order=(1, 1, 1, 12)),
    }

    filas = []
    for nombre, params in candidatos.items():
        try:
            m = ARIMA(
                train,
                order=params["order"],
                seasonal_order=params["seasonal_order"],
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit()

            fc = m.forecast(steps=len(test))
            rmse = float(np.sqrt(np.mean((fc.values - test.values) ** 2)))

            filas.append({
                "Modelo": nombre,
                "AIC": round(m.aic, 2),
                "BIC": round(m.bic, 2),
                "RMSE": round(rmse, 3),
            })
        except Exception:
            filas.append({"Modelo": nombre, "AIC": np.nan, "BIC": np.nan, "RMSE": np.nan})

    return pd.DataFrame(filas)


def metricas_modelo(resultado) -> dict:
    """
    Calcula RMSE, MAE, MAPE y AIC del modelo ajustado.
    (replica accuracy() de R forecast)
    """
    obs   = resultado.model.endog
    ajust = resultado.fittedvalues.values
    resid = obs - ajust

    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae  = float(np.mean(np.abs(resid)))
    mape = float(np.mean(np.abs(resid / obs)) * 100)
    aic  = round(resultado.aic, 1)

    return {"RMSE": round(rmse, 3), "MAE": round(mae, 3), "MAPE": round(mape, 2), "AIC": aic}


def ljung_box_tests(resultado) -> list:
    """Prueba Ljung-Box a rezagos 12 y 24 (replica Box.test de R)."""
    resid = resultado.resid.dropna()
    resultados = []
    for lag in [12, 24]:
        lb = acorr_ljungbox(resid, lags=[lag], return_df=True)
        stat = float(lb["lb_stat"].iloc[0])
        pval = float(lb["lb_pvalue"].iloc[0])
        ok   = pval > 0.05
        resultados.append({
            "lag": lag,
            "estadistico": round(stat, 3),
            "p_valor": round(pval, 4),
            "conclusion": "No se rechaza H₀ (ruido blanco ✔)" if ok else "Se rechaza H₀ ✗",
            "ok": ok,
        })
    return resultados


def datos_residuos(resultado) -> dict:
    """Devuelve arrays de residuos para graficación (QQ-plot, histograma)."""
    resid = resultado.resid.dropna().values
    qq_teorico, qq_muestra = stats.probplot(resid, dist="norm")[:2][0], stats.probplot(resid, dist="norm")[0][1]
    qq_linea_x = [qq_teorico[0], qq_teorico[-1]]
    qq_linea_y = stats.norm.ppf([0.25, 0.75]) * np.std(resid) + np.mean(resid)

    return {
        "residuos": resid,
        "qq_teorico": list(qq_teorico),
        "qq_muestra": list(qq_muestra),
        "qq_linea_x": qq_linea_x,
        "qq_linea_y": list(qq_linea_y),
        "media": float(np.mean(resid)),
        "std": float(np.std(resid)),
    }


def pronostico(resultado, h: int = 12, nivel: str = "both") -> dict:
    """
    Genera el pronóstico a h pasos con intervalos de confianza.
    nivel: 'both' → 80% y 95%, '95' → solo 95%, '80' → solo 80%
    Retorna dict con media, lower80, upper80, lower95, upper95 y fechas.
    """
    alpha_map = {"both": [0.20, 0.05], "80": [0.20], "95": [0.05]}
    alphas = alpha_map.get(nivel, [0.20, 0.05])

    fc = resultado.get_forecast(steps=h)
    media = fc.predicted_mean.values

    # Fecha de inicio del pronóstico: mes siguiente al último dato
    ultimo_idx = resultado.model.data.dates[-1]
    fechas_fc  = pd.date_range(
        start=ultimo_idx + pd.DateOffset(months=1),
        periods=h,
        freq="MS"
    )

    out = {
        "media": media,
        "fechas": fechas_fc,
        "niveles": {},
    }

    for alpha in alphas:
        nivel_pct = int((1 - alpha) * 100)
        ci = fc.conf_int(alpha=alpha)
        out["niveles"][nivel_pct] = {
            "lower": ci.iloc[:, 0].values,
            "upper": ci.iloc[:, 1].values,
        }

    return out


def serie_ajustada(resultado, df: pd.DataFrame) -> tuple:
    """Devuelve (fechas, observado, ajustado) para el gráfico de ajuste."""
    serie  = _build_ts(df)
    fechas = serie.index
    obs    = serie.values
    ajust  = resultado.fittedvalues.values

    # Ajustar longitud si difieren (por NaN al inicio)
    n_min = min(len(fechas), len(obs), len(ajust))
    return fechas[-n_min:], obs[-n_min:], ajust[-n_min:]


def resumen_modelo_texto(resultado) -> str:
    """Retorna el summary como texto (replica summary() de R)."""
    return str(resultado.summary())


def resumen_modelo_estructurado(resultado) -> dict:
    """
    Extrae los datos clave del summary de statsmodels en un dict estructurado
    para renderizar una tarjeta moderna en lugar del html.Pre crudo.
    """
    try:
        params      = resultado.params
        pvalues     = resultado.pvalues
        conf_int    = resultado.conf_int()
        bse         = resultado.bse

        coef_rows = []
        for name in params.index:
            coef_rows.append({
                "nombre":  name,
                "coef":    round(float(params[name]), 4),
                "se":      round(float(bse[name]), 4),
                "pvalue":  round(float(pvalues[name]), 4),
                "ci_low":  round(float(conf_int.loc[name, 0]), 4),
                "ci_high": round(float(conf_int.loc[name, 1]), 4),
                "sig":     (
                    "***" if float(pvalues[name]) < 0.001 else
                    "**"  if float(pvalues[name]) < 0.01  else
                    "*"   if float(pvalues[name]) < 0.05  else
                    "·"   if float(pvalues[name]) < 0.1   else ""
                ),
            })

        info = {
            "aic":    round(float(resultado.aic),  2),
            "bic":    round(float(resultado.bic),  2),
            "llf":    round(float(resultado.llf),  2),
            "nobs":   int(resultado.nobs),
            "orden":  "SARIMA(2,1,2)(1,1,1)[12]",
        }

        return {"coeficientes": coef_rows, "info": info}
    except Exception:
        return None


def tabla_pronostico_df(fc_dict: dict, nivel: str = "both") -> pd.DataFrame:
    """
    Construye el DataFrame de pronósticos para la DataTable
    (replica la tabla del Shiny).
    """
    fechas = fc_dict["fechas"]
    media  = fc_dict["media"]

    df_out = pd.DataFrame({
        "Mes": [f.strftime("%B %Y") for f in fechas],
        "Pronóstico": np.round(media, 3),
    })

    if 95 in fc_dict["niveles"]:
        df_out["IC inf 95%"] = np.round(fc_dict["niveles"][95]["lower"], 3)
        df_out["IC sup 95%"] = np.round(fc_dict["niveles"][95]["upper"], 3)

    if 80 in fc_dict["niveles"]:
        df_out["IC inf 80%"] = np.round(fc_dict["niveles"][80]["lower"], 3)
        df_out["IC sup 80%"] = np.round(fc_dict["niveles"][80]["upper"], 3)

    return df_out
