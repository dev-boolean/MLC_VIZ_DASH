"""Estadística descriptiva de las series (sin modelo)."""
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats as st

from src.explain import acf

from . import engine as E


def clip(c, years=None) -> pd.Series:
    s = E.series(c)
    if years:
        s = s[(s.index.year >= years[0]) & (s.index.year <= years[1])]
    return s


def seasonality(s: pd.Series) -> pd.DataFrame:
    """Cambio mensual medio por mes del año, con intervalo de confianza t del 95 %."""
    d = s.diff().dropna()
    rows = []
    for m in range(1, 13):
        v = d[d.index.month == m]
        n = len(v)
        if n < 3:
            continue
        h = st.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
        rows.append({"mes": m, "media": v.mean(), "h": h, "n": n, "sube": (v > 0).mean()})
    return pd.DataFrame(rows)


def anomalies(s: pd.Series):
    """Cambios mensuales fuera de las vallas externas de Tukey (3 rangos intercuartílicos), como en el paper."""
    d = s.diff().dropna()
    q1, q3 = d.quantile([0.25, 0.75])
    lo, hi = q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1)
    return d, lo, hi, d[(d < lo) | (d > hi)]


def trend(s: pd.Series) -> pd.Series:
    """Media móvil centrada de 12 meses (2×12), la tendencia de la descomposición clásica."""
    return s.rolling(12).mean().rolling(2).mean().shift(-6)


def runs(s: pd.Series):
    """Tramos de la tendencia: mayor subida y mayor bajada entre extremos consecutivos."""
    t = trend(s).dropna()
    if len(t) < 24:
        return None, None
    v = t.values
    best_up = best_dn = (0.0, 0, 0)
    i_min = i_max = 0
    for j in range(1, len(v)):
        if v[j] - v[i_min] > best_up[0]:
            best_up = (v[j] - v[i_min], i_min, j)
        if v[i_max] - v[j] > best_dn[0]:
            best_dn = (v[i_max] - v[j], i_max, j)
        if v[j] < v[i_min]:
            i_min = j
        if v[j] > v[i_max]:
            i_max = j
    up = (best_up[0], t.index[best_up[1]], t.index[best_up[2]])
    dn = (best_dn[0], t.index[best_dn[1]], t.index[best_dn[2]])
    return up, dn


def seasonal_strength(s: pd.Series) -> float:
    """Proporción de la varianza del cambio mensual que explica el mes del año (R² de medias por mes)."""
    d = s.diff().dropna()
    fit = d.groupby(d.index.month).transform("mean")
    return float(1 - ((d - fit) ** 2).sum() / ((d - d.mean()) ** 2).sum())


def describe(v: pd.Series) -> dict:
    return {"Media": v.mean(), "Mediana": v.median(), "Desviación": v.std(), "Mínimo": v.min(), "Máximo": v.max(),
            "Q1": v.quantile(0.25), "Q3": v.quantile(0.75), "Asimetría": v.skew()}


@lru_cache(maxsize=None)
def _stationarity(c, years):
    import warnings

    from statsmodels.tsa.stattools import adfuller, kpss

    s = clip(c, years)
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, x in [("Nivel", s), ("Primera diferencia", s.diff().dropna()), ("Diferencia de 12 meses", s.diff(12).dropna())]:
            a, k = adfuller(x, autolag="AIC")[1], kpss(x, regression="c", nlags="auto")[1]
            verdict = ("Estacionaria" if a < 0.05 and k > 0.05 else "No estacionaria" if a >= 0.05 and k <= 0.05 else "Evidencia mixta")
            out.append((name, a, k, verdict))
    return out


def stationarity(c, years=None):
    return _stationarity(c, tuple(years) if years else None)


def acf_at(s, k):
    return float(acf(s.values, k)[k])
