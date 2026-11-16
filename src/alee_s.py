"""ALEE-S: agregación local entrópica de expertos anclada en SARIMA.

Ronda S del protocolo (``experiments/protocolo.md``, sección 16). Usa el mismo
mecanismo de combinación de ALEE (``src/original_model.py``): pesos locales por
estado, regularización de Kullback-Leibler hacia un previo ``π`` y fuerza de la
regularización adaptativa al tamaño efectivo de la muestra local. La diferencia
es el conjunto de expertos, que incluye a SARIMA(0,1,1)(0,1,1)[12], y el previo,
que puede anclarse en SARIMA:

    π_ancla = (a, (1 - a)/(K - 1), ..., (1 - a)/(K - 1)),   a = peso de SARIMA.

Con ``λ → ∞`` el modelo es SARIMA; con ``λ`` pequeño se aparta de SARIMA solo
donde los errores locales de los otros expertos son menores.

Historial de errores
--------------------
El error del experto ``k`` en la fila ``s`` es ``e_{s,k} = y_s - f_k^{(s)}``,
con ``f_k^{(s)}`` calculado solo con información anterior a ``y_s``:

* persistencia, ``nucleo_cambio`` y estacional: ``_Experts`` ajustado con las
  filas ``0..s-1`` (igual que ALEE-3);
* SARIMA: pronóstico a un paso en ventana expansiva; los parámetros se
  reestiman cada ``refit_every`` filas con los datos disponibles hasta ese
  momento y, entre reestimaciones, se extienden sin reajustar.
"""
import warnings

import numpy as np

from .original_model import ModeloOriginal, _Experts

ALL_EXPERTS = ("sarima", "persistencia", "nucleo_cambio", "estacional")
HIST_START = 24  # primera fila con historial de errores (~3 años de serie)


def sarima_walkforward(series, first_row, n_rows, start=HIST_START, refit_every=12):
    """Pronóstico a un paso de SARIMA para cada fila ``r >= start``.

    La fila ``r`` pronostica el mes ``r + first_row + 1`` de ``series`` con los
    meses ``0..r + first_row``.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    series = np.asarray(series, float)
    pred = np.full(n_rows, np.nan)
    for b0 in range(start, n_rows, refit_every):
        b1 = min(b0 + refit_every, n_rows)
        hist = series[: b0 + first_row + 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = SARIMAX(hist, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False, maxiter=200)
            ext = fit.append(series[b0 + first_row + 1: b1 - 1 + first_row + 1], refit=False) if b1 - 1 > b0 else fit
            p = ext.get_prediction(start=b0 + first_row + 1, end=b1 - 1 + first_row + 1, dynamic=False).predicted_mean
        pred[b0:b1] = np.asarray(p)
    return pred


def expert_history(X, y, cols, sarima_pred, start=HIST_START):
    """Predicciones ``F`` (n, 4) y errores ``E`` de los cuatro expertos, filas ``>= start``."""
    n = len(y)
    F = np.full((n, len(ALL_EXPERTS)), np.nan)
    F[:, 0] = sarima_pred
    for s in range(start, n):
        F[s, 1:] = _Experts(X[:s], y[:s], cols).predict(X[s:s + 1])[0]
    return F, y[:, None] - F


class ALEES:
    """Combinación ALEE de un subconjunto de ``ALL_EXPERTS``.

    Parameters
    ----------
    experts : tuple of str
        Subconjunto de ``ALL_EXPERTS``; si incluye ``"sarima"`` debe ir primero.
    prior : {"ancla", "global"}
        ``"ancla"``: previo fijo con peso ``anchor`` en SARIMA. ``"global"``:
        la combinación global en el símplex (como ALEE-3).
    anchor : float
        Peso de SARIMA en el previo anclado.
    bandwidth, half_life, lam : float
        Hiperparámetros de ALEE.
    feature_names : tuple of str
        Columnas de ``X``.
    """

    def __init__(self, experts=ALL_EXPERTS, prior="ancla", anchor=0.94, bandwidth=1.0, half_life=60.0,
                 lam=0.1, feature_names=None):
        self.experts = tuple(experts)
        self.sel = [ALL_EXPERTS.index(e) for e in self.experts]
        self.prior = prior
        self.anchor = anchor
        self.params = dict(bandwidth=bandwidth, half_life=half_life, lam=lam)
        self.feature_names = feature_names

    def fit(self, X, y, n_train, E_hist, start=HIST_START):
        """Usa el historial de errores de las filas ``m0..n_train-1``."""
        m = ModeloOriginal(feature_names=tuple(self.feature_names), **self.params)
        m._validate_params()
        m0 = int(min(m.min_train, max(start, n_train // 2)))
        if n_train - m0 < 6:
            raise ValueError("historial demasiado corto")
        m.cols_ = {"lag1": m._col(m.lag_col), "mes_sin": m._col("mes_sin"), "mes_cos": m._col("mes_cos")}
        m.errors_ = E_hist[m0:n_train][:, self.sel]
        m.oof_rows_ = np.arange(m0, n_train)
        st = m._state(X[:n_train])
        m.state_mu_, m.state_sd_ = st.mean(0), st.std(0)
        m.state_sd_[m.state_sd_ < 1e-12] = 1.0
        m.state_ = (st[m.oof_rows_] - m.state_mu_) / m.state_sd_
        g = 0.0 if np.isinf(m.half_life) else np.log(0.5) / m.half_life
        m.log_decay_ = g * (n_train - 1 - m.oof_rows_)
        m.n_features_in_ = X.shape[1]
        m.prior_ = m._global_prior()  # también fija n_eff_ref_
        if self.prior == "ancla":
            K = len(self.sel)
            p = np.full(K, (1 - self.anchor) / (K - 1))
            p[0] = self.anchor
            m.prior_ = p
        self.model_ = m
        self.experts_ = _Experts(X[:n_train], y[:n_train], m.cols_)
        return self

    def own_expert_predictions(self, X, sarima_pred):
        """Matriz (n, 4) de los cuatro expertos para filas nuevas, con el SARIMA dado."""
        return np.column_stack([sarima_pred, self.experts_.predict(X)])

    def weights(self, X):
        return self.model_.combination_weights(X)

    def predict(self, X, F_all):
        """``F_all``: (n, 4) pronósticos de los cuatro expertos en el orden de ``ALL_EXPERTS``."""
        W = self.weights(X)
        return (W * F_all[:, self.sel]).sum(axis=1)
