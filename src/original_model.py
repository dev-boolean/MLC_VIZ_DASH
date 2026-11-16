"""Modelo original: agregación local entrópica de expertos (ALEE).

Motivación
----------
En el segundo entregable ningún modelo es el mejor en todos los momentos de la
serie. Con las 240 predicciones fuera de muestra (experimento piloto de
``docs/modelo_original_diseno.md``): XGBoost es el mejor cuando la volatilidad
reciente es baja, Ridge cuando es media y la persistencia (repetir el último
dato) cuando es alta, con un error cuadrático medio cuatro veces menor que el
de XGBoost en ese régimen. Una regla de combinación con pesos fijos no puede
aprovechar ese cambio: los pesos tienen que depender del estado en que está
el mercado laboral cuando se hace el pronóstico.

Nota de alcance (iteración 4): esa comparación original nunca incluyó a
SARIMA ni al experto estacional aislado. Repetida con ellos, ambos dominan
sobre los tres terciles de volatilidad de forma mucho más estable que
XGBoost/Ridge/persistencia (que sí siguen turnándose el primer lugar entre
ellos). La hipótesis de "pesos que cambian porque el mejor modelo cambia"
sigue siendo cierta entre los modelos de aprendizaje automático genéricos,
pero no entre las referencias de series de tiempo bien especificadas -- esto
no invalida la motivación de ALEE, pero acota frente a qué conviene medir su
ventaja.

Algoritmo
---------
Sea ``f_k(x)`` el pronóstico del experto ``k`` (``k = 1..K``) y
``e_{s,k} = y_s - f_k^{(s)}(x_s)`` su error *fuera de muestra* en el mes ``s``
(el experto se entrena solo con los meses anteriores a ``s``). Para predecir en
un punto ``x`` con estado ``z(x)``:

1. Pesos de localización de los meses pasados::

       a_s(x) ∝ exp(-||z(x) - z_s||² / (2 h²)) · γ^(n - s),   Σ_s a_s = 1

2. Matriz local de errores ``Q(x) = Σ_s a_s(x) e_s e_sᵀ``. Como los pesos
   ``w`` suman uno, el error cuadrático local de la combinación es
   ``Σ_s a_s (y_s - wᵀ f_s)² = wᵀ Q w``.
3. Pesos de la combinación::

       w*(x) = argmin_{w ∈ Δ_K}  wᵀ Q(x) w + λ̃ · KL(w || π),   λ̃ = λ · tr(Q) / K

   donde ``π`` son los pesos globales: la misma combinación restringida al
   símplex, pero con todos los meses (solo con decaimiento temporal). El
   término de Kullback-Leibler encoge los pesos locales hacia los globales
   cuando la información local es escasa; escalarlo por ``tr(Q)/K`` (el error
   cuadrático local medio de los expertos) hace que ``λ`` no dependa de las
   unidades de la variable. Con ``shrink_to="uniforme"``, ``π`` es el promedio
   simple (primera versión, iteración 1 del registro de desarrollo). Además
   (iteración 3), ``λ`` se multiplica por ``n_ref / n_eff(a)``, con
   ``n_eff(a) = 1 / Σ a_s²`` el tamaño efectivo de la muestra local: cuantos
   menos meses parecidos haya, más se encoge hacia ``π``. Como ``λ̃`` es una
   constante para cada punto ``x``, las propiedades P1 a P4 no cambian.
4. Se resuelve con descenso de espejo proximal (gradiente exponenciado) con
   paso ``η = 1 / L``, ``L = 2 max_ij |Q_ij|``::

       w_{t+1} ∝ (w_t ⊙ exp(-η ∇g(w_t)))^(1/(1+ηλ̃)) ⊙ π^(ηλ̃/(1+ηλ̃)),   ∇g = 2 Q w

5. Pronóstico: ``ŷ(x) = Σ_k w*_k(x) f_k(x)``.

Propiedades (demostradas en el capítulo 5 del libro y verificadas en
``tests/test_original_model.py``):

* P1. Con ``λ > 0`` el objetivo es estrictamente convexo en el símplex y
  ``w*`` es único.
* P2. Con ``η = 1/L`` cada iteración no aumenta el objetivo (descenso
  monótono), por la suavidad relativa de la parte cuadrática respecto a la
  entropía (desigualdad de Pinsker).
* P3. El pronóstico es una combinación convexa de los expertos: su error
  absoluto no supera el del peor experto y su error cuadrático no supera el
  promedio ponderado de los errores cuadráticos de los expertos.
* P4. Casos límite: ``λ → ∞`` da el previo ``π`` (los pesos globales, o el
  promedio simple con ``shrink_to="uniforme"``); sin localización, los pesos
  son los mismos para todo ``x``.

Expertos (implementados con NumPy, sin modelos de librerías)
-----------------------------------------------------------
* ``persistencia``: ``ŷ = y_t``.
* ``nucleo_cambio``: ``y_t`` más una regresión de Nadaraya-Watson del cambio.
* ``estacional``: ``y_t`` más el cambio medio histórico del mismo mes del año.

Nota (iteración 4, 2026-11-05, exploratoria, posterior a la validación
confirmatoria): hasta la iteración 3 había 5 expertos; se sumaban
``ridge_nivel`` (regresión ridge del nivel) y ``ridge_cambio`` (regresión
ridge del cambio). Se quitaron porque, evaluados en solitario sobre los 240
meses fuera de muestra del problema principal, cada uno comete un RMSE de
~1.6 -- más del doble del experto estacional (~0.78) -- y su presencia solo
diluía el peso del experto estacional en la combinación, incluso cuando el
previo global ``π`` ya lo favorecía. Con los 3 expertos restantes, ALEE pasa
de perder frente al estacional aislado de forma estadísticamente
significativa (Diebold-Mariano, p=0.0044) a empatar con él y con SARIMA
(p=0.14 y p=0.30). El detalle completo está en
``docs/registro_desarrollo.md`` (iteración 4) y ``experiments/protocolo.md``
(sección 11).

Supuesto: las filas de ``X`` están en orden cronológico y la fila ``s`` solo
contiene información disponible antes de conocer ``y_s`` (pronóstico a un
paso). Así se construyen los datos en ``src/data.py``.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .data import AREA_COLS, LAGS, ROLLING_WINDOWS

DEFAULT_FEATURES = tuple(
    [f"td_nacional_lag{lag}" for lag in LAGS]
    + [f"td_nacional_roll{stat}{w}" for w in ROLLING_WINDOWS for stat in ("mean", "std")]
    + ["mes_sin", "mes_cos", "trend_idx"]
    + AREA_COLS
)
EXPERTS = ("persistencia", "nucleo_cambio", "estacional")
# Iteración 4 (exploratoria, 2026-11-05, posterior a la validación confirmatoria):
# se quitaron ridge_nivel y ridge_cambio. En solitario, sobre los 240 meses
# fuera de muestra del problema principal, su RMSE individual es ~1.6 (ridge_nivel)
# y ~1.6 (ridge_cambio) frente a 0.78 del experto estacional -- más del doble de
# error -- y su presencia diluye el peso del experto estacional en la combinación
# incluso cuando el previo global ya lo favorece. Ver docs/registro_desarrollo.md
# (iteración 4) y experiments/protocolo.md (sección 11) para el detalle y las
# cifras de antes/después.
# Regularización del previo global hacia el uniforme y piso de sus pesos: solo
# garantizan que el previo esté en el interior del símplex.
PRIOR_LAM = 1e-2
PRIOR_FLOOR = 1e-6


# ------------------------------------------------------------------ expertos
def _standardize_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return mu, sd


def _ridge_fit(Z, t, alpha):
    """Ridge con intercepto sin penalizar, resuelto por ecuaciones normales."""
    n, p = Z.shape
    A = np.hstack([np.ones((n, 1)), Z])
    P = alpha * np.eye(p + 1)
    P[0, 0] = 0.0
    return np.linalg.solve(A.T @ A + P + 1e-10 * np.eye(p + 1), A.T @ t)


def _ridge_predict(beta, Z):
    return beta[0] + Z @ beta[1:]


def _month_of(sin_v, cos_v):
    """Recupera el mes (1..12) a partir de su codificación cíclica."""
    angle = np.arctan2(sin_v, cos_v)
    return (np.rint(angle * 12 / (2 * np.pi)).astype(int) - 1) % 12 + 1


class _Experts:
    """Los tres expertos ajustados con un tramo de datos.

    Hasta la iteración 3 había 5 expertos (se sumaban ``ridge_nivel`` y
    ``ridge_cambio``). La iteración 4 los quitó: cada uno, en solitario,
    comete sobre los 240 meses fuera de muestra un RMSE de ~1.6 -- más del
    doble del experto estacional (~0.78) -- y su presencia solo diluía el
    peso del experto estacional en la combinación. ``_ridge_fit``/
    ``_ridge_predict`` se conservan en el módulo porque los usan las pruebas
    de regresión por ecuaciones normales, no porque los use ya algún experto.

    Parameters
    ----------
    X, y : ndarray
        Variables y objetivo de entrenamiento.
    cols : dict
        Posición de las columnas con un papel especial (``lag1``, ``mes_sin``, ``mes_cos``).
    """

    def __init__(self, X, y, cols):
        self.cols = cols
        lag1 = X[:, cols["lag1"]]
        self.mu, self.sd = _standardize_fit(X)
        Z = (X - self.mu) / self.sd
        dy = y - lag1
        # Núcleo gaussiano con ancho igual a la mediana de las distancias (heurística de la mediana).
        self.Z, self.dy = Z, dy
        if len(Z) > 1:
            d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
            med = np.median(d2[np.triu_indices(len(Z), 1)])
            self.bw2 = med if med > 0 else 1.0
        else:
            self.bw2 = 1.0
        # Cambio medio por mes del año, encogido hacia cero con una observación ficticia.
        months = _month_of(X[:, cols["mes_sin"]], X[:, cols["mes_cos"]])
        self.month_change = np.zeros(13)
        for m in range(1, 13):
            sel = months == m
            self.month_change[m] = dy[sel].sum() / (sel.sum() + 1.0)

    def predict(self, X):
        """Matriz ``(n, K)`` con el pronóstico de cada experto."""
        lag1 = X[:, self.cols["lag1"]]
        Z = (X - self.mu) / self.sd
        d2 = ((Z[:, None, :] - self.Z[None, :, :]) ** 2).sum(-1)
        k = np.exp(-(d2 - d2.min(axis=1, keepdims=True)) / (2 * self.bw2))
        nw = (k * self.dy).sum(axis=1) / k.sum(axis=1)
        months = _month_of(X[:, self.cols["mes_sin"]], X[:, self.cols["mes_cos"]])
        return np.column_stack([
            lag1,
            lag1 + nw,
            lag1 + self.month_change[months],
        ])


# ------------------------------------------------------------ combinación
def solve_weights(Q, lam, w0=None, max_iter=500, tol=1e-10, return_history=False, prior=None):
    """Pesos de la combinación: minimiza ``wᵀQw + λ̃ KL(w||π)`` en el símplex.

    Descenso de espejo proximal con paso ``1/L``, ``L = 2 max|Q_ij|``.

    Parameters
    ----------
    Q : ndarray (K, K)
        Matriz local (simétrica, semidefinida positiva) de errores.
    lam : float
        Regularización ``λ`` (sin escalar); se usa ``λ̃ = λ tr(Q) / K``.
    w0 : ndarray, optional
        Punto de partida (por defecto, el previo ``π``).
    max_iter : int
        Máximo de iteraciones.
    tol : float
        Se detiene cuando ``||w_{t+1} - w_t||_1 < tol``.
    return_history : bool
        Si se devuelve el valor del objetivo en cada iteración.
    prior : ndarray, optional
        Previo ``π`` del término de Kullback-Leibler, en el interior del
        símplex (por defecto, el uniforme ``u``).

    Returns
    -------
    ndarray or (ndarray, list)
        Pesos óptimos (y la historia del objetivo).
    """
    K = Q.shape[0]
    u = np.full(K, 1.0 / K) if prior is None else np.asarray(prior, dtype=float)
    lam_t = lam * np.trace(Q) / K
    L = 2.0 * np.abs(Q).max()
    w = u.copy() if w0 is None else np.asarray(w0, dtype=float)

    def objective(v):
        kl = np.sum(v * (np.log(np.maximum(v, 1e-300)) - np.log(u)))
        return float(v @ Q @ v + lam_t * kl)

    history = [objective(w)]
    if L <= 0:  # todos los expertos sin error local: cualquier w es óptimo, se deja el uniforme
        return (w, history) if return_history else w
    eta = 1.0 / L
    a = 1.0 / (1.0 + eta * lam_t)
    for _ in range(max_iter):
        grad = 2.0 * Q @ w
        logits = a * (np.log(np.maximum(w, 1e-300)) - eta * grad) + (1 - a) * np.log(u)
        logits -= logits.max()
        w_new = np.exp(logits)
        w_new /= w_new.sum()
        done = np.abs(w_new - w).sum() < tol
        w = w_new
        if return_history:
            history.append(objective(w))
        if done:
            break
    return (w, history) if return_history else w


class ModeloOriginal(BaseEstimator, RegressorMixin):
    """Agregación local entrópica de expertos (ALEE) para pronóstico a un paso.

    Parameters
    ----------
    bandwidth : float
        Ancho ``h`` del núcleo gaussiano sobre el estado estandarizado.
    half_life : float
        Vida media (en meses) del decaimiento temporal: ``γ = 0.5 ** (1 / half_life)``.
        ``np.inf`` desactiva el decaimiento.
    lam : float
        Fuerza de la regularización entrópica hacia el previo ``π``.
    shrink_to : {"global", "uniforme"}
        Previo ``π`` de la regularización: los pesos globales (la combinación
        restringida al símplex con todos los meses, solo con decaimiento
        temporal) o el promedio simple. Con ``"global"``, la localización solo
        se aparta de la mejor combinación global cuando la evidencia local lo
        justifica.
    min_train : int
        Meses mínimos para empezar a registrar errores fuera de muestra de los expertos.
    feature_names : tuple of str
        Nombres de las columnas de ``X``, en orden.
    lag_col, month_cols, state_cols : str, tuple
        Columnas con papel especial: último valor observado, mes codificado y
        variables de estado. El estado incluye además la brecha
        ``y_t - media móvil de 12 meses`` (``gap_cols``).
    gap_cols : tuple of str or None
        ``(último valor, media móvil)`` para la brecha; ``None`` para omitirla.
    localize : bool
        Si es ``False``, todos los meses pasados pesan igual (ablación).
    adaptive_lam : bool
        Si la regularización crece cuando la muestra local efectiva es pequeña
        (ver ``effective_lam``). ``False`` reproduce la iteración 2.
    max_iter : int
        Iteraciones máximas del descenso de espejo.
    random_state : int
        Sin efecto (el algoritmo es determinista); se conserva por compatibilidad.
    """

    def __init__(
        self,
        bandwidth=1.0,
        half_life=60.0,
        lam=0.1,
        shrink_to="global",
        min_train=36,
        feature_names=DEFAULT_FEATURES,
        lag_col="td_nacional_lag1",
        month_cols=("mes_sin", "mes_cos"),
        state_cols=("td_nacional_rollstd3",),
        gap_cols=("td_nacional_lag1", "td_nacional_rollmean12"),
        localize=True,
        adaptive_lam=True,
        max_iter=500,
        random_state=42,
    ):
        self.bandwidth = bandwidth
        self.half_life = half_life
        self.lam = lam
        self.shrink_to = shrink_to
        self.min_train = min_train
        self.feature_names = feature_names
        self.lag_col = lag_col
        self.month_cols = month_cols
        self.state_cols = state_cols
        self.gap_cols = gap_cols
        self.localize = localize
        self.adaptive_lam = adaptive_lam
        self.max_iter = max_iter
        self.random_state = random_state

    # -------------------------------------------------------------- utilidades
    def _validate_params(self):
        if not (self.bandwidth > 0):
            raise ValueError("bandwidth debe ser positivo")
        if not (self.half_life > 0):
            raise ValueError("half_life debe ser positivo (np.inf para no decaer)")
        if not (self.lam >= 0):
            raise ValueError("lam debe ser no negativo")
        if int(self.min_train) < 2:
            raise ValueError("min_train debe ser al menos 2")
        if self.shrink_to not in ("global", "uniforme"):
            raise ValueError('shrink_to debe ser "global" o "uniforme"')

    def _col(self, name):
        try:
            return list(self.feature_names).index(name)
        except ValueError:
            raise ValueError(f"La columna {name!r} no está en feature_names") from None

    def _state(self, X):
        parts = [X[:, self._col(c)] for c in self.state_cols]
        if self.gap_cols is not None:
            parts.append(X[:, self._col(self.gap_cols[0])] - X[:, self._col(self.gap_cols[1])])
        return np.column_stack(parts)

    # ----------------------------------------------------------- aprendizaje
    def fit(self, X, y):
        """Registra los errores fuera de muestra de los expertos y ajusta los expertos finales.

        Para cada mes ``s`` desde ``m0`` hasta ``n - 1`` los expertos se ajustan
        con los meses ``0..s-1`` y pronostican el mes ``s`` (ventana expansiva).

        Parameters
        ----------
        X : array-like (n, p)
            Variables en orden cronológico, con las columnas de ``feature_names``.
        y : array-like (n,)
            Objetivo.

        Returns
        -------
        self
        """
        self._validate_params()
        X, y = check_X_y(X, y, dtype=float, y_numeric=True)
        if X.shape[1] != len(self.feature_names):
            raise ValueError(f"X tiene {X.shape[1]} columnas y feature_names {len(self.feature_names)}")
        n = len(y)
        self.cols_ = {"lag1": self._col(self.lag_col), "mes_sin": self._col(self.month_cols[0]),
                      "mes_cos": self._col(self.month_cols[1])}
        # Arranque: suficientes meses para los expertos y para tener errores que localizar.
        m0 = int(min(self.min_train, max(2, n // 2)))
        errors, rows = [], []
        for s in range(m0, n):
            experts = _Experts(X[:s], y[:s], self.cols_)
            errors.append(y[s] - experts.predict(X[s:s + 1])[0])
            rows.append(s)
        self.errors_ = np.asarray(errors)            # (n - m0, K)
        self.oof_rows_ = np.asarray(rows)
        state = self._state(X)
        self.state_mu_, self.state_sd_ = _standardize_fit(state)
        self.state_ = (state[self.oof_rows_] - self.state_mu_) / self.state_sd_
        gamma = 0.0 if np.isinf(self.half_life) else np.log(0.5) / self.half_life
        self.log_decay_ = gamma * (n - 1 - self.oof_rows_)   # log γ^(n-1-s)
        self.experts_ = _Experts(X, y, self.cols_)
        self.n_features_in_ = X.shape[1]
        self.expert_names_ = EXPERTS
        self.prior_ = self._global_prior()
        return self

    def _global_prior(self):
        """Previo ``π`` de la regularización.

        Con ``shrink_to="global"`` es la combinación restringida al símplex que
        minimiza el error cuadrático de todos los meses registrados (solo con
        el decaimiento temporal), con una regularización mínima hacia el
        uniforme (``PRIOR_LAM``) para que quede en el interior del símplex,
        donde el término de Kullback-Leibler está definido.
        """
        K = self.errors_.shape[1]
        a = np.exp(self.log_decay_ - self.log_decay_.max())
        a /= a.sum()
        self.n_eff_ref_ = 1.0 / np.sum(a ** 2)
        if self.shrink_to == "uniforme":
            return np.full(K, 1.0 / K)
        Q = (self.errors_ * a[:, None]).T @ self.errors_
        prior = solve_weights(Q, PRIOR_LAM, max_iter=5000)
        prior = np.maximum(prior, PRIOR_FLOOR)
        return prior / prior.sum()

    def local_weights(self, X):
        """Pesos ``a_s(x)`` de los meses pasados para cada fila de ``X`` (n_x, n_oof)."""
        check_is_fitted(self, "errors_")
        X = check_array(X, dtype=float)
        if not self.localize:
            logk = np.zeros((len(X), len(self.oof_rows_)))
        else:
            z = (self._state(X) - self.state_mu_) / self.state_sd_
            d2 = ((z[:, None, :] - self.state_[None, :, :]) ** 2).sum(-1)
            logk = -d2 / (2.0 * self.bandwidth ** 2)
        logk = logk + self.log_decay_[None, :]
        logk -= logk.max(axis=1, keepdims=True)
        a = np.exp(logk)
        return a / a.sum(axis=1, keepdims=True)

    def combination_weights(self, X):
        """Pesos ``w*(x)`` de los cinco expertos para cada fila de ``X`` (n_x, K)."""
        A = self.local_weights(X)
        E = self.errors_
        W = np.empty((len(A), E.shape[1]))
        for i, a in enumerate(A):
            Q = (E * a[:, None]).T @ E
            W[i] = solve_weights(Q, self.effective_lam(a), w0=self.prior_, max_iter=self.max_iter, prior=self.prior_)
        return W

    def effective_lam(self, a):
        """Regularización para unos pesos locales ``a``.

        Con ``adaptive_lam=True`` se multiplica ``λ`` por ``n_ref / n_eff(a)``,
        donde ``n_eff(a) = 1 / Σ a_s²`` es el tamaño efectivo de la muestra
        local (de Kish) y ``n_ref`` el de los pesos solo con decaimiento. La
        varianza de ``Q`` como estimador del error local crece como
        ``1 / n_eff``: cuando hay pocos meses parecidos (como en el choque de
        2020) la combinación se queda más cerca de los pesos globales.
        """
        if not self.adaptive_lam:
            return self.lam
        n_eff = 1.0 / np.sum(np.asarray(a) ** 2)
        return self.lam * self.n_eff_ref_ / n_eff

    def expert_predictions(self, X):
        """Pronóstico de cada experto final (n_x, K)."""
        check_is_fitted(self, "experts_")
        return self.experts_.predict(check_array(X, dtype=float))

    def predict(self, X):
        """Pronóstico combinado ``Σ_k w*_k(x) f_k(x)``."""
        check_is_fitted(self, "errors_")
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X tiene {X.shape[1]} columnas; el modelo se ajustó con {self.n_features_in_}")
        return (self.combination_weights(X) * self.expert_predictions(X)).sum(axis=1)
