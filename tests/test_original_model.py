"""Pruebas unitarias del modelo original (``src/original_model.py``).

Cubren los cinco puntos que exige la guía (sección 5.4): aprendizaje en un
problema sintético con solución conocida, reproducibilidad, entradas
inválidas, compatibilidad con ``sklearn.base.clone`` y verificación numérica
de las propiedades teóricas P1 a P4.
"""
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

from src.original_model import EXPERTS, ModeloOriginal, solve_weights

FEATURES = ("lag1", "rollmean12", "rollstd3", "mes_sin", "mes_cos", "x_extra")


def _series(n=200, seed=0, regime_switch=False):
    """Serie sintética mensual con estacionalidad, y sus variables a un paso."""
    rng = np.random.default_rng(seed)
    months = (np.arange(n + 13) % 12) + 1
    season = 0.8 * np.sin(2 * np.pi * months / 12)
    level = np.empty(n + 13)
    level[0] = 10.0
    for t in range(1, n + 13):
        noise_sd = 0.6 if (regime_switch and t > (n + 13) // 2) else 0.15
        level[t] = 10.0 + 0.9 * (level[t - 1] - 10.0) + rng.normal(0, noise_sd)
    serie = level + season
    rows, target = [], []
    for t in range(12, n + 12):
        rows.append([
            serie[t],
            serie[t - 11:t + 1].mean(),
            serie[t - 2:t + 1].std(ddof=1),
            np.sin(2 * np.pi * months[t] / 12),
            np.cos(2 * np.pi * months[t] / 12),
            rng.normal(),
        ])
        target.append(serie[t + 1])
    return np.asarray(rows), np.asarray(target)


def _model(**kw):
    params = dict(feature_names=FEATURES, lag_col="lag1", state_cols=("rollstd3",),
                  gap_cols=("lag1", "rollmean12"), min_train=30)
    params.update(kw)
    return ModeloOriginal(**params)


@pytest.fixture(scope="module")
def datos():
    return _series()


# (i) aprende en un problema con solución conocida ---------------------------
def test_aprende_mejor_que_persistencia(datos):
    X, y = datos
    model = _model().fit(X[:150], y[:150])
    pred = model.predict(X[150:])
    rmse = np.sqrt(np.mean((y[150:] - pred) ** 2))
    rmse_persist = np.sqrt(np.mean((y[150:] - X[150:, 0]) ** 2))
    assert rmse < rmse_persist


def test_los_expertos_exactos_se_llevan_el_peso():
    # Si el objetivo es exactamente y_t (cambio nulo), los tres expertos
    # actuales (persistencia, núcleo-cambio, estacional) son exactos, porque
    # los tres predicen y_t más una corrección de cambio que, entrenada sobre
    # un cambio siempre nulo, también da cero. (Hasta la iteración 3 había un
    # cuarto experto, ridge del nivel, que sí cometía error en este caso -- se
    # quitó en la iteración 4, ver docs/registro_desarrollo.md.) El modelo
    # debe reproducir y_t exactamente sea cual sea el peso que reparta.
    X, _ = _series(seed=3)
    y = X[:, 0].copy()
    model = _model(lam=0.0).fit(X[:150], y[:150])
    np.testing.assert_allclose(model.predict(X[150:]), y[150:], atol=1e-3)


# (ii) reproducibilidad -------------------------------------------------------
def test_reproducible(datos):
    X, y = datos
    p1 = _model(random_state=1).fit(X, y).predict(X)
    p2 = _model(random_state=1).fit(X, y).predict(X)
    np.testing.assert_array_equal(p1, p2)


# (iii) entradas inválidas ----------------------------------------------------
def test_rechaza_nan(datos):
    X, y = datos
    X_bad = X.copy()
    X_bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        _model().fit(X_bad, y)


def test_rechaza_columnas_que_no_coinciden(datos):
    X, y = datos
    with pytest.raises(ValueError):
        _model().fit(X[:, :4], y)
    model = _model().fit(X, y)
    with pytest.raises(ValueError):
        model.predict(X[:, :4])


@pytest.mark.parametrize("bad", [{"bandwidth": 0}, {"half_life": -1}, {"lam": -0.1}, {"min_train": 1}, {"shrink_to": "x"}])
def test_rechaza_hiperparametros_invalidos(datos, bad):
    X, y = datos
    with pytest.raises(ValueError):
        _model(**bad).fit(X, y)


def test_predecir_sin_ajustar_falla(datos):
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        _model().predict(datos[0])


# (iv) compatibilidad con scikit-learn ---------------------------------------
def test_clone_conserva_hiperparametros():
    model = _model(bandwidth=0.7, half_life=24.0, lam=0.3)
    copy = clone(model)
    assert copy.get_params() == model.get_params()
    assert not hasattr(copy, "errors_")


def test_funciona_con_cross_val_score(datos):
    X, y = datos
    scores = cross_val_score(_model(), X, y, cv=TimeSeriesSplit(n_splits=3), scoring="neg_root_mean_squared_error")
    assert np.all(np.isfinite(scores))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_acepta_tipos_numericos_y_dataframes(datos, dtype):
    import pandas as pd

    X, y = datos
    ref = _model().fit(X, y).predict(X)
    np.testing.assert_allclose(_model().fit(X.astype(dtype), y).predict(X.astype(dtype)), ref, rtol=1e-4, atol=1e-4)
    df = pd.DataFrame(X, columns=FEATURES)
    np.testing.assert_allclose(_model().fit(df, pd.Series(y)).predict(df), ref)


# (v) propiedades teóricas ----------------------------------------------------
def _random_Q(K, seed):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(50, K)) * rng.uniform(0.2, 2.0, size=K)
    a = rng.dirichlet(np.ones(50))
    return (E * a[:, None]).T @ E


@pytest.mark.parametrize("seed", range(20))
def test_P2_descenso_monotono(seed):
    Q = _random_Q(5, seed)
    lam = [0.0, 0.05, 0.5, 5.0][seed % 4]
    _, hist = solve_weights(Q, lam, max_iter=300, tol=0.0, return_history=True)
    assert np.all(np.diff(hist) <= 1e-12 * max(1.0, abs(hist[0])))


@pytest.mark.parametrize("seed", range(10))
def test_P1_unicidad_desde_distintos_puntos(seed):
    Q = _random_Q(5, seed)
    rng = np.random.default_rng(seed + 100)
    sols = [solve_weights(Q, 0.2, w0=rng.dirichlet(np.ones(5)), max_iter=20000, tol=1e-14) for _ in range(4)]
    for s in sols[1:]:
        np.testing.assert_allclose(s, sols[0], atol=1e-6)


@pytest.mark.parametrize("seed", range(10))
def test_P1_optimalidad_kkt(seed):
    # En el óptimo interior, ∇F = 2Qw + λ̃(log(w/u) + 1) es constante en todas las coordenadas.
    Q = _random_Q(5, seed)
    lam = 0.3
    w = solve_weights(Q, lam, max_iter=50000, tol=1e-15)
    lam_t = lam * np.trace(Q) / 5
    grad = 2 * Q @ w + lam_t * (np.log(w * 5) + 1)
    assert np.ptp(grad) < 1e-6 * np.abs(grad).max()


def test_pesos_en_el_simplex(datos):
    X, y = datos
    W = _model().fit(X, y).combination_weights(X)
    assert np.all(W >= 0)
    np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-12)


def test_P3_cota_por_el_peor_experto(datos):
    X, y = datos
    model = _model().fit(X[:150], y[:150])
    F = model.expert_predictions(X[150:])
    W = model.combination_weights(X[150:])
    pred = model.predict(X[150:])
    err = np.abs(y[150:] - pred)
    err_k = np.abs(y[150:, None] - F)
    assert np.all(err <= err_k.max(axis=1) + 1e-12)
    assert np.all(err ** 2 <= (W * err_k ** 2).sum(axis=1) + 1e-12)
    assert np.all((pred >= F.min(axis=1) - 1e-12) & (pred <= F.max(axis=1) + 1e-12))


def test_P4_lambda_grande_da_promedio_simple(datos):
    X, y = datos
    W = _model(lam=1e6, shrink_to="uniforme").fit(X, y).combination_weights(X[:20])
    np.testing.assert_allclose(W, 1.0 / len(EXPERTS), atol=1e-3)


def test_P4_lambda_grande_da_los_pesos_globales(datos):
    X, y = datos
    model = _model(lam=1e6).fit(X, y)
    W = model.combination_weights(X[:20])
    assert np.abs(W - model.prior_).max() < 1e-3


def test_previo_global_en_el_interior_y_mejor_que_el_uniforme(datos):
    X, y = datos
    model = _model().fit(X, y)
    assert np.all(model.prior_ > 0) and abs(model.prior_.sum() - 1) < 1e-12
    a = np.exp(model.log_decay_ - model.log_decay_.max())
    a /= a.sum()
    Q = (model.errors_ * a[:, None]).T @ model.errors_
    u = np.full(len(model.prior_), 1 / len(model.prior_))
    assert model.prior_ @ Q @ model.prior_ <= u @ Q @ u


def test_P4_sin_localizacion_ni_decaimiento_pesos_iguales_en_todo_x(datos):
    X, y = datos
    W = _model(localize=False, half_life=np.inf).fit(X, y).combination_weights(X[:30])
    assert np.abs(W - W[0]).max() < 1e-10


def test_localizacion_cambia_los_pesos_entre_regimenes():
    X, y = _series(n=240, seed=5, regime_switch=True)
    # Sin la regularización adaptativa, que a propósito acerca los pesos al previo global.
    model = _model(bandwidth=0.5, half_life=np.inf, adaptive_lam=False).fit(X, y)
    W = model.combination_weights(X)
    vol = X[:, 2]
    calm, rough = W[vol < np.quantile(vol, 0.2)].mean(0), W[vol > np.quantile(vol, 0.8)].mean(0)
    assert np.abs(calm - rough).sum() > 0.05


def test_regularizacion_crece_con_menos_meses_parecidos(datos):
    X, y = datos
    model = _model(lam=0.5).fit(X, y)
    n = len(model.oof_rows_)
    spread = np.full(n, 1.0 / n)
    concentrated = np.zeros(n)
    concentrated[:5] = 0.2
    assert model.effective_lam(concentrated) > model.effective_lam(spread)
    assert _model(lam=0.5, adaptive_lam=False).fit(X, y).effective_lam(concentrated) == 0.5


def test_sin_localizacion_la_regularizacion_no_cambia(datos):
    # Sin núcleo, los pesos locales son los del decaimiento y n_eff = n_ref.
    X, y = datos
    model = _model(lam=0.5, localize=False).fit(X, y)
    a = model.local_weights(X[:1])[0]
    assert abs(model.effective_lam(a) - 0.5) < 1e-9


def test_sin_fuga_de_informacion_futura(datos):
    # Los errores registrados del mes s no pueden depender de datos posteriores a s.
    X, y = datos
    full = _model().fit(X, y)
    part = _model().fit(X[:120], y[:120])
    k = len(part.oof_rows_)
    np.testing.assert_allclose(full.errors_[:k], part.errors_, atol=1e-10)
