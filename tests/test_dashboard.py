"""Pruebas del dashboard: que cada página y cada callback funcionen sin el navegador."""
import numpy as np
import pytest

import app as dashboard  # noqa: F401  (registra las páginas)
from pages import conclusiones, eda, modelos, resultados
from src import chatbot
from src import dash_data as D
from src.explain import conformal_halfwidth, shapley_values

ARTIFACTS = D.artifact_names()


def test_paginas_registradas():
    import dash

    assert {p["path"] for p in dash.page_registry.values()} == {"/", "/eda", "/modelos", "/resultados", "/conclusiones"}


def test_navegacion_marca_la_pagina_activa():
    links = dashboard.update_nav("/resultados")
    assert [link.className for link in links].count("tab tab--selected") == 1


@pytest.mark.parametrize("cov", ["nacional", "area"])
def test_callbacks_eda(cov):
    out = eda.update_main("2001-01-01", "2025-12-31", cov)
    assert len(out) == 9 and all(o is not None for o in out)
    assert eda.update_time_series("2001-01-01", "2025-12-31", cov)[0].data
    assert eda.update_univariate("2010-01-01", "2019-12-31", "tasa_desempleo_area")[0].data
    assert eda.update_bivariate("2001-01-01", "2025-12-31", "tasa_desempleo_area", "tasa_desempleo_nacional", "categoria")[0].data


def test_eda_rango_muy_corto_no_falla():
    out = eda.update_main("2020-01-01", "2020-01-31", "nacional")
    assert "al menos tres meses" in out[1]
    assert "36 meses" in eda.update_time_series("2020-01-01", "2020-12-31", "nacional")[4]


@pytest.mark.parametrize("name", ARTIFACTS)
def test_callbacks_modelos(name):
    assert modelos.update_diagnostics(name) is not None


def test_sin_prediccion_en_vivo():
    assert not hasattr(modelos, "run_prediction") and "Predecir" not in str(modelos.layout)


@pytest.mark.parametrize("task", ["reg", "clf"])
def test_tabla_de_lideres(task):
    data, cols, txt = modelos.update_leaderboard(task)
    assert len(data) >= 7 and txt


def test_shapley_exacto_en_un_modelo_lineal():
    # En un modelo lineal, el valor de Shapley intervencional es w_j (x_j − media de las referencias).
    rng = np.random.default_rng(0)
    w = rng.normal(size=5)
    bg = rng.normal(size=(10, 5))
    x = rng.normal(size=5)
    res = shapley_values(lambda M: M @ w, x, bg, n_permutations=3)
    np.testing.assert_allclose(res["values"], w * (x - bg.mean(axis=0)), atol=1e-10)


def test_cuantil_conforme():
    r = np.arange(1, 101, dtype=float)
    assert conformal_halfwidth(r, 0.9) == 91.0


def test_solo_modelos_del_paper_en_resultados():
    assert list(D.paper_table().columns) == list(D.PAPER_MODELS)
    assert "SARIMA" not in set(D.paper_tests()["baseline"])
    assert "SARIMA" not in set(D.oof("reg").columns.str.replace("pred_", ""))


def test_alee_coincide_con_el_paper():
    assert D.metadata("reg_ALEE")["outer_cv_metrics"]["rmse"]["mean"] == pytest.approx(D.paper_table().loc["colombia", "ALEE"])
    oof = D.oof("reg")
    fold_rmse = oof.groupby("fold").apply(lambda g: np.sqrt(np.mean((g["y_true"] - g["pred_ALEE"]) ** 2)))
    assert fold_rmse.mean() == pytest.approx(D.paper_table().loc["colombia", "ALEE"], abs=1e-9)


@pytest.mark.parametrize("dataset", list(D.DATASETS))
def test_callbacks_resultados(dataset):
    assert resultados.update_tests(dataset)[0]
    fig_abl, txt, fig_reg = resultados.update_ablation(dataset)
    assert fig_abl.data and fig_reg.data and txt


@pytest.mark.parametrize("metric", list(resultados.METRICS))
def test_tabla_de_resultados(metric):
    data, cols, style = resultados.update_table(metric)
    assert len(data) == len(D.DATASETS) and len(style) == len(D.DATASETS)


def test_referencias_del_paper():
    refs = conclusiones.references()
    assert len(refs) > 40 and not any("\\" in str(r) for r in refs)


def test_contexto_del_asistente_sin_sarima(monkeypatch):
    ctx = chatbot.build_context()
    assert "ALEE" in ctx and "SARIMA" not in ctx
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = dashboard.server.test_client()
    r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hola"}]})
    assert r.status_code == 200 and "GROQ_API_KEY" in r.get_json()["respuesta"]
