"""Validación científica de LABOR INSIGHTS:  python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labor import catalog as C  # noqa: E402
from labor import engine as E  # noqa: E402

PAISES = list(C.COUNTRIES)


def rmse_por_pliegue(d, col):
    e2 = (d["y_true"] - d[col]) ** 2
    return float(np.sqrt(e2.groupby(d["pliegue"]).mean()).mean())


@pytest.mark.parametrize("c", PAISES)
def test_pronostico_es_del_mes_siguiente(c):
    f, s = E.forecast(c), E.series(c)
    assert f["last_date"] == s.index[-1]
    assert f["target_date"] == s.index[-1] + pd.offsets.MonthBegin(1)
    assert f["target_date"] not in s.index  # no se mezcla con lo observado
    assert (pd.date_range(s.index[0], s.index[-1], freq="MS") == s.index).all()


@pytest.mark.parametrize("c", PAISES)
def test_pronostico_sale_del_artefacto(c):
    f = E.forecast(c)
    directo = float(E.pipeline(c).predict(E.features(c).iloc[[-1]])[0])
    assert f["pred"] == pytest.approx(directo, abs=1e-10)
    assert f["last"] + f["contrib"].sum() == pytest.approx(f["pred"], abs=1e-10)  # descomposición exacta
    assert f["weights"].sum() == pytest.approx(1) and (f["weights"] >= 0).all()
    assert f["experts"].min() - 1e-9 <= f["pred"] <= f["experts"].max() + 1e-9  # combinación convexa


@pytest.mark.parametrize("c", PAISES)
def test_variables_iguales_a_las_del_entrenamiento(c):
    m, X = E.model(c), E.features(c).dropna().iloc[:-1].values
    assert len(X) == E.meta(c)["n"]
    assert list(E.features(c).columns) == list(m.feature_names) == E.meta(c)["features"]
    assert np.allclose((X - m.experts_.mu) / m.experts_.sd, m.experts_.Z)


@pytest.mark.parametrize("c", PAISES)
def test_sin_fuga_temporal(c):
    """Las variables del mes t no cambian si se borra todo lo posterior a t."""
    from src.data import AREA_COLS, build_features, load_raw

    f, k = E.features(c), len(E.series(c)) // 2
    if c == "colombia":
        corto = build_features(load_raw().iloc[:k])
    else:
        d = E.frame(c).iloc[:k]
        corto = build_features(pd.DataFrame({"fecha": d["fecha"], "tasa_desempleo_nacional": d["valor"], **{a: 0.0 for a in AREA_COLS}}))
    assert np.allclose(corto.iloc[-1][list(f.columns)].astype(float), f.iloc[k - 1].astype(float))


@pytest.mark.parametrize("c", PAISES)
def test_fuera_de_muestra_coincide_con_la_serie_y_las_metricas(c):
    d = E.oof(c)
    assert np.allclose(E.series(c).loc[d["fecha"]].values, d["y_true"], atol=1e-4)
    for m in ["ALEE", *E.MODELS]:
        assert rmse_por_pliegue(d, m) == pytest.approx(E.metric_table().loc[c, m], abs=1e-3), m
    w = d[["w_persistencia", "w_nucleo", "w_estacional"]].values
    f = d[["f_persistencia", "f_nucleo", "f_estacional"]].values
    assert np.allclose((w * f).sum(1), d["ALEE"], atol=1e-3)


@pytest.mark.parametrize("c", PAISES)
def test_escenario_nulo_es_el_caso_base(c):
    f, sc = E.forecast(c), E.scenario(c)
    assert sc["pred"] == pytest.approx(f["pred"], abs=1e-9)
    assert sc["outside"] == []
    otro = E.scenario(c, (0.5, 0, 0))
    v = otro["recent"]
    assert otro["x"]["td_nacional_lag1"] == pytest.approx(f["observed"] + 0.5)
    assert otro["x"]["td_nacional_rollmean3"] == pytest.approx(v[-3:].mean())
    assert otro["x"]["td_nacional_rollstd12"] == pytest.approx(v.std(ddof=1))


def test_colombia_usa_el_artefacto_original():
    from src import dash_data as D

    a = D.model("reg_ALEE").predict(E.features("colombia").iloc[[-1]])[0]
    assert E.forecast("colombia")["pred"] == pytest.approx(float(a))


def test_rutas():
    import app

    cl = app.server.test_client()
    assert cl.get("/health").status_code == 200
    assert [x["id"] for x in cl.get("/api/globo").get_json()] == PAISES
    for p in ["/", "/analytics", "/modelo", "/forecast", "/escenarios", "/global", "/insights"]:
        assert cl.get(p).status_code == 200
    assert cl.post("/api/chat", json={"messages": "x"}).status_code == 400
