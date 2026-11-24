"""Datos, modelo y pronósticos por país.

Es la única puerta al modelo. ALEE no se reentrena ni se modifica aquí: se
cargan los artefactos ya ajustados (``artifacts/reg_ALEE`` para Colombia y
``artifacts/paises/<pais>`` para los demás) y se les pide el pronóstico del mes
siguiente al último dato observado. Todo queda en memoria, así que cambiar de
país no vuelve a calcular nada que ya se haya calculado.
"""
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src import config
from src.data import AREA_COLS, TARGET_COL, build_features, load_raw
from src.explain import conformal_halfwidth, sequential_coverage
from src.original_model import EXPERTS

from . import catalog as C

DATA = config.ROOT_DIR / "data" / "paises"
ART = config.ROOT_DIR / "artifacts"
RESULTS = ART / "resultados" / "solo_curso"
COVERAGE = 0.9
MODELS = [m for m in C.MODEL_LABELS if m != "ALEE"]


# ------------------------------------------------------------------- datos
@lru_cache(maxsize=None)
def frame(c) -> pd.DataFrame:
    """Serie mensual del país: ``fecha`` (primer día del mes), ``valor`` e ``interpolado``."""
    d = pd.read_csv(DATA / f"{c}_serie.csv")
    d["fecha"] = pd.to_datetime(d["fecha"])
    return d


def series(c) -> pd.Series:
    d = frame(c)
    return pd.Series(d["valor"].values, index=pd.DatetimeIndex(d["fecha"]))


@lru_cache(maxsize=None)
def raw_colombia() -> pd.DataFrame:
    d = load_raw().copy()
    d["fecha"] = d["fecha"].dt.to_period("M").dt.to_timestamp()
    return d


@lru_cache(maxsize=None)
def features(c) -> pd.DataFrame:
    """Variables del modelo en cada mes (mismo código del proyecto, ``src.data.build_features``)."""
    if c == "colombia":
        f = build_features(load_raw())
    else:
        d = frame(c)
        base = pd.DataFrame({"fecha": d["fecha"], TARGET_COL: d["valor"]})
        for col in AREA_COLS:
            base[col] = 0.0
        f = build_features(base).drop(columns=AREA_COLS)
    f = f.drop(columns="fecha")
    f.index = pd.DatetimeIndex(frame(c)["fecha"])
    return f


@lru_cache(maxsize=None)
def oof(c) -> pd.DataFrame:
    """Predicciones fuera de muestra de los cinco pliegues externos (todas hechas con meses anteriores)."""
    d = pd.read_csv(DATA / f"{c}_oof.csv")
    d["fecha"] = pd.to_datetime(d["fecha"])
    return d


@lru_cache(maxsize=None)
def later_observations(c):
    """Datos publicados después del corte de la serie. El modelo no los usa."""
    p = DATA / f"{c}_posterior.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d["fecha"] = pd.to_datetime(d["mes"])
    return d


# ------------------------------------------------------------------ modelo
@lru_cache(maxsize=None)
def pipeline(c):
    return joblib.load((ART / "reg_ALEE" if c == "colombia" else ART / "paises" / c) / "model.joblib")


def model(c):
    return pipeline(c).named_steps["model"]


@lru_cache(maxsize=None)
def meta(c) -> dict:
    """Ficha del artefacto con las mismas claves en todos los países."""
    if c == "colombia":
        m = json.loads((ART / "reg_ALEE" / "metadata.json").read_text(encoding="utf-8"))
        return {"hp": m["hyperparameters"], "n": m["training"]["n_observations"],
                "evals": m["final_search"]["n_evaluations"], "features": [v["name"] for v in m["input_schema"]]}
    m = json.loads((ART / "paises" / c / "metadata.json").read_text(encoding="utf-8"))
    return {"hp": m["hiperparametros"], "n": m["entrenamiento"]["observaciones"], "evals": m["busqueda"]["evaluaciones"],
            "features": m["entrenamiento"]["variables"]}


@lru_cache(maxsize=None)
def conformal(c) -> dict:
    """Intervalo conforme partido con los errores fuera de muestra, y su cobertura medida hacia adelante."""
    d = oof(c)
    r = (d["y_true"] - d["ALEE"]).values
    fw = sequential_coverage(r, COVERAGE, warmup=48)
    return {"hw": conformal_halfwidth(r, COVERAGE), "n": len(r), "coverage": fw["coverage"], "n_forward": fw["n"],
            "width": fw["mean_width"]}


def explain_point(c, x) -> dict:
    """Pronóstico de una fila de variables con los votos de los expertos y sus pesos."""
    m = model(c)
    x = np.atleast_2d(np.asarray(x, dtype=float))
    W, F = m.combination_weights(x)[0], m.expert_predictions(x)[0]
    a = m.local_weights(x)[0]
    lag1 = x[0, list(m.feature_names).index(m.lag_col)]
    return {"pred": float(W @ F), "weights": W, "experts": F, "last": float(lag1),
            "contrib": W * (F - lag1), "n_eff": float(1 / np.sum(a ** 2)), "n_ref": float(m.n_eff_ref_),
            "prior": m.prior_, "local": a}


@lru_cache(maxsize=None)
def forecast(c) -> dict:
    """Pronóstico del mes siguiente al último observado."""
    f, s = features(c), series(c)
    x = f.iloc[-1].values
    out = explain_point(c, x)
    conf = conformal(c)
    last, target = s.index[-1], s.index[-1] + pd.offsets.MonthBegin(1)
    m = model(c)
    state_dates = f.index[f.notna().all(axis=1)][m.oof_rows_]
    out.update(country=c, last_date=last, target_date=target, observed=float(s.iloc[-1]), previous=float(s.iloc[-2]),
               change=out["pred"] - float(s.iloc[-1]), lo=out["pred"] - conf["hw"], hi=out["pred"] + conf["hw"],
               year_ago=float(s.iloc[-13]), same_month=float(s.loc[target - pd.DateOffset(years=1)]),
               interpolated_recent=bool(frame(c)["interpolado"].iloc[-12:].any()),
               state=f.iloc[-1][["td_nacional_rollstd3"]].tolist() + [float(x[0] - f.iloc[-1]["td_nacional_rollmean12"])],
               neighbours=pd.Series(out["local"], index=state_dates).nlargest(8))
    return out


# ----------------------------------------------------------- validación
@lru_cache(maxsize=None)
def forward_halfwidths(c, warmup=48) -> np.ndarray:
    """Semiancho del intervalo del 90 % que se habría usado en cada mes de prueba.

    Mismo procedimiento que ``sequential_coverage``: para el mes ``t`` solo se
    usan los errores fuera de muestra de los meses anteriores. Los primeros
    ``warmup`` meses no tienen intervalo (NaN).
    """
    d = oof(c)
    r = np.abs((d["y_true"] - d["ALEE"]).values)
    return np.array([np.nan if t < warmup else conformal_halfwidth(r[:t], COVERAGE) for t in range(len(r))])


def recent_track(c, n=24) -> pd.DataFrame:
    """Últimos ``n`` pronósticos fuera de muestra, con el intervalo que estaba disponible en cada origen."""
    d = oof(c).copy()
    d["error"] = d["y_true"] - d["ALEE"]
    d["hw"] = forward_halfwidths(c)
    d["origen"] = d["fecha"] - pd.offsets.MonthBegin(1)
    d["dentro"] = d["error"].abs() <= d["hw"]
    return d.tail(n)


def later_validation(c) -> pd.DataFrame:
    """Pronósticos de la plataforma frente a datos oficiales publicados después del corte.

    Solo entra un mes si su pronóstico pudo hacerse con información anterior a
    la publicación: el mes siguiente al último dato de la serie. Los demás meses
    posteriores se devuelven sin pronóstico (``pred`` vacío) y solo sirven de
    referencia.
    """
    later = later_observations(c)
    if later is None:
        return pd.DataFrame()
    f, hw = forecast(c), conformal(c)["hw"]
    rows = []
    for _, r in later.iterrows():
        ok = r["fecha"] == f["target_date"]
        rows.append({"origen": f["last_date"] if ok else pd.NaT, "objetivo": r["fecha"], "pred": f["pred"] if ok else np.nan,
                     "obs": float(r["tasa_desempleo_nacional"]), "lo": f["lo"] if ok else np.nan, "hi": f["hi"] if ok else np.nan,
                     "error": float(r["tasa_desempleo_nacional"]) - f["pred"] if ok else np.nan,
                     "dentro": (abs(float(r["tasa_desempleo_nacional"]) - f["pred"]) <= hw) if ok else None, "fuente": r["fuente_principal"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- evaluación
@lru_cache(maxsize=None)
def metrics() -> pd.DataFrame:
    d = pd.read_csv(RESULTS / "metricas_por_pliegue.csv")
    return d[d["modelo"].isin(C.MODEL_LABELS)]


def metric_table(metric="RMSE") -> pd.DataFrame:
    t = metrics().groupby(["conjunto", "modelo"])[metric].mean().unstack()
    return t.reindex(index=list(C.COUNTRIES), columns=list(C.MODEL_LABELS))


@lru_cache(maxsize=None)
def result(name) -> pd.DataFrame:
    return pd.read_csv(RESULTS / f"{name}.csv")


def best_baseline(c):
    row = metric_table().loc[c].drop("ALEE")
    return row.idxmin(), float(row.min())


assert EXPERTS == ("persistencia", "nucleo_cambio", "estacional")
