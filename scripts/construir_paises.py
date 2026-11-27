"""Construye los insumos internacionales de LABOR INSIGHTS a partir del proyecto completo.

    python scripts/construir_paises.py <ruta_del_proyecto_completo> [pais ...]

No cambia el modelo. Para cada país:

1. Copia la serie mensual (con la marca del mes interpolado de EE. UU.).
2. Reajusta ALEE en cada pliegue externo con los hiperparámetros que ya eligió
   la validación anidada y comprueba que reproduce las predicciones guardadas.
   De ahí salen los pesos y los pronósticos de cada experto, mes a mes.
3. Construye el artefacto final con el mismo procedimiento de ``reg_ALEE``
   (``src/artifacts.py``): búsqueda bayesiana de 40 evaluaciones, validación
   interna TimeSeriesSplit(3, gap=1), semilla 42 y ajuste con toda la serie.

Salidas: ``data/paises/<pais>_{serie,oof}.csv`` y ``artifacts/paises/<pais>/``.
"""
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(SRC))
spec = importlib.util.spec_from_file_location("rca", SRC / "experiments" / "run_conjuntos_adicionales.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer  # noqa: E402
from src import config  # noqa: E402
from src.fit_utils import to_numpy  # noqa: E402
from src.original_model import ModeloOriginal  # noqa: E402

PAISES = ["eeuu", "espana", "mexico", "canada", "australia", "japon", "alemania", "corea"]
MODELOS = ["KNN", "Ridge", "Lasso", "BayesianRidge", "DecisionTree", "RandomForest", "XGBoost", "SVR_RBF", "LightGBM"]
# Modelos que en Colombia vienen de las 140 combinaciones se leen de data/dashboard/oof_reg.csv.
OUT = REPO / "data" / "paises"


def datos(pais):
    if pais == "colombia":
        from src.data import assemble_dataset, load_raw
        ds = assemble_dataset()
        raw = load_raw()
        s = pd.Series(raw["tasa_desempleo_nacional"].values, index=raw["fecha"].dt.to_period("M").dt.to_timestamp())
        return s, ds.X, ds.y_reg.values, pd.DatetimeIndex(ds.fechas.dt.to_period("M").dt.to_timestamp())
    s = R.load_series(pais)
    X, y, f = R.build_dataset(s)
    return s, X, y.values, f


def serie(pais, s):
    imp = np.zeros(len(s), dtype=int)
    if pais == "eeuu":
        imp[s.index == pd.Timestamp("2025-10-01")] = 1
    pd.DataFrame({"fecha": s.index.strftime("%Y-%m"), "valor": s.values, "interpolado": imp}).to_csv(
        OUT / f"{pais}_serie.csv", index=False)


def otros_oof(pais, idx):
    """Predicciones fuera de muestra de los modelos de comparación del paper."""
    out = {}
    carpeta = "confirmatorio" if pais == "colombia" else f"adicionales/{pais}"
    for m in MODELOS:
        for base in (SRC / "artifacts" / "resultados" / pais, SRC / "artifacts" / carpeta):
            p = base / f"oof_{m}.csv"
            if p.exists():
                out[m] = pd.read_csv(p).set_index("idx").loc[idx, "y_pred"].values
                break
        else:
            print(f"  {pais}: falta {m}", flush=True)
    return out


def oof(pais, X, y, fechas, pre):
    if pais == "colombia":
        base = SRC / "artifacts" / "confirmatorio" / "iter4"
        hp, ref = pd.read_csv(base / "rmse_K3_sin_ridge.csv"), pd.read_csv(base / "oof_ALEE_K3_sin_ridge.csv")
    else:
        base = SRC / "artifacts" / "adicionales" / pais
        hp, ref = pd.read_csv(base / "hiperparametros_ALEE.csv"), pd.read_csv(base / "oof_ALEE.csv")
    ref = ref.set_index("idx")
    M = X.values
    partes, dif = [], 0.0
    for k, (tr, te) in enumerate(TimeSeriesSplit(n_splits=5, gap=1).split(M)):
        params = hp.iloc[k][["bandwidth", "half_life", "lam"]].astype(float).to_dict()
        m = ModeloOriginal(**params).fit(M[tr], y[tr]) if pais == "colombia" else R.alee_fit(params, M, y, len(tr), pre)
        W, F = m.combination_weights(M[te]), m.expert_predictions(M[te])
        dif = max(dif, float(np.abs((W * F).sum(1) - ref.loc[te, "y_pred"].values).max()))
        assert np.allclose(ref.loc[te, "y_true"].values, y[te])
        partes.append(pd.DataFrame({
            "idx": te, "fecha": (fechas[te] + pd.offsets.MonthBegin(1)).strftime("%Y-%m"), "pliegue": k + 1,
            "y_true": y[te], "ALEE": ref.loc[te, "y_pred"].values,
            "w_persistencia": W[:, 0], "w_nucleo": W[:, 1], "w_estacional": W[:, 2],
            "f_persistencia": F[:, 0], "f_nucleo": F[:, 1], "f_estacional": F[:, 2]}))
    d = pd.concat(partes, ignore_index=True)
    for m, p in otros_oof(pais, d["idx"].values).items():
        d[m] = p
    d.round(5).to_csv(OUT / f"{pais}_oof.csv", index=False)
    return dif


def artefacto(pais, X, y, fechas, pre):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    M = X.values
    inner = list(TimeSeriesSplit(n_splits=config.N_SPLITS_INNER, gap=config.CV_GAP).split(M))
    config.set_global_seed(config.SEED)

    def objective(trial):
        params = R.to_optuna(R.ORIGINAL_MODEL.space, trial)
        return float(np.mean([-np.sqrt(np.mean((y[va] - R.alee_fit(params, M, y, len(tr), pre).predict(M[va])) ** 2))
                              for tr, va in inner]))

    t0 = time.time()
    st = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=config.SEED, n_startup_trials=10))
    st.optimize(objective, n_trials=config.FULL_BUDGETS["bayes_n_trials"], show_progress_bar=False)
    t_busqueda = time.time() - t0
    t0 = time.time()
    m = ModeloOriginal(feature_names=tuple(R.FEATURES), random_state=config.SEED, **st.best_params).fit(M, y)
    t_ajuste = time.time() - t0
    prep = FunctionTransformer(to_numpy)
    prep.fit(X)
    pipe = Pipeline([("prep", prep), ("model", m)])
    d = REPO / "artifacts" / "paises" / pais
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, d / "model.joblib", compress=3)
    assert np.array_equal(pipe.predict(X[-60:]), joblib.load(d / "model.joblib").predict(X[-60:]))
    met = pd.read_csv(SRC / "artifacts" / "resultados" / "solo_curso" / "metricas_por_pliegue.csv")
    met = met[(met["conjunto"] == pais) & (met["modelo"] == "ALEE")]
    meta = {
        "pais": pais, "modelo": "ALEE",
        "estado": "mismo diseño congelado del paper; ajuste final con toda la serie del país",
        "hiperparametros": st.best_params, "semilla": config.SEED,
        "busqueda": {"optimizador": "bayesiano (TPE)", "evaluaciones": len(st.trials),
                     "validacion_interna": "TimeSeriesSplit(n_splits=3, gap=1)",
                     "mejor_rmse_interno": -st.best_value, "segundos": t_busqueda},
        "entrenamiento": {"observaciones": int(len(X)), "primer_mes": str(fechas[0].date()),
                          "ultimo_mes": str(fechas[-1].date()), "segundos": t_ajuste, "variables": list(X.columns)},
        "metricas_bucle_externo": {k: {"media": float(met[k].mean()), "desv": float(met[k].std())}
                                   for k in ["RMSE", "MAE", "MAPE", "R2", "MASE"]},
        "creado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (d / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return st.best_params, t_busqueda


def correr(pais):
    OUT.mkdir(parents=True, exist_ok=True)
    s, X, y, f = datos(pais)
    serie(pais, s)
    pre = None
    if pais != "colombia":
        cols = {"lag1": R.FEATURES.index("td_nacional_lag1"), "mes_sin": R.FEATURES.index("mes_sin"),
                "mes_cos": R.FEATURES.index("mes_cos")}
        pre = R.ExpertErrors(X.values, y, cols)
    dif = oof(pais, X, y, f, pre)
    msg = f"{pais}: OOF reproducido, diferencia maxima {dif:.1e}"
    if pais != "colombia":
        hp, seg = artefacto(pais, X, y, f, pre)
        msg += f" | artefacto {hp} en {seg:.0f} s"
    print(msg, flush=True)


if __name__ == "__main__":
    for p in sys.argv[2:] or ["colombia"] + PAISES:
        correr(p)
