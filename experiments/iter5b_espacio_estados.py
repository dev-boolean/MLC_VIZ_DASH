"""Iteración 5b (exploratoria, posterior a la validación confirmatoria y a la
iteración 4 ya adoptada): ¿un modelo propio de espacio de estados (nivel local
+ estacional, filtro de Kalman) supera al experto estacional aislado, a SARIMA
o a ALEE de 3 expertos?

Metodología de evaluación idéntica a la del baseline oficial de SARIMA
(``src/analysis.py::regression_baselines``): para cada uno de los 5 pliegues
externos (TimeSeriesSplit, gap=1), se ajusta el modelo UNA VEZ con el tramo de
entrenamiento (ventana expansiva desde el inicio de la serie) y se extiende
con ``.append(..., refit=False)`` + ``.get_prediction(dynamic=False)`` para
obtener pronósticos de un paso adelante sin fuga de información sobre el
pliegue de prueba. Misma convención de índices: la fila r del dataset
corresponde al mes r+11 de la serie cruda, con objetivo en r+12.

La especificación (determinístico vs. estocástico, nivel simple vs. nivel con
tendencia) se elige por AIC en el tramo de ENTRENAMIENTO de cada pliegue, sin
mirar el error de prueba.

No modifica `src/original_model.py`, `src/models.py`, el libro ni el paper.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402
from src.data import assemble_dataset, load_raw  # noqa: E402
from src.stats_tests import diebold_mariano_test  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "artifacts", "confirmatorio", "iter5")
os.makedirs(OUT, exist_ok=True)

FIRST_ROW = 11  # la fila r del dataset corresponde al mes r + 11 de la serie cruda


def _fit_ucm(y_train, **kwargs):
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = UnobservedComponents(y_train, **kwargs)
        res = mod.fit(disp=False, maxiter=200)
    return mod, res


SPECS = {
    # nivel local + estacional determinístico (sin varianza propia: la
    # estacionalidad es un patrón fijo que se reestima cada vez que se ajusta)
    "nivel_local_estacional_det": dict(level="local level", freq_seasonal=[{"period": 12, "harmonics": 6}],
                                        stochastic_freq_seasonal=[False]),
    # nivel local + estacional estocástico (permite que el patrón estacional
    # derive con el tiempo, como el nivel)
    "nivel_local_estacional_sto": dict(level="local level", freq_seasonal=[{"period": 12, "harmonics": 6}],
                                        stochastic_freq_seasonal=[True]),
    # nivel local con tendencia + estacional determinístico
    "nivel_tendencia_estacional_det": dict(level="local linear trend", freq_seasonal=[{"period": 12, "harmonics": 6}],
                                            stochastic_freq_seasonal=[False]),
}


def evaluate_fold(y_series, train_end_row, test_rows, spec_choices):
    """Ajusta cada especificación sobre el tramo de entrenamiento, elige la de
    menor AIC en entrenamiento, y devuelve pronósticos de un paso adelante
    sobre test_rows (fuera de muestra) junto con el nombre de la especificación
    elegida y las AIC de todas.
    """
    train_series = y_series[: train_end_row + FIRST_ROW + 1]
    aics = {}
    fits = {}
    for name, kwargs in spec_choices.items():
        try:
            mod, res = _fit_ucm(train_series, **kwargs)
            aics[name] = res.aic
            fits[name] = (mod, res)
        except Exception as e:  # no convergencia u otro problema numérico
            aics[name] = np.inf
            fits[name] = None
    best_name = min(aics, key=aics.get)
    mod, res = fits[best_name]
    last = int(test_rows[-1])
    extended = res.append(y_series[train_end_row + FIRST_ROW + 1: last + FIRST_ROW + 2], refit=False)
    start = int(test_rows[0]) + FIRST_ROW + 1
    end = last + FIRST_ROW + 1
    pred = extended.get_prediction(start=start, end=end, dynamic=False).predicted_mean
    return np.asarray(pred), best_name, aics


def main():
    ds = assemble_dataset()
    y_series = load_raw()["tasa_desempleo_nacional"].values
    outer = TimeSeriesSplit(n_splits=config.N_SPLITS_OUTER, gap=config.CV_GAP)

    rows = []
    fold_choice_log = []
    for f, (train, test) in enumerate(outer.split(ds.X)):
        a = len(train)
        pred, chosen, aics = evaluate_fold(y_series, a, test, SPECS)
        fold_choice_log.append({"pliegue": f, "elegido": chosen, **{f"aic_{k}": v for k, v in aics.items()}})
        rows.append(pd.DataFrame({
            "idx": test, "y_true": ds.y_reg.values[test], "pred": pred, "pliegue": f, "especificacion": chosen,
        }))
    oof_best = pd.concat(rows, ignore_index=True)
    pd.DataFrame(fold_choice_log).to_csv(os.path.join(OUT, "iter5b_eleccion_especificacion.csv"), index=False)
    oof_best.to_csv(os.path.join(OUT, "oof_iter5b_mejor_por_pliegue.csv"), index=False)

    print("=== Elección de especificación por pliegue (AIC en entrenamiento) ===")
    print(pd.DataFrame(fold_choice_log).to_string(index=False))

    # RMSE de la política "mejor especificación por AIC en cada pliegue"
    def rmse_summary(df, pred_col="pred"):
        byfold = df.groupby("pliegue").apply(lambda g: np.sqrt(np.mean((g["y_true"] - g[pred_col]) ** 2)))
        pooled = np.sqrt(np.mean((df["y_true"] - df[pred_col]) ** 2))
        return byfold.mean(), byfold.std(), pooled, byfold

    m, s, p, byfold = rmse_summary(oof_best)
    print(f"\n[iter5b, elegido por AIC] RMSE medio de pliegues = {m:.4f} +/- {s:.4f}   RMSE agrupado = {p:.4f}")
    print("Por pliegue:", byfold.round(3).to_dict())

    # También se evalúa cada especificación FIJA en todos los pliegues (sin
    # selección por AIC), para ver si alguna domina sistemáticamente y para
    # que la tabla final sea más informativa.
    resultados = {"mejor_por_aic": (m, s, p)}
    rows_fixed = {name: [] for name in SPECS}
    for f, (train, test) in enumerate(outer.split(ds.X)):
        a = len(train)
        train_series = y_series[: a + FIRST_ROW + 1]
        last = int(test[-1])
        for name, kwargs in SPECS.items():
            try:
                mod, res = _fit_ucm(train_series, **kwargs)
                extended = res.append(y_series[a + FIRST_ROW + 1: last + FIRST_ROW + 2], refit=False)
                start = int(test[0]) + FIRST_ROW + 1
                end = last + FIRST_ROW + 1
                pred = extended.get_prediction(start=start, end=end, dynamic=False).predicted_mean
                rows_fixed[name].append(pd.DataFrame({"idx": test, "y_true": ds.y_reg.values[test],
                                                        "pred": np.asarray(pred), "pliegue": f}))
            except Exception as e:
                print(f"  [AVISO] especificación {name}, pliegue {f}: fallo de convergencia ({e})")
    for name, parts in rows_fixed.items():
        if not parts:
            print(f"[iter5b, {name}] no convergió en ningún pliegue")
            continue
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(os.path.join(OUT, f"oof_iter5b_{name}.csv"), index=False)
        m, s, p, byfold = rmse_summary(df)
        resultados[name] = (m, s, p)
        print(f"[iter5b, {name}] RMSE medio de pliegues = {m:.4f} +/- {s:.4f}   RMSE agrupado = {p:.4f}   "
              f"por pliegue: {byfold.round(3).tolist()}")

    tabla = pd.DataFrame(resultados, index=["rmse_medio_pliegues", "std_pliegues", "rmse_agrupado"]).T
    tabla.to_csv(os.path.join(OUT, "rmse_iter5b_todas_especificaciones.csv"))
    print("\n=== Tabla resumen iteración 5b ===")
    print(tabla.round(4).to_string())

    # Referencias para comparar
    base = pd.read_csv(os.path.join(os.path.dirname(OUT), "baselines_series_tiempo.csv")).set_index("idx")
    est_aislado = pd.read_csv(os.path.join(os.path.dirname(OUT), "oof_EstacionalALEE.csv")).set_index("idx")
    print(f"\nReferencias: estacional aislado RMSE agrupado = "
          f"{np.sqrt(np.mean((base['y_true'] - est_aislado['y_pred']) ** 2)):.4f}; "
          f"SARIMA RMSE agrupado = {np.sqrt(np.mean((base['y_true'] - base['SARIMA']) ** 2)):.4f}")

    # Si alguna especificación empata o le gana al estacional aislado en el
    # punto estimado (RMSE medio de pliegues <= 0.780), correr DM-HLN.
    candidatos = [name for name, (m, s, p) in resultados.items() if m <= 0.80]
    if not candidatos:
        print("\nNinguna especificación de iter5b se acerca al estacional aislado (0.780); no se corre DM-HLN.")
    for name in candidatos:
        fpath = os.path.join(OUT, f"oof_iter5b_{name}.csv") if name != "mejor_por_aic" else \
            os.path.join(OUT, "oof_iter5b_mejor_por_pliegue.csv")
        df = pd.read_csv(fpath).set_index("idx")
        common = df.index.intersection(est_aislado.index).intersection(base.index)
        e_model = df.loc[common, "y_true"] - df.loc[common, "pred"] if "pred" in df.columns else None
        e_est = est_aislado.loc[common, "y_true"] - est_aislado.loc[common, "y_pred"]
        e_sarima = base.loc[common, "y_true"] - base.loc[common, "SARIMA"]
        dm_est = diebold_mariano_test(e_model.values, e_est.values)
        dm_sarima = diebold_mariano_test(e_model.values, e_sarima.values)
        print(f"\n[{name}] DM-HLN vs. estacional aislado: p={dm_est['p_value']:.4f}, "
              f"diff_perdida_media={dm_est['mean_loss_diff']:.4f}")
        print(f"[{name}] DM-HLN vs. SARIMA: p={dm_sarima['p_value']:.4f}, "
              f"diff_perdida_media={dm_sarima['mean_loss_diff']:.4f}")


if __name__ == "__main__":
    main()
