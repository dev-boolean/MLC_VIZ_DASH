"""Iteración 7b (Track B1, pre-registrada en experiments/protocolo.md sección 13):
Dynamic Model Averaging / Dynamic Model Selection (DMA/DMS), Raftery, Kárný y
Ettler (2010), como arquitectura alternativa de combinación de los 3 expertos
de ALEE (persistencia, nucleo_cambio, estacional).

No modifica src/original_model.py: reutiliza `_Experts` y `EXPERTS` tal cual
están, para garantizar que los expertos son exactamente los mismos que usa
ALEE.

Paso 0 -- predicciones de los expertos en ventana expansiva real
------------------------------------------------------------------
Para cada mes s = 48..287 (los 240 meses de evaluación oficiales) se ajustan
los tres expertos SOLO con los meses 0..s-1 y se pronostica el mes s. Esto es
exactamente lo que hace `ModeloOriginal.fit()` internamente (ver su docstring
y el bucle `for s in range(m0, n)`), solo que aquí se guarda una sola vez para
reutilizarlo en las 5 x 3 corridas de la validación anidada (igual
motivación que `experiments/iter5a_sarima_estacional.py`: el error de un
experto en el mes s no depende de los hiperparámetros del combinador ni del
pliegue, porque todas las particiones -- internas y externas -- son prefijos
cronológicos de la misma serie).

Paso 1 -- filtro DMA/DMS
------------------------
Recursión estándar de Raftery et al. (2010), verosimilitud gaussiana:

    (a) olvido de probabilidades:  pi_{s|s-1,k} ∝ pi_{s-1,k}^alpha           (renormalizado)
    (b) varianza recursiva con olvido: V_{s,k} = delta * V_{s-1,k} + (1-delta) * e_{s-1,k}^2
    (c) actualización bayesiana:   pi_{s,k} ∝ pi_{s|s-1,k} * N(e_{s,k}; 0, V_{s,k})  (renormalizado)

El pronóstico DMA es la media ponderada por pi_{s|s-1} (ANTES de ver y_s, que
es lo único honesto); el DMS es el pronóstico del experto con mayor
pi_{s|s-1}.

Inicialización (decisión de diseño, documentada porque el protocolo pide
"documenta cuál usaste"): pi_0 uniforme (1/K); V_0 (varianza inicial, igual
para los 3 expertos) se fija con la varianza de las primeras diferencias de
la serie observada en el tramo 0..47 (`np.var(np.diff(y[:48]))`), el mismo
tramo que ya se usó para ajustar los primeros expertos en s=48 -- no usa
ninguna información de s=48 en adelante, así que no introduce fuga alguna
hacia el futuro ni hacia ningún pliegue de prueba.

Paso 2 -- validación anidada honesta
-------------------------------------
TimeSeriesSplit(5, gap=1) externo, TimeSeriesSplit(3, gap=1) interno, Optuna
(TPESampler, semilla 42), 40 evaluaciones por pliegue externo. Como todos los
pliegues de entrenamiento (externos e internos) son siempre un PREFIJO
0..k de la serie completa, los índices de las particiones son directamente
índices globales: para cada (alpha, delta) candidato se corre el filtro UNA
sola vez desde s=48 hasta el final de la serie (datos ya calculados en el
paso 0) y se recortan las predicciones en los índices de cada partición --
nunca se usa información del pliegue de prueba externo para elegir
(alpha, delta). Se tunea con el RMSE de DMA (ver nota técnica más abajo); el
DMS de cada pliegue usa el mismo (alpha, delta) ganador.

Nota técnica (limitación, no un cambio de diseño): el pliegue externo 0 tiene
los tres pliegues internos completamente por debajo de s=48 (el entrenamiento
del pliegue externo 0 llega hasta el índice 46, y sus pliegues internos de
prueba llegan como máximo al índice 46), así que NINGÚN punto de ningún
pliegue interno de ese pliegue externo cae dentro del rango s=48..287 donde
existen predicciones de los expertos. La búsqueda bayesiana para ese pliegue
no tiene, por lo tanto, ninguna señal con la que discriminar (alpha, delta):
los 40 ensayos devuelven la misma penalización constante y el "mejor" valor
que reporta Optuna es, en los hechos, arbitrario dentro del rango buscado.
Esto es una consecuencia directa y honesta del m0=48 ya pre-registrado, no un
error de código ni un cambio de diseño -- se documenta aquí y se repite en el
reporte final; el RMSE de prueba del pliegue externo 0 sigue siendo una
evaluación fuera de muestra legítima (el filtro corre hacia adelante,
revelando la verdad mes a mes, con los hiperparámetros fijados sin haber
mirado ese ni ningún otro pliegue de prueba), solo que con hiperparámetros
elegidos sin información.
"""
import os
import sys
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.data import assemble_dataset
from src.original_model import _Experts, EXPERTS
from src.stats_tests import diebold_mariano_test, cohens_d_paired, cliffs_delta, effect_label

optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT = config.ROOT_DIR / "artifacts" / "confirmatorio" / "iter7"
OUT.mkdir(parents=True, exist_ok=True)

M0 = 48
N_TRIALS = 40
SEED = config.SEED
VAR_FLOOR = 1e-8

ds = assemble_dataset()
X_ALL = ds.X.values
Y_ALL = ds.y_reg.values
N = len(Y_ALL)
assert N == 288, f"Se esperaban 288 meses, hay {N}"

COLS = {
    "lag1": list(ds.feature_names).index("td_nacional_lag1"),
    "mes_sin": list(ds.feature_names).index("mes_sin"),
    "mes_cos": list(ds.feature_names).index("mes_cos"),
}


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


# --------------------------------------------------------- Paso 0: expertos OOF
def expert_oof_predictions():
    t0 = time.time()
    preds = np.full((N - M0, 3), np.nan)
    for i, s in enumerate(range(M0, N)):
        experts = _Experts(X_ALL[:s], Y_ALL[:s], COLS)
        preds[i] = experts.predict(X_ALL[s:s + 1])[0]
    print(f"Paso 0: predicciones OOF de los 3 expertos, s={M0}..{N - 1} ({time.time() - t0:.1f}s)")
    return preds  # (240, 3) columnas = EXPERTS = (persistencia, nucleo_cambio, estacional)


def sanity_check(preds):
    y = Y_ALL[M0:]
    print("\nVerificación de cordura (RMSE individual de cada experto, 240 meses):")
    rows = {}
    for k, name in enumerate(EXPERTS):
        r = rmse(y, preds[:, k])
        rows[name] = r
        print(f"  {name:14s}: RMSE = {r:.4f}")
    esperado = {"persistencia": 1.273, "nucleo_cambio": 1.252, "estacional": 0.780}
    for name, exp in esperado.items():
        assert abs(rows[name] - exp) < 0.05, (
            f"RMSE de {name} ({rows[name]:.4f}) se aleja de lo documentado (~{exp}); "
            "probable bug en el bucle de walk-forward."
        )
    print("  -> coincide con lo documentado en docs/registro_desarrollo.md (diagnóstico 4d).")
    return rows


# --------------------------------------------------------- Paso 1: filtro DMA/DMS
def dma_dms_filter(errors, preds, alpha, delta, v_init):
    """Recursión de Raftery, Kárný y Ettler (2010).

    Parameters
    ----------
    errors, preds : ndarray (n, K)
        Error (y_true - pred) y predicción de cada experto en cada mes,
        en el mismo orden temporal (ya alineados con s = M0..N-1).
    alpha : float
        Factor de olvido de las probabilidades posteriores.
    delta : float
        Factor de olvido de la varianza recursiva del error.
    v_init : float
        Varianza inicial común a los 3 expertos (ver docstring del módulo).

    Returns
    -------
    y_dma, y_dms : ndarray (n,)
        Pronóstico DMA (promedio ponderado por pi_{s|s-1}) y DMS (el del
        experto con mayor pi_{s|s-1}), usando en cada mes SOLO información
        estrictamente anterior (pi_{s|s-1} se calcula antes de ver e_s).
    pis : ndarray (n, K)
        Historial de pi_{s|s-1} (la probabilidad usada para pronosticar),
        por si se quiere inspeccionar.
    """
    n, K = errors.shape
    pi_prev = np.full(K, 1.0 / K)
    V = np.full(K, float(v_init))
    y_dma = np.empty(n)
    y_dms = np.empty(n)
    pis = np.empty((n, K))
    for s in range(n):
        pi_pred = pi_prev ** alpha
        pi_pred = pi_pred / pi_pred.sum()
        pis[s] = pi_pred
        y_dma[s] = pi_pred @ preds[s]
        y_dms[s] = preds[s, int(np.argmax(pi_pred))]

        e_s = errors[s]
        Vk = np.maximum(V, VAR_FLOOR)
        loglik = -0.5 * (e_s ** 2) / Vk - 0.5 * np.log(2 * np.pi * Vk)
        loglik -= loglik.max()  # estabilidad numérica, no cambia la normalización
        lik = np.exp(loglik)
        pi_new = pi_pred * lik
        total = pi_new.sum()
        if not np.isfinite(total) or total <= 0:
            pi_new = pi_pred.copy()  # salvaguarda degenerada (no debería ocurrir)
        else:
            pi_new = pi_new / total
        pi_prev = pi_new
        V = delta * V + (1.0 - delta) * (e_s ** 2)
    return y_dma, y_dms, pis


# --------------------------------------------------------- Paso 2: validación anidada
def run_nested_cv(errors_global, preds_global, v_init):
    outer = list(TimeSeriesSplit(config.N_SPLITS_OUTER, gap=config.CV_GAP).split(ds.X))
    fold_rows_dma, fold_rows_dms = [], []
    oof_idx, oof_y, oof_dma, oof_dms = [], [], [], []
    t0 = time.time()

    for f, (tr, te) in enumerate(outer):
        inner = list(TimeSeriesSplit(config.N_SPLITS_INNER, gap=config.CV_GAP).split(tr))
        n_usable_inner = sum(1 for _, ite in inner if np.any(ite >= M0))

        def objective(trial):
            alpha = trial.suggest_float("alpha", 0.90, 1.0, log=True)
            delta = trial.suggest_float("delta", 0.90, 1.0, log=True)
            y_dma_full, _, _ = dma_dms_filter(errors_global, preds_global, alpha, delta, v_init)
            errs = []
            for itr, ite in inner:
                valid = ite[ite >= M0]
                if len(valid) == 0:
                    continue
                rel = valid - M0
                errs.append(rmse(Y_ALL[valid], y_dma_full[rel]))
            return float(np.mean(errs)) if errs else 1e6

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
        best = study.best_params

        y_dma_full, y_dms_full, _ = dma_dms_filter(errors_global, preds_global, best["alpha"], best["delta"], v_init)
        te_rel = te - M0  # todos los pliegues de prueba externos caen dentro de 48..287
        p_dma, p_dms = y_dma_full[te_rel], y_dms_full[te_rel]
        y_te = Y_ALL[te]
        r_dma, r_dms = rmse(y_te, p_dma), rmse(y_te, p_dms)

        flag = " (SIN SEÑAL interna: ver nota técnica)" if n_usable_inner == 0 else ""
        print(f"  pliegue {f}: alpha={best['alpha']:.5f} delta={best['delta']:.5f}  "
              f"RMSE_DMA={r_dma:.4f}  RMSE_DMS={r_dms:.4f}{flag}")

        fold_rows_dma.append({"pliegue": f, "rmse": r_dma, "alpha": best["alpha"], "delta": best["delta"],
                               "pliegues_internos_con_senal": n_usable_inner})
        fold_rows_dms.append({"pliegue": f, "rmse": r_dms, "alpha": best["alpha"], "delta": best["delta"],
                               "pliegues_internos_con_senal": n_usable_inner})
        oof_idx.append(te); oof_y.append(y_te); oof_dma.append(p_dma); oof_dms.append(p_dms)

    print(f"  tiempo total validación anidada: {time.time() - t0:.1f}s")
    oof_idx = np.concatenate(oof_idx); oof_y = np.concatenate(oof_y)
    oof_dma = np.concatenate(oof_dma); oof_dms = np.concatenate(oof_dms)
    return (pd.DataFrame(fold_rows_dma), pd.DataFrame(fold_rows_dms),
            pd.DataFrame({"idx": oof_idx, "y_true": oof_y, "y_pred": oof_dma}),
            pd.DataFrame({"idx": oof_idx, "y_true": oof_y, "y_pred": oof_dms}))


# --------------------------------------------------------- Paso 3: significancia
def run_significance(oof_dma, oof_dms):
    est = pd.read_csv(config.ROOT_DIR / "artifacts" / "confirmatorio" / "oof_EstacionalALEE.csv").set_index("idx")
    base = pd.read_csv(config.ROOT_DIR / "artifacts" / "confirmatorio" / "baselines_series_tiempo.csv").set_index("idx")
    alee3 = pd.read_csv(config.ROOT_DIR / "artifacts" / "confirmatorio" / "iter4" / "oof_ALEE_K3_sin_ridge.csv").set_index("idx")

    refs = {
        "Estacional aislado": est["y_pred"],
        "SARIMA": base["SARIMA"],
        "ALEE-3 (canónico)": alee3["y_pred"],
    }
    y_true_ref = base["y_true"]

    rows = []
    for cand_name, oof in (("DMA", oof_dma), ("DMS", oof_dms)):
        oof_i = oof.set_index("idx")
        idx = oof_i.index
        assert np.allclose(oof_i["y_true"].values, y_true_ref.loc[idx].values), "y_true no coincide al alinear por idx"
        err_cand = (oof_i["y_true"] - oof_i["y_pred"]).values
        for ref_name, ref_series in refs.items():
            ref_aligned = ref_series.loc[idx]
            err_ref = (y_true_ref.loc[idx] - ref_aligned).values
            dm = diebold_mariano_test(err_cand, err_ref, h=1, power=2)
            diffs = np.abs(err_cand) - np.abs(err_ref)
            d_cohen = cohens_d_paired(diffs)
            delta_cliff = cliffs_delta(np.abs(err_cand), np.abs(err_ref))
            gana = dm["p_value"] < 0.05 and dm["mean_loss_diff"] < 0
            pierde = dm["p_value"] < 0.05 and dm["mean_loss_diff"] > 0
            direccion = "GANA (significativo)" if gana else ("PIERDE (significativo)" if pierde else "empate")
            rows.append({
                "candidato": cand_name, "referencia": ref_name,
                "rmse_candidato": rmse(oof_i["y_true"], oof_i["y_pred"]),
                "rmse_referencia": rmse(y_true_ref.loc[idx], ref_aligned),
                "dm_hln": dm["dm_hln"], "p_value": dm["p_value"], "mean_loss_diff": dm["mean_loss_diff"],
                "direccion": direccion,
                "d_cohen": d_cohen, "d_cohen_label": effect_label(d_cohen, "cohen"),
                "cliffs_delta": delta_cliff, "cliffs_delta_label": effect_label(delta_cliff, "cliff"),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "significancia_iter7b_dma.csv", index=False)
    return df


if __name__ == "__main__":
    preds_global = expert_oof_predictions()
    rmse_experts = sanity_check(preds_global)

    pd.DataFrame({
        "idx": np.arange(M0, N),
        "y_true": Y_ALL[M0:],
        "persistencia": preds_global[:, EXPERTS.index("persistencia")],
        "nucleo_cambio": preds_global[:, EXPERTS.index("nucleo_cambio")],
        "estacional": preds_global[:, EXPERTS.index("estacional")],
    }).to_csv(OUT / "expert_oof_predictions.csv", index=False)

    errors_global = Y_ALL[M0:][:, None] - preds_global  # (240, 3)
    v_init = float(np.var(np.diff(Y_ALL[:M0])))
    print(f"\nV_init (varianza inicial común, de las primeras diferencias de y[:{M0}]): {v_init:.5f}")

    print("\nPaso 2: validación anidada honesta (DMA/DMS)")
    rmse_dma_df, rmse_dms_df, oof_dma, oof_dms = run_nested_cv(errors_global, preds_global, v_init)

    rmse_dma_df.to_csv(OUT / "rmse_iter7b_dma.csv", index=False)
    rmse_dms_df.to_csv(OUT / "rmse_iter7b_dms.csv", index=False)
    oof_dma.to_csv(OUT / "oof_iter7b_dma.csv", index=False)
    oof_dms.to_csv(OUT / "oof_iter7b_dms.csv", index=False)

    print("\n=== RESUMEN ===")
    print(f"DMA: RMSE medio de pliegues = {rmse_dma_df['rmse'].mean():.4f} +/- {rmse_dma_df['rmse'].std():.4f}"
          f"   RMSE agrupado = {rmse(oof_dma['y_true'], oof_dma['y_pred']):.4f}")
    print(f"DMS: RMSE medio de pliegues = {rmse_dms_df['rmse'].mean():.4f} +/- {rmse_dms_df['rmse'].std():.4f}"
          f"   RMSE agrupado = {rmse(oof_dms['y_true'], oof_dms['y_pred']):.4f}")

    print("\nPaso 3: Diebold-Mariano-HLN (DMA y DMS vs estacional aislado, SARIMA, ALEE-3)")
    sig = run_significance(oof_dma, oof_dms)
    pd.set_option("display.width", 160)
    print(sig.to_string(index=False))

    print("\n=== Criterio pre-registrado (sección 13 del protocolo) ===")
    for cand in ("DMA", "DMS"):
        sub = sig[sig["candidato"] == cand].set_index("referencia")
        gana_sarima = sub.loc["SARIMA", "direccion"] == "GANA (significativo)"
        gana_estacional = sub.loc["Estacional aislado", "direccion"] == "GANA (significativo)"
        pierde_alee3 = sub.loc["ALEE-3 (canónico)", "direccion"] == "PIERDE (significativo)"
        cumple = gana_sarima and gana_estacional and not pierde_alee3
        print(f"{cand}: gana a SARIMA={gana_sarima}  gana a estacional aislado={gana_estacional}  "
              f"pierde con significancia a ALEE-3={pierde_alee3}  -> CUMPLE CRITERIO: {cumple}")
