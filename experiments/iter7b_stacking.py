"""Iteración 7b (Track B2, pre-registrada en experiments/protocolo.md sección 13):
apilamiento (stacking) con meta-regresor ridge walk-forward, como arquitectura
alternativa de combinación de los 3 expertos de ALEE (persistencia,
nucleo_cambio, estacional). Es la versión "aprendida" de la idea de ALEE: en
vez de una regla cerrada (kernel + entropía), un modelo lineal ajustado a los
datos decide los pesos.

No modifica src/original_model.py: reutiliza `_Experts` y `EXPERTS` tal cual
están, para garantizar que los expertos son exactamente los mismos que usa
ALEE.

Paso 0 -- predicciones de los expertos en ventana expansiva real
------------------------------------------------------------------
El archivo OFICIAL `expert_oof_predictions.csv` guarda, como pide el
protocolo, los 240 meses s=48..287. Esto es exactamente lo que hace
`ModeloOriginal.fit()` internamente (ver su docstring y el bucle
`for s in range(m0, n)`), solo que aquí se calcula una sola vez y se
reutiliza en las 5x3 corridas de la validación anidada (misma motivación que
`experiments/iter5a_sarima_estacional.py` e `iter7b_dma.py`: el error/la
predicción de un experto en el mes s no depende de los hiperparámetros del
combinador ni del pliegue, porque todas las particiones -- internas y
externas -- son prefijos cronológicos de la misma serie).

Nota técnica (necesaria para el meta-regresor, no un cambio de diseño): a
diferencia de DMA/DMS (un filtro bayesiano que arranca con un previo
uniforme y no necesita historia previa), un ridge NO puede "arrancar en
frío": para realizar literalmente el primer paso del reajuste descrito en el
protocolo ("empezando en s=48, ajusta el ridge con todos los datos
0..47; úsalo para predecir los meses 48..59") se necesitan predicciones de
los expertos ANTES de s=48 como insumo de bootstrap para el primer ajuste.
Por eso, además del archivo oficial (s=48..287), se calcula aquí -- con el
mismo procedimiento walk-forward de `_Experts`, nunca con información
posterior a s -- la misma predicción para s=1..47 (s=0 no tiene mes
anterior, así que no puede tener una predicción a un paso; se excluye). Esta
extensión se usa SOLO como historia de entrenamiento del meta-regresor, no
se reporta como predicción de ALEE ni reemplaza el archivo oficial, y no
introduce ninguna mirada al pliegue de prueba: todas las predicciones, en
s=1..287, siguen usando exclusivamente X[:s], y[:s].

Paso 1 -- meta-regresor ridge con reajuste cada 12 meses
----------------------------------------------------------
Para cada fila s con predicciones de expertos disponibles (s=1..287) se
arma el vector de 5 variables: las 3 predicciones de los expertos en s, más
`rollstd3` y la brecha `lag1 - rollmean12` (las mismas que usa ALEE para
localizar), estandarizadas con media/desviación calculadas solo con los
datos de entrenamiento vigentes en cada reajuste (nunca con todo el
conjunto). El reajuste crece cada 12 meses: se entrena con todo lo
disponible hasta el punto de corte, se predicen los siguientes 12 meses, y
esos 12 meses (ya con su verdad conocida) se agregan al entrenamiento del
siguiente reajuste -- igual que `ModeloOriginal` reajusta sus expertos en
cada mes de una ventana expansiva.

Paso 2 -- validación anidada honesta
-------------------------------------
TimeSeriesSplit(5, gap=1) externo, TimeSeriesSplit(3, gap=1) interno, Optuna
(TPESampler, semilla 42), 40 evaluaciones por pliegue externo. Como tr/te e
itr/ite son siempre PREFIJOS 0..k de la serie completa (igual razonamiento
que iter5a/iter7b_dma), el mismo `chunked_ridge_predict` sirve tanto para la
afinación interna (entrenando con itr, reajustando cada 12 meses dentro de
ite) como para la predicción final de cada pliegue externo (entrenando con
tr, reajustando cada 12 meses hasta el final de te). Nunca se usa
información del pliegue de prueba externo para elegir `alpha`, y dentro de
cada reajuste nunca se usa información del mes que se está prediciendo ni de
meses futuros a ese punto de corte.
"""
import os
import sys
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.data import assemble_dataset
from src.original_model import _Experts, EXPERTS
from src.stats_tests import diebold_mariano_test, cohens_d_paired, cliffs_delta, effect_label

optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT = config.ROOT_DIR / "artifacts" / "confirmatorio" / "iter7"
OUT.mkdir(parents=True, exist_ok=True)

M0 = 48            # inicio del rango oficial de evaluacion (240 meses)
S_MIN = 1          # primera fila con prediccion a un paso valida (s=0 no tiene mes anterior)
CHUNK = 12         # reajuste cada 12 meses
N_TRIALS = 40
SEED = config.SEED
STATE_COLS = ("td_nacional_rollstd3",)
GAP_COLS = ("td_nacional_lag1", "td_nacional_rollmean12")

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
def expert_oof_predictions_extended():
    """Predicciones a un paso de los 3 expertos, s = S_MIN..N-1, walk-forward real.

    Devuelve un arreglo (N, 3) con NaN en la fila 0 (sin mes anterior).
    """
    t0 = time.time()
    preds = np.full((N, 3), np.nan)
    for s in range(S_MIN, N):
        experts = _Experts(X_ALL[:s], Y_ALL[:s], COLS)
        preds[s] = experts.predict(X_ALL[s:s + 1])[0]
    print(f"Paso 0: predicciones OOF de los 3 expertos, s={S_MIN}..{N - 1} ({time.time() - t0:.1f}s)")
    return preds  # columnas = EXPERTS = (persistencia, nucleo_cambio, estacional)


def sanity_check(preds):
    """RMSE individual de cada experto sobre los 240 meses OFICIALES (s=48..287)."""
    y = Y_ALL[M0:]
    sub = preds[M0:]
    print("\nVerificación de cordura (RMSE individual de cada experto, 240 meses oficiales):")
    rows = {}
    for k, name in enumerate(EXPERTS):
        r = rmse(y, sub[:, k])
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


# --------------------------------------------------------- variables de estado
def state_features():
    """(N, 2): rollstd3 y la brecha lag1 - rollmean12, para cada fila de X_ALL."""
    c1 = list(ds.feature_names).index(STATE_COLS[0])
    c2a = list(ds.feature_names).index(GAP_COLS[0])
    c2b = list(ds.feature_names).index(GAP_COLS[1])
    return np.column_stack([X_ALL[:, c1], X_ALL[:, c2a] - X_ALL[:, c2b]])


# --------------------------------------------------------- Paso 1: meta-regresor
def chunked_ridge_predict(meta_X, y, train_prefix, predict_rows, alpha, chunk=CHUNK):
    """Reajuste walk-forward del meta-regresor ridge cada `chunk` meses.

    Parameters
    ----------
    meta_X : ndarray (N, 5)
        Variables del meta-regresor (3 predicciones de expertos + 2 variables
        de estado) para cada fila global.
    y : ndarray (N,)
        Objetivo.
    train_prefix : array-like of int
        Índices GLOBALES (ya filtrados a filas válidas, s >= S_MIN) usados
        para el primer ajuste. Son siempre un prefijo cronológico.
    predict_rows : array-like of int
        Índices GLOBALES, en orden creciente, a predecir -- siempre
        posteriores a `train_prefix`.
    alpha : float
        Fuerza de la regularización ridge.

    Returns
    -------
    ndarray
        Predicciones alineadas con `predict_rows`.
    """
    train_rows = list(train_prefix)
    predict_rows = list(predict_rows)
    preds = np.empty(len(predict_rows))
    for start in range(0, len(predict_rows), chunk):
        block = predict_rows[start:start + chunk]
        Xtr = meta_X[train_rows]
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        Xtr_s = (Xtr - mu) / sd
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(Xtr_s, y[train_rows])
        Xte_s = (meta_X[block] - mu) / sd
        preds[start:start + len(block)] = model.predict(Xte_s)
        train_rows = train_rows + list(block)  # ya se conoce su y verdadero: entra al siguiente reajuste
    return preds


# --------------------------------------------------------- Paso 2: validación anidada
def run_nested_cv(meta_X):
    outer = list(TimeSeriesSplit(config.N_SPLITS_OUTER, gap=config.CV_GAP).split(ds.X))
    fold_rows = []
    oof_idx, oof_y, oof_p = [], [], []
    t0 = time.time()

    for f, (tr, te) in enumerate(outer):
        inner = list(TimeSeriesSplit(config.N_SPLITS_INNER, gap=config.CV_GAP).split(tr))
        inner_global = [(tr[itr], tr[ite]) for itr, ite in inner]

        def objective(trial):
            alpha = trial.suggest_float("alpha", 1e-3, 100.0, log=True)
            errs = []
            for itr_g, ite_g in inner_global:
                train_prefix = itr_g[itr_g >= S_MIN]
                predict_rows = ite_g[ite_g >= S_MIN]
                if len(train_prefix) < 2 or len(predict_rows) == 0:
                    continue
                p = chunked_ridge_predict(meta_X, Y_ALL, train_prefix, predict_rows, alpha)
                errs.append(rmse(Y_ALL[predict_rows], p))
            return float(np.mean(errs)) if errs else 1e6

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
        best_alpha = study.best_params["alpha"]

        train_prefix = tr[tr >= S_MIN]
        p_test = chunked_ridge_predict(meta_X, Y_ALL, train_prefix, te, best_alpha)
        y_te = Y_ALL[te]
        r = rmse(y_te, p_test)
        print(f"  pliegue {f}: alpha={best_alpha:.5f}  RMSE={r:.4f}")

        fold_rows.append({"pliegue": f, "rmse": r, "alpha": best_alpha})
        oof_idx.append(te); oof_y.append(y_te); oof_p.append(p_test)

    print(f"  tiempo total validación anidada: {time.time() - t0:.1f}s")
    oof_idx = np.concatenate(oof_idx); oof_y = np.concatenate(oof_y); oof_p = np.concatenate(oof_p)
    return pd.DataFrame(fold_rows), pd.DataFrame({"idx": oof_idx, "y_true": oof_y, "y_pred": oof_p})


# --------------------------------------------------------- Paso 3: significancia
def run_significance(oof):
    est = pd.read_csv(config.ROOT_DIR / "artifacts" / "confirmatorio" / "oof_EstacionalALEE.csv").set_index("idx")
    base = pd.read_csv(config.ROOT_DIR / "artifacts" / "confirmatorio" / "baselines_series_tiempo.csv").set_index("idx")
    alee3 = pd.read_csv(config.ROOT_DIR / "artifacts" / "confirmatorio" / "iter4" / "oof_ALEE_K3_sin_ridge.csv").set_index("idx")

    refs = {
        "Estacional aislado": est["y_pred"],
        "SARIMA": base["SARIMA"],
        "ALEE-3 (canónico)": alee3["y_pred"],
    }
    y_true_ref = base["y_true"]

    oof_i = oof.set_index("idx")
    idx = oof_i.index
    assert np.allclose(oof_i["y_true"].values, y_true_ref.loc[idx].values), "y_true no coincide al alinear por idx"
    err_cand = (oof_i["y_true"] - oof_i["y_pred"]).values

    rows = []
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
            "candidato": "Stacking (ridge walk-forward)", "referencia": ref_name,
            "rmse_candidato": rmse(oof_i["y_true"], oof_i["y_pred"]),
            "rmse_referencia": rmse(y_true_ref.loc[idx], ref_aligned),
            "dm_hln": dm["dm_hln"], "p_value": dm["p_value"], "mean_loss_diff": dm["mean_loss_diff"],
            "direccion": direccion,
            "d_cohen": d_cohen, "d_cohen_label": effect_label(d_cohen, "cohen"),
            "cliffs_delta": delta_cliff, "cliffs_delta_label": effect_label(delta_cliff, "cliff"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "significancia_iter7b_stacking.csv", index=False)
    return df


if __name__ == "__main__":
    preds_ext = expert_oof_predictions_extended()
    rmse_experts = sanity_check(preds_ext)

    expert_file = OUT / "expert_oof_predictions.csv"
    pd.DataFrame({
        "idx": np.arange(M0, N),
        "y_true": Y_ALL[M0:],
        "persistencia": preds_ext[M0:, EXPERTS.index("persistencia")],
        "nucleo_cambio": preds_ext[M0:, EXPERTS.index("nucleo_cambio")],
        "estacional": preds_ext[M0:, EXPERTS.index("estacional")],
    }).to_csv(expert_file, index=False)
    print(f"\nGuardado: {expert_file}")

    state = state_features()
    meta_X = np.column_stack([preds_ext, state])  # (N, 5): 3 expertos + rollstd3 + brecha
    print(f"meta_X listo: {meta_X.shape}, filas válidas (s>={S_MIN}): {N - S_MIN}")

    print("\nPaso 2: validación anidada honesta (stacking ridge walk-forward)")
    rmse_df, oof = run_nested_cv(meta_X)

    rmse_df.to_csv(OUT / "rmse_iter7b_stacking.csv", index=False)
    oof.to_csv(OUT / "oof_iter7b_stacking.csv", index=False)

    print("\n=== RESUMEN ===")
    print(f"Stacking: RMSE medio de pliegues = {rmse_df['rmse'].mean():.4f} +/- {rmse_df['rmse'].std():.4f}"
          f"   RMSE agrupado = {rmse(oof['y_true'], oof['y_pred']):.4f}")
    print(rmse_df.to_string(index=False))

    print("\nPaso 3: Diebold-Mariano-HLN (stacking vs estacional aislado, SARIMA, ALEE-3)")
    sig = run_significance(oof)
    pd.set_option("display.width", 160)
    print(sig.to_string(index=False))

    print("\n=== Criterio pre-registrado (sección 13 del protocolo) ===")
    sub = sig.set_index("referencia")
    gana_sarima = sub.loc["SARIMA", "direccion"] == "GANA (significativo)"
    gana_estacional = sub.loc["Estacional aislado", "direccion"] == "GANA (significativo)"
    pierde_alee3 = sub.loc["ALEE-3 (canónico)", "direccion"] == "PIERDE (significativo)"
    cumple = gana_sarima and gana_estacional and not pierde_alee3
    print(f"Stacking: gana a SARIMA={gana_sarima}  gana a estacional aislado={gana_estacional}  "
          f"pierde con significancia a ALEE-3={pierde_alee3}  -> CUMPLE CRITERIO: {cumple}")
