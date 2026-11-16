"""Análisis posterior a la corrida de los 140 modelos.

Lee los registros que deja ``src.runner`` (tabla maestra, historiales de
búsqueda y predicciones fuera de muestra) y calcula lo que se presenta en el
capítulo: métricas agrupadas de todos los pliegues externos, modelos de
referencia, diagnóstico de residuos, calibración de probabilidades y
reconstrucción de los modelos finales para interpretarlos.

Las "predicciones fuera de muestra" son las que cada modelo hizo sobre el
tramo de prueba de cada pliegue externo. Los cinco tramos de prueba son
contiguos, así que apilados forman una serie de 240 pronósticos mensuales
hechos siempre con información anterior.
"""
import pickle
from typing import Dict, List, Optional

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from statsmodels.stats.diagnostic import acorr_ljungbox, het_white
from statsmodels.tsa.stattools import bds

from . import config, metrics as M, targets
from .balancing import apply_balancing
from .data import assemble_dataset, load_raw
from .models import CLASSIFIER_BY_NAME, REGRESSOR_BY_NAME
from .nested_cv import _fit_final

ALTO = config.CLASS_LABELS.index("Alto")

BALANCING_LABELS = {"none": "Ninguno", "smote": "SMOTE", "adasyn": "ADASYN", "class_weight": "Pesos de clase"}
OPTIMIZER_LABELS = {"grid": "Grid", "random": "Random", "bayesian": "Bayesiano", "genetic": "Genético"}
MODEL_LABELS = {
    "KNN": "KNN", "NaiveBayes": "Naive Bayes", "LogisticRegression": "Regresión logística",
    "DecisionTree": "Árbol de decisión", "RandomForest": "Random Forest", "XGBoost": "XGBoost",
    "SVM": "SVM", "SVR": "SVR", "Ridge": "Ridge", "Lasso": "Lasso",
}


# ------------------------------------------------------------------ carga
def load_master() -> pd.DataFrame:
    """Tabla maestra: una fila por combinación."""
    return pd.read_csv(config.RESULTS_MASTER_PATH)


def load_folds() -> pd.DataFrame:
    """Métricas por combinación, pliegue y métrica (formato largo)."""
    return pd.read_csv(config.RESULTS_FOLDS_PATH)


def load_history(run_id: str) -> dict:
    """Historial de una combinación: métricas y mejores hiperparámetros por
    pliegue, historia de cada búsqueda y registro del algoritmo genético."""
    with open(config.HISTORIES_DIR / f"{run_id}.pkl", "rb") as f:
        return pickle.load(f)


def pooled_oof(run_id: str) -> dict:
    """Predicciones fuera de muestra de los cinco pliegues externos, apiladas.

    Parameters
    ----------
    run_id : str
        Identificador de la combinación.

    Returns
    -------
    dict
        ``idx`` (posición de cada mes en el conjunto de datos), ``y_true``,
        ``y_pred``, ``y_proba`` (clasificación) y ``fold`` de cada predicción.
    """
    data = np.load(config.OOF_DIR / f"{run_id}.npz")
    folds = sorted({int(k.split("_")[-1]) for k in data.files})
    out = {"idx": [], "y_true": [], "y_pred": [], "y_proba": [], "fold": []}
    for k in folds:
        out["idx"].append(data[f"test_idx_{k}"])
        out["y_true"].append(data[f"y_true_{k}"])
        out["y_pred"].append(data[f"y_pred_{k}"])
        out["fold"].append(np.full(len(data[f"test_idx_{k}"]), k))
        if f"y_proba_{k}" in data.files:
            out["y_proba"].append(data[f"y_proba_{k}"])
    result = {key: np.concatenate(v) for key, v in out.items() if v}
    return result


def best_per_model(master: pd.DataFrame, task: str) -> pd.DataFrame:
    """Mejor combinación de cada modelo según la métrica principal.

    Se elige por F1-macro promedio en clasificación y por RMSE promedio en
    regresión.

    Returns
    -------
    pandas.DataFrame
        Una fila por modelo, ordenada de mejor a peor.
    """
    sub = master[master["task"] == task]
    if task == "clf":
        idx = sub.groupby("model")["f1_macro_mean"].idxmax()
        return sub.loc[idx].set_index("model").sort_values("f1_macro_mean", ascending=False)
    idx = sub.groupby("model")["rmse_mean"].idxmin()
    return sub.loc[idx].set_index("model").sort_values("rmse_mean")


def mean_std(row: pd.Series, key: str, digits: int = 3) -> str:
    """Formatea ``media ± desviación`` de una métrica de la tabla maestra."""
    return f"{row[f'{key}_mean']:.{digits}f} ± {row[f'{key}_std']:.{digits}f}"


# ----------------------------------------------------- clasificación agrupada
def pooled_classification_metrics(pooled: dict) -> dict:
    """Métricas de clasificación sobre las 240 predicciones apiladas.

    Además de las métricas macro, incluye el recall y la precisión de la
    clase "Alto" (los meses de crisis), que es la que más importa detectar.
    """
    y, p = pooled["y_true"], pooled["y_pred"]
    out = {
        "accuracy": accuracy_score(y, p),
        "precision_macro": precision_score(y, p, average="macro", zero_division=0),
        "recall_macro": recall_score(y, p, average="macro", zero_division=0),
        "f1_macro": f1_score(y, p, average="macro", zero_division=0),
        "recall_alto": recall_score(y == ALTO, p == ALTO, zero_division=0),
        "precision_alto": precision_score(y == ALTO, p == ALTO, zero_division=0),
    }
    if "y_proba" in pooled:
        out["auc_ovr"] = M.macro_ovr_auc(y, pooled["y_proba"])
        out["auc_alto"] = float(M.roc_auc_score((y == ALTO).astype(int), pooled["y_proba"][:, ALTO]))
    return out


def classification_baselines(ds=None) -> pd.DataFrame:
    """Modelos de referencia de clasificación sobre los mismos pliegues externos.

    * Mayoría: predice siempre la clase más frecuente del tramo de
      entrenamiento.
    * Persistencia: predice que el próximo mes tendrá la misma clase que el
      mes actual.
    * Estacional: predice la clase que tuvo el mes correspondiente un año antes.
    """
    ds = ds or assemble_dataset()
    outer = TimeSeriesSplit(n_splits=config.N_SPLITS_OUTER, gap=config.CV_GAP)
    y = ds.y_clf.values
    current = targets.to_codes(targets.categorize(ds.X["td_nacional_lag1"])).values
    seasonal = targets.to_codes(targets.categorize(ds.X["td_nacional_lag12"])).values
    truth, majority, persist, season = [], [], [], []
    for train, test in outer.split(ds.X):
        truth.append(y[test])
        majority.append(np.full(len(test), np.bincount(y[train]).argmax()))
        persist.append(current[test])
        season.append(seasonal[test])
    truth = np.concatenate(truth)
    rows = {}
    for name, pred in [("Mayoría", majority), ("Persistencia", persist), ("Estacional", season)]:
        pred = np.concatenate(pred)
        rows[name] = pooled_classification_metrics({"y_true": truth, "y_pred": pred})
    return pd.DataFrame(rows).T


# ---------------------------------------------------------- regresión agrupada
def pooled_regression_errors(run_id: str) -> dict:
    """Errores de pronóstico (observado menos predicho) apilados de una combinación."""
    pooled = pooled_oof(run_id)
    pooled["error"] = pooled["y_true"] - pooled["y_pred"]
    return pooled


def regression_baselines(ds=None) -> pd.DataFrame:
    """Pronósticos de referencia a un mes sobre los mismos 240 meses.

    * Persistencia: el mes siguiente igual al mes actual.
    * Estacional: el mes siguiente igual al mismo mes del año anterior.
    * SARIMA(0,1,1)(0,1,1)[12]: el modelo del primer entregable, reajustado en
      cada pliegue con el tramo de entrenamiento y usado para pronosticar un
      mes adelante con la información observada hasta ese momento. Se ajusta
      con la restricción de invertibilidad (la opción por defecto de
      statsmodels): con solo 59 meses en el primer pliegue, sin ella el
      coeficiente de media móvil estacional se dispara y la estimación no converge.

    Returns
    -------
    pandas.DataFrame
        Una fila por mes pronosticado con ``idx``, ``y_true`` y una columna
        por modelo de referencia.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    ds = ds or assemble_dataset()
    outer = TimeSeriesSplit(n_splits=config.N_SPLITS_OUTER, gap=config.CV_GAP)
    y_series = load_raw()["tasa_desempleo_nacional"].values
    first_row = 11  # la fila r del conjunto de datos corresponde al mes r + 11 de la serie
    rows = []
    for train, test in outer.split(ds.X):
        a = len(train)
        last = int(test[-1])
        # información disponible al cierre del entrenamiento: serie hasta el objetivo de la última fila
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = SARIMAX(
                y_series[: a + first_row + 1], order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)
            ).fit(disp=False, maxiter=200)
        extended = fit.append(y_series[a + first_row + 1: last + first_row + 2], refit=False)
        start = int(test[0]) + first_row + 1
        end = last + first_row + 1
        sarima = extended.get_prediction(start=start, end=end, dynamic=False).predicted_mean
        frame = pd.DataFrame(
            {
                "idx": test,
                "y_true": ds.y_reg.values[test],
                "Persistencia": ds.X["td_nacional_lag1"].values[test],
                "Estacional": ds.X["td_nacional_lag12"].values[test],
                "SARIMA": np.asarray(sarima),
            }
        )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def residual_tests(errors: np.ndarray, fitted: np.ndarray) -> dict:
    """Pruebas sobre los residuos de un modelo de regresión.

    Parameters
    ----------
    errors : ndarray
        Errores de pronóstico (observado menos predicho), en orden temporal.
    fitted : ndarray
        Valores predichos. Como los modelos no son lineales en las variables,
        la prueba de White se hace contra el valor predicho y su cuadrado.

    Returns
    -------
    dict
        p-valores de White (homocedasticidad), BDS (independencia),
        Ljung-Box a 12 y 24 rezagos (autocorrelación), Shapiro-Wilk y
        Jarque-Bera (normalidad).
    """
    white_p = het_white(errors, np.column_stack([np.ones(len(errors)), fitted]))[1]
    bds_p = float(bds(errors, max_dim=2)[1])
    lb = acorr_ljungbox(errors, lags=[12, 24], return_df=True)["lb_pvalue"]
    return {
        "white_p": float(white_p),
        "bds_p": bds_p,
        "ljungbox12_p": float(lb.loc[12]),
        "ljungbox24_p": float(lb.loc[24]),
        "shapiro_p": float(stats.shapiro(errors)[1]),
        "jarque_bera_p": float(stats.jarque_bera(errors)[1]),
    }


# --------------------------------------------------------------- calibración
def reliability_bins(y_bin: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Tabla del diagrama de confiabilidad: probabilidad media predicha y
    frecuencia observada en cada intervalo de probabilidad."""
    edges = np.linspace(0, 1, n_bins + 1)
    which = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = which == b
        rows.append(
            {
                "bin": b, "n": int(mask.sum()),
                "prob_media": float(prob[mask].mean()) if mask.any() else np.nan,
                "frecuencia": float(y_bin[mask].mean()) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def calibration_row(pooled: dict) -> dict:
    """Brier score y ECE de la probabilidad de la clase "Alto" (uno contra resto)."""
    y_bin = (pooled["y_true"] == ALTO).astype(int)
    prob = pooled["y_proba"][:, ALTO]
    return {
        "brier_alto": M.binary_brier(y_bin, prob),
        "ece_alto": M.expected_calibration_error(y_bin, prob),
        "brier_multiclase": M.multiclass_brier(pooled["y_true"], pooled["y_proba"]),
    }


def _platt(p_cal, y_cal, p_test):
    clip = lambda p: np.clip(p, 1e-4, 1 - 1e-4)
    logit = lambda p: np.log(clip(p) / (1 - clip(p)))
    model = LogisticRegression(C=1e6).fit(logit(p_cal).reshape(-1, 1), y_cal)
    return model.predict_proba(logit(p_test).reshape(-1, 1))[:, 1]


def _isotonic(p_cal, y_cal, p_test):
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_cal, y_cal).predict(p_test)


def recalibration_experiment(model_name: str, balancing: str, optimizer: str, ds=None, calib_fraction: float = 0.25) -> dict:
    """Efecto de recalibrar con Platt e isotónica la probabilidad de "Alto".

    En cada pliegue externo, el tramo de entrenamiento se parte en orden
    cronológico: con el primer 75 % se ajusta el modelo (con sus mejores
    hiperparámetros del pliegue) y con el 25 % final se ajusta el recalibrador;
    el pliegue de prueba solo se usa para evaluar. Los tres juegos de
    probabilidades se comparan sobre las predicciones apiladas de los cinco
    pliegues.

    Returns
    -------
    dict
        Brier score y ECE de la probabilidad de "Alto" sin recalibrar, con
        Platt y con regresión isotónica.
    """
    ds = ds or assemble_dataset()
    spec = CLASSIFIER_BY_NAME[model_name]
    history = load_history(f"clf_{model_name}_{balancing}_{optimizer}")
    outer = TimeSeriesSplit(n_splits=config.N_SPLITS_OUTER, gap=config.CV_GAP)
    y_all, truth, raw, platt, iso = ds.y_clf, [], [], [], []
    for fold, (train, test) in enumerate(outer.split(ds.X)):
        cut = int(len(train) * (1 - calib_fraction))
        base, calib = train[:cut], train[cut:]
        params = history["fold_results"][fold]["best_params"]
        X_b, y_b, sw = apply_balancing(balancing, ds.X.iloc[base], y_all.iloc[base])
        est, X_test_t = _fit_final(spec, params, X_b, y_b, sw, ds.X.iloc[test])
        scaler_apply = _scaled(spec, X_b, ds.X.iloc[calib])
        p_calib = M.expand_proba(est.predict_proba(scaler_apply), est.classes_)[:, ALTO]
        p_test = M.expand_proba(est.predict_proba(X_test_t), est.classes_)[:, ALTO]
        y_calib = (y_all.iloc[calib].values == ALTO).astype(int)
        truth.append((y_all.iloc[test].values == ALTO).astype(int))
        raw.append(p_test)
        if y_calib.min() == y_calib.max():  # sin variación no se puede calibrar
            platt.append(p_test)
            iso.append(p_test)
        else:
            platt.append(_platt(p_calib, y_calib, p_test))
            iso.append(_isotonic(p_calib, y_calib, p_test))
    y_bin = np.concatenate(truth)
    out = {}
    for name, probs in [("sin_recalibrar", raw), ("platt", platt), ("isotonica", iso)]:
        p = np.concatenate(probs)
        out[f"brier_{name}"] = M.binary_brier(y_bin, p)
        out[f"ece_{name}"] = M.expected_calibration_error(y_bin, p)
    return out


def _scaled(spec, X_fit, X_new):
    """Aplica al tramo ``X_new`` el escalado ajustado con ``X_fit`` (si el modelo lo usa)."""
    from sklearn.preprocessing import StandardScaler

    if spec.needs_scaling:
        return StandardScaler().fit(X_fit).transform(X_new)
    return X_new.values


# --------------------------------------------------- reconstrucción de modelos
def refit_fold(task: str, model_name: str, balancing: str, optimizer: str, fold: int, ds=None):
    """Reconstruye el modelo final de un pliegue externo con sus mejores hiperparámetros.

    Repite exactamente el ajuste final de ``run_nested_cv`` (balanceo,
    escalado y modelo sobre todo el tramo de entrenamiento), de modo que las
    predicciones coinciden con las guardadas. Sirve para interpretar el
    modelo (SHAP, LIME) y para repetir su ajuste con otras semillas.

    Returns
    -------
    dict
        ``est`` (estimador ajustado), ``scaler`` (o None), ``X_train`` (ya
        balanceado), ``X_test`` (sin escalar), ``X_test_t`` (listo para el
        modelo), ``y_test``, ``test_idx`` y ``params``.
    """
    from sklearn.preprocessing import StandardScaler

    ds = ds or assemble_dataset()
    registry = CLASSIFIER_BY_NAME if task == "clf" else REGRESSOR_BY_NAME
    spec = registry[model_name]
    run_id = f"{task}_{model_name}_{balancing}_{optimizer}"
    params = load_history(run_id)["fold_results"][fold]["best_params"]
    y = ds.y_clf if task == "clf" else ds.y_reg
    train, test = list(TimeSeriesSplit(n_splits=config.N_SPLITS_OUTER, gap=config.CV_GAP).split(ds.X))[fold]
    X_fit, y_fit, sw = ds.X.iloc[train], y.iloc[train], None
    if task == "clf" and balancing != "none":
        X_fit, y_fit, sw = apply_balancing(balancing, X_fit, y_fit)
    est, X_test_t = _fit_final(spec, params, X_fit, y_fit, sw, ds.X.iloc[test])
    scaler = StandardScaler().fit(X_fit) if spec.needs_scaling else None
    return {
        "est": est, "scaler": scaler, "X_train": X_fit, "X_test": ds.X.iloc[test],
        "X_test_t": X_test_t, "y_test": y.iloc[test], "test_idx": test, "params": params,
    }


def fit_on_all(task: str, model_name: str, balancing: str, params: dict, ds=None):
    """Ajusta un modelo con todas las observaciones (para interpretarlo, no para evaluarlo).

    Se usa para la interpretabilidad global: el gráfico resumen de SHAP se
    construye sobre el conjunto de datos completo, con el modelo entrenado con
    todos los meses y los hiperparámetros elegidos en el último pliegue externo.

    Returns
    -------
    tuple
        Estimador ajustado, escalador (o None) y variables de entrenamiento ya
        balanceadas.
    """
    from sklearn.preprocessing import StandardScaler

    ds = ds or assemble_dataset()
    registry = CLASSIFIER_BY_NAME if task == "clf" else REGRESSOR_BY_NAME
    spec = registry[model_name]
    y = ds.y_clf if task == "clf" else ds.y_reg
    X_fit, y_fit, sw = ds.X, y, None
    if task == "clf" and balancing != "none":
        X_fit, y_fit, sw = apply_balancing(balancing, ds.X, y)
    est, _ = _fit_final(spec, params, X_fit, y_fit, sw, ds.X.iloc[:1])
    scaler = StandardScaler().fit(X_fit) if spec.needs_scaling else None
    return est, scaler, X_fit


def ablation_area_features(task: str, model_name: str, balancing: str, optimizer: str, ds=None) -> dict:
    """Desempeño del mejor modelo con y sin los indicadores de las 13 áreas.

    Repite el ajuste final de cada pliegue externo con los hiperparámetros
    elegidos en la corrida real, primero con todas las variables y luego sin
    las tres de las áreas metropolitanas. Sirve para poner a prueba si el
    desempleo de las áreas aporta información sobre el desempleo nacional.

    Returns
    -------
    dict
        Métrica agrupada de los cinco pliegues con y sin las variables de área
        (F1-macro en clasificación, RMSE en regresión).
    """
    from .data import AREA_COLS

    ds = ds or assemble_dataset()
    registry = CLASSIFIER_BY_NAME if task == "clf" else REGRESSOR_BY_NAME
    spec = registry[model_name]
    history = load_history(f"{task}_{model_name}_{balancing}_{optimizer}")
    y = ds.y_clf if task == "clf" else ds.y_reg
    result = {}
    for label, columns in [("con_areas", list(ds.X.columns)), ("sin_areas", [c for c in ds.X.columns if c not in AREA_COLS])]:
        X = ds.X[columns]
        truth, pred = [], []
        for fold, (train, test) in enumerate(TimeSeriesSplit(n_splits=config.N_SPLITS_OUTER, gap=config.CV_GAP).split(X)):
            params = history["fold_results"][fold]["best_params"]
            X_fit, y_fit, sw = X.iloc[train], y.iloc[train], None
            if task == "clf" and balancing != "none":
                X_fit, y_fit, sw = apply_balancing(balancing, X_fit, y_fit)
            est, X_test_t = _fit_final(spec, params, X_fit, y_fit, sw, X.iloc[test])
            truth.append(y.iloc[test].values)
            pred.append(est.predict(X_test_t))
        truth, pred = np.concatenate(truth), np.concatenate(pred)
        result[label] = (
            float(f1_score(truth, pred, average="macro", zero_division=0)) if task == "clf"
            else float(np.sqrt(np.mean((truth - pred) ** 2)))
        )
    return result
