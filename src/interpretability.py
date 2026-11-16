"""Interpretabilidad global y local con valores de Shapley (SHAP) y LIME.

SHAP atribuye a cada variable su contribución a una predicción usando la
teoría de juegos cooperativos (propiedades de eficiencia, simetría y
aditividad). Para modelos de árboles y modelos lineales el cálculo es exacto
(``TreeExplainer`` y ``LinearExplainer``); para el resto se aproxima con
``KernelExplainer``. LIME, en cambio, ajusta un modelo lineal simple alrededor
de cada observación a partir de perturbaciones, por lo que es una aproximación
local.
"""
from typing import Optional

import lime.lime_tabular
import numpy as np
import pandas as pd
import shap
from scipy import stats

TREE_MODELS = (
    "RandomForestClassifier", "RandomForestRegressor", "DecisionTreeClassifier", "DecisionTreeRegressor",
    "XGBClassifier", "XGBClassifierSafe", "XGBRegressor",
)
LINEAR_MODELS = ("Ridge", "Lasso", "LogisticRegression")


def scale(scaler, X) -> np.ndarray:
    """Aplica el escalador del modelo (si lo hay) y devuelve un arreglo."""
    if scaler is not None:
        return scaler.transform(X)
    return X.values if hasattr(X, "values") else np.asarray(X)


def compute_shap(est, scaler, X_background: pd.DataFrame, X_explain: pd.DataFrame, task: str):
    """Calcula los valores SHAP de un modelo ya ajustado.

    Parameters
    ----------
    est : estimator
        Modelo ajustado.
    scaler : StandardScaler or None
        Escalador que el modelo espera (None si no usa).
    X_background : pandas.DataFrame
        Datos de referencia (entrenamiento) para los explicadores que los necesitan.
    X_explain : pandas.DataFrame
        Observaciones a explicar.
    task : {"clf", "reg"}
        Tipo de problema.

    Returns
    -------
    explainer : objeto de SHAP
    values : ndarray
        Forma (n, variables) en regresión y (n, variables, clases) en
        clasificación.
    """
    background, explain = scale(scaler, X_background), scale(scaler, X_explain)
    name = type(est).__name__
    if name in TREE_MODELS:
        explainer = shap.TreeExplainer(est)
        values = explainer.shap_values(explain)
    elif name in LINEAR_MODELS:
        explainer = shap.LinearExplainer(est, background)
        values = explainer.shap_values(explain)
    else:
        predict = est.predict_proba if task == "clf" else est.predict
        explainer = shap.KernelExplainer(predict, shap.sample(background, min(50, len(background)), random_state=42))
        values = explainer.shap_values(explain, nsamples="auto", silent=True)
    if isinstance(values, list):
        values = np.stack(values, axis=-1)
    return explainer, np.asarray(values)


def base_value(explainer, class_index: Optional[int] = None) -> float:
    """Valor esperado del modelo (punto de partida de una explicación local)."""
    expected = np.atleast_1d(explainer.expected_value)
    return float(expected[class_index]) if class_index is not None and len(expected) > 1 else float(expected[0])


def local_explanation(explainer, values: np.ndarray, X_explain: pd.DataFrame, row: int, class_index: Optional[int] = None):
    """Arma el objeto ``shap.Explanation`` de una observación (para el gráfico de cascada)."""
    contributions = values[row, :, class_index] if class_index is not None else values[row]
    return shap.Explanation(
        values=contributions, base_values=base_value(explainer, class_index),
        data=X_explain.iloc[row].values, feature_names=list(X_explain.columns),
    )


def mean_abs_shap(values: np.ndarray, columns, class_index: Optional[int] = None) -> pd.Series:
    """Importancia global: media del valor absoluto de SHAP por variable."""
    v = values[:, :, class_index] if class_index is not None else values
    return pd.Series(np.abs(v).mean(axis=0), index=columns).sort_values(ascending=False)


def lime_importance(
    est, scaler, X_train: pd.DataFrame, X_explain: pd.DataFrame, task: str,
    class_index: Optional[int] = None, n_instances: int = 30, seed: int = 42,
) -> pd.Series:
    """Importancia global según LIME: media del peso absoluto por variable.

    Se explican ``n_instances`` observaciones de ``X_explain`` y se promedia el
    peso absoluto que LIME asigna a cada variable en su aproximación lineal
    local, lo que permite compararla con la importancia media de SHAP.

    Parameters
    ----------
    class_index : int, optional
        Clase a explicar (obligatorio en clasificación).
    n_instances : int
        Número de observaciones a explicar (las primeras de ``X_explain``).

    Returns
    -------
    pandas.Series
        Peso absoluto medio por variable, de mayor a menor.
    """
    train, explain = scale(scaler, X_train), scale(scaler, X_explain)
    mode = "classification" if task == "clf" else "regression"
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=train, feature_names=list(X_train.columns), mode=mode,
        discretize_continuous=False, random_state=seed,
    )
    predict = est.predict_proba if task == "clf" else est.predict
    weights = np.zeros(train.shape[1])
    n = min(n_instances, len(explain))
    for i in range(n):
        kwargs = {"labels": (class_index,)} if task == "clf" else {}
        exp = explainer.explain_instance(explain[i], predict, num_features=train.shape[1], **kwargs)
        pairs = exp.as_map()[class_index if task == "clf" else 1]
        for feature, weight in pairs:
            weights[feature] += abs(weight)
    return pd.Series(weights / n, index=X_train.columns).sort_values(ascending=False)


def agreement(shap_importance: pd.Series, lime_weights: pd.Series, top: int = 5) -> dict:
    """Acuerdo entre la importancia de SHAP y la de LIME.

    Returns
    -------
    dict
        Correlación de Spearman entre ambas importancias y número de variables
        en común entre los dos rankings de las ``top`` más importantes.
    """
    both = pd.concat([shap_importance.rename("shap"), lime_weights.rename("lime")], axis=1).fillna(0)
    rho = stats.spearmanr(both["shap"], both["lime"])[0]
    overlap = len(set(shap_importance.index[:top]) & set(lime_weights.index[:top]))
    return {"spearman": float(rho), "top_en_comun": overlap, "top": top}
