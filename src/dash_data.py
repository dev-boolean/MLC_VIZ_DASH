"""Datos y modelos del dashboard, cargados una sola vez al iniciar el servidor.

Todo se lee de archivos ya calculados (``data/``, ``artifacts/`` y
``data/dashboard/``, ver ``scripts/build_dashboard_data.py``): el dashboard
nunca entrena modelos ni ejecuta búsquedas de hiperparámetros.
"""
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from . import config
from .data import assemble_dataset, load_raw

DASH_DIR = config.ROOT_DIR / "data" / "dashboard"
ARTIFACTS_DIR = config.ROOT_DIR / "artifacts"

MODEL_LABELS = {
    "KNN": "KNN", "NaiveBayes": "Naive Bayes", "LogisticRegression": "Regresión logística",
    "DecisionTree": "Árbol de decisión", "RandomForest": "Random Forest", "XGBoost": "XGBoost",
    "SVM": "SVM", "SVR": "SVR", "Ridge": "Ridge", "Lasso": "Lasso", "ALEE": "ALEE (modelo original)",
    "BayesianRidge": "Regresión bayesiana", "SVR_RBF": "SVR (kernel RBF)", "LightGBM": "LightGBM",
}
# Artefactos de regresión que no vienen de las 140 combinaciones: el modelo
# original y los modelos que agregó el paper.
PAPER_ARTIFACTS = ("reg_ALEE", "reg_BayesianRidge", "reg_SVR_RBF", "reg_LightGBM")

# Resultados del paper (experiments/resultados_curso.py): ALEE frente a los
# ocho modelos de regresión del curso y LightGBM, en nueve conjuntos de datos.
RESULTS_DIR = ARTIFACTS_DIR / "resultados"
PAPER_DIR = RESULTS_DIR / "solo_curso"
PAPER_MODELS = ("ALEE", "KNN", "Ridge", "Lasso", "BayesianRidge", "DecisionTree", "RandomForest", "XGBoost",
                "SVR_RBF", "LightGBM")
DATASETS = {"colombia": "Colombia", "eeuu": "Estados Unidos", "espana": "España", "mexico": "México",
            "canada": "Canadá", "australia": "Australia", "japon": "Japón", "alemania": "Alemania", "corea": "Corea del Sur"}
ABLATION_ORDER = ("ALEE completo", "Sin localización", "Sin λ adaptativa", "Previo uniforme", "Promedio simple",
                  "Solo persistencia", "Solo núcleo", "Solo estacional")
BALANCING_LABELS = {"none": "Ninguno", "smote": "SMOTE", "adasyn": "ADASYN", "class_weight": "Pesos de clase"}
OPTIMIZER_LABELS = {"grid": "Grid", "random": "Random", "bayesian": "Bayesiano", "genetic": "Genético"}

RAW_LABELS = {
    "tasa_desempleo_nacional": "Desempleo nacional",
    "tasa_ocupacion_nacional": "Ocupación nacional",
    "tasa_global_participacion_nacional": "Participación nacional",
    "tasa_desempleo_area": "Desempleo 13 áreas",
    "tasa_ocupacion_area": "Ocupación 13 áreas",
    "tasa_global_participacion_area": "Participación 13 áreas",
}

# Diccionario de las 17 variables de los modelos: etiqueta, grupo y descripción.
FEATURE_INFO = {
    "td_nacional_lag1": ("Desempleo nacional del mes t", "Rezagos", "Último dato conocido (%)"),
    "td_nacional_lag2": ("Desempleo nacional de t−1", "Rezagos", "Un mes antes (%)"),
    "td_nacional_lag3": ("Desempleo nacional de t−2", "Rezagos", "Dos meses antes (%)"),
    "td_nacional_lag6": ("Desempleo nacional de t−5", "Rezagos", "Cinco meses antes (%)"),
    "td_nacional_lag12": ("Desempleo nacional de t−11", "Rezagos", "Once meses antes (%)"),
    "td_nacional_rollmean3": ("Media móvil 3 meses", "Estadísticos móviles", "Promedio de t−2 a t (%)"),
    "td_nacional_rollstd3": ("Desviación móvil 3 meses", "Estadísticos móviles", "Volatilidad reciente (pp)"),
    "td_nacional_rollmean6": ("Media móvil 6 meses", "Estadísticos móviles", "Promedio de t−5 a t (%)"),
    "td_nacional_rollstd6": ("Desviación móvil 6 meses", "Estadísticos móviles", "(pp)"),
    "td_nacional_rollmean12": ("Media móvil 12 meses", "Estadísticos móviles", "Promedio de t−11 a t (%)"),
    "td_nacional_rollstd12": ("Desviación móvil 12 meses", "Estadísticos móviles", "(pp)"),
    "mes_sin": ("Mes (seno)", "Calendario", "Codificación cíclica del mes t"),
    "mes_cos": ("Mes (coseno)", "Calendario", "Codificación cíclica del mes t"),
    "trend_idx": ("Índice de tendencia", "Calendario", "Meses desde enero de 2001"),
    "tasa_global_participacion_area": ("Participación 13 áreas", "13 áreas metropolitanas", "Mes t (%)"),
    "tasa_desempleo_area": ("Desempleo 13 áreas", "13 áreas metropolitanas", "Mes t (%)"),
    "tasa_ocupacion_area": ("Ocupación 13 áreas", "13 áreas metropolitanas", "Mes t (%)"),
}


@lru_cache(maxsize=None)
def raw() -> pd.DataFrame:
    """Serie mensual original (300 meses)."""
    return load_raw()


@lru_cache(maxsize=None)
def dataset():
    """Conjunto de datos de pronóstico a un mes (288 filas, 17 variables)."""
    return assemble_dataset()


@lru_cache(maxsize=None)
def selection() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS_DIR / "seleccion.csv")


@lru_cache(maxsize=None)
def artifact_names() -> tuple:
    return tuple(sorted(p.name for p in ARTIFACTS_DIR.glob("*_*") if (p / "model.joblib").exists()))


@lru_cache(maxsize=None)
def metadata(name: str) -> dict:
    return json.loads((ARTIFACTS_DIR / name / "metadata.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def model(name: str):
    return joblib.load(ARTIFACTS_DIR / name / "model.joblib")


@lru_cache(maxsize=None)
def oof(task: str) -> pd.DataFrame:
    df = pd.read_csv(DASH_DIR / f"oof_{task}.csv", parse_dates=["fecha"])
    return df


@lru_cache(maxsize=None)
def mcs() -> pd.DataFrame:
    return pd.read_csv(DASH_DIR / "mcs.csv")


@lru_cache(maxsize=None)
def friedman() -> dict:
    return json.loads((DASH_DIR / "friedman_clf.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def conformal() -> dict:
    return json.loads((DASH_DIR / "conformal.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def shap_global(name: str) -> dict:
    d = np.load(DASH_DIR / "shap" / f"{name}.npz")
    return {"rows": d["rows"], "phi": d["phi"], "columns": list(d["columns"])}


@lru_cache(maxsize=None)
def paper_metrics() -> pd.DataFrame:
    """Métricas por pliegue de los modelos del paper en los nueve conjuntos."""
    df = pd.read_csv(PAPER_DIR / "metricas_por_pliegue.csv")
    return df[df["modelo"].isin(PAPER_MODELS)]


def paper_table(metric: str = "RMSE") -> pd.DataFrame:
    """Media de una métrica sobre los cinco pliegues externos: conjuntos en filas, modelos en columnas."""
    t = paper_metrics().groupby(["conjunto", "modelo"])[metric].mean().unstack()
    return t.reindex(index=list(DATASETS), columns=list(PAPER_MODELS))


@lru_cache(maxsize=None)
def paper_tests() -> pd.DataFrame:
    """Pruebas de ALEE frente a cada modelo en cada conjunto (Diebold-Mariano-HLN, Giacomini-White, ROPE)."""
    return pd.read_csv(PAPER_DIR / "pruebas_por_baseline.csv")


@lru_cache(maxsize=None)
def paper_between() -> pd.DataFrame:
    """Pruebas entre conjuntos (Wilcoxon, Conover y signed-rank bayesiano) de ALEE frente a cada modelo."""
    return pd.read_csv(PAPER_DIR / "pruebas_entre_conjuntos.csv")


@lru_cache(maxsize=None)
def paper_friedman() -> dict:
    """Friedman con Iman-Davenport, rangos medios y diferencia crítica de Nemenyi."""
    s = pd.read_csv(PAPER_DIR / "friedman.csv", index_col=0).iloc[:, 0]
    ranks = paper_table().rank(axis=1).mean().sort_values()
    from .stats_tests import critical_difference

    return {"chi2": float(s["friedman_chi2"]), "p_value": float(s["friedman_p"]),
            "iman_davenport_F": float(s["iman_davenport_F"]), "avg_ranks": ranks.to_dict(),
            "cd": critical_difference(len(PAPER_MODELS), len(DATASETS))}


@lru_cache(maxsize=None)
def paper_ablation() -> pd.DataFrame:
    """Ablación de ALEE en los nueve conjuntos, con Diebold-Mariano-HLN y Holm frente a ALEE completo."""
    return pd.read_csv(PAPER_DIR / "ablacion.csv")


@lru_cache(maxsize=None)
def paper_regimes() -> pd.DataFrame:
    """RMSE por tercil de volatilidad reciente, por conjunto y modelo."""
    df = pd.read_csv(PAPER_DIR / "errores_por_regimen.csv")
    return df[df["modelo"].isin(PAPER_MODELS)]


@lru_cache(maxsize=None)
def paper_targets() -> pd.DataFrame:
    """Descripción de la variable objetivo en cada conjunto."""
    return pd.read_csv(PAPER_DIR / "variable_objetivo.csv")


MONTHS_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
             "octubre", "noviembre", "diciembre"]


def month_es(fecha) -> str:
    """«mayo de 2020»: nombre del mes en español sin depender del idioma del servidor."""
    fecha = pd.Timestamp(fecha)
    return f"{MONTHS_ES[fecha.month - 1]} de {fecha.year}"


def artifact_label(name: str) -> str:
    task, model_name = name.split("_", 1)
    kind = "Regresión" if task == "reg" else "Clasificación"
    return f"{kind} · {MODEL_LABELS.get(model_name, model_name)}"


def leaderboard(task: str) -> pd.DataFrame:
    """Tabla de líderes: la mejor combinación de cada modelo y, en regresión, el modelo original."""
    sel = selection()[selection()["task"] == task]
    rows = []
    for _, r in sel.iterrows():
        rows.append({
            "Modelo": MODEL_LABELS[r["model"]], "Balanceo": BALANCING_LABELS[r["balancing"]] if task == "clf" else "—",
            "Optimizador": OPTIMIZER_LABELS[r["optimizer"]],
            "Métrica": "F1-macro" if task == "clf" else "RMSE",
            "Media": round(r["metric_mean"], 3), "Desviación": round(r["metric_std"], 3),
            "Media ± desv.": r["metric_mean_std"],
            "Inferencia (ms)": round(r["predict_time_s"] * 1000, 2),
            "Finalista estadístico": "Sí" if r["in_statistical_finalists"] else "No",
            "Estado": "Mejor combinación (2.º entregable)" if r["is_global_best"] else "",
        })
    if task == "reg":
        for name in (a for a in PAPER_ARTIFACTS if a in artifact_names()):
            model_name = name.split("_", 1)[1]
            m = metadata(name)["outer_cv_metrics"]["rmse"]
            rows.append({
                "Modelo": MODEL_LABELS[model_name], "Balanceo": "—", "Optimizador": "Bayesiano", "Métrica": "RMSE",
                "Media": round(m["mean"], 3), "Desviación": round(m["std"], 3),
                "Media ± desv.": f"{m['mean']:.3f} ± {m['std']:.3f}", "Inferencia (ms)": None,
                "Finalista estadístico": "Sí" if bool(mcs().set_index("modelo").loc[model_name, "en_conjunto"]) else "No",
                "Estado": "Modelo original" if model_name == "ALEE" else "Agregado en el paper",
            })
    asc = task == "reg"
    return pd.DataFrame(rows).sort_values("Media", ascending=asc).reset_index(drop=True)
