"""Configuración global del proyecto: semilla, rutas, esquema de validación y
presupuestos de búsqueda.

Todo lo que debe documentarse como control de reproducibilidad (una única
semilla global, rutas de los registros de experimentos, tamaños de los
bucles de validación anidada) vive aquí para que el resto de los módulos no
repita valores.
"""
import os
import random
from pathlib import Path

import numpy as np

# --- Reproducibilidad -------------------------------------------------------
SEED = 42

# --- Rutas ------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "MLC.xlsx"
DATA_SHEET = "Series de datos"

EXPERIMENTS_DIR = ROOT_DIR / "experiments"
RESULTS_MASTER_PATH = EXPERIMENTS_DIR / "results_master.csv"
RESULTS_FOLDS_PATH = EXPERIMENTS_DIR / "results_folds.csv"
HISTORIES_DIR = EXPERIMENTS_DIR / "histories"
OOF_DIR = EXPERIMENTS_DIR / "oof"


def set_experiments_dir(path) -> None:
    """Redirige todas las rutas de salida a otra carpeta (útil para pruebas).

    Parameters
    ----------
    path : str or pathlib.Path
        Carpeta donde se escribirán los registros de experimentos.
    """
    global EXPERIMENTS_DIR, RESULTS_MASTER_PATH, RESULTS_FOLDS_PATH, HISTORIES_DIR, OOF_DIR
    EXPERIMENTS_DIR = Path(path)
    RESULTS_MASTER_PATH = EXPERIMENTS_DIR / "results_master.csv"
    RESULTS_FOLDS_PATH = EXPERIMENTS_DIR / "results_folds.csv"
    HISTORIES_DIR = EXPERIMENTS_DIR / "histories"
    OOF_DIR = EXPERIMENTS_DIR / "oof"


def set_global_seed(seed: int = SEED) -> None:
    """Fija la semilla de todas las fuentes de aleatoriedad globales.

    Los estimadores de scikit-learn y XGBoost, SMOTE/ADASYN, el muestreador de
    Optuna y el muestreo aleatorio de hiperparámetros reciben la semilla de
    forma explícita; esta función cubre el generador global de ``random`` (que
    es el que usan los operadores de DEAP) y el de NumPy.

    Parameters
    ----------
    seed : int
        Semilla global del proyecto.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# --- Validación anidada (walk-forward, ver src/nested_cv.py) ----------------
# Los datos son una serie mensual (300 observaciones) y las variables incluyen
# rezagos de la propia serie, así que un K-fold aleatorio mezclaría meses casi
# idénticos entre entrenamiento y prueba. Por eso ambos bucles usan
# TimeSeriesSplit (ventana expansiva).
N_SPLITS_OUTER = 5
N_SPLITS_INNER = 3

# Meses que se dejan sin usar entre el final del entrenamiento y el inicio de
# la prueba de cada partición.
CV_GAP = 1

# --- Presupuestos de búsqueda ------------------------------------------------
# "smoke": presupuestos mínimos, solo para comprobar que el pipeline completo
#          corre de principio a fin.
# "full":  presupuestos de la corrida real (≈ 40 evaluaciones por método).
RUN_MODE = "smoke"

SMOKE_BUDGETS = {
    "grid_max_combinations": 4,
    "random_n_iter": 4,
    "bayes_n_trials": 6,
    "genetic_population": 6,
    "genetic_budget": 12,
}

FULL_BUDGETS = {
    # Se iguala el número de evaluaciones (≈ 40) entre los cuatro métodos para
    # que comparar "calidad por unidad de presupuesto" tenga sentido. En el
    # genético el presupuesto cuenta evaluaciones, no generaciones: la
    # búsqueda continúa hasta agotar las 40.
    "grid_max_combinations": 40,
    "random_n_iter": 40,
    "bayes_n_trials": 40,
    "genetic_population": 8,
    "genetic_budget": 40,
}


def get_budgets() -> dict:
    """Devuelve los presupuestos del modo de ejecución activo."""
    return SMOKE_BUDGETS if RUN_MODE == "smoke" else FULL_BUDGETS


# --- Umbrales de clasificación (justificación en src/targets.py) ------------
CLASS_THRESHOLDS = {
    "bajo_max": 10.0,   # < 10 %  -> "Bajo"
    "alto_min": 14.0,   # >= 14 % -> "Alto" (crisis 2001-2003 y 2020)
}
CLASS_LABELS = ["Bajo", "Medio", "Alto"]
