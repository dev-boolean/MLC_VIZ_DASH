"""Registro de los 14 modelos base (7 de clasificación y 7 de regresión) y su
espacio de búsqueda de hiperparámetros.

Cada entrada (``ModelSpec``) expone:

* ``estimator()``: fabrica un estimador de scikit-learn o XGBoost sin ajustar,
  con la semilla fija.
* ``space``: lista de ``HParam`` (ver ``search_space.py``), la misma para los
  cuatro optimizadores.
* ``needs_scaling``: si el modelo es sensible a la escala (KNN, SVM, modelos
  lineales regularizados); en ese caso se estandarizan las variables, siempre
  ajustando el escalador solo con el tramo de entrenamiento.
* ``supports_class_weight``: si el estimador tiene el parámetro nativo
  ``class_weight``. El balanceo por pesos de clase se implementa igual para
  todos con pesos por observación (ver ``balancing.py``).
"""
from dataclasses import dataclass
from typing import Callable, List

import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from . import config
from .search_space import HParam

SEED = config.SEED


class XGBClassifierSafe(XGBClassifier):
    """XGBClassifier que acepta cualquier subconjunto de clases.

    XGBoost (>= 2.0) exige que las etiquetas sean enteros consecutivos desde
    cero. En los pliegues más tempranos de la validación anidada puede faltar
    una clase por completo (por ejemplo, solo aparecen "Medio" y "Alto"), y el
    XGBClassifier estándar se niega a entrenar. Esta subclase codifica las
    etiquetas presentes con un ``LabelEncoder`` antes de ajustar y las
    decodifica al predecir, de modo que ``classes_`` y ``predict`` se
    comportan como en el resto de clasificadores de scikit-learn.

    Mientras dura el ``fit`` de XGBoost, ``classes_`` devuelve ``0..k-1``
    porque el ``fit`` original compara las etiquetas que recibe con esa
    propiedad; una vez termina, devuelve las etiquetas originales.
    """

    def fit(self, X, y, sample_weight=None, **kwargs):
        self._label_encoder = LabelEncoder().fit(y)
        y_encoded = self._label_encoder.transform(y)
        self._fitting = True
        try:
            super().fit(X, y_encoded, sample_weight=sample_weight, **kwargs)
        finally:
            self._fitting = False
        return self

    @property
    def classes_(self):
        if getattr(self, "_fitting", False):
            return np.arange(self.n_classes_)
        return self._label_encoder.classes_

    def predict(self, X, **kwargs):
        encoded = super().predict(X, **kwargs)
        return self._label_encoder.inverse_transform(np.asarray(encoded).astype(int))


@dataclass
class ModelSpec:
    name: str
    estimator: Callable[[], object]
    space: List[HParam]
    needs_scaling: bool = False
    supports_class_weight: bool = False


# ---------------------------------------------------------------- Clasificacion
CLASSIFIERS: List[ModelSpec] = [
    ModelSpec(
        name="KNN",
        estimator=lambda: KNeighborsClassifier(),
        space=[
            HParam("n_neighbors", "int", 3, 30),
            HParam("weights", "cat", choices=["uniform", "distance"]),
            HParam("p", "cat", choices=[1, 2]),
        ],
        needs_scaling=True,
    ),
    ModelSpec(
        name="NaiveBayes",
        estimator=lambda: GaussianNB(),
        space=[HParam("var_smoothing", "float", 1e-12, 1e-6, log=True)],
    ),
    ModelSpec(
        name="LogisticRegression",
        estimator=lambda: LogisticRegression(solver="saga", max_iter=5000, random_state=SEED),
        space=[
            HParam("C", "float", 1e-3, 1e2, log=True),
            HParam("penalty", "cat", choices=["l1", "l2"]),
        ],
        needs_scaling=True,
        supports_class_weight=True,
    ),
    ModelSpec(
        name="DecisionTree",
        estimator=lambda: DecisionTreeClassifier(random_state=SEED),
        space=[
            HParam("max_depth", "int", 2, 20),
            HParam("min_samples_split", "int", 2, 20),
            HParam("min_samples_leaf", "int", 1, 10),
            HParam("criterion", "cat", choices=["gini", "entropy"]),
        ],
        supports_class_weight=True,
    ),
    ModelSpec(
        name="RandomForest",
        estimator=lambda: RandomForestClassifier(random_state=SEED, n_jobs=1),
        space=[
            HParam("n_estimators", "int", 50, 300),
            HParam("max_depth", "int", 3, 20),
            HParam("min_samples_leaf", "int", 1, 10),
            HParam("max_features", "cat", choices=["sqrt", "log2"]),
        ],
        supports_class_weight=True,
    ),
    ModelSpec(
        name="XGBoost",
        estimator=lambda: XGBClassifierSafe(
            random_state=SEED, eval_metric="mlogloss", n_jobs=1, verbosity=0
        ),
        space=[
            HParam("n_estimators", "int", 50, 300),
            HParam("max_depth", "int", 2, 10),
            HParam("learning_rate", "float", 1e-3, 0.3, log=True),
            HParam("subsample", "float", 0.5, 1.0),
            HParam("colsample_bytree", "float", 0.5, 1.0),
        ],
    ),
    ModelSpec(
        name="SVM",
        estimator=lambda: SVC(probability=True, random_state=SEED),
        space=[
            HParam("C", "float", 1e-2, 1e2, log=True),
            HParam("gamma", "float", 1e-4, 1e1, log=True),
            HParam("kernel", "cat", choices=["rbf", "linear"]),
        ],
        needs_scaling=True,
        supports_class_weight=True,
    ),
]

# ------------------------------------------------------------------- Regresion
REGRESSORS: List[ModelSpec] = [
    ModelSpec(
        name="KNN",
        estimator=lambda: KNeighborsRegressor(),
        space=[
            HParam("n_neighbors", "int", 3, 30),
            HParam("weights", "cat", choices=["uniform", "distance"]),
            HParam("p", "cat", choices=[1, 2]),
        ],
        needs_scaling=True,
    ),
    ModelSpec(
        name="Ridge",
        estimator=lambda: Ridge(random_state=SEED),
        space=[HParam("alpha", "float", 1e-3, 1e3, log=True)],
        needs_scaling=True,
    ),
    ModelSpec(
        name="Lasso",
        estimator=lambda: Lasso(random_state=SEED, max_iter=10000),
        space=[HParam("alpha", "float", 1e-4, 10, log=True)],
        needs_scaling=True,
    ),
    ModelSpec(
        name="DecisionTree",
        estimator=lambda: DecisionTreeRegressor(random_state=SEED),
        space=[
            HParam("max_depth", "int", 2, 20),
            HParam("min_samples_split", "int", 2, 20),
            HParam("min_samples_leaf", "int", 1, 10),
        ],
    ),
    ModelSpec(
        name="RandomForest",
        estimator=lambda: RandomForestRegressor(random_state=SEED, n_jobs=1),
        space=[
            HParam("n_estimators", "int", 50, 300),
            HParam("max_depth", "int", 3, 20),
            HParam("min_samples_leaf", "int", 1, 10),
            HParam("max_features", "cat", choices=["sqrt", "log2"]),
        ],
    ),
    ModelSpec(
        name="XGBoost",
        estimator=lambda: XGBRegressor(random_state=SEED, n_jobs=1, verbosity=0),
        space=[
            HParam("n_estimators", "int", 50, 300),
            HParam("max_depth", "int", 2, 10),
            HParam("learning_rate", "float", 1e-3, 0.3, log=True),
            HParam("subsample", "float", 0.5, 1.0),
            HParam("colsample_bytree", "float", 0.5, 1.0),
        ],
    ),
    ModelSpec(
        name="SVR",
        estimator=lambda: SVR(),
        space=[
            HParam("C", "float", 1e-2, 1e2, log=True),
            HParam("gamma", "float", 1e-4, 1e1, log=True),
            HParam("epsilon", "float", 1e-3, 1.0, log=True),
            HParam("kernel", "cat", choices=["rbf", "linear"]),
        ],
        needs_scaling=True,
    ),
]

CLASSIFIER_BY_NAME = {m.name: m for m in CLASSIFIERS}
REGRESSOR_BY_NAME = {m.name: m for m in REGRESSORS}


# --------------------------------------------------------- Modelo original
# Se registra aparte de REGRESSORS para no alterar las 140 combinaciones del
# segundo entregable. Escala su estado internamente, así que no necesita el
# escalador del pipeline.
def _original_estimator():
    from .original_model import ModeloOriginal

    return ModeloOriginal(random_state=SEED)


# Iteración 4 (2026-11-05, exploratoria, posterior a la validación
# confirmatoria): ALEE pasó de 5 a 3 expertos (se quitaron ridge_nivel y
# ridge_cambio, ver original_model.py y docs/registro_desarrollo.md), así que
# ``ridge_alpha`` ya no es un hiperparámetro del modelo. El rango de ``lam``
# se amplió hasta 100 (antes 10) para confirmar con una validación anidada
# honesta -- no solo reescalando a mano sobre el pliegue de prueba -- que un
# ``lam`` más agresivo no mejora el resultado por sí solo (ver
# experiments/iter4_retune_lam.py): el optimizador, con este rango más ancho,
# sigue convergiendo a valores de lam pequeños.
ORIGINAL_MODEL = ModelSpec(
    name="ALEE",
    estimator=_original_estimator,
    space=[
        HParam("bandwidth", "float", 0.05, 3.0, log=True),
        HParam("half_life", "float", 6.0, 120.0, log=True),
        HParam("lam", "float", 1e-3, 100.0, log=True),
    ],
)


# --------------------------------------------------- Referencia SOTA (protocolo congelado)
# LightGBM se registra aparte de REGRESSORS por la misma razón que ALEE: no
# alterar las 140 combinaciones ya corridas del segundo entregable. Es uno de
# los dos métodos recientes del estado del arte fijados en el protocolo
# confirmatorio (experiments/protocolo.md, decisión 1).
def _lightgbm_estimator():
    from lightgbm import LGBMRegressor

    return LGBMRegressor(random_state=SEED, n_jobs=1, verbosity=-1)


LIGHTGBM_MODEL = ModelSpec(
    name="LightGBM",
    estimator=_lightgbm_estimator,
    space=[
        HParam("n_estimators", "int", 50, 300),
        HParam("max_depth", "int", 2, 10),
        HParam("learning_rate", "float", 1e-3, 0.3, log=True),
        HParam("subsample", "float", 0.5, 1.0),
        HParam("colsample_bytree", "float", 0.5, 1.0),
        HParam("num_leaves", "int", 7, 63),
    ],
)


# --------------------------------------------- Modelos del curso que faltaban (2026-11-11)
# La guía de resultados del curso exige comparar contra todos los modelos vistos
# en clase. En regresión faltaba la regresión bayesiana (BayesianRidge), y la
# guía pide SVR con kernel RBF (el SVR del segundo entregable podía elegir
# también el lineal). Se registran aparte para no alterar las 140 combinaciones.
from sklearn.linear_model import BayesianRidge  # noqa: E402

BAYESIAN_RIDGE_MODEL = ModelSpec(
    name="BayesianRidge",
    estimator=lambda: BayesianRidge(max_iter=500),
    space=[
        HParam("alpha_1", "float", 1e-8, 1e-2, log=True),
        HParam("alpha_2", "float", 1e-8, 1e-2, log=True),
        HParam("lambda_1", "float", 1e-8, 1e-2, log=True),
        HParam("lambda_2", "float", 1e-8, 1e-2, log=True),
    ],
    needs_scaling=True,
)

SVR_RBF_MODEL = ModelSpec(
    name="SVR_RBF",
    estimator=lambda: SVR(kernel="rbf"),
    space=[
        HParam("C", "float", 1e-2, 1e2, log=True),
        HParam("gamma", "float", 1e-4, 1e1, log=True),
        HParam("epsilon", "float", 1e-3, 1.0, log=True),
    ],
    needs_scaling=True,
)
