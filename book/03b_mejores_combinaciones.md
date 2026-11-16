---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python (MLC ML)
  language: python
  name: mlc-ml
---

# Mejores combinaciones y artefactos

El dashboard no muestra las 140 combinaciones, sino la mejor de cada uno de los siete modelos base por tarea. El criterio es el que fija la guía: la mejor **métrica media del bucle externo** de la validación anidada, F1-macro en clasificación y RMSE en regresión. El código está en `src/artifacts.py`.

```{code-cell} ipython3
:tags: [remove-cell]
import os, sys, json
if os.path.basename(os.getcwd()) == "book":
    os.chdir("..")
sys.path.insert(0, os.getcwd())
import warnings
warnings.filterwarnings("ignore")  # avisos de versiones de librerías que no afectan los resultados
import numpy as np
import pandas as pd
from src import analysis as A
from src import dash_data as D
sel = pd.read_csv("artifacts/seleccion.csv")
master = A.load_master()
```

(tabla-seleccion)=
## Tabla de selección

```{code-cell} ipython3
tabla = sel.assign(
    Modelo=sel["model"].map(D.MODEL_LABELS),
    Balanceo=sel["balancing"].map(D.BALANCING_LABELS),
    Optimizador=sel["optimizer"].map(D.OPTIMIZER_LABELS),
    Tarea=sel["task"].map({"clf": "Clasificación", "reg": "Regresión"}),
)[["Tarea", "Modelo", "Balanceo", "Optimizador", "metric", "metric_mean_std", "rank_in_task", "is_global_best", "in_statistical_finalists"]]
tabla.columns = ["Tarea", "Modelo", "Balanceo", "Optimizador", "Métrica", "Media ± desv.", "Puesto", "Mejor global", "Finalista"]
tabla[["Mejor global", "Finalista"]] = tabla[["Mejor global", "Finalista"]].replace({True: "Sí", False: "No"})
tabla["Métrica"] = tabla["Métrica"].map({"f1_macro": "F1-macro", "rmse": "RMSE"})
tabla.set_index(["Tarea", "Puesto"])
```

Cada fila es la variante, entre las 16 de clasificación (cuatro balanceos por cuatro optimizadores) o las 4 de regresión, con la mejor métrica media. La pertenencia a los finalistas se comprobó con la misma batería del segundo entregable.

```{code-cell} ipython3
for criterio in sel["finalist_criterion"].unique():
    print(criterio)
```

En clasificación Friedman no rechaza la igualdad entre los siete modelos, y en regresión el Model Confidence Set conserva los 28. Las dos mejores globales, KNN y Ridge, son finalistas, pero ninguna se distingue estadísticamente de las demás. La selección por métrica media es entonces un criterio de desempate, no una prueba de superioridad.

## Media por pliegue frente a RMSE agrupado

El segundo entregable destacaba a SVR como el mejor modelo de regresión porque usaba el RMSE agrupado de las 240 predicciones. La guía pide la media de los pliegues, y con ese criterio gana Ridge:

```{code-cell} ipython3
filas = []
for _, r in sel[sel["task"] == "reg"].iterrows():
    p = A.pooled_oof(r["run_id"])
    filas.append({"Modelo": D.MODEL_LABELS[r["model"]], "RMSE medio por pliegue": r["metric_mean"],
                  "RMSE agrupado (240 meses)": np.sqrt(np.mean((p["y_true"] - p["y_pred"]) ** 2))})
pd.DataFrame(filas).set_index("Modelo").round(3)
```

Los dos criterios intercambian tres pares de posiciones: Ridge y SVR, XGBoost y Random Forest, y árbol de decisión y Lasso. Las diferencias dentro de cada par son pequeñas frente a la variabilidad entre pliegues. El RMSE agrupado pesa más el pliegue con el choque de 2020, porque sus errores son los más grandes, y eso perjudica a los modelos de árboles, que no extrapolan más allá de los valores vistos al entrenar.

## Reentrenamiento final y serialización

En la validación anidada cada pliegue externo eligió sus propios hiperparámetros, así que no existe un único conjunto óptimo que reutilizar. Para el modelo final se repitió la búsqueda con el mismo optimizador, el mismo presupuesto (40 evaluaciones) y la misma validación interna, ahora con los 288 meses. Después se ajustó el pipeline completo (balanceo, escalado y modelo) con todos los datos y se guardó con `joblib` (`compress=3`) en `artifacts/`. El balanceo solo actúa al entrenar, de modo que el pipeline guardado contiene únicamente el escalado y el modelo, y el dashboard no necesita `imbalanced-learn`.

```{code-cell} ipython3
filas = []
for nombre in D.artifact_names():
    m = D.metadata(nombre)
    filas.append({"Artefacto": nombre, "Hiperparámetros finales": json.dumps(m["hyperparameters"], ensure_ascii=False),
                  "Evaluaciones": m["final_search"]["n_evaluations"], "Tamaño (MB)": m["file"]["size_mb"],
                  "Ida y vuelta exacta": m["roundtrip_test"]["passed"]})
pd.set_option("display.max_colwidth", 90)
pd.DataFrame(filas).set_index("Artefacto")
```

Junto a cada modelo hay un `metadata.json` con las versiones de las librerías, la semilla, los hiperparámetros, el esquema de entrada (nombre, tipo y rango de las 17 variables) y las métricas del bucle externo. La **prueba de ida y vuelta** carga cada artefacto desde el disco y comprueba que sus predicciones sobre los 288 meses coinciden exactamente con las del modelo en memoria. La celda siguiente la repite al construir el libro:

```{code-cell} ipython3
fallas = []
for nombre in D.artifact_names():
    ref = np.load(f"artifacts/{nombre}/roundtrip.npz")
    X = pd.DataFrame(ref["X"], columns=list(ref["columns"]))
    if not np.array_equal(D.model(nombre).predict(X), ref["y_pred"]):
        fallas.append(nombre)
print(f"{len(D.artifact_names())} artefactos cargados; predicciones distintas: {fallas or 'ninguna'}")
```

Los artefactos pesan en total menos de 2 MB, así que caben sin problema en el plan gratuito de Render y en GitHub sin Git LFS. Además están `reg_ALEE`, el modelo original de tres expertos (capítulo 5), y los tres modelos que agregó el paper: `reg_BayesianRidge`, `reg_SVR_RBF` y `reg_LightGBM` (`python -m src.artifacts --original --curso`), con el mismo procedimiento de búsqueda bayesiana y la misma prueba de ida y vuelta.
