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

# Reproducibilidad y bibliografía

```{code-cell} ipython3
:tags: [remove-cell]
import os, sys, platform
if os.path.basename(os.getcwd()) == "book":
    os.chdir("..")
sys.path.insert(0, os.getcwd())
import warnings
warnings.filterwarnings("ignore")  # avisos de versiones de librerías que no afectan los resultados
from importlib.metadata import version
import pandas as pd
import psutil
from src import config
from src.models import CLASSIFIERS, ORIGINAL_MODEL, REGRESSORS
master = pd.read_csv(config.RESULTS_MASTER_PATH)
```

## Cómo reproducir todo

Las tablas del paper (`experiments/reproducir_todo.py`) se generaron con el entorno de `requirements-paper.txt` (Python 3.13); el resto, con `requirements.txt` (Python 3.9).

```bash
python -m venv .venv
.venv\Scripts\activate                     # en Linux o macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name mlc-ml

python -m src.runner full                  # las 140 combinaciones (cerca de una hora)
python -m src.artifacts                    # mejores combinaciones, reentrenamiento y serialización
python -m src.artifacts --original --curso # ALEE, regresión bayesiana, SVR (RBF) y LightGBM
python experiments/reproducir_todo.py      # validación confirmatoria, nueve conjuntos y tablas del paper
python scripts/build_dashboard_data.py     # datos precalculados del dashboard
python -m pytest                           # pruebas
jupyter-book build book/                   # este libro (jupyter-book 0.15.1)
python app.py                              # dashboard local
```

(lista-reproducibilidad)=
## Lista de verificación

```{code-cell} ipython3
libs = ["numpy", "pandas", "scikit-learn", "xgboost", "imbalanced-learn", "optuna", "deap", "scipy", "statsmodels", "dash", "plotly"]
filas = [("Python", platform.python_version()), ("Sistema", f"{platform.system()} {platform.release()}"),
         ("Procesador", platform.processor() or platform.machine()), ("Núcleos lógicos", psutil.cpu_count()),
         ("Memoria RAM (GB)", round(psutil.virtual_memory().total / 2**30, 1)),
         ("GPU", "AMD Radeon RX 6600 (sin CUDA; no se usó aceleración por GPU)")]
filas += [(lib, version(lib)) for lib in libs]
pd.DataFrame(filas, columns=["Elemento", "Valor"]).set_index("Elemento")
```

| Elemento | Detalle |
|---|---|
| Semilla | Una sola semilla global, `SEED = 42` en `src/config.py`, propagada a NumPy, scikit-learn, XGBoost, SMOTE y ADASYN, Optuna y DEAP |
| Entorno | `requirements.txt` (todo el proyecto) y `requirements-dash.txt` (dashboard), con versiones exactas |
| Validación | Anidada: `TimeSeriesSplit` de 5 pliegues externos y 3 internos, con 1 mes de separación |
| Presupuesto | 40 evaluaciones de hiperparámetros por pliegue externo y por método |
| Ejecuciones | 140 combinaciones del segundo entregable, 18 reentrenamientos finales, 10 iteraciones de diseño del modelo original y la evaluación de 10 modelos en 9 conjuntos de datos |
| Dispersión | Todas las métricas se reportan como media ± desviación estándar de los pliegues externos |
| Pruebas | `pytest` en cada *push* con GitHub Actions (`.github/workflows/tests.yml`) |
| Datos | `data/MLC.xlsx`, las series de los ocho países adicionales en `data/adicionales/` (`scripts/descargar_conjuntos_adicionales.py`) y `DATA_STATEMENT.md` |
| Código | Repositorio de GitHub (pendiente de publicar) |
| Archivo persistente | DOI de Zenodo a partir de una *release* de GitHub (pendiente) |

## Rangos de hiperparámetros

```{code-cell} ipython3
filas = []
for tarea, specs in [("Clasificación", CLASSIFIERS), ("Regresión", REGRESSORS), ("Modelo original", [ORIGINAL_MODEL])]:
    for spec in specs:
        for h in spec.space:
            rango = ", ".join(map(str, h.choices)) if h.kind == "cat" else f"[{h.low:g}, {h.high:g}]" + (" log" if h.log else "")
            filas.append({"Tarea": tarea, "Modelo": spec.name, "Hiperparámetro": h.name, "Tipo": h.kind, "Rango": rango})
pd.DataFrame(filas).set_index(["Tarea", "Modelo"])
```

(tabla-maestra)=
## Tabla maestra de experimentos

`experiments/results_master.csv` tiene una fila por combinación, con las métricas del bucle externo (media, desviación y número de pliegues), los tiempos, el número de evaluaciones, la semilla y los hiperparámetros de cada pliegue en JSON.

```{code-cell} ipython3
cols = ["run_id", "task", "model", "balancing", "optimizer", "seed", "f1_macro_mean", "rmse_mean", "total_evaluations", "total_wall_time_s"]
print(f"{len(master)} combinaciones; tiempo total de cómputo: {master['total_wall_time_s'].sum() / 60:.1f} minutos")
master[cols].head(8)
```

## Referencias

```{bibliography}
```

### Referencias de los entregables anteriores

```{include} _referencias_e2.md
```
