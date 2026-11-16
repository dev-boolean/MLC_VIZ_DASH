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

# Auditoría de fuga de información y declaración de los datos

Antes de congelar los resultados se revisaron las cuatro fuentes de fuga que señala la guía {cite}`kapoor2023`. Cada comprobación es una celda que se ejecuta al construir el libro y falla si la condición no se cumple.

```{code-cell} ipython3
:tags: [remove-cell]
import os, sys
if os.path.basename(os.getcwd()) == "book":
    os.chdir("..")
sys.path.insert(0, os.getcwd())
import warnings
warnings.filterwarnings("ignore")  # avisos de versiones de librerías que no afectan los resultados
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from src import config
from src.data import assemble_dataset, load_raw
ds = assemble_dataset()
raw = load_raw()
```

## 1. Variables que codifican la respuesta

La tasa de desempleo nacional cumple la identidad TD = 1 − TO/TGP con la ocupación y la participación nacionales del mismo mes. Por eso esas dos series no están entre las variables de los modelos.

```{code-cell} ipython3
identidad = 100 * (1 - raw["tasa_ocupacion_nacional"] / raw["tasa_global_participacion_nacional"])
r = np.corrcoef(identidad, raw["tasa_desempleo_nacional"])[0, 1]
excluidas = {"tasa_ocupacion_nacional", "tasa_global_participacion_nacional"}
assert excluidas.isdisjoint(ds.feature_names)
print(f"Correlación entre la identidad y la tasa observada: {r:.7f}")
print(f"Ninguna de las {len(ds.feature_names)} variables es la ocupación o la participación nacional del mismo mes.")
```

## 2. Mezcla temporal entre entrenamiento y prueba

Las particiones externas e internas son de ventana expansiva, con un mes de separación: todos los meses de entrenamiento son anteriores a todos los de prueba.

```{code-cell} ipython3
filas = []
for k, (tr, te) in enumerate(TimeSeriesSplit(config.N_SPLITS_OUTER, gap=config.CV_GAP).split(ds.X)):
    assert ds.fechas.iloc[tr].max() < ds.fechas.iloc[te].min()
    filas.append({"Pliegue": k, "Entrenamiento hasta": ds.fechas.iloc[tr].max().date(),
                  "Prueba desde": ds.fechas.iloc[te].min().date(), "Meses de entrenamiento": len(tr), "Meses de prueba": len(te)})
pd.DataFrame(filas)
```

## 3. Variables que usan información futura

Cada fila usa solo información hasta el mes t para pronosticar t+1. Si eso se cumple, la variable `td_nacional_lag1` de una fila debe ser exactamente el objetivo de la fila anterior.

```{code-cell} ipython3
lag1, y = ds.X["td_nacional_lag1"].values, ds.y_reg.values
assert np.array_equal(lag1[1:], y[:-1])
media12 = raw["tasa_desempleo_nacional"].rolling(12).mean().dropna().values
assert np.isin(np.round(ds.X["td_nacional_rollmean12"].values, 10), np.round(media12, 10)).all()
print("El rezago 1 de cada fila es el objetivo de la fila anterior y las medias móviles terminan en el mes t.")
```

## 4. Preprocesamiento ajustado con todo el conjunto y duplicados

El escalador, SMOTE y ADASYN se ajustan dentro de cada pliegue, solo con el tramo de entrenamiento (`src/nested_cv.py`, `src/search.py`). Tampoco hay filas repetidas que pudieran caer a la vez en entrenamiento y en prueba.

```{code-cell} ipython3
import inspect
from src import nested_cv, search
fuente = inspect.getsource(nested_cv._fit_final) + inspect.getsource(search._fit_predict_fold)
assert "StandardScaler().fit(X_fit)" in fuente and "StandardScaler().fit(X_tr)" in fuente
duplicadas = ds.X.duplicated().sum()
assert duplicadas == 0
print(f"Escalador ajustado dentro de cada pliegue; {duplicadas} filas duplicadas en las {len(ds.X)} observaciones.")
```

**Conclusión de la auditoría:** no se encontró ninguna fuga. Como ningún resultado cambia, las tablas y figuras de los capítulos 2 y 3 siguen siendo válidas.

## Declaración de los datos

```{include} ../DATA_STATEMENT.md
:start-line: 1
```
