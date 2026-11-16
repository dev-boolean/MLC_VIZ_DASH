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

# Modelo original: agregación local entrópica de expertos (ALEE)

```{admonition} Versión del modelo
:class: note
Este capítulo describe la versión de ALEE del paper, con tres expertos. La versión pre-registrada tenía cinco; los dos expertos ridge se quitaron en una iteración de diseño posterior a la validación confirmatoria. Las diez iteraciones evaluadas están registradas en `docs/registro_desarrollo.md` y `experiments/protocolo.md`.
```

```{code-cell} ipython3
:tags: [remove-cell]
import os, sys
if os.path.basename(os.getcwd()) == "book":
    os.chdir("..")
sys.path.insert(0, os.getcwd())
import warnings
warnings.filterwarnings("ignore")  # avisos de versiones de librerías que no afectan los resultados
import time, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from src import config
from src import dash_data as D
from src.data import assemble_dataset
from src.original_model import EXPERTS, ModeloOriginal
ds = assemble_dataset()
oof = D.oof("reg")
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
```

## Limitación que motiva el modelo

Si se separan las 240 predicciones fuera de muestra del segundo entregable por terciles de la volatilidad reciente (la desviación de los últimos tres meses, que se conoce al pronosticar), el mejor modelo cambia de un tercil a otro:

```{code-cell} ipython3
modelos = ["XGBoost", "Ridge", "SVR", "KNN", "RandomForest", "Lasso", "DecisionTree"]
vol = ds.X["td_nacional_rollstd3"].values[oof["idx"]]
tercil = pd.qcut(vol, 3, labels=["Baja", "Media", "Alta"])
mse = pd.DataFrame({D.MODEL_LABELS[m]: (oof["y_true"] - oof[f"pred_{m}"]) ** 2 for m in modelos}).groupby(tercil, observed=True).mean().T
mse.columns.name = "Volatilidad reciente"
mse.round(3)
```

```{code-cell} ipython3
:tags: [remove-input]
mejor = mse.idxmin()
print("Mejor modelo por tercil: " + "; ".join(f"{t}: {m}" for t, m in mejor.items()))
print("Error del mejor y del peor modelo por tercil: " + "; ".join(f"{t}: {mse[t].min():.2f} y {mse[t].max():.2f}" for t in mse.columns))
```

El modelo con menor error no es el mismo en los tres terciles. Una regla de combinación con pesos fijos (el promedio simple, los pesos por error global o el *stacking* lineal {cite}`timmermann2006`) no puede aprovechar ese cambio. **Hipótesis:** una combinación cuyos pesos dependan del estado del mercado laboral reduce el error frente a cualquier modelo único y frente a las combinaciones con pesos fijos.

## Formulación

Hay $K = 3$ expertos programados con NumPy: persistencia ($\hat y = y_t$), regresión de Nadaraya-Watson del cambio, y cambio medio del mismo mes del año (el experto estacional). Hasta la iteración 3 había dos expertos más —ridge del nivel y ridge del cambio $y_{t+1} - y_t$— que se quitaron en la iteración 4 (exploratoria, posterior a la validación confirmatoria): evaluados en solitario se equivocaban más del doble que el experto estacional y diluían su peso en la combinación sin aportar precisión (`docs/registro_desarrollo.md`). Sea $e_s \in \mathbb{R}^K$ el vector de errores **fuera de muestra** de los expertos en el mes $s$: cada experto se entrena solo con los meses anteriores a $s$. Para pronosticar en un punto $x$ con estado $z(x)$ (la volatilidad reciente y la brecha entre el último dato y la media de 12 meses, estandarizadas), los meses pasados reciben pesos

$$
a_s(x) \propto \exp\!\left(-\frac{\lVert z(x) - z_s\rVert^2}{2h^2}\right)\gamma^{\,n-1-s}, \qquad \sum_s a_s(x) = 1, \qquad \gamma = 0.5^{1/T_{1/2}}.
$$ (eq-localizacion)

Como los pesos de la combinación $w$ suman uno, $y_s - w^\top f_s = w^\top e_s$, y el error cuadrático local de la combinación es $w^\top Q(x) w$ con $Q(x) = \sum_s a_s(x)\, e_s e_s^\top$. Los pesos resuelven

$$
w^*(x) = \arg\min_{w \in \Delta_K} \; F(w) = w^\top Q(x)\, w + \tilde\lambda\, \mathrm{KL}(w \,\Vert\, \pi), \qquad
\tilde\lambda = \lambda\, \frac{n_{\mathrm{ref}}}{n_{\mathrm{eff}}(x)}\, \frac{\operatorname{tr} Q(x)}{K},
$$ (eq-objetivo)

donde $\Delta_K$ es el símplex y $\pi$ son los **pesos globales**: la misma combinación con todos los meses y solo el decaimiento temporal. $n_{\mathrm{eff}} = 1/\sum_s a_s^2$ es el tamaño efectivo de la muestra local {cite}`kish1965` y $n_{\mathrm{ref}}$ el de los pesos globales. Cuando hay pocos meses parecidos al actual, como en el choque de 2020, la regularización crece y la combinación se queda cerca de $\pi$. El factor $\operatorname{tr}Q/K$, el error local medio de los expertos, hace que $\lambda$ no dependa de las unidades. El pronóstico es $\hat y(x) = \sum_k w^*_k(x)\, f_k(x)$.

El problema se resuelve con descenso de espejo proximal (gradiente exponenciado {cite}`kivinen1997`) con paso $\eta = 1/L$ y $L = 2\max_{ij}\lvert Q_{ij}\rvert$:

$$
w_{t+1} \propto \left(w_t \odot e^{-2\eta Q w_t}\right)^{\frac{1}{1+\eta\tilde\lambda}} \odot \pi^{\frac{\eta\tilde\lambda}{1+\eta\tilde\lambda}}.
$$ (eq-actualizacion)

## Pseudocódigo

```text
Entrada: X (n × p, en orden cronológico), y, hiperparámetros h, T½, λ
Salida: pronóstico ŷ(x) y pesos w*(x)

Entrenamiento
1. para s = m0, …, n−1:
       ajustar los K expertos con las filas 0..s−1
       e_s ← y_s − f^(s)(x_s)
2. z_s ← estado estandarizado de cada fila registrada
3. π ← argmin_{w ∈ Δ} wᵀ Q_glob w + 0.01 · tr(Q_glob)/K · KL(w ‖ u)      (u: uniforme)
4. ajustar los K expertos finales con todas las filas

Predicción en x
5. a ← pesos de la ecuación (1);  n_eff ← 1 / Σ a_s²
6. Q ← Σ a_s e_s e_sᵀ;  λ̃ ← λ · (n_ref / n_eff) · tr(Q)/K;  η ← 1 / (2 max|Q_ij|)
7. w ← π
   repetir hasta ‖w_nuevo − w‖₁ < 10⁻¹⁰ o 500 iteraciones:  w ← actualización (3)
8. devolver ŷ = Σ_k w_k f_k(x)
```

## Sustento teórico

**P1 (existencia y unicidad).** $\Delta_K$ es compacto y $F$ es continua, así que existe un mínimo. $w^\top Q w$ es convexa porque $Q \succeq 0$, y $\mathrm{KL}(\cdot\Vert\pi)$ es estrictamente convexa en $\Delta_K$. Con $\tilde\lambda > 0$, $F$ es estrictamente convexa y el minimizador es único. Como el gradiente de la divergencia tiende a menos infinito en la frontera, el óptimo está en el interior y cumple $\nabla F(w^*) = c\,\mathbf{1}$.

**P2 (descenso monótono).** Para $d = w' - w$ se cumple $d^\top (2Q) d \le 2\max_{ij}\lvert Q_{ij}\rvert\, \lVert d\rVert_1^2 = L\lVert d\rVert_1^2$. Por la desigualdad de Pinsker, $\tfrac12\lVert d\rVert_1^2 \le \mathrm{KL}(w'\Vert w)$, de donde $g(w') \le g(w) + \nabla g(w)^\top (w'-w) + L\,\mathrm{KL}(w'\Vert w)$ con $g(w) = w^\top Q w$: $g$ es $L$-suave relativa a la entropía {cite}`bauschke2017,lu2018`. La actualización {eq}`eq-actualizacion` es el minimizador de $\nabla g(w_t)^\top v + \tilde\lambda\,\mathrm{KL}(v\Vert\pi) + L\,\mathrm{KL}(v\Vert w_t)$ en $\Delta_K$; evaluando ese problema en $v = w_t$ se obtiene $F(w_{t+1}) \le F(w_t)$. Además, $F(w_T) - F(w^*) \le L\,\mathrm{KL}(w^*\Vert w_0)/T$ {cite}`lu2018`. Como $\tilde\lambda$ es constante para cada $x$, la regularización adaptativa no altera la prueba.

**P3 (cota por los expertos).** $\hat y$ es una combinación convexa de los $f_k$, así que $\lvert y - \hat y\rvert \le \max_k \lvert y - f_k\rvert$ y, por la desigualdad de Jensen, $(y-\hat y)^2 \le \sum_k w_k (y - f_k)^2$.

**P4 (casos límite).** Si $\lambda \to \infty$, $w^* \to \pi$, que es la combinación global. Sin localización, los pesos no dependen de $x$. Con $\pi$ uniforme y $\lambda \to \infty$ se obtiene el promedio simple. Las reglas estándar son casos particulares, lo que define la ablación.

**Supuestos y plausibilidad.** Las filas están en orden cronológico y cada una solo usa información anterior al mes que pronostica, lo que se verificó en {doc}`02b_auditoria`. Las garantías P1 a P3 no dependen de la distribución de los datos. La que sí depende de ella es que los errores pasados en estados parecidos anticipen los errores futuros, y en un choque sin precedentes como el de 2020 ese supuesto falla.

## Implementación

La clase `ModeloOriginal` hereda de `BaseEstimator` y `RegressorMixin`, de modo que funciona con `clone`, `cross_val_score` y la validación anidada del segundo entregable.

````{dropdown} Código del solucionador y de la clase (src/original_model.py)
```{literalinclude} ../src/original_model.py
:language: python
:pyobject: solve_weights
```

```{literalinclude} ../src/original_model.py
:language: python
:pyobject: ModeloOriginal
```
````

## Pruebas unitarias

`tests/test_original_model.py` cubre los cinco puntos que exige la guía: aprendizaje con solución conocida, reproducibilidad, entradas inválidas, compatibilidad con `clone` y verificación numérica de P1 a P4. Se ejecutan al construir el libro:

```{code-cell} ipython3
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_original_model.py"], capture_output=True, text=True)
print(r.stdout.strip().splitlines()[-1])
```

## Complejidad

El entrenamiento cuesta $O(n^3 p)$ en tiempo y $O(n^2)$ en memoria: en cada uno de los $n$ meses del historial, el experto de núcleo recalcula las distancias entre todos los meses anteriores para fijar su ancho de banda con la heurística de la mediana. Predecir en un punto cuesta $O(nK^2 + I K^2)$, con $I$ iteraciones del descenso de espejo. Con los tamaños del problema (hasta 288 meses) las operaciones vectorizadas de NumPy hacen que el exponente medido quede por debajo de 3:

```{code-cell} ipython3
filas = []
for n in [72, 144, 216, 288]:
    t0 = time.perf_counter(); m = ModeloOriginal().fit(ds.X.values[:n], ds.y_reg.values[:n]); fit = time.perf_counter() - t0
    t0 = time.perf_counter(); m.predict(ds.X.values[-48:]); pred = time.perf_counter() - t0
    filas.append({"n": n, "Entrenamiento (s)": fit, "Predicción de 48 meses (s)": pred})
tiempos = pd.DataFrame(filas).set_index("n")
pendiente = np.polyfit(np.log(tiempos.index), np.log(tiempos["Entrenamiento (s)"]), 1)[0]
print(f"Exponente empírico del entrenamiento: {pendiente:.2f}")
tiempos.round(3)
```

## Iteraciones de diseño y ablación

Antes de la versión actual se evaluaron diez iteraciones de diseño sobre los 240 meses de Colombia, todas con validación anidada y registradas con fecha en `docs/registro_desarrollo.md`. Las tres primeras ajustaron el previo y la regularización de la versión de cinco expertos. La cuarta, posterior a la validación confirmatoria, quitó los dos expertos ridge: evaluados solos se equivocaban más del doble que el experto estacional y diluían su peso en la combinación. Las seis siguientes probaron otros expertos, otras variables de estado y otras reglas de combinación, y ninguna se adoptó. Como tantas iteraciones sobre los mismos meses pueden volver optimista el resultado en Colombia, la evidencia principal está en los ocho conjuntos adicionales, evaluados una sola vez con el modelo congelado (capítulo 7).

Para ver de dónde viene el desempeño, cada uno de los tres expertos se evaluó solo en los mismos pliegues externos, con los hiperparámetros elegidos para la versión actual:

```{code-cell} ipython3
hp = pd.read_csv("artifacts/confirmatorio/iter4/rmse_K3_sin_ridge.csv")
res = {k: [] for k in list(EXPERTS) + ["promedio simple", "ALEE"]}
for f, (tr, te) in enumerate(TimeSeriesSplit(config.N_SPLITS_OUTER, gap=config.CV_GAP).split(ds.X)):
    params = hp.iloc[f][["bandwidth", "half_life", "lam"]].to_dict()
    m = ModeloOriginal(**params).fit(ds.X.values[tr], ds.y_reg.values[tr])
    F = m.expert_predictions(ds.X.values[te])
    for k, nombre in enumerate(EXPERTS):
        res[nombre].append(np.sqrt(np.mean((ds.y_reg.values[te] - F[:, k]) ** 2)))
    res["promedio simple"].append(np.sqrt(np.mean((ds.y_reg.values[te] - F.mean(axis=1)) ** 2)))
    res["ALEE"].append(np.sqrt(np.mean((ds.y_reg.values[te] - m.predict(ds.X.values[te])) ** 2)))
ablacion = pd.DataFrame({k: [np.mean(v), np.std(v, ddof=1)] for k, v in res.items()}, index=["RMSE medio", "Desviación"]).T
ablacion.sort_values("RMSE medio").round(3)
```

El experto estacional es el componente más fuerte, y el promedio simple de los tres queda muy por detrás de la combinación. El capítulo 7 repite la ablación completa (sin localización, sin regularización adaptativa, previo uniforme, promedio simple y cada experto solo) en los nueve conjuntos, con pruebas de significancia.

## Interpretabilidad

Valores de Shapley {cite}`strumbelj2014` de los pronósticos de ALEE y de la mejor combinación de regresión del segundo entregable (Ridge), precalculados en `data/dashboard/shap/`:

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=False)
for ax, nombre, color in [(axes[0], "reg_ALEE", "#7c3aed"), (axes[1], "reg_Ridge", "#2563eb")]:
    sg = D.shap_global(nombre)
    imp = pd.Series(np.abs(sg["phi"]).mean(axis=0), index=[D.FEATURE_INFO[c][0] for c in sg["columns"]]).sort_values().tail(8)
    ax.barh(imp.index, imp.values, color=color)
    ax.set_title(D.MODEL_LABELS[nombre.split("_", 1)[1]])
    ax.set_xlabel("Impacto medio |Shapley| (pp)")
fig.tight_layout()
plt.show()
```

```{code-cell} ipython3
:tags: [remove-input]
for nombre in ["reg_ALEE", "reg_Ridge"]:
    sg = D.shap_global(nombre)
    imp = pd.Series(np.abs(sg["phi"]).mean(axis=0), index=[D.FEATURE_INFO[c][0] for c in sg["columns"]]).sort_values(ascending=False)
    imp = imp / imp.sum()
    print(f"{D.MODEL_LABELS[nombre.split('_', 1)[1]]}: " + "; ".join(f"{k} {v:.0%}" for k, v in imp.head(3).items()))
```

Los dos modelos usan la información de forma muy distinta. ALEE concentra la atribución en el último dato conocido y en el mes del año (seno y coseno), que son la huella de sus expertos de persistencia y estacional. Ridge la reparte entre muchas variables.
