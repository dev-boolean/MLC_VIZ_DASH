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

# Evaluación ampliada y robustez

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
import matplotlib.pyplot as plt
from src import dash_data as D

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
VIOLETA, AZUL, GRIS, AMBAR = "#7c3aed", "#2563eb", "#64748b", "#d97706"
NOMBRES = {**D.MODEL_LABELS, "ALEE": "ALEE"}
R = D.RESULTS_DIR

def p_fmt(p):
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"
```

## Conjuntos de datos

Además de Colombia se usaron ocho tasas de desempleo mensuales sin desestacionalizar, de fuentes públicas y elegidas antes de ver resultados sobre ellas. Tres se fijaron en el protocolo original: Estados Unidos (FRED), España (Eurostat) y México (OCDE, que redistribuye la encuesta del INEGI). Las otras cinco (Canadá, Australia, Japón, Alemania y Corea del Sur) se tomaron de la OCDE con criterios de inclusión fijados antes de descargarlas: serie mensual sin desestacionalizar, al menos 240 meses consecutivos y sin promedios móviles de tres meses. Como esos países no tienen indicadores de áreas metropolitanas, en ellos todos los modelos usan las 14 variables nacionales.

```{code-cell} ipython3
obj = D.paper_targets().assign(Conjunto=lambda d: d["conjunto"].map(D.DATASETS))
obj = obj[["Conjunto", "desde", "hasta", "meses", "media", "sd", "min", "max", "asimetria", "atipicos"]]
obj.columns = ["Conjunto", "Desde", "Hasta", "Meses", "Media (%)", "Desv. (pp)", "Mín. (%)", "Máx. (%)", "Asimetría", "Atípicos"]
obj.set_index("Conjunto").round(2)
```

Los niveles van del 0.9 % de Japón al 27.1 % de España, y la distribución es asimétrica a la derecha en todos los conjuntos: los choques suben el desempleo de golpe y la recuperación es lenta.

## Desempeño en los nueve conjuntos

RMSE medio de los cinco pliegues externos. La fila en negrita de cada conjunto es el menor error:

```{code-cell} ipython3
tabla = D.paper_table().rename(index=D.DATASETS, columns=NOMBRES)
tabla.round(3).style.highlight_min(axis=1, props="font-weight: bold; background-color: #f5f3ff").format("{:.3f}")
```

```{code-cell} ipython3
:tags: [remove-input]
t = D.paper_table()
otros = t.drop(columns="ALEE")
red = 1 - t["ALEE"] / otros.min(axis=1)
print(f"ALEE tiene el menor RMSE en {(t['ALEE'] < otros.min(axis=1)).sum()} de {len(t)} conjuntos. "
      f"Frente al mejor modelo de comparación, el error baja entre {red.min():.0%} ({D.DATASETS[red.idxmin()]}) "
      f"y {red.max():.0%} ({D.DATASETS[red.idxmax()]}).")
```

```{code-cell} ipython3
filas = []
for m in ["MAE", "MAPE", "R2", "MASE"]:
    tm = D.paper_table(m)
    mejor = tm.idxmax(axis=1) if m == "R2" else tm.idxmin(axis=1)
    filas.append({"Métrica": m, "Conjuntos donde ALEE es el mejor": f"{(mejor == 'ALEE').sum()} de {len(tm)}",
                  "ALEE en Colombia": round(tm.loc["colombia", "ALEE"], 3)})
pd.DataFrame(filas).set_index("Métrica")
```

El resultado no depende de la métrica: ALEE también es el mejor en MAE, MAPE, R² y MASE en los nueve conjuntos. El MASE se escala con el error absoluto medio del pronóstico ingenuo estacional dentro del entrenamiento de cada pliegue {cite}`hyndman2006`.

## Pruebas por conjunto y entre conjuntos

```{code-cell} ipython3
pr = D.paper_tests()
resumen = pr.groupby("conjunto").agg(
    dm=("dm_p_holm", lambda p: f"{(p < 0.05).sum()} de {len(p)}"), dm_max=("dm_p_holm", "max"),
    gw=("gw_p_holm", lambda p: f"{(p < 0.05).sum()} de {len(p)}"), gw_max=("gw_p_holm", "max"),
    bayes=("p_bayes_mejor", "min")).reindex(list(D.DATASETS))
resumen.index = resumen.index.map(D.DATASETS)
resumen["dm_max"] = resumen["dm_max"].map(p_fmt)
resumen["gw_max"] = resumen["gw_max"].map(p_fmt)
resumen.columns = ["DM-HLN significativas", "p máximo DM-HLN", "GW significativas", "p máximo GW", "P(ALEE mejor) mínima"]
resumen.round(3)
```

En los nueve conjuntos, Diebold-Mariano-HLN y Giacomini-White con Holm rechazan la igualdad frente a cada uno de los nueve modelos. Entre conjuntos, la prueba de Friedman compara los rangos de los diez modelos:

```{code-cell} ipython3
fr = D.paper_friedman()
rangos = pd.Series(fr["avg_ranks"]).sort_values()
fig, ax = plt.subplots(figsize=(9, 2.6))
ax.hlines(0, 1, len(rangos), color="#0f172a")
for i, (m, r) in enumerate(rangos.items()):
    y = 0.35 + 0.22 * (i % 3) if i % 2 == 0 else -0.35 - 0.22 * (i % 3)
    ax.plot([r, r], [0, y], color=VIOLETA if m == "ALEE" else GRIS, lw=1)
    ax.text(r, y, f"{NOMBRES[m]} ({r:.2f})", ha="center", va="bottom" if y > 0 else "top", fontsize=8,
            color=VIOLETA if m == "ALEE" else "#0f172a", fontweight="bold" if m == "ALEE" else "normal")
ax.plot([1, 1 + fr["cd"]], [-1.25, -1.25], color=AMBAR, lw=3)
ax.text(1 + fr["cd"] / 2, -1.33, f"Diferencia crítica de Nemenyi = {fr['cd']:.2f}", ha="center", va="top", fontsize=8, color=AMBAR)
ax.set_xlim(0.5, len(rangos) + 0.5); ax.set_ylim(-1.7, 1.3); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlabel("Rango medio en los nueve conjuntos (1 = mejor)")
ax.set_title(f"Friedman con Iman-Davenport: χ² = {fr['chi2']:.1f}, p = {p_fmt(fr['p_value'])}", fontsize=10)
plt.show()
```

```{code-cell} ipython3
entre = D.paper_between()
t = pd.DataFrame({"Modelo": entre["baseline"].map(D.MODEL_LABELS),
                  "Conjuntos que gana ALEE": entre["conjuntos_ALEE_gana"].astype(str) + " de 9",
                  "Wilcoxon p (Holm)": entre["wilcoxon_p_holm"].round(3),
                  "Conover p (Holm)": entre["conover_p_holm"].map(p_fmt),
                  "P(ALEE mejor), signed-rank bayesiano": entre["p_bayes_mejor"].round(3)})
t.set_index("Modelo")
```

ALEE queda primero en los nueve conjuntos (rango medio 1.00). Con nueve conjuntos, el Wilcoxon no puede bajar de p = 0.0039 sin corregir; tras Holm queda en 0.035 frente a cada modelo. El Conover distingue a ALEE de ocho de los nueve modelos; frente a la regresión bayesiana no alcanza la significancia. Nueve conjuntos dan una potencia moderada: la guía recomienda diez o más para Friedman.

## Ablación

Cada variante quita una pieza de ALEE y se compara con el modelo completo con Diebold-Mariano-HLN y Holm. Las hipótesis, anunciadas en la metodología del paper, son que cada pieza mejora al modelo: la localización por estado (H1), la regularización adaptativa al tamaño efectivo (H2), el previo global (H3), la combinación aprendida frente al promedio simple (H4) y la combinación frente a cada experto solo (H5).

```{code-cell} ipython3
abl = D.paper_ablation()
tabla = abl.pivot(index="variante", columns="conjunto", values="RMSE").reindex(list(D.ABLATION_ORDER))[list(D.DATASETS)]
tabla.columns = [D.DATASETS[c] for c in tabla.columns]
sig = abl[(abl["dm_p_holm"] < 0.05) & (abl["dif_perdida"] < 0)].groupby("variante")["conjunto"].apply(
    lambda s: ", ".join(D.DATASETS[c] for c in s))
tabla["Significativamente peor que ALEE completo en"] = sig.reindex(tabla.index).fillna("—")
tabla.round(3)
```

```{code-cell} ipython3
:tags: [remove-input]
peor = abl[(abl["dm_p_holm"] < 0.05) & (abl["dif_perdida"] < 0)]
mejor = abl[(abl["dm_p_holm"] < 0.05) & (abl["dif_perdida"] > 0)]
for v in D.ABLATION_ORDER[1:]:
    n = (peor["variante"] == v).sum()
    print(f"{v}: significativamente peor en {n} de 9 conjuntos")
print(f"Variantes significativamente mejores que ALEE completo en algún conjunto: {len(mejor)}")
```

La mayor parte de la ventaja viene de los expertos. La combinación aprendida supera al promedio simple en siete de los nueve conjuntos (H4) y, en Canadá, Australia y Japón, también al experto estacional solo (H5). La localización por estado, que era la idea central del modelo, solo aporta de forma significativa en México y Japón (H1), y la regularización adaptativa en Canadá y Japón (H2). El previo global no cambia el resultado en ningún conjunto (H3). El cambio de ganador según el régimen existe entre los modelos genéricos, pero no tanto entre los expertos de ALEE, porque el estacional domina en casi todos los regímenes {cite}`timmermann2006`.

## Robustez

### Semillas del optimizador

```{code-cell} ipython3
sem = pd.read_csv(R / "sensibilidad_semillas.csv")
sem = sem[sem["semilla"].astype(str).str.isdigit()]
t = sem.groupby("conjunto")["rmse_medio"].agg(["mean", "std", "min", "max"]).reindex([c for c in D.DATASETS if c in set(sem["conjunto"])])
t.index = t.index.map(D.DATASETS)
t.columns = ["RMSE medio", "Desviación", "Mínimo", "Máximo"]
t.round(4)
```

Con las semillas 42 a 46 del optimizador bayesiano, el RMSE de ALEE cambia en la tercera cifra decimal: la búsqueda de hiperparámetros es estable.

### Sensibilidad a los hiperparámetros

```{code-cell} ipython3
sh = pd.read_csv(R / "colombia" / "sensibilidad_hiperparametros.csv")
mapa = sh.pivot(index="lam", columns="bandwidth", values="rmse_medio")
fig, ax = plt.subplots(figsize=(6.5, 4.2))
im = ax.imshow(mapa.values, origin="lower", cmap="Purples_r", aspect="auto")
ax.set_xticks(range(mapa.shape[1])); ax.set_xticklabels([f"{v:.2g}" for v in mapa.columns])
ax.set_yticks(range(mapa.shape[0])); ax.set_yticklabels([f"{v:.2g}" for v in mapa.index])
ax.set_xlabel("Ancho de banda h"); ax.set_ylabel("λ")
ax.set_title("RMSE de ALEE en Colombia (vida media fija de 60 meses)", fontsize=10)
fig.colorbar(im, ax=ax, label="RMSE medio por pliegue (pp)")
plt.show()
print(f"El RMSE va de {sh['rmse_medio'].min():.3f} a {sh['rmse_medio'].max():.3f}; "
      f"la mitad de las {len(sh)} combinaciones queda por debajo de {sh['rmse_medio'].median():.3f}.")
```

El error es casi plano en una región amplia de la rejilla. Solo empeora con un ancho de banda muy pequeño y λ muy baja, cuando cada pronóstico depende de muy pocos meses parecidos. En toda la rejilla ALEE queda muy por debajo del mejor modelo de comparación en Colombia.

### Tamaño de muestra

```{code-cell} ipython3
ca = pd.read_csv(R / "curva_aprendizaje_eeuu.csv")
ca = ca[ca["modelo"].isin(["ALEE", "Ridge"])]
fig, ax = plt.subplots(figsize=(6.5, 3.6))
for m, color, mk in [("ALEE", VIOLETA, "D"), ("Ridge", AZUL, "o")]:
    g = ca[ca["modelo"] == m]
    ax.plot(g["meses_entrenamiento"], g["rmse"], marker=mk, color=color, label=NOMBRES[m])
ax.set_xscale("log"); ax.set_xlabel("Meses de entrenamiento"); ax.set_ylabel("RMSE (pp)")
ax.set_title("Curva de aprendizaje en EE. UU. (últimos 156 meses como prueba)", fontsize=10); ax.legend()
plt.show()
```

ALEE alcanza su error con solo 60 meses de entrenamiento y casi no cambia al tener más historia, porque sus expertos aprovechan la estructura de la serie desde el principio. Ridge necesita mucha más historia y aun con 777 meses queda por encima.

### Errores por régimen de volatilidad

```{code-cell} ipython3
reg = D.paper_regimes()
mejor = reg.loc[reg.groupby(["conjunto", "volatilidad"])["rmse"].idxmin()]
print(f"ALEE es el mejor modelo en {(mejor['modelo'] == 'ALEE').sum()} de {len(mejor)} combinaciones de conjunto y tercil de volatilidad.")
t = reg[reg["conjunto"] == "colombia"].pivot(index="modelo", columns="volatilidad", values="rmse")[["baja", "media", "alta"]]
t.index = t.index.map(NOMBRES)
t.columns = ["Volatilidad baja", "Media", "Alta"]
t.sort_values("Media").round(3)
```

### Incertidumbre

```{code-cell} ipython3
c = D.conformal()["ALEE"]
f = c["forward"]
print(f"Intervalo de predicción conforme al 90 % de ALEE en Colombia: semiancho de {c['halfwidth_90']:.2f} pp. "
      f"Medida hacia adelante (cada mes con los errores anteriores), la cobertura fue {f['coverage']:.1%} en {f['n']} meses, "
      f"con un ancho medio de {f['mean_width']:.2f} pp.")
```

La cobertura medida hacia adelante queda cerca de la nominal {cite}`angelopoulos2023`.

### Costo computacional

```{code-cell} ipython3
cost = pd.read_csv(R / "costo_computacional.csv")
cost = cost[cost["modelo"].isin(D.PAPER_MODELS)].set_index("modelo")
cost.index = cost.index.map(NOMBRES)
cost = cost.assign(**{"Entrenamiento (ms)": cost["entrenamiento_s"] * 1000, "Inferencia (ms)": cost["inferencia_s"] * 1000})
cost[["Entrenamiento (ms)", "Inferencia (ms)", "memoria_pico_MB", "parametros"]].rename(
    columns={"memoria_pico_MB": "Memoria pico (MB)", "parametros": "Parámetros"}).sort_values("Entrenamiento (ms)").round(2)
```

Medido en el último pliegue de Colombia, con los hiperparámetros de la búsqueda bayesiana y cada modelo en un proceso independiente. ALEE es el más lento, pero entrena en menos de un segundo y pronostica 48 meses en medio segundo: para un pronóstico mensual ese costo es irrelevante.
