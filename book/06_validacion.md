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

# Protocolo y validación estadística

```{code-cell} ipython3
:tags: [remove-cell]
import os, sys
if os.path.basename(os.getcwd()) == "book":
    os.chdir("..")
sys.path.insert(0, os.getcwd())
import warnings
warnings.filterwarnings("ignore")  # avisos de versiones de librerías que no afectan los resultados
import pandas as pd
from src import dash_data as D

def p_fmt(p):
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"

pruebas = D.paper_tests()
rmse = D.paper_table()
```

## Protocolo pre-registrado

El protocolo experimental se escribió y se congeló el 5 de noviembre de 2026, antes de la validación confirmatoria. Fijó la métrica primaria (el RMSE medio de los cinco pliegues externos), las particiones, el nivel de significancia, las pruebas de hipótesis y los conjuntos de datos adicionales. Todas las desviaciones posteriores y las iteraciones de diseño quedaron registradas con fecha en `experiments/protocolo.md` y `docs/registro_desarrollo.md`.

La versión pre-registrada de ALEE tenía cinco expertos. La versión de tres expertos que se presenta en este libro resulta de una iteración de diseño posterior a la validación confirmatoria, ajustada también con validación anidada. En total se evaluaron diez iteraciones de diseño sobre los 240 meses de Colombia. Por eso los ocho conjuntos adicionales, evaluados una sola vez con el modelo ya congelado, son la evidencia principal de que el resultado generaliza (capítulo 7).

| Elemento | Estado | Detalle |
|---|---|---|
| Métrica primaria, particiones y α | Pre-registrado | Protocolo congelado el 5 de noviembre de 2026 |
| Diebold-Mariano-HLN, Giacomini-White y Friedman | Pre-registrado | — |
| Conjuntos adicionales: EE. UU., España y México | Pre-registrado | Evaluados una sola vez, con el modelo congelado |
| LightGBM | Pre-registrado | Método reciente fijado en el protocolo |
| ALEE de tres expertos | Posterior | Iteración de diseño que quitó los dos expertos ridge |
| Siete iteraciones de diseño más | Posterior | Otros expertos, otros estados de localización y otras reglas de combinación; ninguna se adoptó |
| Canadá, Australia, Japón, Alemania y Corea del Sur | Posterior | Países y criterios de inclusión fijados antes de descargar los datos; ALEE se evaluó una sola vez, congelado |
| Regresión bayesiana y SVR con kernel RBF | Posterior | Exigidos por la guía del curso |
| Prueba bayesiana con ROPE | Posterior | ROPE de ±1 % fijado antes de calcularla |
| MAPE, R², MASE, Holm por prueba y Wilcoxon | Posterior | Exigidos por la guía |

## Modelos y validación

ALEE se compara con los ocho modelos de regresión del curso (KNN, Ridge, Lasso, regresión bayesiana, árbol de decisión, Random Forest, XGBoost y SVR con kernel RBF) y con LightGBM. Todos usan las mismas particiones, el mismo preprocesamiento y el mismo presupuesto: validación anidada con `TimeSeriesSplit` de 5 pliegues externos y 3 internos con un mes de separación, y optimización bayesiana de 40 evaluaciones con semilla 42 en el bucle interno.

## Pruebas de hipótesis

Como los errores de pronóstico forman una serie de tiempo, se usaron las pruebas que la guía indica para datos con dependencia temporal:

| Nivel | Prueba | Hipótesis nula |
|---|---|---|
| Por conjunto | Diebold-Mariano con la corrección de Harvey-Leybourne-Newbold | ALEE y el modelo de comparación tienen el mismo error cuadrático medio |
| Por conjunto | Giacomini-White condicional | Igual capacidad predictiva condicional a la información disponible |
| Por conjunto | t bayesiana correlacionada con ROPE de ±1 % | La diferencia relativa de RMSE está dentro de la región de equivalencia práctica |
| Entre conjuntos | Friedman con Iman-Davenport | Los diez modelos tienen el mismo rango medio |
| Entre conjuntos | Wilcoxon de rangos con signo y Conover, con Holm | ALEE y cada modelo tienen el mismo desempeño en los nueve conjuntos |

En cada conjunto se aplica la corrección de Holm sobre los nueve contrastes de ALEE frente a cada modelo. El tamaño del efecto se reporta con la diferencia media de pérdida cuadrática (con su intervalo del 95 % por bootstrap estacionario de bloque medio de 12 meses), la d de Cohen y la A₁₂ de Vargha-Delaney.

## Resultados en Colombia

```{code-cell} ipython3
col = pruebas[pruebas["conjunto"] == "colombia"]
tabla = pd.DataFrame({
    "Modelo": col["baseline"].map(D.MODEL_LABELS),
    "RMSE del modelo": col["baseline"].map(rmse.loc["colombia"]).round(3),
    "Diferencia de pérdida [IC 95 %]": col["dif_perdida_media"].round(3).astype(str) + " [" + col["ic95_inf"].round(3).astype(str)
                                       + ", " + col["ic95_sup"].round(3).astype(str) + "]",
    "DM-HLN p (Holm)": col["dm_p_holm"].map(p_fmt),
    "GW p (Holm)": col["gw_p_holm"].map(p_fmt),
    "P(ALEE mejor)": col["p_bayes_mejor"].round(3),
    "d de Cohen": col["d_cohen"].round(2),
}).set_index("Modelo")
print(f"RMSE medio por pliegue de ALEE en Colombia: {rmse.loc['colombia', 'ALEE']:.3f}")
tabla
```

```{code-cell} ipython3
:tags: [remove-input]
otros = rmse.loc["colombia"].drop("ALEE")
print(f"El mejor modelo de comparación en Colombia es {D.MODEL_LABELS[otros.idxmin()]} ({otros.min():.3f}); "
      f"ALEE tiene {1 - rmse.loc['colombia', 'ALEE'] / otros.min():.0%} menos error. "
      f"Diebold-Mariano-HLN rechaza en {(col['dm_p_holm'] < 0.05).sum()} de {len(col)} comparaciones "
      f"y Giacomini-White en {(col['gw_p_holm'] < 0.05).sum()} de {len(col)}.")
```

La diferencia de pérdida es la de ALEE menos la del modelo de comparación: un valor negativo indica que ALEE se equivoca menos. En Colombia las dos pruebas frecuentistas rechazan la igualdad frente a los nueve modelos después de la corrección de Holm. Las probabilidades de la prueba bayesiana son orientativas, porque esa prueba supone particiones intercambiables y aquí se aplica sobre pliegues temporales dependientes.

## Criterio de superioridad

La condición R3 de la guía pide que el modelo original supere a los modelos de comparación con al menos tres pruebas significativas después de corregir por comparaciones múltiples, y con un tamaño del efecto no despreciable.

```{code-cell} ipython3
:tags: [remove-input]
entre = D.paper_between()
fr = D.paper_friedman()
d_col = pruebas.loc[pruebas["conjunto"] == "colombia", "d_cohen"]
filas = [
    ("Diebold-Mariano-HLN (Holm), por conjunto", f"{(pruebas['dm_p_holm'] < 0.05).sum()} de {len(pruebas)} rechazan; p máximo {p_fmt(pruebas['dm_p_holm'].max())}"),
    ("Giacomini-White (Holm), por conjunto", f"{(pruebas['gw_p_holm'] < 0.05).sum()} de {len(pruebas)} rechazan; p máximo {p_fmt(pruebas['gw_p_holm'].max())}"),
    ("Friedman con Iman-Davenport", f"χ² = {fr['chi2']:.1f}, p = {p_fmt(fr['p_value'])}"),
    ("Wilcoxon (Holm), entre conjuntos", f"p = {entre['wilcoxon_p_holm'].max():.3f} frente a cada uno de los {len(entre)} modelos"),
    ("Conover (Holm), entre conjuntos", f"p < 0.05 frente a {(entre['conover_p_holm'] < 0.05).sum()} de {len(entre)} modelos"),
    ("d de Cohen en Colombia", f"entre {d_col.max():.2f} y {d_col.min():.2f} (negativo: favorece a ALEE)"),
]
pd.DataFrame(filas, columns=["Prueba", "Resultado"]).set_index("Prueba")
```

Las pruebas por conjunto y las pruebas entre conjuntos rechazan la igualdad a favor de ALEE. El Wilcoxon alcanza el mínimo posible con nueve conjuntos, y el Conover no distingue a ALEE de la regresión bayesiana, la segunda mejor en rango medio. El capítulo 7 presenta los resultados en los nueve conjuntos y los estudios de robustez.

El protocolo completo, con todas las desviaciones declaradas y las iteraciones de diseño, está en `experiments/protocolo.md`.
