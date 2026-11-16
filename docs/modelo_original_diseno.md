# Modelo original: agregación local entrópica de expertos (ALEE)

Borrador de los capítulos 4 y 5 del libro. El código está en `src/original_model.py` y las pruebas en `tests/test_original_model.py`.

## 1. Limitación identificada (experimento piloto)

Con las 240 predicciones fuera de muestra del segundo entregable (cinco pliegues externos, diciembre de 2005 a noviembre de 2025), la mejor combinación de cada modelo de regresión y la persistencia, separando los meses por terciles de la volatilidad reciente (`td_nacional_rollstd3`, conocida al pronosticar):

| Modelo | MSE con volatilidad baja | media | alta |
|---|---|---|---|
| XGBoost (grid) | **1.804** | 1.116 | 4.453 |
| Ridge (random) | 2.313 | **0.895** | 1.999 |
| Persistencia | 2.540 | 1.438 | **1.041** |

- El mejor modelo cambia con el régimen. Con volatilidad alta, XGBoost comete un error cuatro veces mayor que la persistencia.
- El mejor modelo también cambia de un pliegue a otro: Random Forest, Ridge, XGBoost, persistencia y árbol de decisión, cada uno en un pliegue distinto.
- Si se eligiera cada mes el mejor modelo (un oráculo, que no es alcanzable), el MSE sería 0.467, frente a 1.673 del mejor modelo único. Hay un margen grande para una regla que adapte los pesos al estado del mercado laboral.

**Limitación:** las reglas de combinación estándar (promedio simple, pesos por error global, *stacking* lineal) usan los mismos pesos en todos los meses y no pueden aprovechar que la ventaja relativa de cada modelo depende del régimen.

## 2. Algoritmo

**Entradas:** `X` (n × p, en orden cronológico), `y`, hiperparámetros `h` (ancho del núcleo), `T½` (vida media del decaimiento), `λ` (regularización) y `α` (ridge de los expertos).
**Salida:** pronóstico `ŷ(x)` y pesos `w*(x)` de los expertos.

```
Entrenamiento
1. Para s = m0, …, n−1:
     ajustar los K expertos con las filas 0..s−1
     e_s ← y_s − f^(s)(x_s)                              # errores fuera de muestra (K)
2. z_s ← estado estandarizado de la fila s: [rollstd3, y_t − rollmean12]
3. π ← argmin_{w ∈ Δ} wᵀ Q_glob w + 0.01·tr(Q_glob)/K · KL(w‖u)    # pesos globales
       con Q_glob = Σ_s d_s e_s e_sᵀ,  d_s ∝ γ^(n−1−s),  γ = 0.5^(1/T½)
4. Ajustar los K expertos finales con todas las filas.

Predicción en x
5. a_s ∝ exp(−‖z(x) − z_s‖² / (2h²)) · γ^(n−1−s),   Σ a_s = 1
6. Q ← Σ_s a_s e_s e_sᵀ,  n_eff ← 1/Σ_s a_s²,  λ̃ ← λ · (n_ref/n_eff) · tr(Q)/K
   L ← 2·max_ij |Q_ij|,  η ← 1/L          # n_ref: n_eff de los pesos solo con decaimiento
7. w ← π
   repetir hasta ‖w⁺ − w‖₁ < 10⁻¹⁰ o 500 iteraciones:
       w⁺ ∝ (w ⊙ exp(−2ηQw))^(1/(1+ηλ̃)) ⊙ π^(ηλ̃/(1+ηλ̃))
8. ŷ(x) ← Σ_k w_k f_k(x)
```

Los expertos (todos programados con NumPy) son: persistencia, ridge del nivel, ridge del cambio, Nadaraya-Watson del cambio y el cambio medio del mismo mes del año.

## 3. Sustento teórico

Sea `F(w) = g(w) + λ̃ KL(w‖π)` con `g(w) = wᵀQw`, `Q ⪰ 0`, `π` en el interior del símplex `Δ`.

**Identidad del error local.** Como `Σ_k w_k = 1`, `y_s − wᵀf_s = wᵀ(y_s·1 − f_s) = wᵀe_s`, y entonces `Σ_s a_s (y_s − wᵀf_s)² = wᵀQw`. El objetivo es exactamente el error cuadrático local de la combinación.

**P1 (existencia y unicidad).** `Δ` es compacto y `F` es continua, así que hay mínimo. `g` es convexa (`Q ⪰ 0`) y `KL(·‖π)` es estrictamente convexa en `Δ`, de modo que con `λ̃ > 0` `F` es estrictamente convexa y el minimizador es único. Además, `KL` tiene gradiente que tiende a menos infinito en la frontera, así que el óptimo está en el interior y cumple `∇F = c·1` (prueba `test_P1_optimalidad_kkt`).

**P2 (descenso monótono).** Para `d = w' − w`, `dᵀ(2Q)d ≤ 2 max|Q_ij| (Σ_i |d_i|)² = L‖d‖₁²`, así que `g(w') ≤ g(w) + ∇g(w)ᵀd + (L/2)‖d‖₁²`. Por la desigualdad de Pinsker, `(1/2)‖d‖₁² ≤ KL(w'‖w)`, de donde

`g(w') ≤ g(w) + ∇g(w)ᵀ(w' − w) + L·KL(w'‖w)`.

El paso 7 con `η = 1/L` es `w⁺ = argmin_{v ∈ Δ} ∇g(w)ᵀv + λ̃ KL(v‖π) + L·KL(v‖w)`; la fórmula cerrada sale de igualar a cero la derivada del lagrangiano. Evaluando en `v = w`:

`F(w⁺) ≤ g(w) + [∇g(w)ᵀ(w⁺ − w) + λ̃ KL(w⁺‖π) + L·KL(w⁺‖w)] ≤ g(w) + λ̃ KL(w‖π) = F(w)`.

Este es el argumento de suavidad relativa de Bauschke, Bolte y Teboulle (2017) y de Lu, Freund y Nesterov (2018), que además da una tasa `F(w_T) − F(w*) ≤ L·KL(w*‖w_0)/T` (pruebas `test_P2_descenso_monotono` y `test_P1_unicidad_desde_distintos_puntos`).

**P3 (cota por los expertos).** `ŷ` es una combinación convexa, así que `min_k f_k ≤ ŷ ≤ max_k f_k`, `|y − ŷ| ≤ max_k |y − f_k|` y, por Jensen, `(y − ŷ)² ≤ Σ_k w_k (y − f_k)²` (prueba `test_P3_cota_por_el_peor_experto`).

**P4 (casos límite).** Si `λ → ∞`, `w* → π` (los pesos globales, o el promedio simple con `shrink_to="uniforme"`). Si no se localiza (`localize=False`), los pesos no dependen de `x` (pruebas `test_P4_*`). Estos casos son las reglas estándar contra las que la guía pide contrastar la regla nueva, y sirven directamente para la ablación.

**Complejidad.** Entrenamiento: `O(n·(p³ + n·p))`. En cada uno de los `n` meses se resuelven dos sistemas ridge de tamaño `p + 1` y se arma la matriz de distancias del experto de núcleo, de tamaño `s × s`. Predicción en un punto: `O(n·K² + I·K²)`, con `I` iteraciones del descenso de espejo. Medido: unos 0.5 s para ajustar con 240 meses y unos 0.3 s para predecir 48.

## 4. Trabajo relacionado (búsqueda preliminar, no sistemática)

Búsqueda web general del 2026-11-04. **No reemplaza** la revisión de la sección 5.2 de la guía: faltan Scopus o Web of Science, Google Scholar y arXiv con al menos cinco cadenas registradas, la matriz de novedad y el *snowballing*.

| Trabajo | Qué hace | En qué se diferencia ALEE |
|---|---|---|
| Li y Tkacz (2001), Banco de Canadá | Combinación no lineal de pronósticos con pesos estimados por regresión de núcleo | Revisar en el texto completo si sus pesos están restringidos al símplex y regularizados. ALEE localiza por un estado de régimen y resuelve un problema convexo con regularización entrópica |
| Cerqueira, Torgo y otros (2017, 2019), *Arbitrated Dynamic Ensemble* | Meta-aprendices (bosques aleatorios) que predicen el error de cada modelo; pesos con softmax | ALEE no entrena un meta-modelo: los pesos salen de un problema de optimización con solución única y garantías (P1 a P3) |
| Combinación dependiente del régimen (arXiv 2209.01697; Kazak y Tetereva) | Pesos que cambian por régimen, con regímenes discretos estimados | ALEE usa una localización continua (núcleo) en lugar de regímenes discretos |
| Mezclas regularizadas de densidades predictivas (arXiv 2012.11649) y gradiente exponenciado (Kivinen y Warmuth) | Penalización KL hacia el uniforme y actualización multiplicativa en el símplex | ALEE usa esas herramientas como solucionador; la parte nueva es la localización por estado y el previo global |
| Orozco-Castañeda, Sierra-Suárez y Vidal (2024); UNAL en *Computation* (2025); Lasso-Valderrama y Zárate-Solano (Banco de la República) | Pronóstico del desempleo en Colombia con ML o con combinación de pronósticos | Ninguno usa una combinación localizada por régimen; hay que verificarlo en el texto completo |

Fuentes: <https://oar-rao.bank-banque-canada.ca/record/877/files/wp01-12.pdf>, <https://bigfish.cs.dal.ca/publication/2019_ctpss19/2019_CTPSS19.pdf>, <https://arxiv.org/pdf/2209.01697v2.pdf>, <https://arxiv.org/pdf/2012.11649>, <https://hunch.net/?p=286>, <https://perfilesycapacidades.javeriana.edu.co/en/publications/labor-market-forecasting-in-unprecedented-times-a-machine-learnin/>, <https://repositorio.unal.edu.co/items/6c13ff7b-4b51-4ffc-aed4-abaf42f25ea7/full>.

**Riesgo de novedad:** la combinación localizada de pronósticos tiene antecedentes, sobre todo Li y Tkacz (2001) y ADE. Para que la declaración de novedad se sostenga, el aporte debe presentarse como el mecanismo concreto (localización continua por estado, problema convexo en el símplex con previo global y las propiedades P1 a P3), contrastado con esos trabajos.

La regularización adaptativa (paso 6) no altera P1 a P4: para cada punto `x`, `λ̃` es una constante positiva del problema de optimización.

## 5. Estado tras tres iteraciones de diseño

Ver `docs/registro_desarrollo.md`.

| Versión | RMSE medio por pliegue | RMSE agrupado |
|---|---|---|
| Iteración 1 (previo uniforme) | 0.989 ± 0.595 | 1.154 |
| Iteración 2 (previo global) | 0.973 ± 0.629 | 1.159 |
| Iteración 3 (regularización por tamaño efectivo) | 0.917 ± 0.520 | 1.054 |
| Experto estacional solo | 0.780 ± 0.299 | — |
| SARIMA | 0.818 ± 0.567 | 0.995 |
| Ridge (mejor del segundo entregable) | 1.196 ± 0.552 | 1.317 |

ALEE supera a las siete mejores combinaciones del segundo entregable y a la persistencia, pero **no** al SARIMA ni a su propio experto estacional. Casi toda la diferencia está en el pliegue con el choque de 2020. Hoy no cumple R3; el tope esperado es 4.0.
