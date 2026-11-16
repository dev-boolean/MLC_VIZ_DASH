# Protocolo experimental pre-registrado

> **Estado: congelado (2026-11-05).** Las 5 decisiones que estaban marcadas `[DECIDIR]` se resolvieron y confirmó el equipo el 2026-11-05 (ver sección 9). La validación confirmatoria del problema principal ya se corrió bajo este protocolo (`experiments/run_confirmatorio.py`, resultados en `artifacts/confirmatorio/`). **Nota de reproducibilidad:** esta carpeta de trabajo no es un repositorio Git clonado (`git log` falla aquí), así que no se pudo crear el tag `protocolo-v1` real que pide este párrafo. El equipo debe crear ese tag en su repositorio real al incorporar estos cambios: `git tag -a protocolo-v1 -m "Protocolo congelado 2026-11-05"`.

## 1. Hipótesis

**Confirmatoria (H1):** el modelo original (ALEE) tiene menor error cuadrático de pronóstico a un mes que el mejor modelo de referencia, en los 240 meses fuera de muestra del problema principal.

**Exploratorias:** ALEE es mejor que cada una de las siete mejores combinaciones; la localización por estado mejora frente a los pesos globales (ablación); las ventajas se mantienen en los conjuntos adicionales.

**Resultado confirmatorio (2026-11-05): H1 se rechaza.** Ver sección 10.

## 2. Métricas

- **Principal:** RMSE medio de los pliegues externos (el criterio del bloque B).
- **Secundarias:** RMSE y MAE agrupados de las 240 predicciones, cobertura y ancho de los intervalos de predicción (predicción conforme) y CRPS si el modelo entrega distribuciones.

## 3. Modelos comparados

- **ALEE:** los hiperparámetros `bandwidth`, `half_life`, `lam` y `ridge_alpha` se ajustan con el optimizador bayesiano (40 evaluaciones) dentro de la validación anidada.
- **Las siete mejores combinaciones del segundo entregable:** `artifacts/seleccion.csv`.
- **Referencias:** persistencia, estacional (`y_{t-11}`), SARIMA(0,1,1)(0,1,1)[12] y el experto estacional de ALEE (último valor más el cambio medio del mismo mes), que en el desarrollo superó a todos.
- **Dos métodos recientes del estado del arte (decisión 1, confirmada 2026-11-05): LightGBM y TabPFN.** Se descartó N-BEATS/N-HiTS: con 288 observaciones una red profunda está en desventaja estructural de datos y el riesgo de implementación no se justificaba para el plazo de entrega.
  - **Desviación declarada (2026-11-05):** LightGBM sí se corrió bajo este protocolo. **TabPFN no se pudo ejecutar**: el paquete `tabpfn` (verificado instalable, versión pública 9.2.0) descarga sus pesos pre-entrenados desde un repositorio de Hugging Face con licencia *gated* (`Prior-Labs/tabpfn_3_5`), que exige que una persona inicie sesión en una cuenta de Hugging Face y acepte los términos en el navegador (`hf auth login` o un `HF_TOKEN`) antes de poder descargar el modelo. Eso no se pudo resolver de forma automática en el entorno de ejecución. **Pendiente para el equipo:** alguien del equipo acepta la licencia en <https://huggingface.co/Prior-Labs/tabpfn_3_5> con su propia cuenta, genera un `HF_TOKEN` de lectura, y se vuelve a correr `experiments/run_confirmatorio.py` (ya tiene todo lo demás cacheado; solo falta esa pieza) para completar esta decisión tal como se aprobó.

## 4. Datos, particiones y semillas

- **Problema principal:** `data/MLC.xlsx`, 288 filas, mismas 17 variables para todos los modelos. Particiones: `TimeSeriesSplit(n_splits=5, gap=1)` externo y `TimeSeriesSplit(n_splits=3, gap=1)` interno. Semilla 42.
- **Validación externa final (cambio de distribución) — decisión 2, confirmada 2026-11-05:** meses de enero a septiembre de 2026 publicados por el DANE (vía Banco de la República), fecha de corte 2026-11-05.
  - **Desviación declarada (2026-11-05):** se verificaron y documentaron los 9 valores reales de la tasa de desempleo nacional de enero a septiembre de 2026 (fuente DANE/GEIH, cruzados contra múltiples medios y los boletines técnicos oficiales del DANE) en `artifacts/confirmatorio/colombia_2026_tasa_nacional_verificada.csv`. **Pero la validación externa en sí no se pudo ejecutar**: requiere también los valores reales de 2026 de los tres indicadores de las 13 áreas metropolitanas (`tasa_global_participacion_area`, `tasa_desempleo_area`, `tasa_ocupacion_area`), que son parte de las 17 variables de entrada de todos los modelos, y esas series no están en boletines de prensa — hay que descargarlas de las tablas estadísticas del DANE/Banco de la República, algo que el entorno de ejecución usado no pudo hacer (sin acceso a herramientas de descarga de páginas web en esa corrida). **Pendiente para el equipo:** descargar esas tres series para 2026 y correr el pronóstico de los 9 meses con los modelos ya entrenados.
- **Repeticiones (decisión 3, confirmada 2026-11-05):** 5 orígenes de la ventana expansiva, consistente con el `n_splits=5` del problema principal.
- **Semillas:** los benchmarks estocásticos (Random Forest, XGBoost, LightGBM) se corren con diez semillas (42 a 51). ALEE es determinista dados sus hiperparámetros; su variabilidad por semilla viene solo del optimizador.

## 5. Conjuntos adicionales (elegidos antes de ver resultados)

Criterio: tasas de desempleo mensuales sin desestacionalizar, de fuentes públicas, con al menos 20 años de historia y de naturaleza distinta. Se construyen las mismas variables con `src/data.py`.

**Decisión 4, confirmada 2026-11-05: Estados Unidos (FRED, `UNRATENSA`), España (Eurostat, `une_rt_m`) y México (INEGI, ENOE tasa de desocupación mensual nacional).** Chile queda fuera de la ronda confirmatoria porque su serie oficial es trimestre móvil, estructuralmente distinta a una serie mensual puntual.

**Desviación declarada (2026-11-05): ninguno de los tres conjuntos se pudo ensamblar en esta corrida.** Se confirmó que las tres fuentes publican series mensuales reales con suficiente historia (verificado por búsqueda web: FRED actualizado a septiembre de 2026, Eurostat con reportes de paro de septiembre de 2026, INEGI con boletines ENOE mensuales regulares hasta agosto de 2026), pero construirlas requiere descargar series completas (CSV/API), y el entorno de ejecución usado para esta corrida no tuvo acceso a herramientas de descarga de páginas web (solo pudo ejecutar código y búsquedas de texto). **Pendiente para el equipo:** descargar las tres series completas y correr `experiments/run_confirmatorio.py` extendido a estos conjuntos (la infraestructura — modelos, optimización, pruebas estadísticas — ya está lista y probada sobre el problema principal).

| Conjunto | Fuente | Por qué es distinto |
|---|---|---|
| Estados Unidos, `UNRATENSA` | FRED | Economía grande, estacionalidad débil, choque de 2020 extremo |
| España, desempleo mensual sin ajustar | Eurostat `une_rt_m` | Nivel alto y estacionalidad turística fuerte |
| México, tasa de desocupación | INEGI (ENOE) | Mercado latinoamericano con informalidad alta, cambio metodológico en 2020 |

## 6. Pruebas de hipótesis (series de tiempo, Tabla 3 de la guía)

1. **Model Confidence Set** sobre todos los modelos con ALEE incluido (pérdida cuadrática, α = 0.10, bloques de 12 meses, 5000 remuestras). **Corrido.**
2. **Giacomini-White** condicional: ALEE frente a cada finalista del MCS. **Corrido.**
3. **Diebold-Mariano con corrección HLN**: ALEE frente al mejor de referencia, con intervalo de bootstrap estacionario (bloque medio de 12 meses). **Corrido.**
4. **SPA de Hansen**: control de *data snooping* sobre todo el universo. **Corrido.**
5. **Entre conjuntos:** Friedman con Iman-Davenport y Wilcoxon-Holm por pares. **No corrido** (depende de los conjuntos adicionales de la sección 5, no ensamblados).

Corrección por comparaciones múltiples: Holm-Bonferroni sobre todas las pruebas dirigidas, incluidas las de ablación y las de los conjuntos adicionales. Nivel α = 0.05 (0.10 para el MCS). Tamaño del efecto: d de Cohen sobre las diferencias de pérdida y delta de Cliff entre pliegues.

**Criterio de superioridad (sección 6.4 de la guía):** se cumplen a la vez (a) mejor métrica media que el mejor de referencia y que cada una de las siete combinaciones, (b) al menos tres pruebas rechazan con p ajustado menor que α, (c) efecto no despreciable (|δ| ≥ 0.147 o |d| ≥ 0.2) y (d) mejor en al menos la mitad de los conjuntos adicionales, sin ser significativamente peor en ninguno.

**Resultado (2026-11-05): no se cumple.** Ver sección 10 para el detalle completo.

## 7. Presupuesto

**Decisión 5, confirmada 2026-11-05:** 40 evaluaciones de optimización bayesiana por modelo y por pliegue externo (ya fijado), más un tope de referencia de 15 minutos por modelo por pliegue externo (no se alcanzó en ningún caso: el modelo más lento, Random Forest con diez semillas, promedió ~45 segundos por pliegue). Presupuesto realmente consumido: ver `artifacts/confirmatorio/presupuesto_consumido.csv`.

## 8. Desviaciones

Toda desviación posterior a la etiqueta se declara aquí con fecha y motivo. Los análisis no previstos aquí son exploratorios.

- **2026-11-05 — TabPFN no ejecutado:** licencia *gated* de Hugging Face, requiere aceptación manual por una persona del equipo (sección 3).
- **2026-11-05 — Validación externa (ene-sep 2026) no ejecutada:** faltan los indicadores de las 13 áreas metropolitanas para 2026, que no se consiguen por búsqueda de texto (sección 4). Los 9 valores de la tasa nacional sí quedaron verificados y documentados para cuando el equipo complete los datos faltantes.
- **2026-11-05 — Conjuntos adicionales (EE.UU./España/México) no ensamblados:** requieren descarga de series completas, no disponible en el entorno de ejecución usado (sección 5). Por lo tanto, la prueba de Friedman/Wilcoxon-Holm entre conjuntos (sección 6, punto 5) y la condición (d) del criterio de superioridad no se pudieron evaluar.

## 9. Registro de las 5 decisiones (confirmadas por el equipo el 2026-11-05)

| # | Decisión | Resuelto como |
|---|---|---|
| 1 | Métodos SOTA | LightGBM + TabPFN (TabPFN pendiente de ejecutar, ver sección 3) |
| 2 | Validación externa | Ene-sep 2026, fuente Banco de la República/DANE (pendiente de ejecutar, ver sección 4) |
| 3 | Repeticiones ventana expansiva | 5 orígenes |
| 4 | Conjuntos adicionales | EE.UU., España, México (pendientes de ensamblar, ver sección 5) |
| 5 | Tope de cómputo | 40 evaluaciones + 15 min/modelo/pliegue de referencia |

## 10. Resultado de la validación confirmatoria sobre el problema principal (2026-11-05)

Corrida completa en `experiments/run_confirmatorio.py`, resultados en `artifacts/confirmatorio/` (`tabla_rmse_confirmatorio.csv`, `mcs_confirmatorio.csv`, `giacomini_white_confirmatorio.csv`, `criterio_superioridad.csv`).

**RMSE medio de los pliegues externos (240 meses), de mejor a peor:** Estacional(ALEE) 0.836 < SARIMA 0.995 < **ALEE 1.054** < Persistencia 1.293 < Ridge 1.346 < SVR 1.370 < KNN 1.373 < RandomForest 1.493 < XGBoost 1.567 < LightGBM 1.704 < DecisionTree 1.748 < Lasso 1.846 < Estacional(y_{t-11}) 2.107.

**Condición (a) — mejor métrica que la mejor referencia: NO SE CUMPLE.** ALEE (1.054) es peor que el experto estacional de ALEE aislado (0.836) y que SARIMA (0.995). Sí es mejor que las siete combinaciones del entregable anterior y que LightGBM.

**Model Confidence Set (α=0.10):** los 13 candidatos, incluido ALEE, quedan en el conjunto de confianza — no se puede descartar estadísticamente a ninguno como inferior al mejor, con esta cantidad de meses.

**Diebold-Mariano (HLN), ALEE vs. Estacional(ALEE):** p = 0.0044, diferencia de pérdida media = +0.413 a favor del experto estacional aislado — es decir, se rechaza la igualdad de precisión predictiva, y la dirección favorece al experto estacional aislado, no a ALEE.

**Giacomini-White (Holm):** ALEE es significativamente distinto (mejor) de Persistencia, Ridge, SVR, LightGBM, DecisionTree, Lasso y Estacional(y_{t-11}); no se distingue significativamente de Estacional(ALEE), SARIMA, KNN, RandomForest ni XGBoost tras la corrección.

**SPA de Hansen (referencia = Estacional(ALEE)):** p = 0.671 — ningún candidato, incluido ALEE, supera de forma estadísticamente robusta al experto estacional aislado.

**Tamaño del efecto ALEE vs. Estacional(ALEE):** d de Cohen = 0.19 (muy pequeño), δ de Cliff = 0.03 (insignificante).

**Veredicto del criterio de superioridad (a ∧ b ∧ c ∧ d): NO SE CUMPLE.** Falla ya en la condición (a): el "mejor de referencia" en este problema no es SARIMA ni ninguna de las siete combinaciones, sino el propio experto estacional de ALEE aislado, y la combinación completa de ALEE no logra superarlo. Las condiciones (b) y (c) tampoco se sostienen para esa comparación específica. La condición (d) no se pudo evaluar (sección 5).

**Lectura honesta:** este resultado confirma, con el protocolo ya congelado y no solo con el piloto exploratorio, que la localización y regularización de ALEE no agregan valor frente a usar directamente su experto estacional solo, al menos en este conjunto de datos y en el periodo evaluado. ALEE sí mejora de forma clara y estadísticamente significativa sobre las siete combinaciones del entregable anterior, sobre la persistencia y sobre LightGBM — su valor está ahí, no frente a un modelo estacional simple. Esto no invalida la originalidad del mecanismo (sigue sin encontrarse un antecedente con el mismo mecanismo exacto, ver `book/04_literatura.md`), pero sí invalida, con los datos actuales, la afirmación de que ALEE sea un mejor pronosticador que la mejor alternativa simple disponible.

## 11. Iteración 4 (2026-11-05, exploratoria, posterior a este resultado confirmatorio)

**Esta sección documenta un ajuste de diseño hecho DESPUÉS de ver el resultado de la sección 10.** Por definición ya no es parte de la validación confirmatoria original — es una nueva iteración de diseño, declarada como tal (ver `docs/registro_desarrollo.md`, iteración 4), no una re-interpretación de los mismos resultados.

**Diagnóstico que la motivó:** evaluados en solitario sobre los mismos 240 meses, dos de los cinco expertos de ALEE (`ridge_nivel` RMSE 1.605, `ridge_cambio` RMSE 1.620) se equivocan más del doble que el experto estacional (0.780) y más que los otros dos (`nucleo_cambio` 1.252, `persistencia` 1.273). Un primer intento de corregir esto forzando a mano una regularización más fuerte hacia el previo global resultó ser inválido (ajustado mirando el pliegue de prueba, ver iteración 4b del registro) y se descartó. La corrección que sí se sostuvo bajo validación anidada honesta fue quitar esos dos expertos.

**Cambio:** `ModeloOriginal` pasa de 5 a 3 expertos (persistencia, núcleo-cambio, estacional); desaparece el hiperparámetro `ridge_alpha`. Re-ejecutado con el mismo presupuesto del protocolo (40 evaluaciones bayesianas, `TimeSeriesSplit(5,gap=1)`/`TimeSeriesSplit(3,gap=1)`), con el rango de `lam` ampliado de `[1e-3,10]` a `[1e-3,100]` en la búsqueda (confirmado que no cambia la conclusión: el optimizador sigue prefiriendo `lam` pequeño).

**RMSE medio de los pliegues externos (240 meses), de mejor a peor:** Estacional(ALEE) 0.836 < **ALEE 0.869** < SARIMA 0.995 < Persistencia 1.293 < Ridge 1.346 < SVR 1.370 < KNN 1.373 < RandomForest 1.493 < XGBoost 1.567 < LightGBM 1.704 < DecisionTree 1.748 < Lasso 1.846 < Estacional(y_{t-11}) 2.107. (RMSE medio de pliegues, métrica primaria: 0.817 ± 0.328, frente a 0.780 ± 0.299 del estacional aislado.)

**Diebold-Mariano (HLN), ALEE vs. Estacional(ALEE):** p = 0.140 (antes 0.0044) — **ya no se rechaza la igualdad**. Frente a SARIMA: p = 0.302, tampoco se rechaza, y la diferencia de pérdida favorece nominalmente a ALEE.

**Giacomini-White (Holm):** ALEE gana con significancia a 9 de los 11 modelos restantes (todos menos Estacional(ALEE), SARIMA y, por poco, XGBoost: p_holm=0.059).

**SPA de Hansen (referencia = Estacional(ALEE)):** p = 0.674; el candidato con mejor pérdida media es el propio ALEE, aunque sin robustez estadística frente a *data snooping*.

**Tamaño del efecto ALEE vs. Estacional(ALEE):** d de Cohen = 0.094 (muy pequeño), δ de Cliff = 0.018 (insignificante) — antes 0.19 y 0.03.

**Veredicto del criterio de superioridad (a ∧ b ∧ c ∧ d) con la iteración 4: SIGUE SIN CUMPLIRSE.** Falla la condición (a): en el punto estimado, ALEE (0.817/0.869 según la métrica) sigue siendo numéricamente peor que el estacional aislado (0.780/0.836), aunque la diferencia ya no es significativa. La condición (b) exige al menos 3 pruebas que rechacen a favor de ALEE frente a la mejor referencia específicamente, y ni DM-HLN ni SPA lo hacen. La condición (d) sigue sin evaluarse (conjuntos adicionales pendientes).

**Lectura honesta:** la iteración 4 es una mejora real y metodológicamente bien fundamentada — no un ajuste cosmético ni *p-hacking* — que lleva a ALEE de "pierde de forma significativa contra su propio mejor componente" a "empate estadístico con las dos referencias más fuertes del problema, superando con significancia a todas las demás". No alcanza, sin embargo, a cumplir el criterio de superioridad pre-registrado. El equipo adoptó esta versión de 3 expertos como la definitiva (`src/original_model.py`); las cifras de la sección 10 (5 expertos) quedan como registro histórico en `artifacts/confirmatorio/oof_ALEE_K5_original_iter3.csv`.

## 12. Iteraciones 5 y 6 y validación externa (2026-11-06, exploratorias, posteriores a la iteración 4)

**Motivación:** tras adoptar la iteración 4, el equipo pidió explícitamente buscar un diseño que superara con significancia tanto a SARIMA como al estacional aislado (las dos referencias con las que ALEE-3 solo empata). Se probaron tres propuestas razonadas y, al evaluar la tercera contra datos externos, se encontró y corrigió un defecto de diseño adicional. Todo lo siguiente se validó con el mismo protocolo de nested CV honesto (`TimeSeriesSplit(5,gap=1)`/`TimeSeriesSplit(3,gap=1)`, optimización bayesiana de 40 evaluaciones, semilla 42, sin mirar nunca el pliegue de prueba al elegir hiperparámetros), salvo donde se declara explícitamente lo contrario.

**Iteración 5a — ALEE combinando SARIMA y el estacional aislado (con variantes de persistencia y un estacional robusto de mediana):** cuatro conjuntos de expertos, todos con RMSE medio de pliegues entre 0.810 y 0.830 — todos peor que el estacional solo (0.780). Un control adicional de peso fijo global (sin el mecanismo de localización de ALEE, elegido solo con pliegues internos) da 0.810, prácticamente igual, con el peso óptimo variando entre 0.23 y 1.00 de SARIMA según el pliegue — señal de que los pliegues internos no identifican un peso estable. Como ningún conjunto empató o superó al estacional aislado en el punto estimado, no se corrieron pruebas de Diebold-Mariano (hacerlo habría sido reportar significancia sobre algo que ya pierde en el punto estimado). **Causa raíz:** en el pliegue del choque de 2020 SARIMA se equivoca casi el doble que el estacional, y cualquier mezcla que le dé peso ahí empeora frente al estacional solo; los pliegues internos no contienen un choque representativo para aprender a evitarlo. Esto confirma, bajo validación honesta, lo que el intento inválido de la iteración 4b había insinuado de forma incorrecta: la aparente mejora de mezclar con SARIMA (RMSE~0.76 mirando el pliegue de prueba) desaparece al elegir el peso sin esa fuga.

**Iteración 5b — modelo de espacio de estados (`UnobservedComponents`: nivel local + estacional):** evaluado con la misma metodología walk-forward que el baseline oficial de SARIMA (replicada con diferencia ~1e-15 frente a ese baseline, sin error de indexación). Con política honesta (especificación elegida por AIC en los datos de entrenamiento de cada pliegue, nunca mirando la prueba): RMSE medio 0.815, agrupado 0.990 — empata con SARIMA (p=0.40, DM-HLN) y con el estacional aislado (p=0.22) — mismo resultado cualitativo que ALEE. Entre tres especificaciones fijas comparadas después por su RMSE de prueba —esto sí es una comparación múltiple leve, no una elección pre-registrada, y se declara como tal—, la mejor (nivel local + estacional estocástico, RMSE medio 0.797, agrupado 0.972) empata con el estacional aislado (p=0.265) pero le gana con significancia a SARIMA (p=0.0003). Es un hallazgo sugerente, no concluyente, por el procedimiento con el que se eligió esa especificación entre las tres.

**Validación externa con los 9 meses reales de 2026 (ene-sep), nunca vistos por ningún diseño:** se evaluaron los modelos que no requieren los indicadores de las 13 áreas metropolitanas todavía pendientes (sección 8, punto 2): persistencia (RMSE 1.328), estacional aislado (0.816), SARIMA (0.763), y un primer intento de ALEE-3 que dio RMSE 1.502 pero resultó **inválido** (ver iteración 6). Con n=9 ninguna prueba tiene potencia estadística; es solo comparación directa. Hallazgo honesto: en esta muestra pequeña SARIMA queda primero, invirtiendo el orden observado en los 240 meses oficiales (donde el estacional le ganaba a SARIMA, 0.780 vs. 0.818).

**Iteración 6 — corrección del kernel de `nucleo_cambio`:** al investigar el mal desempeño de ALEE en 2026 se encontró que el experto `nucleo_cambio` calcula su distancia de Nadaraya-Watson sobre las 17 columnas completas, incluidas las 3 de área (`AREA_COLS`), que en 2026 no existen y se habían rellenado en cero, distorsionando el kernel. Se probó una variante que excluye `AREA_COLS` del kernel —motivada también, independientemente, por la multicolinealidad severa entre área y nacional ya documentada desde el segundo entregable—, validada con el protocolo completo. Resultado: RMSE medio 0.826 (vs. 0.817 de ALEE-3 actual), agrupado 0.878 (vs. 0.869) — estadísticamente indistinguible de ALEE-3 (DM-HLN p=0.556): **no es una mejora de desempeño, es un empate técnico con ligera pérdida numérica.** Repetida la validación de 2026 con esta versión corregida: RMSE 1.491 (vs. 1.502) — mejora marginal, no sustancial. Más importante: el error de enero 2026 (salto real de 7.98 % a 10.9 %) casi no cambió (3.856 vs. 3.719 del kernel original) — la causa de ese fallo **no es el kernel**, sino que el mecanismo de localización/regularización de ALEE le quita peso al experto estacional (que sí predice bien el salto) en ese estado particular. Es un problema distinto y más profundo, no resuelto en esta ronda.

**Veredicto de esta ronda:** ninguna de las tres propuestas (combinar con SARIMA, espacio de estados, corregir el kernel) logra, bajo validación honesta, que ALEE supere con significancia a la vez a SARIMA y al estacional aislado. El criterio de superioridad pre-registrado (sección 10) sigue sin cumplirse. **Decisión del equipo (2026-11-06): se cierra, por ahora, la búsqueda de un diseño superior.** El modelo canónico sigue siendo el de la iteración 4; la corrección del kernel de la iteración 6 queda documentada y disponible como mejora de robustez (permite evaluar ALEE sin datos de área) pero no se adopta por defecto, al ser neutra en desempeño. La pista más concreta para una eventual iteración 7 —el mecanismo de localización le quita peso al estacional en saltos bruscos no vistos— queda registrada en `docs/registro_desarrollo.md` sin perseguirse, para no seguir ajustando el diseño sobre el mismo problema sin un límite claro.

Archivos: `experiments/iter5a_sarima_estacional.py`, `experiments/iter5b_espacio_estados.py`, `experiments/iter6_nucleo_sin_area.py`, `experiments/iter6_validacion_externa_2026.py`, `experiments/iter6_validacion_externa_2026_corregida.py`, `artifacts/confirmatorio/iter5/`, `artifacts/confirmatorio/iter6/`, `artifacts/confirmatorio/externa_2026/`.

## 13. Protocolo pre-registrado de la iteración 7 (ajuste del mecanismo de localización de ALEE) y de dos arquitecturas alternativas de combinación (2026-11-08)

**Declarado ANTES de correr nada sobre los pliegues de prueba.** El equipo pidió explícitamente reabrir la búsqueda de un diseño superior, en dos frentes en paralelo: (A) corregir el modo de fallo de ALEE ya identificado en la iteración 6 (le quita peso al experto estacional en saltos bruscos no vistos, como enero de 2026), y (B) explorar arquitecturas de combinación distintas al mecanismo entrópico de ALEE. Se exige pre-registrar esta ronda, sin excepción, porque ya se acumulan seis iteraciones de diseño sobre el mismo conjunto de 240 meses (sección 12) y el propio equipo declaró ese riesgo de sobreajuste creciente; esta es la séptima, y agrava ese riesgo — se declara aquí, no después de ver resultados. Todo lo que sigue se decidió usando solo el diagnóstico ya escrito en `docs/registro_desarrollo.md` ("Observaciones para la siguiente iteración") y literatura establecida, **nunca mirando el pliegue de prueba**, y no se modificará después de ver resultados: cualquier cambio de diseño posterior sería una octava iteración declarada como tal.

**Disciplina de validación (idéntica a las rondas anteriores):** `TimeSeriesSplit(5, gap=1)` externo, `TimeSeriesSplit(3, gap=1)` interno, optimización bayesiana (Optuna, TPESampler, semilla 42) con **40 evaluaciones** por pliegue externo, hiperparámetros elegidos solo con los pliegues internos. Batería de pruebas idéntica a la sección 2: Model Confidence Set (α=0.10), Giacomini-White con Holm, Diebold-Mariano-HLN, SPA de Hansen, d de Cohen y δ de Cliff.

**Criterio de éxito, declarado ahora:** para que un candidato de esta ronda se adopte como nueva versión canónica tiene que, con el mismo criterio de superioridad de la sección 10 (condiciones a-c; la d sigue sin poder evaluarse por falta de los conjuntos adicionales), superar con significancia tanto a SARIMA como al estacional aislado **a la vez**, Y no perder con significancia frente a ALEE-3 (la versión canónica actual) en ningún candidato que se proponga como reemplazo. Un candidato que solo empate con SARIMA/estacional, como ALEE-3, se documenta como resultado adicional pero no reemplaza al modelo canónico.

### Track A — Iteración 7: ajustar el mecanismo de localización de ALEE

**Hipótesis (motivada por el diagnóstico de la iteración 6, ya escrito antes de esta ronda):** el espacio de estado que usa ALEE para localizar (`rollstd3` de la volatilidad reciente, y la brecha `lag1 - rollmean12`) no incluye ninguna medida de la magnitud de un salto puntual reciente. Un mes con un salto grande pero con volatilidad *reciente* todavía baja (el patrón de enero de 2026) puede terminar con vecinos históricos poco relevantes en ese espacio, y la regularización adaptativa (que se encoge hacia el previo global cuando el tamaño efectivo de muestra local `n_eff` es bajo) interpreta ese estado como "distinto a la historia" justo cuando el experto estacional sí acertaba.

Dos variantes, elegidas ahora y no después:

- **7.1 — variable de estado adicional:** agregar `|y_t - y_{t-1}|` (magnitud del último cambio mensual) como tercera coordenada del estado de localización (`state_cols`), junto a `rollstd3` y la brecha ya existentes. Mismos hiperparámetros que ALEE-3 (`bandwidth`, `half_life`, `lam`), buscados en los mismos rangos ya establecidos en el protocolo (`bandwidth`: log-uniforme [0.05, 3.0]; `half_life`: log-uniforme [6, 120]; `lam`: log-uniforme [1e-3, 100]), sin hiperparámetros nuevos.
- **7.2 — piso mínimo de peso para el experto estacional:** imponer una cota inferior `w_estacional ≥ floor` directamente en la solución del símplex (proyectando el resto de la masa entre los demás expertos tras resolver sin la cota), motivada por que el estacional ha sido, en las seis iteraciones anteriores, consistentemente el componente individual más fuerte. Hiperparámetro nuevo `floor`, buscado en log-uniforme [0.01, 0.5], junto con `bandwidth`, `half_life`, `lam` en los mismos rangos de siempre.

Ambas variantes se corren con los 3 expertos de la iteración 4 (persistencia, núcleo-cambio, estacional) y el presupuesto estándar de 40 evaluaciones bayesianas por pliegue externo.

### Track B — Dos arquitecturas alternativas de "súper modelo"

Ambas usan los mismos tres expertos de ALEE-3 (persistencia, núcleo-cambio, estacional) como insumos, pronóstico a un paso en ventana expansiva real (nunca se usa información del mes que se pronostica), y se comparan contra las mismas referencias.

- **B1 — Promedio/Selección Dinámica de Modelos (DMA/DMS), Raftery, Kárný y Ettler (2010):** actualización bayesiana recursiva de la probabilidad posterior de cada experto, con un factor de olvido `α` sobre las probabilidades (`π_{t|t-1,k} ∝ π_{t-1|t-1,k}^α`, renormalizado) y un segundo factor de olvido `δ` sobre la varianza recursiva del error de cada experto, asumiendo verosimilitud gaussiana. El pronóstico final es el promedio ponderado por probabilidad posterior (DMA) o el del modelo de mayor probabilidad posterior (DMS); se reportan ambos. Es un método establecido en la literatura econométrica para exactamente este problema — combinación de pronósticos con pesos que cambian en el tiempo según qué modelo viene acertando — y le da a ALEE un comparador directo con un mecanismo de "elegir según la información" distinto al suyo (kernel + regularización entrópica) en vez de una variación sobre el mismo mecanismo. Hiperparámetros `α` y `δ`, buscados en log-uniforme [0.90, 1.0] cada uno (`α=δ=1` reduce el método a un promedio acumulado simple de verosimilitud, sin olvido), con el mismo presupuesto de 40 evaluaciones.
- **B2 — Apilamiento (*stacking*) con meta-regresor walk-forward:** un meta-modelo (ridge con intercepto, mismas variables de estado que usa ALEE para localizar más las tres predicciones de los expertos) entrenado sobre los errores pasados y reajustado cada 12 meses en ventana expansiva (nunca con información del mes pronosticado ni de meses futuros al reajuste). Hiperparámetro: la fuerza de regularización `alpha` del ridge, en log-uniforme [1e-3, 100] — mismo rango usado para `lam` en ALEE, por consistencia. Es la versión "aprendida" de la idea de ALEE: en vez de una regla cerrada (kernel + entropía), un modelo lineal ajustado a los datos decide los pesos.

### Qué se reporta, gane o pierda

Como en las rondas 5-6, se documentan los cuatro resultados completos — ganen, empaten o pierdan frente al criterio — con sus pruebas de significancia, en una actualización de esta sección y en `docs/registro_desarrollo.md`. Ningún resultado se omite ni se re-intenta con un diseño distinto sin declararlo como una octava iteración.

Archivos de esta ronda: `experiments/iter7_estado_salto.py`, `experiments/iter7_piso_estacional.py`, `experiments/iter7b_dma.py`, `experiments/iter7b_stacking.py`, resultados en `artifacts/confirmatorio/iter7/`.

### Resultados (2026-11-08, ejecutados exactamente según lo pre-registrado arriba)

**Verificación de cordura:** antes de confiar en cualquier resultado, se comprobó que el RMSE de cada experto en solitario sobre los mismos 240 meses coincide con lo ya documentado (diagnóstico 4d): estacional 0.780, núcleo-cambio 1.252, persistencia 1.273 (RMSE medio de pliegues). Coincidió en las cuatro implementaciones independientes (una por cada script de esta ronda). Nota metodológica sin impacto en los resultados: al validar esta cifra, los scripts de Track B compararon inicialmente contra el RMSE **agrupado** de persistencia (1.293) en vez del RMSE **medio de pliegues** (1.273) — son las dos métricas que este protocolo reporta siempre por separado (sección 2), no un error de cálculo; verificado de forma independiente que ambas cifras son correctas para su propia definición.

**RMSE medio de los pliegues externos (métrica primaria) y RMSE agrupado:**

| Candidato | RMSE medio de pliegues | RMSE agrupado |
|---|---|---|
| Estacional aislado (referencia) | 0.780 ± 0.299 | 0.836 |
| **DMA** | **0.777 ± 0.343** | **0.835** |
| DMS | 0.780 ± 0.344 | 0.839 |
| SARIMA (referencia) | 0.818 ± 0.567 | 0.995 |
| 7.2 — piso mínimo del estacional | 0.810 ± 0.292 | 0.861 |
| 7.1 — estado de salto | 0.812 ± 0.286 | 0.861 |
| ALEE-3 (canónico, referencia) | 0.817 ± 0.328 | 0.869 |
| Stacking (ridge walk-forward) | 0.791 ± 0.404 | 0.869 |

**Diebold-Mariano-HLN, cada candidato contra estacional aislado / SARIMA / ALEE-3 (12 pruebas en total, Track A; 9 pruebas, Track B — 21 en total):** ninguna de las 21 comparaciones resulta significativa (todos los p-valores entre 0.11 y 0.99); todos los tamaños del efecto son insignificantes o muy pequeños (|d| ≤ 0.15, |δ| ≤ 0.08). Detalle completo con dirección y tamaño del efecto en `artifacts/confirmatorio/iter7/significancia_iter7b_dma.csv`, `significancia_iter7b_stacking.csv` y en el reporte de los forks (no se generó un CSV aparte para 7.1/7.2; las cifras están en el log de ejecución).

**Lectura honesta, candidato por candidato:**

- **DMA (Track B1) es, en el punto estimado, el primer mecanismo de combinación de todo el proyecto cuyo RMSE medio de pliegues (0.777) iguala o roza por debajo al del propio experto estacional aislado (0.780)** — ni ALEE en ninguna de sus siete versiones (iteraciones 0 a 7.2) ni el espacio de estados ni el stacking lo habían logrado en el punto estimado. Sin embargo, la diferencia es minúscula y nada significativa (DM-HLN p=0.955, diferencia de pérdida media −0.0013): es, en la práctica, un empate estadístico perfecto, no una mejora demostrada. También empata con SARIMA (p=0.195) y con ALEE-3 (p=0.114, con dirección a favor de DMA mayor que en cualquier comparación anterior pero aún lejos de significancia).
- **DMS** (el modelo de mayor probabilidad posterior en cada mes, sin promediar) da un resultado casi idéntico a DMA (0.780 vs 0.777), también sin significancia en ninguna comparación.
- **7.1 y 7.2** (las dos correcciones directas al mecanismo de localización de ALEE) dan resultados prácticamente idénticos entre sí (0.812 y 0.810) y muy parecidos a ALEE-3 (0.817) — mejoran apenas en el punto estimado sobre ALEE-3 pero sin ningún cambio cualitativo: siguen empatando con SARIMA y con el estacional aislado, igual que ALEE-3. La pista de la iteración 6 (el mecanismo de localización le quita peso al estacional en saltos bruscos) **no se traduce en una mejora medible** con ninguna de las dos correcciones probadas, al menos en estos 240 meses.
- **Stacking** (el meta-regresor "aprendido") da el peor resultado de los cuatro candidatos nuevos (0.791 de RMSE medio, pero con la desviación estándar entre pliegues más alta de toda la ronda, 0.404 — muy sensible al pliegue 3, el del choque de 2020) y empata, sin destacar, con las tres referencias.
- **Caveat declarado por el fork de DMA/DMS:** el pliegue externo 0 no tiene pliegues internos con información disponible para afinar `(alpha, delta)` (los tres caen antes de `m0=48`), una consecuencia directa de `m0=48` ya pre-registrado, no un error de diseño; sus hiperparámetros para ese pliegue quedan sin informar por el proceso de búsqueda. Los pliegues 2 y 3 convergieron al mismo `(alpha, delta)` hasta el décimo decimal, verificado como una superficie de validación interna casi plana en esa región, no un bug.

**Veredicto del criterio de superioridad pre-registrado (ganar con significancia a SARIMA Y al estacional aislado a la vez, sin perder con significancia frente a ALEE-3): NO SE CUMPLE para ninguno de los cuatro candidatos.** Esta séptima ronda de iteraciones tampoco encuentra un diseño que supere de forma demostrable a la mejor referencia simple del problema. **Decisión del equipo (2026-11-08): el modelo canónico sigue siendo ALEE-3** (iteración 4, `src/original_model.py`). Ninguna de las cuatro propuestas de esta ronda se adopta por defecto. DMA queda documentado como el hallazgo más prometedor de toda la búsqueda — el primero en empatar en punto estimado con el estacional aislado en vez de perder por un margen visible — y como la pista concreta para una eventual octava iteración (por ejemplo, ampliar el rango de `delta` más allá de 1.0, o probar una verosimilitud con colas más pesadas que la gaussiana para el choque de 2020), no perseguida en esta ronda para no seguir ajustando el diseño sobre el mismo conjunto de 240 meses sin límite.

**Total de iteraciones de diseño evaluadas sobre los datos del problema principal, acumulado: 10** (las seis de las secciones 10-12, más 7.1, 7.2, DMA/DMS y stacking de esta ronda; DMA y DMS se cuentan como una sola iteración de diseño con dos variantes de lectura, igual que SARIMA+estacional en 5a tuvo cuatro variantes contadas como una iteración).

Archivos de resultados: `artifacts/confirmatorio/iter7/{oof,rmse}_iter7_1_estado_salto.csv`, `{oof,rmse}_iter7_2_piso_estacional.csv`, `{oof,rmse}_iter7b_dma.csv`, `{oof,rmse}_iter7b_dms.csv`, `{oof,rmse}_iter7b_stacking.csv`, `expert_oof_predictions.csv`, `significancia_iter7b_dma.csv`, `significancia_iter7b_stacking.csv`.

## 14. Conjuntos adicionales: pre-registro de la ejecución (2026-11-08, antes de ver resultados)

> Esta sección se escribió **antes** de correr ningún modelo sobre los tres conjuntos adicionales. Ejecuta lo que ya fijaron las secciones 5 y 6 (decisión 4) y declara aquí, por adelantado, las desviaciones inevitables. Ningún diseño de ALEE se modifica en esta ronda: se evalúa **ALEE-3 tal como está congelado** (iteración 4, `src/original_model.py`, mismo espacio de búsqueda de `src/models.py`).

**Datos (descargados el 2026-11-08, copias crudas en `data/adicionales/`):**

| Conjunto | Fuente efectiva | Serie | Periodo |
|---|---|---|---|
| Estados Unidos | FRED | `UNRATENSA` (sin ajuste estacional) | 1948-01 a 2026-10 |
| España | Eurostat API SDMX | `une_rt_m`, `M.NSA.TOTAL.PC_ACT.T.ES` | 1986-04 a 2026-09 |
| México | API SDMX de la OCDE (`DSD_LFS@DF_IALFS_UNE_M`, MEX, sin ajuste, total, 15+ años), que redistribuye la ENE/ENOE de INEGI | tasa de desocupación mensual | 2000-01 a 2026-09 |

Se usa la historia completa disponible de cada serie (decidido ahora, antes de ver resultados).

**Desviaciones declaradas por adelantado:**

1. **México se toma de la OCDE, no directamente de INEGI:** la API de INEGI exige un token personal con registro de cuenta. La OCDE publica la misma encuesta (ENE hasta 2004, ENOE desde 2005; en abril-mayo de 2020 la ENOE se suspendió y el dato proviene de la ETOE telefónica, tal como lo publica la OCDE).
2. **14 variables en vez de 17:** los tres indicadores de las 13 áreas metropolitanas no existen para otros países. Todos los modelos usan las mismas 14 variables nacionales (rezagos 1, 2, 3, 6, 12; medias y desviaciones móviles de 3, 6 y 12; mes cíclico; tendencia). Para ALEE esto equivale a la variante de núcleo sin área de la iteración 6, que en Colombia fue neutra en desempeño.
3. **Una sola semilla (42) para los modelos estocásticos** (Random Forest, XGBoost, LightGBM), en vez de diez, por tiempo de cómputo. La comparación principal del protocolo ya usa la semilla 42; las otras nueve solo medían variabilidad.
4. **TabPFN sigue sin ejecutarse** (licencia *gated*, sección 3).
5. **Versiones de librerías:** entorno nuevo (`.venv`, Python 3.13, scikit-learn 1.9, optuna 5.0, xgboost 3.4, lightgbm 4.7), distinto del fijado en `requirements.txt`.
6. **Mes faltante en EE.UU. (agregado el 2026-11-08, antes de ver resultados):** FRED no tiene octubre de 2025 porque la BLS no levantó la encuesta durante el cierre del gobierno federal. Se rellena por interpolación lineal entre septiembre y noviembre de 2025 (ambos 4.3). Se detectó al ver que el conjunto tenía 921 filas en vez de 934; la corrida de EE.UU. se detuvo y se reinició antes de producir ningún resultado.
7. **Implementación de la búsqueda de ALEE:** se usa la misma optimización bayesiana (TPE de Optuna, semilla 42, 10 arranques aleatorios, 40 evaluaciones, menos RMSE medio de los 3 pliegues internos), pero los errores fuera de muestra de los expertos se calculan una sola vez por serie y se reusan en cada ajuste sobre un prefijo. Es exactamente el mismo cálculo que `ModeloOriginal.fit` (el error del mes *s* solo depende de los meses anteriores); se verifica numéricamente contra `.fit()` en cada conjunto y se registra la diferencia máxima.

**Modelos:** ALEE-3; las siete combinaciones (Ridge, Lasso, KNN, DecisionTree, RandomForest, XGBoost, SVR) y LightGBM, todas con optimización bayesiana de 40 evaluaciones como en `run_confirmatorio.py`; referencias: persistencia, estacional ingenuo (`y_{t-11}`), SARIMA(0,1,1)(0,1,1)[12] con la misma metodología de `regression_baselines` y el experto estacional de ALEE aislado. Particiones: `TimeSeriesSplit` 5 externos / 3 internos, `gap=1`.

**Pruebas por conjunto (sección 6):** MCS (α = 0.10, bloques de 12, 5000 remuestras), Giacomini-White condicional y Diebold-Mariano-HLN de ALEE frente al mejor de referencia y frente a SARIMA, y SPA de Hansen. **Entre conjuntos:** Friedman/Iman-Davenport y Wilcoxon-Holm con los cuatro conjuntos (Colombia más los tres nuevos).

**Mejor de referencia de cada conjunto:** el modelo (distinto de ALEE) con menor RMSE medio de pliegues externos en ese conjunto, sea cual sea.

**Condición (d):** ALEE tiene menor RMSE medio de pliegues que el mejor de referencia en al menos 2 de los 3 conjuntos adicionales, y en ninguno es significativamente peor que él (DM-HLN, p < 0.05, Holm sobre los tres conjuntos).

**Compromiso:** se reportan todos los resultados, ganen, empaten o pierdan. Si ALEE no cumple, no se rediseña sobre estos conjuntos; cualquier cambio posterior se declara como iteración exploratoria nueva.

### 14.1 Resultado (2026-11-08)

Corrida: `experiments/run_conjuntos_adicionales.py` (predicciones en `artifacts/adicionales/<pais>/`), análisis: `experiments/analizar_conjuntos_adicionales.py` (tablas en `artifacts/adicionales/`). La implementación rápida de ALEE coincidió con `ModeloOriginal.fit()` con diferencia máxima 0 en los tres conjuntos.

**RMSE medio de pliegues externos (posición de ALEE entre 13 modelos):**

| Conjunto | Meses de prueba | Mejor de referencia | ALEE | Mejor ref. | Estacional ALEE | DM-HLN ALEE vs mejor ref. (Holm) |
|---|---|---|---|---|---|---|
| Colombia (problema principal) | 240 | Estacional ALEE | 0.817 (2.º) | 0.780 | 0.780 | p = 0.141, empate |
| EE.UU. | 775 | SARIMA | 0.380 (2.º) | 0.366 | 0.391 | p = 0.535, empate |
| España | 395 | SARIMA | 0.280 (3.º) | 0.217 | 0.279 | p < 0.001, **ALEE peor** |
| México | 255 | SARIMA | 0.281 (2.º) | 0.253 | 0.283 | p = 0.013, **ALEE peor** |

**Condición (d): NO SE CUMPLE.** ALEE no es mejor que el mejor de referencia en ninguno de los tres conjuntos adicionales (0/3) y es significativamente peor que SARIMA en dos (España y México). En consecuencia, **el criterio de superioridad (a ∧ b ∧ c ∧ d) no se cumple**, y ahora la condición (d) ya está evaluada, no pendiente.

**Lo que sí se sostiene en los cuatro conjuntos:** ALEE supera a los ocho modelos de aprendizaje automático (siete combinaciones y LightGBM) y a la persistencia en 4 de 4 conjuntos. Friedman con Iman-Davenport rechaza la igualdad de los 13 modelos (χ² = 42.2, F = 22.0, p < 0.001); rangos medios: SARIMA 1.50, ALEE 2.25, Estacional ALEE 2.25, persistencia 4.75 y los modelos de ML entre 5.5 y 12.0. Con solo cuatro conjuntos, el Wilcoxon por pares no puede bajar de p = 0.125 aunque ALEE gane en los cuatro, así que ninguna comparación por pares es significativa tras Holm (limitación estructural del número de conjuntos, no del modelo).

**Hallazgo exploratorio (no forma parte del criterio):** en EE.UU., la serie más larga (775 meses de prueba), ALEE supera con significancia a su propio experto estacional aislado (DM-HLN p = 0.033, sin corregir por multiplicidad) y su RMSE agrupado (0.461) es menor que el de SARIMA (0.469), aunque en RMSE medio de pliegues SARIMA sigue adelante. Es la primera vez que la localización de ALEE aporta una mejora medible sobre su mejor experto, y es coherente con la hipótesis de que el mecanismo necesita muchos meses comparables. En España y México, en cambio, ALEE y su estacional quedan prácticamente iguales, y ambos pierden contra SARIMA: el límite de ALEE en esos conjuntos lo marcan sus expertos, no la combinación.

**Decisión:** conforme al compromiso de la sección 14, no se rediseña ALEE sobre estos conjuntos. El modelo canónico sigue siendo ALEE-3.

## 15. Estudio de robustez con datos simulados (2026-11-08, exploratorio)

Archivos: `experiments/simulacion/` (`simular.py`, `analizar.py`, `calibrar.py`, `predicciones_simulacion.csv`). Dos procesos generadores de 900 meses calibrados con la serie de Colombia (nivel ≈ 11, patrón estacional real, ruido ≈ 0.46): **estable** (patrón estacional fijo) y **con crisis** (régimen de Markov con choques tipo 2020 cada ~10 años: salto de nivel, ruido ×2.7, estacionalidad al 20 %). Validación 5/3 con `gap=1`; los hiperparámetros de ALEE se eligieron por rejilla de 18 combinaciones en los pliegues internos (no bayesiana); 5 semillas por escenario, 750 meses de prueba cada una.

| Escenario | RMSE ALEE | RMSE estacional | ALEE mejor en | Con p < 0.05 (DM-HLN) |
|---|---|---|---|---|
| Estable | 0.660 | 0.660 | 2/5 semillas | 0/5 |
| Con crisis | 1.016 | 1.019 (+0.4 % a favor de ALEE) | 4/5 semillas | 1/5 |

Lectura: el resultado depende del proceso generador que se eligió, así que no es evidencia sobre datos reales. Aun en un escenario construido para favorecer a ALEE, la ventaja es mínima. En los meses de crisis la persistencia acierta más que ambos, pero ALEE reacciona tarde, porque sus variables de estado (volatilidad de 3 meses y brecha frente a la media de 12) se basan en meses ya observados.

## 16. Pre-registro: modelo anclado en SARIMA y confirmación en países nuevos (2026-11-09)

> **Estado: pre-registrado el 2026-11-09, antes de diseñar el candidato y antes de descargar ningún dato de confirmación.** Esta sección fija las reglas de una ronda nueva con dos fases separadas: una fase de **desarrollo** sobre datos ya vistos y una fase de **confirmación** sobre datos que ningún diseño ha visto. Solo la fase de confirmación puede sustentar una afirmación de superioridad.

### 16.1 Motivación e hipótesis

En los cuatro conjuntos ya evaluados (sección 14.1), SARIMA fue el mejor modelo en rango medio (1.50) y ALEE-3 quedó segundo (2.25). En España y México el límite de ALEE lo pusieron sus expertos, que son peores que SARIMA, y no el mecanismo de combinación. La hipótesis de esta ronda es que **un ALEE que use a SARIMA como experto ancla** conserva el desempeño de SARIMA en régimen normal y lo mejora cuando la evidencia local indica que otro experto acierta más (crisis, saltos bruscos).

**H1 (confirmatoria):** el candidato congelado tiene menor error cuadrático de pronóstico a un mes que el mejor modelo de referencia de cada país, en los países de confirmación.

**Diseño inicial del candidato (ALEE-S), declarado ahora:** expertos SARIMA(0,1,1)(0,1,1)[12] (pronóstico a un paso en ventana expansiva, como `regression_baselines`), persistencia, `nucleo_cambio` y estacional; pesos locales con regularización entrópica adaptativa, como en ALEE-3; previo π anclado en SARIMA. La fase de desarrollo puede modificar este diseño (ver 16.2), pero cada cambio queda registrado.

**Originalidad:** las combinaciones "SARIMA + otro modelo" (por ejemplo, ARIMA con redes neuronales o con ML sobre los residuos) ya existen en la literatura de pronóstico de desempleo. La contribución que se reclama no es combinar con SARIMA, sino el mecanismo de ALEE (pesos locales por estado con regularización entrópica adaptativa al tamaño efectivo de la muestra) aplicado con SARIMA como ancla. La revisión de literatura (`book/04_literatura.md`) se ampliará para documentarlo.

### 16.2 Fase de desarrollo (exploratoria)

- **Datos:** solo los cuatro conjuntos ya vistos: Colombia, EE.UU., España y México (secciones 10 a 14). Se declaran "quemados": cualquier resultado sobre ellos es exploratorio.
- **Libertad de diseño:** se pueden probar variantes (expertos, previo, variables de estado, regularización). Cada variante evaluada se registra en `docs/registro_desarrollo.md` con fecha y resultado, como "iteración S.k", sin omitir ninguna.
- **Tope:** como máximo **8 variantes**. Si ninguna mejora a SARIMA en desarrollo, se documenta y la ronda se cierra sin pasar a confirmación.
- **Regla de selección (fijada ahora):** pasa a confirmación **un solo candidato**: la variante con mejor rango medio, según RMSE medio de pliegues, sobre los cuatro conjuntos de desarrollo. En caso de empate pasa la más simple (menos expertos y menos hiperparámetros). Si ninguna variante tiene mejor rango medio que SARIMA en desarrollo, no hay confirmación.

### 16.3 Congelamiento (antes de descargar los datos de confirmación)

Antes de descargar cualquier serie de confirmación se agrega a este archivo una subsección **16.6 Congelamiento** con:
1. el código exacto del candidato y la huella SHA-256 de cada archivo de `src/` que usa;
2. su espacio de hiperparámetros y su presupuesto de búsqueda;
3. la fecha y hora del congelamiento.

Después de congelar no se modifica ningún archivo del candidato. Si hay que corregir un error de ejecución que no cambia el modelo (por ejemplo, de lectura de datos), se declara como desviación con su motivo.

### 16.4 Datos de confirmación (elegidos ahora, sin haberlos descargado)

**Fuente:** API SDMX de la OCDE, flujo `OECD.SDD.TPS,DSD_LFS@DF_IALFS_UNE_M`, la misma que se usó para México: `MEASURE=UNE_LF_M`, `ADJUSTMENT=N` (sin ajuste estacional), `SEX=_T`, `FREQ=M`. Para la edad se usa `AGE=Y_GE15`; si no existe, `Y15T74`; si tampoco, `Y15T64`.

**Países principales (5):** Canadá (CAN), Australia (AUS), Japón (JPN), Alemania (DEU) y Corea del Sur (KOR).

**Reserva, en este orden:** Italia (ITA), Países Bajos (NLD), Suecia (SWE).

**Criterios de inclusión.** Se verifican solo con los metadatos de la serie (existencia, longitud, huecos, tipo de ajuste), nunca con resultados de ningún modelo:
- serie mensual sin desestacionalizar;
- al menos 240 meses consecutivos;
- no puede ser un promedio móvil de tres meses por metodología nacional. Por esto quedan excluidos de entrada el Reino Unido, Chile y Noruega.

Si un país principal no cumple, se reemplaza por el siguiente de la reserva y se declara. Se usa el tramo sin huecos más largo que termine en el último mes disponible. Un único mes faltante aislado se rellena por interpolación lineal (mismo criterio que EE.UU., sección 14) y se declara.

### 16.5 Evaluación de confirmación (una sola vez)

- **Modelos:** el candidato congelado y los mismos 13 de la sección 14: ALEE-3, las siete combinaciones, LightGBM, SARIMA, persistencia, estacional ingenuo y estacional ALEE.
- **Validación y búsqueda:** mismas 14 variables nacionales, la misma validación `TimeSeriesSplit` de 5 pliegues externos y 3 internos con `gap=1`, optimización bayesiana de 40 evaluaciones con semilla 42 para todos los modelos con hiperparámetros, y una sola semilla para los modelos estocásticos.
- **Mejor de referencia de cada país:** el modelo distinto del candidato con menor RMSE medio de pliegues en ese país.
- **Se ejecuta una sola vez.** No se repite con otro candidato sobre estos mismos países.

**Pruebas (α = 0.05, Holm-Bonferroni sobre las cuatro pruebas de la condición b):**
- **T1.** Diebold-Mariano-HLN unilateral del candidato frente al mejor de referencia en cada país, combinado entre países con el método de Stouffer.
- **T2.** Wilcoxon de rangos con signo, unilateral, sobre el RMSE medio de pliegues del candidato frente al mejor de referencia en los 5 países. Con n = 5, el mínimo p unilateral es 1/32 ≈ 0.031: exige ganar en los 5 países.
- **T3.** Giacomini-White condicional en cada país, combinado con el método de Fisher. Cuenta a favor solo si la diferencia media de pérdida agrupada favorece al candidato.
- **T4.** SPA de Hansen en cada país, con el mejor de referencia como benchmark y el universo de modelos como candidatos, combinado con Fisher. Cuenta a favor solo si el candidato es el "mejor candidato" de la SPA en al menos 3 de los 5 países.
- **Además se reportan, sin entrar en el criterio:** el MCS por país (α = 0.10, bloques de 12 meses) y Friedman con Iman-Davenport sobre los 5 países.

**Criterio de superioridad (las cuatro condiciones a la vez):**
- **(a)** RMSE medio de pliegues menor que el mejor de referencia en al menos 3 de los 5 países.
- **(b)** Al menos 3 de las 4 pruebas T1 a T4 rechazan a favor del candidato, con p ajustado por Holm menor que 0.05.
- **(c)** Efecto no despreciable: d de Cohen ≥ 0.2 o δ de Cliff ≥ 0.147. Se calcula sobre las diferencias de pérdida cuadrática frente al mejor de referencia, normalizadas por la RMSE de SARIMA de cada país y apiladas entre países.
- **(d)** En ningún país el candidato es significativamente peor que el mejor de referencia (Diebold-Mariano-HLN bilateral, Holm sobre los 5 países).

**Compromiso:** se reportan todos los resultados, cumplan o no el criterio. Si el candidato no cumple, la conclusión es negativa, y cualquier diseño posterior es exploratorio y necesitaría otros datos de confirmación nuevos.

## 17. Adiciones para cumplir la guía de resultados del curso (2026-11-11)

> Estas adiciones se hacen **después** de conocer los resultados de las secciones 10 a 14, porque la guía de redacción de resultados del curso las exige. Se declaran aquí como desviaciones del protocolo; ninguna cambia el modelo ALEE-3 ni las predicciones ya obtenidas de los demás modelos.

1. **Modelo del curso que faltaba: regresión bayesiana (`BayesianRidge`).** Se agrega como baseline con el mismo protocolo (validación anidada 5/3, `gap=1`, 40 evaluaciones bayesianas, semilla 42) en los cuatro conjuntos.
2. **SVR con kernel RBF.** La guía pide SVR con kernel RBF; el SVR del segundo entregable podía elegir también el lineal. Se corre `SVR_RBF` con el mismo protocolo, y es el que se reporta como "SVM" en las tablas. El SVR anterior se conserva en los archivos, pero no entra en las tablas.
3. **Prueba bayesiana con ROPE.** La guía la exige para pronóstico, junto con Diebold-Mariano-HLN y Giacomini-White. Se fija **antes de calcularla**: región de equivalencia práctica de ±1 % en la diferencia relativa de RMSE, `(RMSE_ALEE − RMSE_b)/RMSE_b`. Por conjunto se usa la t bayesiana correlacionada (Corani y Benavoli, 2015) sobre los 5 pliegues externos, con ρ = proporción media de prueba. Entre conjuntos se usa el signed-rank bayesiano (Benavoli et al., 2014). Por la dependencia temporal, se interpreta con cautela, como indica la guía.
4. **Corrección de Holm por prueba**, sobre los 13 contrastes de ALEE frente a cada baseline, en cada conjunto.
5. **Métricas secundarias** que pide la guía: MAPE, R² y MASE. La escala del MASE es el MAE del ingenuo estacional (m = 12) dentro del entrenamiento de cada pliegue.
6. **Tamaño de efecto**: diferencia media de pérdida cuadrática con IC del 95 % por bootstrap estacionario (bloque medio de 12 meses, 2000 remuestras), d de Cohen y A12 de Vargha-Delaney.
7. **Entre conjuntos**: Friedman con Iman-Davenport, post-hoc de Conover con Holm frente a ALEE como modelo de control, y diagrama de diferencia crítica.
8. **Ablación con prueba de significancia**, en los cuatro conjuntos y con el mismo protocolo: sin localización, sin λ adaptativa, previo uniforme, promedio simple y cada experto solo, todas frente a ALEE completo con DM-HLN y Holm.
9. **Sensibilidad**: semillas del optimizador 42 a 46, mapa de RMSE sobre (h, λ) con vida media fija de 60 meses, y curva de aprendizaje con los 5 pliegues externos de EE.UU.
10. **Costo computacional**: tiempo de entrenamiento e inferencia, memoria pico y número de parámetros, medidos en el último pliegue de Colombia con los hiperparámetros por defecto de cada modelo.

Todo se reproduce con `experiments/resultados_finales.py`; las salidas quedan en `artifacts/resultados/`.

**Nota sobre ALEE en Colombia:** las tablas usan las predicciones canónicas de la iteración 4 (`artifacts/confirmatorio/iter4/oof_ALEE_K3_sin_ridge.csv`). Esa búsqueda incluyó una cuarta dimensión inerte (`ridge_alpha`), que el modelo de 3 expertos ya no usa pero que altera la secuencia de TPE. El recálculo con el espacio de 3 hiperparámetros (`artifacts/resultados/colombia/alee_completo.csv`) se usa solo para la ablación, la figura de pesos y la sensibilidad, y su diferencia con la versión canónica se reporta como parte de la variabilidad del optimizador.

**Ronda S (sección 16):** la fase de desarrollo terminó con S.4 como candidato, pero la confirmación en países nuevos **no se ejecutó**. La guía del curso aclara que un resultado no superior, bien analizado, es válido. Por eso el equipo decidió no gastar los datos de confirmación y presentar ALEE-3 como modelo propuesto. S.4 queda documentado como exploración de desarrollo, sin afirmación de superioridad.

## 18. Ajustes de credibilidad del paper (2026-11-11)

Cambios de análisis y de presentación posteriores a la sección 17. Ninguno modifica las predicciones de ALEE ni de los demás modelos.

1. **ALEE canónico de Colombia reconstruido.** Se reconstruyó con los hiperparámetros guardados en `artifacts/confirmatorio/iter4/rmse_K3_sin_ridge.csv` (diferencia máxima con las predicciones guardadas: 1.8e-14). La figura de pesos y la importancia por permutación ahora salen de ese modelo, el mismo de las tablas, y no del recálculo con 3 hiperparámetros.
2. **Curva de aprendizaje rehecha.** Periodo de prueba fijo (últimos 156 meses de EE.UU.) y ventanas de entrenamiento de 60, 120, 240, 480 y 777 meses. Los hiperparámetros de ALEE se fijan en la mediana de los elegidos en los pliegues de EE.UU.
3. **Costo computacional repetido** con los hiperparámetros elegidos por búsqueda bayesiana en el último pliegue de Colombia. Cada modelo se mide en un proceso independiente, y la memoria se toma del proceso completo, lo que incluye la memoria nativa.
4. **Nuevos análisis:** variabilidad entre semillas de Random Forest, XGBoost y LightGBM (semillas 42 a 51, ya calculadas en la validación confirmatoria), gráficos de predicho frente a observado y de residuos, y tablas completas de pruebas para EE.UU., España y México.
5. **Valores p exactos** (notación científica cuando p < 0.001).
6. **Reproducibilidad:** `experiments/reproducir_todo.py` (punto de entrada único), `scripts/descargar_conjuntos_adicionales.py` y `requirements-paper.txt` (versiones exactas).
7. **Paper:**
   - el criterio (a) vuelve al texto pre-registrado ("siete combinaciones"), y se declara que se aplica también a los modelos agregados;
   - las hipótesis de la ablación (H1 a H5) se anuncian en la metodología;
   - se agrega una tabla que separa lo pre-registrado de lo agregado después;
   - se incluyen proposiciones con esbozo de demostración;
   - la contribución se reformula en una sola frase, citando a Makridakis et al. (2018, 2020).

## 19. Cinco conjuntos adicionales para la comparación con los modelos del curso (2026-11-12, antes de ver resultados)

> Se escribe antes de correr ningún modelo sobre estos datos. Los países son los que la sección 16.4 ya había elegido, y verificado con sus criterios de inclusión, para confirmar la variante anclada en SARIMA. Esa confirmación no se ejecutó (sección 17), así que los datos siguen sin haberse visto.

**Motivo:** la guía de resultados del curso recomienda al menos 5 conjuntos de datos para que las pruebas entre conjuntos tengan potencia. Con 4, el Wilcoxon no puede bajar de p = 0.125. Con 9, el mínimo es 0.0039.

**Datos (API SDMX de la OCDE, `DSD_LFS@DF_IALFS_UNE_M`, ajuste N, sexo total, edad 15+, mensual; descargados el 2026-11-12):**

| País | Periodo | Meses |
|---|---|---|
| Canadá (CAN) | 1955-01 a 2026-10 | 862 |
| Australia (AUS) | 1978-02 a 2026-09 | 584 |
| Japón (JPN) | 1955-01 a 2026-09 | 861 |
| Alemania (DEU) | 1991-01 a 2026-09 | 429 |
| Corea del Sur (KOR) | 1989-01 a 2026-09 | 453 |

Los cinco cumplen los criterios de §16.4. Las series están filtradas con `scripts/preparar_oecd.py`, y en ninguna hubo que interpolar meses.

**Modelo evaluado:** ALEE-3 congelado (iteración 4), con el mismo espacio y presupuesto de búsqueda. No se cambia nada del diseño.

**Comparación:** los ocho modelos del curso. Por consistencia con el pipeline también se corren LightGBM y las referencias de series de tiempo, aunque el paper solo reporta los modelos del curso.

**Protocolo:** el mismo de los conjuntos anteriores: 14 variables, validación anidada 5/3 con `gap=1`, 40 evaluaciones bayesianas, semilla 42, las mismas pruebas, Holm por prueba y la misma ablación.

**Compromiso:** se reportan los resultados en los nueve conjuntos, sean cuales sean.

### 19.1 Resultado (2026-11-12)

ALEE-3 congelado tiene el menor RMSE en los nueve conjuntos y también el mejor MAE, MAPE, R² y MASE en todos. Frente al mejor modelo del curso, el RMSE baja entre 21 % (Japón) y 33 % (Colombia).

- **Por conjunto:** las 81 comparaciones (nueve modelos por nueve conjuntos) de Diebold-Mariano-HLN y de Giacomini-White son significativas con Holm (máximos 0.0074 y 0.020).
- **Entre conjuntos:** Friedman da χ² = 62.7 (p = 4.0e-10). El Wilcoxon con Holm da p = 0.035 frente a cada uno de los nueve modelos, y el Conover con Holm da p < 0.05 frente a ocho de ellos (frente a la regresión bayesiana, p = 0.071).

*Corrección del 2026-11-15:* la versión anterior de estos dos puntos decía 72 comparaciones, χ² = 55.7, Wilcoxon p = 0.031 y Conover p ≤ 0.026, cifras de una corrida sin LightGBM. Las cifras de arriba son las de `artifacts/resultados/solo_curso/` y las del paper.
- **Ablación:**
  - el promedio simple es peor en 7 de 9 conjuntos;
  - la localización es significativa solo en México y Japón;
  - la λ adaptativa es significativa en Canadá y Japón;
  - el previo global no es significativo en ningún conjunto;
  - el experto estacional solo es significativamente peor en Canadá, Australia y Japón.

Detalle en `artifacts/resultados/solo_curso/`, generado con `experiments/resultados_curso.py`.
