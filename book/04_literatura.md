# Revisión de literatura y novedad

```{admonition} Estado de la revisión
:class: note
La búsqueda en **Scopus** se hizo el 2026-11-15 con el acceso institucional de la Universidad del Norte: las cinco cadenas pre-registradas, cuatro cadenas complementarias y *snowballing* hacia adelante, con conteos exactos (`docs/revision_sistematica/registro_busquedas.md`). Scopus es la base de la revisión. Las filas de Google Scholar y de arXiv o RePEc son de una búsqueda web preliminar anterior y se conservan solo como complemento. **Pendiente del equipo:** leer completos los antecedentes más cercanos y repetir la búsqueda en Scopus antes de la entrega.
```

## Protocolo de búsqueda

**Bases de datos:** Scopus (o Web of Science), Google Scholar y arXiv o RePEc/SSRN, porque el problema es de economía aplicada.

**Cadenas propuestas**, que combinan (i) la familia del modelo, (ii) el problema de aplicación y (iii) sinónimos en inglés y en español:

1. `("forecast combination" OR "ensemble") AND ("locally weighted" OR "kernel weights" OR "state-dependent weights") AND "unemployment"`
2. `("dynamic ensemble" OR "mixture of experts") AND ("unemployment rate" OR "labor market") AND forecast*`
3. `("exponentiated gradient" OR "entropic regularization" OR "Kullback-Leibler") AND "forecast combination"`
4. `("regime-dependent" OR "regime switching") AND "combination weights" AND (unemployment OR "labour market")`
5. `("combinación de pronósticos" OR "ensamble") AND ("tasa de desempleo" OR "mercado laboral") AND Colombia`

## Matriz de novedad

| Base de datos | Cadena | Fecha | Resultados | Revisados | Relevantes | Observación |
|---|---|---|---|---|---|---|
| Scopus | 1 a 5 (pre-registradas) | 2026-11-15 | 0, 0, 4, 0 y 0 | 4 | 1 | Las cadenas pre-registradas resultaron demasiado específicas |
| Scopus | C1 a C4 (complementarias) y *snowballing* hacia adelante desde los tres trabajos más cercanos | 2026-11-15 | 14, 45, 4, 11 y 12 (90 registros con duplicados) | 90 | 9 | Detalle de cada cadena en `docs/revision_sistematica/registro_busquedas.md` |
| Google Scholar | 1 a 5 (más sinónimos: *state-dependent forecast combination*, *kernel-weighted ensemble*, *entropic regularization simplex weights*, *adaptive shrinkage forecast combination*) | 2026-11-05 | ~60 (agregado entre cadenas y sinónimos) | ~25 | 6 | Búsqueda web preliminar, no un export formal de Scholar con todos los resultados — ver detalle abajo |
| arXiv o RePEc | 1 a 4 (mismas variantes) | 2026-11-05 | ~40 | ~18 | 4 | Incluye *snowballing* desde Li 2001, Cerqueira 2017 y Elliott-Timmermann 2005; sin acceso a WebFetch en esta pasada, varios candidatos quedan solo revisados por título/resumen de buscador, no por texto completo (marcado explícitamente abajo) |

## Antecedentes más cercanos (Scopus, 2026-11-15)

El mecanismo de ALEE tiene cuatro piezas: (1) pesos que cambian con el estado; (2) localización con un núcleo continuo sobre ese estado; (3) pesos en el símplex; (4) regularización hacia la combinación global, más fuerte cuando hay pocos meses parecidos. La clasificación siguiente se hizo con el título y el resumen; los cuatro primeros trabajos deben leerse completos para confirmarla.

| Trabajo | (1) | (2) | (3) | (4) | Diferencia principal con ALEE |
|---|---|---|---|---|---|
| {cite:t}`oelrich2024` | Sí | Parcial: ventana local en el espacio de covariables | Sí | Parcial: se encoge hacia pesos **iguales** con pocos datos locales | Combina **densidades** con puntaje logarítmico. ALEE combina pronósticos puntuales con pérdida cuadrática y se encoge hacia la combinación global óptima con divergencia de Kullback-Leibler, con una fuerza ajustada por el tamaño efectivo de Kish; el problema tiene solución única y un algoritmo con descenso monótono |
| {cite:t}`dang2027` | Sí | No: K celdas discretas de contexto | Sí | No: descenso de espejo entrópico por celda, sin previo global | Aprendizaje en línea con cotas de arrepentimiento; ALEE localiza con un núcleo continuo y regulariza hacia la combinación global |
| {cite:t}`chenmaung2023` | Sí, en el tiempo | Sí, lineal local en el tiempo | No consta | No: penalización SCAD | Los pesos dependen del tiempo, no del estado del mercado laboral; aplicación a inflación y desempleo de EE. UU. |
| {cite:t}`li2001` (versión publicada en 2004) | Sí | Sí, regresión de núcleo | No consta | No | Regresión de combinación no paramétrica; aplicación a inflación |
| {cite:t}`elliotttimmermann2005` | Sí | No: régimen latente discreto | — | — | Pesos por régimen de Markov |

Ningún trabajo encontrado reúne las cuatro piezas. El más cercano es {cite:t}`oelrich2024`, que comparte la idea de pesos condicionados a variables de estado que se encogen cuando hay poca información local.

## Trabajos cercanos (búsqueda preliminar)

| Trabajo | Familia | Qué hace | Qué conserva ALEE | Qué es distinto |
|---|---|---|---|---|
| {cite:t}`li2001` | Combinación no lineal de pronósticos | Pesos de combinación estimados por regresión de núcleo | La idea de pesos que varían con una variable de estado | ALEE resuelve un problema convexo en el símplex con previo global y regularización adaptativa. Hay que verificar en el texto completo si sus pesos están restringidos |
| {cite:t}`cerqueira2017` (ADE) | Ensambles dinámicos | Meta-aprendices que predicen el error de cada modelo; pesos con softmax | Pesos que dependen del punto que se pronostica | ALEE no entrena un meta-modelo: los pesos tienen solución única y garantías (P1 a P3) |
| Combinación por regímenes (arXiv 2209.01697; Kazak y Tetereva) | Combinación dependiente del régimen | Pesos distintos para regímenes discretos | Pesos que cambian con el estado económico | Localización continua por núcleo, sin estimar regímenes |
| {cite:t}`kivinen1997` | Aprendizaje en línea | Gradiente exponenciado en el símplex | La actualización multiplicativa | Se usa como solucionador de un problema local, no como aprendizaje en línea |
| {cite:t}`timmermann2006` | Revisión de combinación de pronósticos | Promedio simple, pesos por error y *stacking*; el «enigma de la combinación» | Las reglas estándar como casos límite (P4) | — |
| {cite:t}`orozcocastaneda2024` | Pronóstico del desempleo en Colombia | SVR y redes neuronales para pronosticar empleo y desempleo mensual, con variables de Google Trends, enfocado en la tercera ola de COVID-19 | El problema de aplicación | No usa procesos gaussianos ni combinación de pronósticos; no hay localización por régimen |
| {cite:t}`perezrosero2025` (artículo derivado de una tesis de maestría de la UNAL, 2025) | Pronóstico del desempleo en Colombia | Procesos gaussianos con reducción de dimensionalidad interpretable (UMAP y *Local Biplot*); supera a ARIMA, Lasso, ElasticNet y SVR | El problema de aplicación y la comparación directa contra Lasso | No usa combinación de expertos ni localización por régimen; su reducción de dimensionalidad resuelve interpretabilidad, no el cambio del mejor modelo entre regímenes |
| Banco de la República, *Forecasting the Colombian Unemployment Rate Using Labour Force Flows* | Pronóstico del desempleo en Colombia a partir de flujos del mercado laboral | **Pendiente de verificar en el texto completo** — no se confirmó si usa combinación de pronósticos; no citar su metodología hasta leerlo | El problema de aplicación | Por verificar |
| {cite:t}`elliotttimmermann2005` | Combinación de pronósticos dependiente del régimen | Pesos de combinación que cambian según un régimen latente tipo Markov; aplicado a variables macroeconómicas en general | Pesos que dependen del estado de la economía | Régimen discreto y latente, no localización continua por núcleo; no se menciona regularización KL hacia un previo. Es el antecedente conceptual más cercano — pendiente de leer el texto completo, no solo el resumen |
| {cite:t}`cumingsmenon2020` | Combinación de pronósticos probabilísticos | Regularización entrópica aplicada a una distancia de Wasserstein entre distribuciones; aplicado a inflación de EE.UU. | La idea de regularización entrópica en la combinación | Es para pronósticos de densidad, no para pesos puntuales en el símplex; no hay previo global ni localización por estado |
| {cite:t}`nuthanakaluva2026` (preprint 2026) | Pesos de combinación tipo Gibbs/softmax | Pesos exponenciados sobre pérdidas acumuladas, actualizados con descenso exponenciado; aplicado a M4 y series de tráfico/electricidad, no a macroeconomía | La actualización multiplicativa en el símplex (técnica clásica que el propio trabajo cita de Vovk 1990) | Sin regularización KL hacia un previo, sin localización por núcleo sobre un estado económico, sin aplicación macroeconómica |
| {cite:t}`wanghyndman2023` (revisión) | Revisión de combinación de pronósticos (continuación de `timmermann2006`) | Panorama de 50+ años de métodos de combinación, incluidos los basados en aprendizaje automático y los adaptativos en el tiempo | Confirma, desde una revisión independiente y más reciente, que no identifica un método con el mecanismo exacto de ALEE | Es una revisión, no propone un método nuevo; se usó para *snowballing* adicional sin encontrar un antecedente con el mecanismo exacto |
| «Split-then-Combine» (STC) con selección CAS, Arroyo y de Juan (arXiv 2012.11935, basado en Arroyo & de Juan 2014) | Combinación de pronósticos en el símplex, enfoque composicional (geometría de Aitchison) | Pesos restringidos al símplex, calculados en forma cerrada (media geométrica de pesos inversos al error cuadrado) por panel estacional discreto (mes o trimestre del calendario); selección de pronosticadores por *clustering* y biplots cuando hay muchos modelos | **Verificado en texto completo (2026-11-05).** Comparte únicamente la restricción al símplex. No tiene núcleo de localización por estado continuo (solo agrupa por panel estacional discreto), no tiene término de regularización KL/entrópica hacia un previo global (el centro del símplex es la propia estimación, no un objetivo de encogimiento), y no ajusta la fuerza de ninguna regularización por tamaño de muestra efectivo porque no existe tal regularización en el método |
| Lee y Lee, «Solving the Forecast Combination Puzzle» (arXiv 2308.05263); Liu, «...Using Double Shrinkages» (2024); Radchenko y Vasnev, «Too Similar to Combine?» (2023) | Encogimiento (*shrinkage*) de pesos de combinación | Abordan por qué el promedio simple suele superar a los pesos óptimos, mediante distintas formas de encogimiento hacia pesos iguales | El símplex como espacio de los pesos | Resuelven el "enigma de la combinación" con encogimiento estático o por similitud entre pronósticos, no con localización por estado ni regularización entrópica adaptativa por tamaño de muestra efectivo; revisados solo por título y resumen de buscador, no por texto completo |
| «Dynamic TMoE» (arXiv 2605.20678) y «Dynamic Mixture of Experts Against Severe Distribution Shifts» (arXiv 2511.18987) | Mezcla de expertos (*mixture of experts*) para series de tiempo no estacionarias | Redes de compuertas (*gating networks*) que aprenden a ponderar expertos ante *drift* o cambios de distribución | La idea de pesos que se adaptan a cambios de régimen | Las compuertas se entrenan como redes neuronales (aprendizaje por gradiente de una función de pérdida de la red), no como la solución de un problema convexo con solución única y garantías formales; no hay núcleo de localización por estado observado ni regularización KL hacia un previo global; no aplicado a macroeconomía. Revisados solo por título y resumen de buscador |

**Nota sobre duplicados evitados:** la búsqueda confirmó que «Combining Forecasts with Nonparametric Kernel Regressions» (*Studies in Nonlinear Dynamics & Econometrics*, 2004, vol. 8, núm. 4) es la versión publicada en revista del mismo trabajo ya citado como {cite:t}`li2001` (Bank of Canada Working Paper 2001-12, mismo número de serie `wp01-12`); no se agrega como entrada separada para no duplicar el antecedente.

## Declaración de novedad

> Al mejor de nuestro conocimiento, y según la búsqueda descrita, no se encontraron estudios que combinen pronósticos puntuales con pesos localizados por un núcleo continuo sobre el estado del mercado laboral, restringidos al símplex y regularizados con divergencia de Kullback-Leibler hacia la combinación global con una fuerza que depende del tamaño efectivo de la muestra local, ni que apliquen esa regla al pronóstico de la tasa de desempleo. Los antecedentes más cercanos son {cite:t}`oelrich2024`, que condiciona los pesos de una combinación de densidades a la precisión local de cada experto y los encoge hacia pesos iguales cuando hay pocos datos; {cite:t}`dang2027`, que usa descenso de espejo entrópico por celdas discretas de contexto en aprendizaje en línea; y {cite:t}`li2001` y {cite:t}`chenmaung2023`, que estiman pesos con núcleos sobre los pronósticos o sobre el tiempo, sin restricción al símplex ni regularización hacia la combinación global.

La contribución que se reclama es, por tanto, el mecanismo conjunto (localización continua por estado, previo global, regularización entrópica adaptativa y su solución con garantías) y su aplicación al desempleo, no cada pieza por separado: la combinación condicionada a covariables, el encogimiento y el gradiente exponenciado ya existen en la literatura.

```{admonition} Alcance
:class: warning
La clasificación se basa en títulos y resúmenes de Scopus y en la búsqueda preliminar. Antes de la entrega el equipo debe leer completos {cite:t}`oelrich2024`, {cite:t}`dang2027`, {cite:t}`chenmaung2023` y {cite:t}`li2001`, y repetir la búsqueda en Scopus. Si alguno resulta tener las cuatro piezas, la contribución se reduce a la aplicación y el tope de la nota es 3.0 (condición R1 de la guía).
```
