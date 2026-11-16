# Registro de la revisión sistemática

Búsquedas hechas el 2026-11-15 en Scopus (acceso institucional de la Universidad del Norte), con el campo `TITLE-ABS-KEY`, salvo donde se indica. La lectura de cada trabajo se hizo sobre el título y el resumen de Scopus; los trabajos marcados como antecedentes cercanos deben leerse completos para confirmar la clasificación.

## Cadenas pre-registradas (capítulo 4)

| # | Cadena | Resultados |
|---|---|---|
| 1 | `("forecast combination" OR "ensemble") AND ("locally weighted" OR "kernel weights" OR "state-dependent weights") AND "unemployment"` | 0 |
| 2 | `("dynamic ensemble" OR "mixture of experts") AND ("unemployment rate" OR "labor market") AND forecast*` | 0 |
| 3 | `("exponentiated gradient" OR "entropic regularization" OR "Kullback-Leibler") AND "forecast combination"` | 4 |
| 4 | `("regime-dependent" OR "regime switching") AND "combination weights" AND (unemployment OR "labour market")` | 0 |
| 5 | `("combinación de pronósticos" OR "ensamble") AND ("tasa de desempleo" OR "mercado laboral") AND Colombia` | 0 |

Las cadenas pre-registradas resultaron demasiado específicas: cuatro de cinco no devolvieron nada. Para no dejar sin revisar la literatura cercana se agregaron cadenas complementarias más amplias, declaradas como tales.

## Cadenas complementarias

| # | Cadena | Resultados |
|---|---|---|
| C1 | `"forecast combination" AND unemployment` | 14 |
| C2 | `("forecast combination" OR "combining forecasts") AND (kernel OR "locally weighted" OR nonparametric OR "state-dependent" OR "state dependent")` | 45 |
| C3 | `("forecast combination" OR "combining forecasts" OR "prediction pool*" OR "forecast averaging") AND ("exponentiated gradient" OR "mirror descent" OR entropic OR "Kullback-Leibler divergence" OR "local predictive")` | 4 |
| C4 | `(unemployment OR desempleo) AND (forecast* OR pronóstico* OR prediction) AND Colombia` | 11 |
| S1 | *Snowballing* hacia adelante: `REF("local prediction pools") OR REF("combining forecasts with nonparametric kernel regressions") OR REF("context-gated exponentiated gradient")` | 12 |

Total: 90 registros (con duplicados entre cadenas). Tras el filtro por título y resumen quedaron los trabajos de las tablas siguientes; el resto se descartó por no tratar combinación de pronósticos (por ejemplo, pronóstico de energía, volatilidad o demanda con un solo modelo) o por combinar con pesos fijos.

## Antecedentes cercanos al mecanismo de ALEE

Las cuatro piezas del mecanismo son: (1) pesos que cambian con el estado o el contexto; (2) localización con un núcleo continuo sobre ese estado; (3) pesos en el símplex; (4) regularización hacia una combinación global, más fuerte cuando hay pocos datos parecidos.

| Trabajo | (1) | (2) | (3) | (4) | Diferencia principal con ALEE |
|---|---|---|---|---|---|
| Oelrich, Villani y Ankargren (2024), *Local prediction pools*, Journal of Forecasting 43(1), 103–117 | Sí | Parcial: ventana local (*caliper*) en el espacio de covariables | Sí | Parcial: los pesos se acercan a los **iguales** cuando hay pocos datos locales | Combina **densidades** con puntaje logarítmico; ALEE combina pronósticos puntuales con pérdida cuadrática, regulariza hacia la combinación global óptima (no hacia pesos iguales) con divergencia de Kullback-Leibler y fuerza ajustada por el tamaño efectivo de Kish, y tiene solución única con un algoritmo de descenso monótono |
| Dang (2027, en prensa), *Context-gated exponentiated gradient for online forecast combination*, Operations Research Letters 71, 107567 | Sí | No: el contexto parte el tiempo en K celdas discretas | Sí | No: descenso de espejo entrópico por celda, sin previo global | Aprendizaje en línea con cotas de arrepentimiento y celdas discretas; ALEE localiza con un núcleo continuo y regulariza hacia la combinación global |
| Chen y Maung (2023), *Time-varying forecast combination for high-dimensional data*, Journal of Econometrics 237(2), 105418 | Sí, en el **tiempo** | Sí, lineal local en el tiempo | No consta en el resumen | No: penalización SCAD para seleccionar pronósticos | Los pesos dependen del tiempo y no del estado del mercado laboral; aplicación a inflación y desempleo de EE. UU. |
| Li y Tkacz (2004), *Combining forecasts with nonparametric kernel regressions*, Studies in Nonlinear Dynamics and Econometrics 8(4) | Sí | Sí, regresión de núcleo | No consta en el resumen | No | Regresión de combinación no paramétrica; aplicación a inflación |
| Oelrich y Villani (2025), *Modeling local predictive ability using power-transformed Gaussian processes* | Sí | Sí, proceso gaussiano sobre covariables | — | — | Modela la capacidad predictiva local con puntajes logarítmicos; continúa los *local prediction pools* |
| Chernis, Hauzenberger, Huber, Koop y Mitchell (2025), *Predictive density combination using Bayesian machine learning*, International Economic Review 66(3) | Sí | Aprendizaje automático bayesiano | — | — | Síntesis predictiva bayesiana de densidades |
| Elliott y Timmermann (2005), *Optimal forecast combination under regime switching* | Sí | No: regímenes discretos | — | — | Pesos por régimen de Markov |

**Ningún trabajo encontrado tiene las cuatro piezas a la vez.** El más cercano es *Local prediction pools*, que comparte la idea de pesos condicionados a variables de estado que se encogen cuando hay poca información local.

## Pronóstico del desempleo en Colombia (C4)

| Trabajo | Método |
|---|---|
| Conto-López (2026), Sankhya B | Holt-Winters, SARIMA, redes neuronales y un modelo híbrido |
| Pérez-Rosero y otros (2025), Computation 13(5) | *Biplots* locales y procesos gaussianos |
| *Nonlinear macroeconomic time series forecasting with TSARX* (2025) | TSARX (umbral con estacionalidad multiplicativa) en tres economías |
| Orozco-Castañeda, Sierra-Suárez y Vidal (2024), Bulletin of Economic Research | Aprendizaje automático |

Ninguno usa una combinación de pronósticos con pesos localizados por el estado del mercado laboral.

## Pendiente

El equipo decidió usar Scopus como única base de la revisión; la búsqueda web preliminar en Google Scholar, arXiv y RePEc (capítulo 4) se conserva solo como complemento.


- **Lectura completa** de *Local prediction pools*, Dang (2027), Chen y Maung (2023) y Li y Tkacz (2004), para confirmar las columnas marcadas como «no consta» o «parcial».
- **Repetir la búsqueda** justo antes de la entrega y anotar la nueva fecha.
