# Ficha del modelo: ALEE (agregación local entrópica de expertos)

## Detalles del modelo

- **Autores:** Andrés Parejo, Santiago Hurtado y Juan Marín (Pregrado en Ciencia de Datos, curso de Machine Learning, Universidad del Norte).
- **Tipo:** combinación de tres expertos de pronóstico programados en NumPy: persistencia, regresión de Nadaraya-Watson del cambio mensual y cambio estacional medio. Los pesos dependen del estado del mercado laboral (volatilidad de los últimos tres meses y brecha frente a la media de doce meses) y se obtienen resolviendo un problema convexo en el símplex, regularizado con divergencia de Kullback-Leibler hacia una combinación global, con más fuerza cuando hay pocos meses parecidos en la historia.
- **Implementación:** `src/original_model.py`, clase `ModeloOriginal`, compatible con scikit-learn. El artefacto del dashboard es `artifacts/reg_ALEE/model.joblib`.
- **Hiperparámetros:** ancho de banda del núcleo (`bandwidth`), vida media del decaimiento temporal (`half_life`) y fuerza de la regularización (`lam`), elegidos con optimización bayesiana de 40 evaluaciones en una validación interna de ventana expansiva.
- **Garantías:** solución única (P1), descenso monótono del algoritmo (P2), error nunca mayor que el del peor experto (P3) y casos límite conocidos (P4). Se comprueban en `tests/test_original_model.py`.
- **Versión:** la de tres expertos descrita en el paper. Las versiones anteriores y las demás iteraciones de diseño están registradas en `docs/registro_desarrollo.md` y `experiments/protocolo.md`.

## Uso previsto

- Pronóstico académico a un mes de la tasa de desempleo nacional de Colombia, con la información disponible al cierre del mes actual.
- Comparación de reglas de combinación de pronósticos en un curso de machine learning.

## Usos no recomendados

- Decisiones de política económica, laboral o financiera sin el criterio de expertos y sin las cifras oficiales del DANE.
- Pronósticos a más de un mes: el modelo solo se entrenó y evaluó a un paso.
- Otras series sin repetir la validación: los hiperparámetros dependen de los datos.

## Datos

- **Colombia:** 288 meses (objetivos de enero de 2002 a diciembre de 2025) con 17 variables construidas solo con información pasada. Ver `DATA_STATEMENT.md`.
- **Conjuntos adicionales:** Estados Unidos (FRED), España (Eurostat), México, Canadá, Australia, Japón, Alemania y Corea del Sur (OCDE), con 14 variables nacionales, porque no tienen indicadores de área. Se evaluaron una sola vez, con el modelo ya congelado.
- **Validación:** anidada con `TimeSeriesSplit` (5 pliegues externos y 3 internos, un mes de separación) y semilla 42.

## Métricas

RMSE medio de los cinco pliegues externos frente al mejor de los ocho modelos de regresión del curso (KNN, Ridge, Lasso, regresión bayesiana, árbol de decisión, Random Forest, XGBoost y SVR con kernel RBF) y LightGBM en cada conjunto. Tabla completa, con MAE, MAPE, R² y MASE, en el paper y en `artifacts/resultados/solo_curso/`.

| Conjunto | ALEE | Mejor modelo de comparación | Reducción del error |
|---|---|---|---|
| Colombia | 0.817 | Ridge (1.219) | 33% |
| Estados Unidos | 0.380 | Lasso (0.541) | 30% |
| España | 0.280 | Lasso (0.369) | 24% |
| México | 0.281 | SVR (kernel RBF) (0.369) | 24% |
| Canadá | 0.437 | Lasso (0.601) | 27% |
| Australia | 0.240 | Regresión bayesiana (0.342) | 30% |
| Japón | 0.149 | Regresión bayesiana (0.188) | 21% |
| Alemania | 0.226 | Regresión bayesiana (0.288) | 22% |
| Corea del Sur | 0.325 | Regresión bayesiana (0.438) | 26% |

- **Pruebas por conjunto:** Diebold-Mariano-HLN y Giacomini-White con corrección de Holm rechazan la igualdad en las 81 comparaciones (nueve modelos por nueve conjuntos).
- **Entre conjuntos:** Friedman con Iman-Davenport rechaza la igualdad de los diez modelos; el Wilcoxon con Holm da p = 0.035 frente a cada modelo.
- **Intervalo de predicción al 90 %** (predicción conforme sobre los 240 errores fuera de muestra de Colombia): semiancho de 1.18 pp; cobertura medida hacia adelante de 93.8% en 192 meses, con un ancho medio de 2.55 pp.

## Ablación

La mayor parte de la ventaja viene de los expertos. La combinación aprendida supera al promedio simple en siete de los nueve conjuntos; la localización por estado solo aporta de forma significativa en México y Japón, y la regularización adaptativa en Canadá y Japón.

## Limitaciones y riesgos

- Nueve conjuntos dan una potencia moderada a las pruebas entre conjuntos.
- En Colombia se evaluaron diez iteraciones de diseño sobre los mismos 240 meses, lo que puede volver optimista el resultado en ese conjunto. Los ocho conjuntos adicionales, evaluados una sola vez con el modelo congelado, son la evidencia más confiable.
- El error se concentra en el choque de 2020: con pocos meses parecidos en la historia, la localización tiene poca información y el modelo se acerca a los pesos globales.
- Los datos son la versión vigente de las series y no la publicada en tiempo real cada mes.
- El modelo supone que las filas están en orden cronológico y que cada fila solo contiene información anterior al mes que se pronostica.

## Consideraciones éticas

- La tasa de desempleo resume la situación de millones de personas: un error de pronóstico puede retrasar o adelantar respuestas de política. El modelo debe presentarse siempre con su incertidumbre.
- No usa datos individuales ni información personal.
