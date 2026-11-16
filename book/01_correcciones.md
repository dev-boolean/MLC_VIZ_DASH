# Trazabilidad de correcciones

El segundo entregable obtuvo una nota de **4.5**. El docente valoró la validación anidada que respeta el tiempo, el preprocesamiento sin fuga, la identidad contable detectada y evitada, la comparación honesta con la persistencia y el SARIMA, la batería estadística y la comparación de optimizadores. Descontó por cinco puntos, que se recogen en la tabla siguiente junto con dos hallazgos propios. Una observación sin fila en esta tabla se considera no corregida.

## Observaciones del docente

| # | Observación del docente | Acción realizada | Ubicación en el Jupyter Book | Evidencia | Estado |
|---|---|---|---|---|---|
| 1 | No se entrega `src/` | Se publica la carpeta `src/` con los módulos del pipeline y tres módulos nuevos: `artifacts.py` (bloque B), `original_model.py` (modelo original) y `explain.py` (explicaciones del dashboard) | Capítulo 10 | Carpeta `src/` del repositorio | Hecho; falta publicar el repositorio |
| 2 | No se entrega `requirements.txt` | Se publica con versiones exactas. `pywin32` queda marcado solo para Windows, para que la instalación funcione en Linux (Render y GitHub Actions), y se agrega `pytest`. El dashboard tiene su propio `requirements-dash.txt`, más liviano | Capítulo 10 | `requirements.txt`, `requirements-dash.txt` | Hecho; falta publicar el repositorio |
| 3 | No se entrega la tabla maestra | Se publica `experiments/results_master.csv` (140 filas, con métricas del bucle externo, tiempos, número de evaluaciones, semilla e hiperparámetros por pliegue), junto con `results_folds.csv`, `histories/` y `oof/` | Capítulos 3 y 10 | {ref}`tabla-maestra` | Hecho; los resultados del modelo original y de los nueve conjuntos están en `artifacts/resultados/` (capítulo 7) |
| 4 | Se presenta como un solo cuaderno y no como JBook | El cuaderno se dividió en capítulos con `_toc.yml` siguiendo la estructura mínima de la guía (`scripts/split_book.py`), sin cambiar las salidas de los entregables anteriores | Todo el libro | `book/_toc.yml` | Hecho |
| 5 | No se mide la GPU | No se mide. La aceleración que pide la guía es la de XGBoost con `device="cuda"`, que exige una GPU NVIDIA. El equipo del proyecto tiene una AMD Radeon RX 6600 (gfx1032), sin CUDA, y XGBoost no la puede usar en Windows: la alternativa de AMD (ROCm) no tiene soporte completo para esta tarjeta en Windows. Se prefirió no reportar un factor de aceleración medido en otro equipo | Capítulo 10 (hardware) | Especificación del hardware en {ref}`lista-reproducibilidad` | Justificada, sin medición |

## Hallazgos propios

| # | Hallazgo | Acción | Ubicación | Estado |
|---|---|---|---|---|
| P1 | La celda de carga de datos leía `C:/Users/boolean/Downloads/MLC.xlsx`, una ruta absoluta de un solo equipo | Ahora lee `../data/MLC.xlsx`, que es el mismo archivo (huella SHA-256 idéntica), por lo que sus resultados no cambian | Capítulo 2 | Hecho |
| P2 | Con el criterio de la guía (métrica **media** del bucle externo), la mejor combinación de regresión es Ridge (RMSE 1.196 ± 0.552). El segundo entregable destacaba a SVR porque usaba el RMSE **agrupado** de las 240 predicciones (1.314) | Se reportan los dos criterios y se explica la diferencia | {doc}`03b_mejores_combinaciones` | Hecho |
| P3 | Un pronóstico estacional muy sencillo (el último valor más el cambio medio del mismo mes del año) supera a las 140 combinaciones en RMSE medio por pliegue (0.780 frente a 1.196 de Ridge) | Se incorpora como uno de los tres expertos de ALEE y se evalúa solo en la ablación | Capítulos 5 y 7 | Hecho |

## Resultados que cambian

Las correcciones 1 a 4 y P1 no cambian ningún resultado del segundo entregable: son de entrega, organización y rutas. P2 cambia qué modelo de regresión se presenta como el mejor (Ridge en lugar de SVR) y P3 motivó el experto estacional de ALEE. Ninguno de los dos cambia el ranking de los 140 modelos.
