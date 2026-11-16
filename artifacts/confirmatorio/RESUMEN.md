# Resumen de la validación confirmatoria (2026-11-05, actualizado con la iteración 4)

Protocolo congelado: `experiments/protocolo.md`. Script: `experiments/run_confirmatorio.py` (corrida principal) + `experiments/analizar_confirmatorio.py` (pruebas estadísticas). Iteración 4 (exploratoria, posterior a este resultado): `experiments/iter4_reducir_expertos.py`, `experiments/protocolo.md` sección 11, `docs/registro_desarrollo.md`.

## Veredicto en una frase

**H1 se rechaza, con las dos versiones de ALEE.** Con los 5 expertos originales (iteración 3), ALEE perdía de forma estadísticamente significativa contra su propio experto estacional aislado. Con 3 expertos (iteración 4, adoptada como versión canónica tras quitar los dos expertos ridge que individualmente eran el doble de malos), ALEE **empata estadísticamente** con el estacional aislado y con SARIMA, y le gana con significancia a todo lo demás — pero sigue sin cumplir, en el punto estimado, el criterio de superioridad pre-registrado.

## Tabla de RMSE (240 meses fuera de muestra, de mejor a peor) — con ALEE de 3 expertos (iteración 4)

| Modelo | RMSE | MAE |
|---|---|---|
| Estacional (experto ALEE aislado) | 0.836 | 0.564 |
| **ALEE (3 expertos, iteración 4)** | **0.869** | **0.586** |
| SARIMA(0,1,1)(0,1,1)[12] | 0.995 | 0.562 |
| Persistencia | 1.293 | 0.905 |
| Ridge | 1.346 | 0.906 |
| SVR | 1.370 | 0.879 |
| KNN | 1.373 | 0.883 |
| Random Forest | 1.493 | 0.911 |
| XGBoost | 1.567 | 0.888 |
| LightGBM | 1.704 | 1.048 |
| Decision Tree | 1.748 | 1.085 |
| Lasso | 1.846 | 1.260 |
| Estacional (y_{t-11}, ingenuo) | 2.107 | 1.244 |

*(ALEE original de 5 expertos, iteración 3: RMSE 1.054 — significativamente peor que el estacional aislado, p=0.0044. Conservado en `oof_ALEE_K5_original_iter3.csv`.)*

## Pruebas estadísticas (con ALEE de 3 expertos)

- **Model Confidence Set (α=0.10):** los 13 candidatos quedan dentro del conjunto de confianza — no se puede descartar estadísticamente a ninguno, ni siquiera a los peores, con 240 meses.
- **Diebold-Mariano (HLN), ALEE vs. Estacional(ALEE):** p=0.140 (antes 0.0044) — **ya no se rechaza la igualdad**. Frente a SARIMA: p=0.302, tampoco se rechaza, y el signo favorece nominalmente a ALEE.
- **Giacomini-White (Holm):** ALEE gana con significancia a 9 de los 11 modelos restantes (todos menos Estacional(ALEE) y SARIMA; con XGBoost queda al borde, p_holm=0.059).
- **SPA de Hansen (referencia Estacional-ALEE):** p=0.674; el candidato con mejor pérdida media es el propio ALEE, sin robustez estadística frente a *data snooping*.
- **Tamaño del efecto ALEE vs. Estacional(ALEE):** d de Cohen 0.094 (muy pequeño, antes 0.19), δ de Cliff 0.018 (insignificante, antes 0.03).

## Criterio de superioridad del protocolo (a ∧ b ∧ c ∧ d)

**Sigue sin cumplirse, incluso con la mejora de la iteración 4.** Falla en la condición (a): en el punto estimado, ALEE (0.869 agrupado / 0.817 medio de pliegues) sigue siendo numéricamente peor que el estacional aislado (0.836 / 0.780), aunque la diferencia ya no es estadísticamente significativa. Ver detalle completo en `experiments/protocolo.md` secciones 10 (resultado original) y 11 (iteración 4).

## Lo que SÍ se cumplió del protocolo

- Las 5 decisiones `[DECIDIR]` se resolvieron y están registradas en `experiments/protocolo.md` sección 9.
- Se corrió el problema principal completo bajo el protocolo congelado: 7 combinaciones del entregable anterior + ALEE + LightGBM, con optimización bayesiana de 40 evaluaciones, `TimeSeriesSplit(5,gap=1)`/`TimeSeriesSplit(3,gap=1)`, 10 semillas en los modelos estocásticos.
- Se corrió la batería completa de pruebas sobre el problema principal: MCS, Giacomini-White, Diebold-Mariano HLN, SPA de Hansen, tamaño del efecto.
- Presupuesto consumido real: `presupuesto_consumido.csv` (ningún modelo se acercó al tope de 15 min/pliegue; el más lento, Random Forest, promedió ~45 s).

## Lo que NO se pudo completar en esta corrida (declarado como desviación en el protocolo, sección 8)

1. **TabPFN no se ejecutó.** El paquete está instalado y funciona, pero sus pesos están en un repositorio de Hugging Face con licencia *gated*: alguien del equipo tiene que aceptarla manualmente en <https://huggingface.co/Prior-Labs/tabpfn_3_5> con su propia cuenta y generar un token de lectura. Con eso, `experiments/run_confirmatorio.py` ya está listo para correrlo (falta solo agregar el `ModelSpec` de TabPFN en `src/models.py`, siguiendo el mismo patrón que `LIGHTGBM_MODEL`).
2. **Validación externa (enero-septiembre 2026) no se ejecutó.** Sí se verificaron y documentaron los 9 valores reales de la tasa de desempleo nacional (`colombia_2026_tasa_nacional_verificada.csv`, cruzados contra boletines oficiales del DANE y múltiples medios), pero faltan los tres indicadores de las 13 áreas metropolitanas para esos mismos meses — no se publican en boletines de prensa, hay que descargarlos de las tablas del DANE/Banco de la República, y el entorno de ejecución de esta corrida no tuvo acceso a herramientas de descarga de páginas web.
3. **Los tres conjuntos adicionales (EE.UU., España, México) no se ensamblaron.** Se confirmó que las tres fuentes públicas (FRED, Eurostat, INEGI) tienen series mensuales vigentes y con suficiente historia, pero construir el dataset completo requiere descargar las series (API o CSV masivo), que tampoco estuvo disponible en este entorno de ejecución. Por esto, la prueba de Friedman/Wilcoxon-Holm entre conjuntos y la condición (d) del criterio de superioridad quedan sin evaluar.

Estas tres cosas requieren o bien acceso de red más amplio (descarga de páginas/APIs) en la próxima corrida, o que alguien del equipo las resuelva manualmente (la licencia de TabPFN, especialmente, solo la puede aceptar una persona).

## Archivos generados

- `oof_<modelo>.csv`, `oof_<modelo>_seed<N>.csv`: predicciones fuera de muestra por modelo y semilla.
- `baselines_series_tiempo.csv`: Persistencia, Estacional, SARIMA.
- `oof_EstacionalALEE.csv`: experto estacional de ALEE aislado.
- `tabla_rmse_confirmatorio.csv`, `mcs_confirmatorio.csv`, `giacomini_white_confirmatorio.csv`, `criterio_superioridad.csv`: resultados de la sección de análisis.
- `presupuesto_consumido.csv`, `variabilidad_semillas_*.csv`: control de presupuesto y estabilidad entre semillas.
- `colombia_2026_tasa_nacional_verificada.csv`: los 9 valores reales de 2026 ya verificados, listos para cuando se complete la validación externa.
- `log_ejecucion.txt`, `stdout.log`: bitácora completa de la corrida.
