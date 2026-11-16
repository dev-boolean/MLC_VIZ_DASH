# -*- coding: utf-8 -*-
"""Capítulo: reproducibilidad y conclusiones."""

CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


md(r"""
## Reproducibilidad del análisis

Todo lo que se muestra en este capítulo se puede repetir desde cero. El código no está en el cuaderno sino en la carpeta `src`, dividido por tareas para evitar bloques repetidos entre modelos: `data.py` carga los datos y construye las variables, `targets.py` define las categorías, `models.py` registra los 14 modelos y sus espacios de búsqueda, `balancing.py` aplica SMOTE, ADASYN o los pesos de clase, `search.py` contiene los cuatro optimizadores, `nested_cv.py` hace la validación anidada, `metrics.py` y `stats_tests.py` calculan las métricas y las pruebas estadísticas, `interpretability.py` reúne SHAP y LIME, y `analysis.py` y `plots.py` leen los registros y arman las tablas y gráficos de este capítulo. Cada función que usan varios modelos tiene su docstring con los parámetros, lo que devuelve y los supuestos.

La versión exacta de cada librería está en `requirements.txt`, la semilla global (42) está en `src/config.py`, y cada una de las 140 combinaciones quedó registrada en `experiments/results_master.csv` con sus métricas, tiempos, número de evaluaciones e hiperparámetros. Los comandos para repetir todo son:

```bash
pip install -r requirements.txt
python -m src.runner full                       # las 140 combinaciones y sus registros
python scripts/seed_sensitivity.py              # sensibilidad a la semilla
python scripts/multifidelity_experiment.py      # Successive Halving
python scripts/benchmark_complexity.py          # escalamiento y estándar frente a optimizado
python scripts/benchmark_parallel.py            # paralelización con joblib
python scripts/profile_pipeline.py              # perfilamiento de CPU y memoria
python scripts/diagnostico_hilos.py             # hilos frente a procesos en joblib
```

Los experimentos pesados se corrieron aparte y este cuaderno solo lee sus resultados desde la carpeta `experiments`, de modo que ejecutarlo no vuelve a lanzar la búsqueda de los 140 modelos.
""")

md(r"""
La tabla maestra guarda, para cada combinación, la semilla, las métricas del bucle externo (media y desviación entre los cinco pliegues), los tiempos, el número de evaluaciones y los hiperparámetros que eligió el optimizador en cada pliegue. Así se ven seis de sus 140 filas:
""")

code(r"""
import json

muestra = master.sample(6, random_state=config.SEED).sort_values('run_id')
metrica = [f"F1-macro {f['f1_macro_mean']:.3f} ± {f['f1_macro_std']:.3f}" if f['task'] == 'clf' else f"RMSE {f['rmse_mean']:.3f} ± {f['rmse_std']:.3f}"
           for _, f in muestra.iterrows()]
cuadro_maestro = pd.DataFrame({
    'Semilla': muestra['seed'].values, 'Métrica del bucle externo': metrica,
    'Tiempo de búsqueda (s)': muestra['search_time_s'].round(1).values, 'Evaluaciones': muestra['total_evaluations'].values,
    'Hiperparámetros (último pliegue)': muestra['hiperparametros_por_pliegue'].map(lambda texto: json.loads(texto)[-1]).values,
}, index=muestra['run_id'])
cuadro_maestro
""")

code(r"""
# Resumen: el mejor modelo de machine learning de cada tarea frente a la mejor referencia, sobre las 240 predicciones fuera de muestra
mejor_clf_nombre, mejor_ref_clf = agrupadas['F1-score'].idxmax(), referencias_clf['F1-score'].idxmax()
solo_ml = comparacion_reg[comparacion_reg['Tipo'] == 'Machine learning']
solo_ref = comparacion_reg[comparacion_reg['Tipo'] == 'Referencia']
mejor_reg_nombre, mejor_ref_reg = solo_ml['RMSE agrupado'].idxmin(), solo_ref['RMSE agrupado'].idxmin()
pd.DataFrame([
    {'Tarea': 'Clasificación', 'Métrica': 'F1-macro (mayor es mejor)',
     'Mejor modelo de machine learning': f"{mejor_clf_nombre}: {agrupadas.loc[mejor_clf_nombre, 'F1-score']:.3f}",
     'Mejor referencia': f"{mejor_ref_clf}: {referencias_clf.loc[mejor_ref_clf, 'F1-score']:.3f}"},
    {'Tarea': 'Regresión', 'Métrica': 'RMSE (menor es mejor)',
     'Mejor modelo de machine learning': f"{mejor_reg_nombre}: {solo_ml.loc[mejor_reg_nombre, 'RMSE agrupado']:.3f}",
     'Mejor referencia': f"{mejor_ref_reg}: {solo_ref.loc[mejor_ref_reg, 'RMSE agrupado']:.3f}"},
]).set_index('Tarea')
""")

md(r"""
## Conclusiones del modelado con machine learning

Este capítulo comparó siete modelos de clasificación y siete de regresión con cuatro técnicas de balanceo (solo en clasificación) y cuatro métodos de optimización de hiperparámetros, es decir, 140 modelos evaluados con una validación anidada que respeta el orden del tiempo. El resultado principal es poco halagador para el machine learning y por eso mismo es el más útil: **ningún modelo de regresión superó al SARIMA(0,1,1)(0,1,1)[12] del capítulo anterior** (RMSE de 0.995 frente a 1.314 del mejor, SVR) ni, con significancia, a la simple persistencia (1.293), y en clasificación **solo KNN superó a la regla de repetir la categoría del mes actual** (F1-macro de 0.815 frente a 0.752). La razón se ve en los residuos y en los pronósticos: la serie es en buena parte estacionalidad y persistencia, que el SARIMA modela de forma explícita, mientras que los modelos de machine learning, con 288 observaciones y 17 variables muy correlacionadas, aprenden esa estructura solo a medias (los residuos conservan autocorrelación y un pico en el rezago 12) y fallan donde la serie se rompe, como en abril de 2020.

Dentro de lo que sí se pudo comparar, en clasificación KNN, Random Forest, XGBoost y Naive Bayes son equivalentes en rango según Nemenyi y SVM es el peor; ADASYN es la técnica de balanceo más segura (mejora media de 0.049 de F1-macro, delta de Cliff de 0.40), aunque no ayuda a todos los modelos y a algunos, como el árbol de decisión y XGBoost, no les sirve ninguna; y las probabilidades de XGBoost, SVM, Random Forest y regresión logística salen mal calibradas y mejoran con el escalado de Platt (el ECE de XGBoost baja de 0.126 a 0.057), mientras que recalibrar KNN o el árbol las empeora. En regresión, entre los modelos de machine learning no hay diferencias significativas; los árboles son los mejores antes de la pandemia y los peores durante ella, y los regularizados (Ridge y SVR) son los más robustos al choque.

Entre los métodos de optimización, el bayesiano fue el más eficiente: alcanza el 10 % del mejor puntaje con una mediana de 15 evaluaciones (Random necesita 35.5, el genético 39 y Grid casi nunca llega), tiene el área bajo la curva de arrepentimiento más baja (0.28 contra 0.36, 0.38 y 0.49) y es significativamente mejor que Grid, aunque no que Random Search ni que el genético. El genético sufre convergencia prematura, porque su población pierde casi 90 % de su diversidad en siete generaciones. La calidad final del modelo depende poco del optimizador, y Successive Halving consigue una calidad parecida a la de Random Search con la sexta parte del costo en árboles. Los resultados son reproducibles con la semilla global, y la aleatoriedad de los modelos mueve mucho menos el resultado (desviaciones de hasta 0.06 de F1-macro y 0.04 de RMSE) que la que hay entre pliegues.

En el costo computacional, Random Forest y XGBoost consumen el 82 % de los 57 minutos de la corrida. Las mejoras que no cambian el modelo son casi gratis (los histogramas de XGBoost lo aceleran 15.6 veces, la parada temprana 2.4 y repartir pliegues entre procesos entre 3.6 y 4.3), mientras que las que lo aproximan cambian exactitud por velocidad (LinearSVC es 29 veces más rápido pero pierde 8.6 puntos porcentuales de accuracy, y FAISS aproximado es 6.1 veces más rápido y pierde una milésima). Los hilos de Python no sirven para este trabajo, porque el GIL los frena y, además, cambian el resultado del algoritmo genético, y no fue posible medir la aceleración con GPU porque el equipo no tiene tarjeta con CUDA.

Respecto a las hipótesis del proyecto, la primera hipótesis secundaria se sostiene: la tasa de desempleo de las 13 áreas metropolitanas es la variable más importante en Ridge (13.7 % de la importancia) y en XGBoost para la categoría Alto, y quitar los tres indicadores de área empeora el RMSE en cinco de los siete modelos (hasta 9 % en XGBoost), aunque la evidencia es moderada. La ruptura de la pandemia (segunda hipótesis) se ve en el desempeño: el error se multiplica entre 2 y 6 veces, según el modelo, entre los periodos tranquilos y 2020-2021 (de 0.51 a 3.03 en el SARIMA, de 0.83 a 4.93 en XGBoost), y ningún modelo anticipa el salto de abril de 2020. La estacionalidad (tercera hipótesis) es modelable, pero el SARIMA la modela mucho mejor que los modelos de machine learning con la codificación de seno y coseno que se usó.

Hay límites que condicionan todo lo anterior. Con 288 meses y solo 17 meses Alto en las predicciones fuera de muestra, cualquier conclusión sobre esa categoría tiene mucha incertidumbre; los cortes de las categorías (10 % y 14 %) son una decisión propia, y las cifras de la mejor variante de cada modelo son algo optimistas porque se escogió la mejor de 16. Los pasos siguientes que más prometen son enriquecer las variables de estacionalidad (más armónicos o variables por mes) y trabajar con diferencias en lugar de niveles, combinar el SARIMA con un modelo de machine learning sobre sus residuos, incorporar variables externas (por ejemplo, de actividad económica o de precios), y probar poblaciones más grandes en el algoritmo genético.
""")
