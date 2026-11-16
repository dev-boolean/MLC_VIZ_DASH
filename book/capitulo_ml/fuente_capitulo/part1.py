# -*- coding: utf-8 -*-
"""Capítulo: planteamiento, datos, validación, modelos y métodos de optimización."""

CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


md(r"""
# Modelado Predictivo: Machine Learning
""")

md(r"""
## Justificación del enfoque

En la sección anterior el modelo SARIMA se apoyó únicamente en la historia de la propia tasa de desempleo nacional. Para este segundo entregable se quiere ir un paso más allá y comparar qué tan bien lo hacen distintos modelos de machine learning sobre el mismo problema, tanto para predecir el valor de la tasa (regresión) como la categoría en la que va a caer (clasificación). Para cada tarea se entrenan siete modelos, y cada uno se combina con las técnicas de balanceo de clases (solo en clasificación) y con los cuatro métodos de optimización de hiperparámetros que pide la actividad: Grid Search, Random Search, optimización bayesiana y algoritmo genético.

Como las alternativas se multiplican y no se suman, el número de modelos crece rápido. En clasificación son 7 modelos por 4 técnicas de balanceo por 4 optimizadores, es decir 112 combinaciones; en regresión son 7 modelos por 4 optimizadores, es decir 28, porque el balanceo de clases no aplica. En total se entrenaron, optimizaron y evaluaron 140 modelos de forma independiente, y cada corrida quedó registrada en una tabla maestra (`experiments/results_master.csv`) para poder filtrarla y compararla después.

Antes de entrenar hubo que tomar algunas decisiones sobre los datos que condicionan todo lo que sigue, así que se explican primero.
""")

md(r"""
## Una identidad contable que hay que evitar

Las tres tasas del mercado laboral no son independientes entre sí. La tasa de desempleo es la proporción de la fuerza de trabajo que no tiene empleo, de modo que se puede escribir como

$$TD = \frac{PEA - OCU}{PEA} = 1 - \frac{OCU}{PEA} = 1 - \frac{TO}{TGP}$$

donde TO es la tasa de ocupación y TGP la tasa global de participación. Si esto se cumple en los datos, usar la ocupación y la participación nacionales del mismo mes para explicar el desempleo nacional no sería predecir nada: el modelo solo tendría que despejar una ecuación. Lo comprobamos con los datos:
""")

code(r"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from src import config, targets, plots
from src import analysis as A
from src import optim_analysis as OA
from src import stats_tests as ST
from src import interpretability as I
from src.data import assemble_dataset, load_raw
from src.models import CLASSIFIERS, REGRESSORS, CLASSIFIER_BY_NAME, REGRESSOR_BY_NAME
from src.nested_cv import run_nested_cv

# Se calcula 1 - TO/TGP con las variables nacionales y se compara con la tasa de desempleo nacional
datos = load_raw()
implicita = 1 - datos['tasa_ocupacion_nacional'] / datos['tasa_global_participacion_nacional']
correlacion = np.corrcoef(implicita, datos['tasa_desempleo_nacional'] / 100)[0, 1]
print(f'Correlación entre 1 - TO/TGP y la tasa de desempleo nacional: {correlacion:.7f}')
""")

md(r"""
La correlación es de 0.9999993, es decir, la identidad se cumple casi a la perfección. Por esa razón la ocupación y la participación nacionales no se usan como variables explicativas. En cambio, los indicadores de las 13 áreas metropolitanas sí se conservan: corresponden a una población distinta (las ciudades principales) y no quedan amarrados a esa identidad. De hecho, en el análisis bivariado ya se había visto que el desempleo de área y el nacional se mueven casi juntos (correlación de 0.96), lo que los hace buenos candidatos como predictores y permite poner a prueba la primera hipótesis secundaria.
""")

md(r"""
## Un pronóstico a un mes

Para no depender de información del mismo mes, el problema se plantea igual que en el SARIMA: con lo que se sabe hasta el mes *t* se pronostica la tasa de desempleo nacional del mes *t+1*. Como variables explicativas se construyeron los rezagos de la propia tasa (1, 2, 3, 6 y 12 meses), los promedios y las desviaciones móviles a 3, 6 y 12 meses, el mes del año codificado con seno y coseno para capturar la estacionalidad, un índice de tendencia y los tres indicadores de las áreas metropolitanas del mes *t*. Al construir los rezagos se pierden los primeros meses de la serie, así que quedan 288 observaciones utilizables.
""")

code(r"""
ds = assemble_dataset()
fechas_objetivo = ds.fechas + pd.offsets.MonthEnd(1)  # mes que se pronostica

print('Observaciones utilizables:', ds.X.shape[0], '| variables explicativas:', ds.X.shape[1])
print('Primer mes pronosticado:', fechas_objetivo.iloc[0].date(), '| último:', fechas_objetivo.iloc[-1].date())
ds.X.head()
""")

md(r"""
Cada fila de la tabla resume lo que se sabe en un mes y sirve para pronosticar el siguiente: el primer mes que se pronostica es enero de 2002 y el último diciembre de 2025.
""")

md(r"""
## Las categorías para clasificar

El dataset original solo tiene variables continuas, así que la variable de clasificación se construyó a partir de la tasa de desempleo del mes siguiente. Se definieron tres categorías: **Bajo** (menos de 10 %), **Medio** (entre 10 % y 14 %) y **Alto** (14 % o más). Los cortes no son terciles; se eligieron mirando la distribución histórica que se vio en el EDA (media de 11.6 %, mediana de 11.2 %, primer cuartil de 9.7 % y tercer cuartil de 12.9 %), de forma que la categoría Alto recogiera los periodos de crisis, es decir, la recesión de comienzos de los 2000 y el choque de la pandemia en 2020.
""")

code(r"""
informe = targets.class_balance_report(ds.y_clf_labels)
informe.columns = ['Meses', '% del total']
informe
""")

code(r"""
fig = plots.plot_target_classes(fechas_objetivo, ds.y_reg.values, config.CLASS_THRESHOLDS)
plt.show()
""")

md(r"""
De los 288 meses, 91 son Bajo (31.6 %), 156 son Medio (54.2 %) y 41 son Alto (14.2 %). El desbalance no se forzó: aparece porque los meses de crisis son pocos, y es justamente lo que le da sentido a comparar más adelante SMOTE, ADASYN y los pesos de clase. En la figura se ve que los meses Alto se concentran en 2002-2004, con algún caso suelto hasta 2010, y en 2020-2021, mientras que desde 2011 hasta la pandemia casi todos los meses son Medio o Bajo.
""")

md(r"""
## Una validación que respeta el orden del tiempo

La validación cruzada tampoco puede hacerse mezclando los meses al azar. Como las variables incluyen rezagos y promedios móviles, el mes *t* y el mes *t+1* comparten casi toda su información (el segundo rezago de uno es el primer rezago del otro), y un K-fold aleatorio dejaría meses casi idénticos repartidos entre entrenamiento y prueba (Bergmeir & Benítez, 2012). El resultado sería una métrica demasiado optimista, que es justo el sesgo de selección que se busca evitar con la validación anidada (Cawley & Talbot, 2010).

Por eso se usó una validación anidada con ventana expansiva. El bucle externo, de 5 pliegues, mide qué tan bien generaliza el modelo ya optimizado; dentro de cada tramo de entrenamiento, un bucle interno de 3 pliegues es el que usa el optimizador para elegir los hiperparámetros. En los dos bucles la prueba queda siempre después del entrenamiento, con un mes de separación entre ambos para que los rezagos no se crucen en la frontera.
""")

code(r"""
particiones = list(TimeSeriesSplit(n_splits=config.N_SPLITS_OUTER, gap=config.CV_GAP).split(ds.X))
fig = plots.plot_walk_forward(fechas_objetivo, particiones)
plt.show()

# Tamaño y composición de cada pliegue externo
etiquetas = ds.y_clf_labels
filas = []
for i, (entrenamiento, prueba) in enumerate(particiones):
    filas.append({
        'Pliegue': i + 1,
        'Meses de entrenamiento': len(entrenamiento),
        'Prueba desde': fechas_objetivo.iloc[prueba[0]].strftime('%Y-%m'),
        'Prueba hasta': fechas_objetivo.iloc[prueba[-1]].strftime('%Y-%m'),
        'Bajo (entrenamiento)': int((etiquetas.iloc[entrenamiento] == 'Bajo').sum()),
        'Alto (entrenamiento)': int((etiquetas.iloc[entrenamiento] == 'Alto').sum()),
        'Alto (prueba)': int((etiquetas.iloc[prueba] == 'Alto').sum()),
    })
pd.DataFrame(filas).set_index('Pliegue')
""")

md(r"""
Con 288 observaciones, 5 pliegues externos dejan pruebas de 48 meses (cuatro años) y un primer entrenamiento de 47 meses; más pliegues habrían dado pruebas demasiado cortas y menos habrían entregado muy pocas estimaciones del desempeño. En el bucle interno se usaron 3 pliegues para que los tramos de validación siguieran siendo razonables incluso en el primer pliegue externo. La métrica que se reporta siempre es la del bucle externo, nunca el mejor valor que encontró la búsqueda en el bucle interno.

La tabla muestra algo que condiciona los resultados: las clases no están repartidas parejo en el tiempo. El primer pliegue entrena con apenas 1 mes Bajo, y los 13 meses Alto de la pandemia caen en la prueba del pliegue 4, de modo que el modelo los enfrenta después de haber visto solo los 27 meses Alto de 2002 a 2010. En el total de las predicciones fuera de muestra (2006-2025) solo hay 17 meses Alto, y eso limita lo que se puede concluir sobre esa clase.
""")

md(r"""
## Modelos y espacios de búsqueda

Los siete modelos de cada tarea son los pedidos en la actividad. En clasificación: KNN, Naive Bayes (gaussiano), regresión logística (con regularización L1 y L2), árbol de decisión, Random Forest, XGBoost y SVM. En regresión: KNN, Ridge, Lasso, árbol de decisión, Random Forest, XGBoost y SVR. Se usaron las implementaciones de scikit-learn (Pedregosa et al., 2011) y de XGBoost (Chen & Guestrin, 2016). A todos se les define el mismo espacio de búsqueda para los cuatro optimizadores, de manera que ninguno tenga ventaja por buscar en una región distinta.

Los rangos no son arbitrarios. Los parámetros de regularización (`C`, `alpha`, `gamma`, `var_smoothing` y la tasa de aprendizaje) se exploran en escala logarítmica, porque sus efectos cambian por órdenes de magnitud. Los parámetros de complejidad se acotan según el tamaño de los datos: con 288 observaciones no tiene sentido un árbol de más de 20 niveles ni un KNN con más de 30 vecinos. La siguiente tabla resume el espacio de cada modelo. Una limitación del diseño es que, con 17 variables, `sqrt` y `log2` dan ambas 4 variables por división, así que en los bosques `max_features` en la práctica no cambia el modelo; se dejó igual para no alterar el espacio definido desde el inicio.
""")

code(r"""
filas = []
for tarea, modelos in [('Clasificación', CLASSIFIERS), ('Regresión', REGRESSORS)]:
    for modelo in modelos:
        for h in modelo.space:
            if h.kind == 'cat':
                rango, escala = ', '.join(str(c) for c in h.choices), 'categórica'
            else:
                rango, escala = f'{h.low:g} a {h.high:g}', 'logarítmica' if h.log else 'lineal'
            filas.append({'Tarea': tarea, 'Modelo': modelo.name, 'Hiperparámetro': h.name, 'Rango': rango, 'Escala': escala})
pd.DataFrame(filas).set_index(['Tarea', 'Modelo'])
""")

md(r"""
Para el desbalance de clases se compararon tres técnicas, siempre aplicadas solo sobre el tramo de entrenamiento de cada pliegue para no filtrar información. SMOTE genera ejemplos sintéticos de la clase minoritaria interpolando entre un caso y sus vecinos más cercanos (Chawla et al., 2002). ADASYN hace algo parecido pero crea más ejemplos donde la clase es más difícil de aprender (He et al., 2008). Los pesos de clase no crean datos: simplemente hacen que equivocarse en la clase minoritaria cueste más. Se aplicaron como pesos por observación, igual para todos los modelos; KNN no admite pesos, así que para ese modelo los pesos de clase equivalen a no balancear. Como en los primeros pliegues internos puede faltar una clase por completo (los primeros 13 meses de la serie son todos Alto), en esos casos no se remuestrea.
""")

md(r"""
## Métodos de optimización de hiperparámetros

Los cuatro métodos evalúan cada combinación de hiperparámetros de la misma manera: reconstruyen a mano el pipeline completo (balanceo, escalado y modelo) dentro de cada pliegue interno y lo ajustan solo con el tramo de entrenamiento, de modo que el preprocesamiento nunca ve los datos de validación. El puntaje que se maximiza es el F1-macro en clasificación y el RMSE (con signo negativo) en regresión.

- **Grid Search** recorre una grilla de 4 valores por hiperparámetro numérico. Cuando la grilla completa tiene más de 40 combinaciones se elige al azar una submuestra de 40, que se recorre en el orden natural de la grilla (de los valores bajos a los altos de cada hiperparámetro); cuando tiene menos (Naive Bayes, Ridge y Lasso tienen 4, la regresión logística 8, KNN 16 y SVM 32) se recorre completa.
- **Random Search** evalúa 40 combinaciones muestreadas al azar del espacio de búsqueda, una alternativa que suele encontrar buenas configuraciones con menos evaluaciones que una grilla cuando pocos hiperparámetros importan de verdad (Bergstra & Bengio, 2012).
- **Optimización bayesiana**, con Optuna (Akiba et al., 2019), usa como modelo sustituto el estimador de Parzen de estructura de árbol (TPE; Bergstra et al., 2011). Después de 10 evaluaciones iniciales al azar, separa lo evaluado en un grupo bueno (el 10 % mejor) y otro malo, ajusta una densidad a cada uno, *l(x)* y *g(x)*, y propone el punto que maximiza *l(x)/g(x)*, que equivale a maximizar la mejora esperada. Esa función de adquisición favorece explorar cerca de donde ya se obtuvieron buenos resultados, sin abandonar del todo las zonas inciertas. Se escogió TPE porque maneja de forma natural un espacio mixto (enteros, continuos en escala logarítmica y categóricos) y no necesita invertir una matriz de covarianza, como haría un proceso gaussiano.
- **Algoritmo genético**, con DEAP (Fortin et al., 2012), representa cada hiperparámetro como un gen entre 0 y 1 que se traduce al rango real. La población es de 8 individuos y evoluciona con selección por torneo de tamaño 3, cruce por mezcla (`cxBlend`, con probabilidad 0.6), mutación gaussiana (probabilidad 0.3 por individuo y 0.3 por gen, desviación 0.15) y elitismo de un individuo, de modo que el mejor siempre pasa intacto a la siguiente generación. Solo se vuelven a evaluar los individuos que cambiaron, y la búsqueda termina al completar 40 evaluaciones. Las elecciones se justifican así: el torneo de tamaño 3 da una presión de selección moderada y no obliga a reescalar las aptitudes (que son F1 en clasificación y RMSE con signo negativo en regresión); el cruce por mezcla es el adecuado para genes continuos; la mutación gaussiana de desviación 0.15 (el 15 % del rango de cada gen) explora vecindades sin saltos bruscos; el elitismo evita perder la mejor solución, y una población de 8 permite repartir las 40 evaluaciones en unas 8 generaciones.

Para que la comparación de eficiencia sea justa, los cuatro disponen de unas 40 evaluaciones por búsqueda (menos en Grid cuando la grilla es más pequeña). Toda la corrida usa una única semilla global (42), que se propaga a NumPy, a scikit-learn, a XGBoost, a SMOTE/ADASYN, al muestreador de Optuna y al generador que usa DEAP; por eso repetir la corrida da exactamente los mismos resultados. Además de la tabla maestra, cada búsqueda guardó el puntaje y el tiempo de cada evaluación, la diversidad de la población en cada generación del algoritmo genético y las predicciones de cada pliegue externo.
""")
