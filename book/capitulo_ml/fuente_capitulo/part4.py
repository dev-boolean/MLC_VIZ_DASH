# -*- coding: utf-8 -*-
"""Capítulo: comparación de los métodos de optimización."""

CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


md(r"""
## Comparación de los métodos de optimización

Comparar los cuatro métodos no es solo ver cuál terminó con la mejor métrica. Se miden tres cosas: cuánto cuesta cada método (evaluaciones, tiempo total y tiempo por evaluación), qué tan rápido se acerca cada uno a su mejor resultado (el desempeño en cualquier momento, es decir, el mejor puntaje encontrado hasta la evaluación *t* en función de *t*) y si las diferencias entre métodos son estadísticamente significativas. Para esto se guardó, en cada una de las 700 búsquedas (140 combinaciones por 5 pliegues), el puntaje y el tiempo de cada evaluación.

Como los problemas tienen escalas distintas (F1-macro en clasificación y RMSE en regresión), el desempeño en cualquier momento se mide con un arrepentimiento normalizado. En cada problema (un modelo, una técnica de balanceo y un pliegue externo) se toma como cero el mejor puntaje encontrado por cualquiera de los cuatro métodos y como uno el puntaje mediano de todas las evaluaciones; el arrepentimiento de un método tras *t* evaluaciones mide qué tan lejos queda del mejor puntaje conocido. Un método que llega pronto a cero es un método eficiente.
""")

code(r"""
# Historial de cada búsqueda: puntaje y tiempo de cada evaluación, y registro por generación del algoritmo genético
evaluaciones, generaciones, busquedas = OA.load_search_records(master)

costo = OA.cost_table(busquedas)
costo.index = [A.OPTIMIZER_LABELS[o] for o in costo.index]
costo.columns = ['Evaluaciones por búsqueda', 'Tiempo por búsqueda (s)', 'Tiempo total (min)', 'Costo del optimizador (%)', 'Tiempo por evaluación (s)']
costo.round(2)
""")

md(r"""
Los cuatro métodos cuestan casi lo mismo por evaluación, así que comparar su calidad por presupuesto es justo. Random, el bayesiano y el genético hicieron 40 evaluaciones por búsqueda; Grid hizo en promedio 25.8, porque cuando la grilla completa es pequeña (Naive Bayes, Ridge y Lasso tienen 4 combinaciones y la regresión logística 8) se recorre entera y termina antes. El tiempo por evaluación va de 0.12 s (bayesiano) a 0.17 s (Grid), y el tiempo total de las búsquedas fue de 12.5 minutos para Grid, 14.6 para Random, 13.6 para el bayesiano y 15.9 para el genético. El costo del propio optimizador, es decir, lo que gasta en decidir qué evaluar sin contar el entrenamiento de los modelos, es despreciable en Random (0.02 %) y en el genético (0.07 %), y llega a 1.75 % en el bayesiano, que es el único que ajusta un modelo sustituto en cada iteración. Frente al costo de entrenar un Random Forest esto no cambia nada, pero sí importaría si cada evaluación tardara milisegundos.
""")

code(r"""
# Desempeño en cualquier momento: arrepentimiento normalizado tras t evaluaciones
arrepentimiento = OA.normalized_regret(evaluaciones)
fig = plots.plot_regret(arrepentimiento)
plt.show()

puntos = OA.regret_checkpoints(arrepentimiento)
puntos.index = [A.OPTIMIZER_LABELS[o] for o in puntos.index]
puntos.columns = [f'Tras {c} evaluaciones' if isinstance(c, (int, np.integer)) else c.capitalize() for c in puntos.columns]
puntos.round(3)
""")

code(r"""
umbral = OA.evaluations_to_threshold(arrepentimiento, threshold=0.10)
umbral.index = [A.OPTIMIZER_LABELS[o] for o in umbral.index]
umbral.columns = ['Evaluaciones necesarias (mediana)', 'Tiempo necesario (mediana, s)', 'Búsquedas que lo alcanzan (%)']
umbral.round(2)
""")

md(r"""
La guía define el desempeño en cualquier momento como el mejor valor de la métrica encontrado hasta la evaluación *t*. El arrepentimiento normalizado es esa misma curva reescalada para poder promediar problemas con unidades distintas; para ver los valores sin transformar, la siguiente figura muestra dos casos, uno de clasificación y uno de regresión. El puntaje es el de la validación interna (F1-macro o RMSE) y se promedia entre los cinco pliegues externos.
""")

code(r"""
fig = plots.plot_best_so_far(evaluaciones, [
    ('XGBoost, clasificación sin balanceo', 'clf', 'XGBoost', 'none'),
    ('Random Forest, regresión', 'reg', 'RandomForest', 'none')])
plt.show()
""")

md(r"""
Con el arrepentimiento normalizado (0 es el mejor puntaje conocido y 1 el puntaje mediano de todas las evaluaciones), el bayesiano es el que más rápido baja y el único que sigue mejorando hasta el final. Tras 5 evaluaciones su arrepentimiento es de 0.50, frente a 0.65 del genético, 0.69 de Random y 0.76 de Grid; tras 20 es de 0.16 (contra 0.27, 0.24 y 0.30), y tras 40 de 0.06 (contra 0.21, 0.18 y 0.24). El área bajo la curva, que resume todo el recorrido, es de 0.28 para el bayesiano, 0.36 para Random, 0.38 para el genético y 0.49 para Grid. Para llegar a un arrepentimiento de 0.10, es decir, a quedar a una décima de la distancia entre el puntaje típico y el mejor, el bayesiano necesita una mediana de 15 evaluaciones y lo logra en el 86 % de las búsquedas; Random necesita 35.5 y lo logra en el 53 %, el genético 39 y el 52 %, y Grid no llega en la mayoría (solo lo alcanza el 44 %). En otras palabras, para una misma calidad de solución el bayesiano usa menos de la mitad de las evaluaciones, que es lo que la actividad pide demostrar con números. El panel derecho muestra lo mismo contra el tiempo: la curva del bayesiano queda por debajo de las otras casi todo el recorrido, así que su ventaja no se pierde por el costo extra de ajustar el modelo sustituto. Grid es el último en parte por cómo se diseñó: recorre su submuestra de la grilla en el orden natural de la grilla, de los valores bajos a los altos de cada hiperparámetro, y no tiene ningún criterio para empezar por las zonas buenas.

Los dos ejemplos en unidades originales cuentan lo mismo con más detalle. En XGBoost de clasificación el bayesiano termina con el mejor F1-macro interno (0.55) mientras que los otros tres se quedan cerca de 0.535; el genético sube más rápido al principio (llega a 0.53 con 7 evaluaciones) pero se estanca enseguida. En Random Forest de regresión, Random Search encuentra una zona buena de golpe en la sexta evaluación, el bayesiano llega a una zona parecida hacia la quinta y sigue refinando hasta el RMSE más bajo (1.26), y Grid se queda en 1.29.
""")

code(r"""
# ¿Hay diferencias significativas entre los optimizadores? Cada bloque es un modelo con una técnica de balanceo
# y se compara el área bajo la curva de arrepentimiento (menor es mejor) promediada entre pliegues.
area = arrepentimiento.groupby(['task', 'model', 'balancing', 'optimizer'])['regret'].mean().unstack('optimizer')[OA.OPTIMIZER_ORDER]
area.columns = [A.OPTIMIZER_LABELS[o] for o in area.columns]
f_area = ST.friedman_test(area, higher_is_better=False)
print(f'Friedman sobre el área de arrepentimiento ({len(area)} bloques): chi2 = {f_area.statistic:.2f}, p = {f_area.p_value:.4g}')

cd_area = ST.critical_difference(4, len(area))
fig, ax = plt.subplots(figsize=(9, 2.6))
ST.plot_cd_diagram(f_area.avg_ranks, cd_area, ax=ax, title='Diagrama de diferencia crítica: calidad de la búsqueda')
plt.show()

nemenyi_area = ST.nemenyi_posthoc(area, higher_is_better=False)
nemenyi_area.round(4)
""")

code(r"""
# Calidad final de los modelos entrenados con cada optimizador (métrica del bucle externo)
clf_optimizador = clasificacion.pivot_table(index=['model', 'balancing'], columns='optimizer', values='f1_macro_mean')[OA.OPTIMIZER_ORDER]
reg_optimizador = regresion.pivot_table(index='model', columns='optimizer', values='rmse_mean')[OA.OPTIMIZER_ORDER]
for nombre, tabla, mejor_alto in [('clasificación (F1-macro)', clf_optimizador, True), ('regresión (RMSE)', reg_optimizador, False)]:
    tabla = tabla.copy()
    tabla.columns = [A.OPTIMIZER_LABELS[o] for o in tabla.columns]
    f = ST.friedman_test(tabla, higher_is_better=mejor_alto)
    print(f'{nombre}: Friedman con {len(tabla)} bloques, chi2 = {f.statistic:.2f}, p = {f.p_value:.4f}')
    print('   rango promedio:', ', '.join(f'{k} {v:.2f}' for k, v in f.avg_ranks.items()), '\n')
""")

md(r"""
La diferencia entre los métodos en desempeño en cualquier momento es estadísticamente significativa: la prueba de Friedman sobre el área bajo la curva de arrepentimiento, con 32 bloques (las 35 combinaciones de modelo y balanceo menos tres de Naive Bayes, en las que todas las evaluaciones dieron el mismo puntaje porque su único hiperparámetro, `var_smoothing`, no cambia las predicciones en ese rango), da χ² = 10.79 y p = 0.013. El diagrama de diferencia crítica ordena a los métodos por rango promedio (menor es mejor): bayesiano 1.95, Random 2.45, genético 2.59 y Grid 3.00, con una diferencia crítica de 0.83. Solo el bayesiano y Grid se separan de manera significativa (p de Nemenyi de 0.0065); entre los demás, incluidos el bayesiano frente a Random (p = 0.41) y frente al genético (p = 0.19), las diferencias no llegan a ser significativas. Con esta evidencia se puede afirmar que el bayesiano es mejor que Grid, pero no que supere de forma significativa a Random Search o al genético; lo que sí muestra es una tendencia clara (rango de 1.95 contra 2.45 y 2.59) y la diferencia en evaluaciones necesarias que se vio antes.

Sobre la calidad de los modelos finales (la métrica del bucle externo), el efecto del optimizador es más débil y no tiene un ganador consistente. En clasificación sí hay diferencias (Friedman con 28 bloques: χ² = 8.60, p = 0.035), pero a favor de Grid, el bayesiano y el genético (rangos de 2.29, 2.29 y 2.36) frente a Random (3.07); en regresión, con solo 7 bloques, no hay diferencias significativas (p = 0.086) y el orden es otro: Random queda primero (1.57), seguido del bayesiano (2.29), Grid (3.00) y el genético (3.14). Que el método que encuentra antes las buenas soluciones no siempre dé el mejor modelo en el bucle externo es esperable: con 40 evaluaciones los cuatro llegan a zonas de calidad parecida y, a partir de ahí, la diferencia entre ellos es menor que el ruido de pasar de la validación interna a la prueba externa.
""")

md(r"""
### Diversidad genética del algoritmo genético
""")

code(r"""
diversidad = OA.genetic_diversity_summary(generaciones, collapse_threshold=0.05)
fig = plots.plot_genetic_diversity(diversidad['by_generation'])
plt.show()

print(f"Búsquedas genéticas analizadas: {diversidad['n_searches']}")
print(f"Proporción cuya población colapsa (distancia media < 0.05) antes de agotar el presupuesto: {diversidad['share_collapsed']:.1%}")
print(f"Generación promedio en que colapsa: {diversidad['mean_collapse_generation']:.1f}")
print(f"Mejora promedio de la mejor aptitud después de colapsar: {diversidad['gain_after_collapse']:.4f}")
diversidad['by_generation'].rename(columns={
    'diversidad_pares': 'Distancia media entre pares', 'diversidad_por_gen': 'Desviación por gen',
    'configuraciones_distintas': 'Configuraciones distintas', 'busquedas': 'Búsquedas'}).round(3)
""")

md(r"""
La población del algoritmo genético pierde diversidad muy rápido. La distancia promedio entre individuos (en la escala de 0 a 1 de los genes) baja de 0.39 en la generación 0 (la población inicial) a 0.14 en la generación 3 y a 0.05 en la 7, una caída de casi 90 %, y la desviación por gen pasa de 0.27 a 0.03. En individuos distintos, de los 8 que tiene la población quedan en promedio unos 5 en la generación 1 y 2.5 en la 8. En el 60 % de las 175 búsquedas genéticas la población colapsa (distancia promedio menor que 0.05) antes de agotar el presupuesto de 40 evaluaciones, en promedio hacia la generación 4, y en esas búsquedas la mejor aptitud solo mejora 0.0044 después del colapso: cuando los individuos se parecen tanto, el cruce casi no produce nada nuevo y solo la mutación sigue explorando.

Esta es la señal de convergencia prematura que la guía pide vigilar, y es consistente con el desempeño del genético: arranca bien (es el método con menor arrepentimiento en la primera evaluación, 1.83 contra entre 2.0 y 2.7 de los demás) pero se estanca y termina tercero, con un arrepentimiento de 0.21 tras 40 evaluaciones. La causa más probable es la combinación de una población pequeña (8 individuos), un torneo de tamaño 3 con elitismo, que ejerce mucha presión de selección, y un presupuesto que solo alcanza para unas 8 generaciones. Una población mayor o una mutación más alta mantendrían la diversidad por más tiempo, pero reducirían las generaciones disponibles y no se probaron.
""")

md(r"""
### Estabilidad frente a la semilla
""")

code(r"""
# Parte A: aleatoriedad del modelo (10 semillas, mismos hiperparámetros elegidos en la corrida)
semillas_modelo = pd.read_csv(config.EXPERIMENTS_DIR / 'seed_sensitivity_models.csv')
semillas_modelo['modelo'] = semillas_modelo['modelo'].map(A.MODEL_LABELS)
semillas_modelo = semillas_modelo.rename(columns={'tarea': 'Tarea', 'modelo': 'Modelo', 'metrica': 'Métrica', 'media': 'Media',
                                                  'desviacion': 'Desviación', 'minimo': 'Mínimo', 'maximo': 'Máximo'})
semillas_modelo['Tarea'] = semillas_modelo['Tarea'].map({'clf': 'Clasificación', 'reg': 'Regresión'})
semillas_modelo.set_index(['Tarea', 'Modelo'])[['Métrica', 'Media', 'Desviación', 'Mínimo', 'Máximo']].round(4)
""")

code(r"""
# Parte B: aleatoriedad del optimizador (5 semillas por método estocástico)
semillas_opt = pd.read_csv(config.EXPERIMENTS_DIR / 'seed_sensitivity_optimizers.csv')
agrupado = semillas_opt.groupby(['tarea', 'modelo', 'optimizador'])
resumen_opt = agrupado['valor'].agg(['mean', 'std', 'min', 'max', 'count'])
resumen_opt['distintas'] = agrupado['mejores_hiperparametros_ultimo_pliegue'].nunique()  # configuraciones distintas elegidas
resumen_opt.columns = ['Media', 'Desviación', 'Mínimo', 'Máximo', 'Semillas', 'Configuraciones distintas (último pliegue)']
resumen_opt.index = pd.MultiIndex.from_tuples(
    [({'clf': 'Clasificación', 'reg': 'Regresión'}[t], A.MODEL_LABELS[m], A.OPTIMIZER_LABELS[o]) for t, m, o in resumen_opt.index],
    names=['Tarea', 'Modelo', 'Optimizador'])
resumen_opt.round(4)
""")

code(r"""
# Reproducibilidad: se repiten cuatro combinaciones con la misma semilla y se comparan con la tabla maestra
filas = []
for tarea, modelo, balanceo, optimizador in [('clf', 'KNN', 'smote', 'genetic'), ('clf', 'NaiveBayes', 'adasyn', 'bayesian'),
                                              ('reg', 'Ridge', 'none', 'genetic'), ('reg', 'Lasso', 'none', 'bayesian')]:
    especificacion = (CLASSIFIER_BY_NAME if tarea == 'clf' else REGRESSOR_BY_NAME)[modelo]
    objetivo = ds.y_clf if tarea == 'clf' else ds.y_reg
    clave = 'f1_macro' if tarea == 'clf' else 'rmse'
    repeticion = run_nested_cv(especificacion, ds.X, objetivo, tarea, balanceo, optimizador, budgets=config.FULL_BUDGETS)
    nuevo = np.mean([fr.metrics[clave] for fr in repeticion.fold_results])
    original = master.loc[master['run_id'] == f'{tarea}_{modelo}_{balanceo}_{optimizador}', f'{clave}_mean'].iloc[0]
    filas.append({'Combinación': f'{A.MODEL_LABELS[modelo]} · {A.BALANCING_LABELS[balanceo]} · {A.OPTIMIZER_LABELS[optimizador]}',
                  'Métrica': 'F1-macro' if tarea == 'clf' else 'RMSE', 'Corrida original': original, 'Repetición': nuevo,
                  'Diferencia absoluta': abs(original - nuevo)})
pd.DataFrame(filas).set_index('Combinación')
""")

md(r"""
La aleatoriedad del modelo (parte A) mueve poco el resultado en comparación con las diferencias entre modelos. Con diez semillas, la desviación estándar del F1-macro agrupado es de entre 0.004 y 0.025 en seis de los siete clasificadores y de 0.062 en Naive Bayes, cuya única fuente de azar es el remuestreo con ADASYN (KNN, que es determinista, varía 0.009 solo por SMOTE). En regresión, Ridge, SVR, KNN y Lasso no cambian nada (desviación de 0, son deterministas, lo que sirve de control) y los modelos con aleatoriedad varían 0.021 (árbol), 0.029 (Random Forest) y 0.040 (XGBoost) de RMSE. Así, la diferencia entre KNN (0.805) y Random Forest (0.670) en F1-macro, o entre SVR y XGBoost en RMSE (1.31 contra 1.56), es mucho mayor que lo que mueve cualquier semilla. Entre los modelos del grupo del medio la historia cambia: Random Forest (0.670) y Naive Bayes (0.673) están a una diferencia menor que la variación de una semilla de Naive Bayes, y entre SVR y Ridge (1.314 y 1.317) la diferencia es de tres milésimas con un azar de cero, de modo que la incertidumbre viene de los datos y no del algoritmo. La variabilidad entre pliegues (desviaciones de entre 0.14 y 0.21 de F1-macro y de entre 0.5 y 0.9 de RMSE) es mucho mayor que la de las semillas, y por eso hacen falta las pruebas de la sección siguiente.

La aleatoriedad del optimizador (parte B) importa más. Con cinco semillas, el resultado que se obtiene con Random, el bayesiano o el genético varía entre 0.02 y 0.06 de F1-macro en clasificación (árbol de decisión y KNN) y entre 0.002 y 0.029 de RMSE en regresión (KNN y Ridge), márgenes del mismo tamaño que las diferencias entre modelos vecinos. Casi siempre cada semilla termina en una configuración distinta (5 diferentes en 5 semillas, en 9 de las 12 combinaciones), pero con un desempeño parecido, lo que indica que hay muchas configuraciones casi equivalentes, una meseta, y no un único óptimo. Las excepciones son el bayesiano en la regresión de KNN, que solo encontró 2 configuraciones distintas y varió apenas 0.002, y el genético en esa misma tarea y Random en KNN de clasificación, con 4 de 5. En regresión el bayesiano es también el más estable (desviaciones de entre 0.002 y 0.003) y el genético el menos (entre 0.024 y 0.029). Grid, al ser determinista, da siempre lo mismo, pero no es mejor: gana en KNN de clasificación (0.667) y pierde en Ridge (1.318 contra entre 1.20 y 1.22 de los otros tres), porque su grilla de cuatro valores (0.001, 0.1, 10 y 1000) se salta la zona buena de alpha, que está cerca de 70.

Para comprobar que todo esto es reproducible se repitieron cuatro combinaciones con la misma semilla (42): las métricas coinciden con las de la tabla maestra en todos los decimales (diferencias de entre 0 y 2·10⁻¹⁶, es decir, error de redondeo), incluidas las del algoritmo genético y la optimización bayesiana, lo que confirma que la semilla global controla todas las fuentes de azar.
""")

md(r"""
### Optimización multi-fidelidad

Todas las búsquedas anteriores evalúan cada configuración con el costo completo del modelo, incluso las que desde el comienzo se ven mal. El *Successive Halving* (Jamieson & Talwalkar, 2016) reparte mejor ese costo: empieza con muchas configuraciones entrenadas con un presupuesto pequeño, descarta las peores y le da más presupuesto a las que van quedando. Se implementó para Random Forest y XGBoost, que son los modelos que más tiempo consumen en la corrida de los 140 modelos.

La fidelidad es el número de árboles. Entrenar un bosque cuesta aproximadamente lo mismo por cada árbol, y con pocos árboles una configuración ya muestra si es prometedora, aunque su puntaje sea más ruidoso. El factor de reducción es 3: se arranca con 27 configuraciones entrenadas con 11 árboles, las 9 mejores pasan a 33 árboles, las 3 mejores a 100 y la mejor termina con 300. Con ese factor cada ronda cuesta casi lo mismo (unos 300 árboles por pliegue interno) y la búsqueda completa gasta 3582 árboles por pliegue externo, lo que equivale a entrenar cuatro modelos de 300 árboles. Para saber si ahorrar presupuesto así vale la pena se compara con dos variantes de Random Search: una que gasta el mismo costo en árboles (4 configuraciones completas) y la estándar de 40 evaluaciones, con entre 50 y 300 árboles. Cada método se repitió con tres semillas. Hyperband (Li et al., 2018) generaliza esta idea repitiendo el esquema con distintos números de configuraciones iniciales; aquí se usó la versión básica.
""")

code(r"""
mf = pd.read_csv(config.EXPERIMENTS_DIR / 'multifidelity.csv')
resumen_mf = mf.groupby(['tarea', 'modelo', 'metodo']).agg(
    metrica=('metrica_externa', 'mean'), desviacion=('metrica_externa', 'std'),
    tiempo=('tiempo_busqueda_s', 'mean'), costo=('costo_arboles_por_pliegue', 'mean'))
resumen_mf.columns = ['Métrica externa', 'Desviación entre semillas', 'Tiempo de búsqueda (s)', 'Costo en árboles por pliegue']
resumen_mf.index = pd.MultiIndex.from_tuples(
    [({'clf': 'Clasificación (F1-macro)', 'reg': 'Regresión (RMSE)'}[t], A.MODEL_LABELS[m], k) for t, m, k in resumen_mf.index],
    names=['Tarea', 'Modelo', 'Método'])
resumen_mf.round(3)
""")

md(r"""
Successive Halving gasta 3582 árboles por pliegue externo, la sexta parte de lo que gasta Random Search con sus 40 evaluaciones (unos 21 mil árboles por pliegue, con 175 árboles de media por evaluación) y casi lo mismo que Random con el mismo costo (3600, cuatro configuraciones completas). Frente a esta última comparación, la justa, Successive Halving gana en los cuatro casos: sube el F1-macro de 0.484 a 0.492 en Random Forest y de 0.503 a 0.513 en XGBoost, y baja el RMSE de 1.374 a 1.270 en Random Forest y de 1.363 a 1.311 en XGBoost. Frente a Random Search estándar, que gasta seis veces más, consigue resultados parecidos: mejores en Random Forest (0.492 contra 0.483 de F1-macro y 1.270 contra 1.295 de RMSE) y peores por poco en XGBoost (0.513 contra 0.528 y 1.311 contra 1.306), con el tiempo de búsqueda dividido por entre 2.8 y 5.5 (18.5 s contra 102.3 s en Random Forest de clasificación, 12.9 s contra 35.7 s en XGBoost de clasificación, 14.6 s contra 78.9 s y 7.6 s contra 32.9 s en regresión). Conviene ser cautos: con tres semillas, la desviación entre semillas va de 0.005 a 0.083, así que diferencias de menos de 0.02 de F1-macro o de 0.05 de RMSE no son concluyentes. Lo que sí se puede afirmar es que reasignar el presupuesto hacia las configuraciones prometedoras permite acercarse a la calidad de la búsqueda completa con una fracción del costo.

Un detalle: el costo en árboles no es lo mismo que el tiempo. Con el mismo costo en árboles, Successive Halving tardó más que Random (18.5 s contra 16.6 s en Random Forest, 12.9 s contra 5.8 s en XGBoost de clasificación), porque entrena 40 modelos pequeños, de 11 y 33 árboles, en lugar de 4 grandes, y cada ajuste tiene un costo fijo de preparar los datos, balancear y escalar.
""")
