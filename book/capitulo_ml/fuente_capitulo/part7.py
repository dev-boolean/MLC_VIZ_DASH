# -*- coding: utf-8 -*-
"""Capítulo: optimización computacional."""

CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


md(r"""
## Optimización computacional

Con 288 observaciones y 17 variables el costo computacional no es un problema: la corrida completa de los 140 modelos tomó 57 minutos sin paralelizar nada. Aun así, conviene entender cómo escalaría cada modelo si los datos fueran mucho más grandes, y qué se gana al usar la variante pensada para ese caso. Esta sección hace tres cosas: contrasta la complejidad teórica de cada modelo con el tiempo medido en el proyecto, mide el escalamiento con n y con p sobre datos sintéticos de tamaño controlado (cada medición se repitió tres veces, o dos en los modelos más lentos, y se reporta la mediana), y compara la variante estándar de cada modelo con la optimizada: KNN exhaustivo frente a KD-Tree y FAISS (Johnson et al., 2019), Ridge con solver directo frente a SAGA (Defazio et al., 2014), Naive Bayes con `fit` frente a `partial_fit`, XGBoost con método exacto frente a histogramas y SVM con núcleo frente a `LinearSVC` y gradiente estocástico. Todas las mediciones de esta sección se hicieron con BLAS y OpenMP limitados a un hilo: en una primera versión el Lasso saltó de 0.01 s a 1 s entre 10 mil y 20 mil observaciones, y resultó ser un artefacto de los 12 hilos de OpenBLAS compitiendo con otros procesos y no un comportamiento del algoritmo (con un hilo el tiempo crece de forma lineal).
""")

md(r"""
### Complejidad teórica y tiempo medido en el proyecto

| Modelo | Entrenamiento | Inferencia | Variante pensada para datos grandes |
|---|---|---|---|
| KNN | $O(1)$ (solo guarda los datos) | $O(n \cdot p)$ por consulta con búsqueda exhaustiva | KD-Tree o Ball-Tree, $O(p \log n)$ en pocas dimensiones; FAISS para vecinos aproximados |
| Ridge | $O(n p^2 + p^3)$ con la ecuación normal | $O(p)$ | SAGA, con costo lineal en $n$ y $p$ por iteración |
| Lasso | $O(n p \cdot \text{iter})$ por descenso de coordenadas | $O(p)$ | gradiente estocástico |
| Naive Bayes | $O(n p)$ | $O(p)$ | `partial_fit`, que aprende por lotes sin cargar todo en memoria |
| Regresión logística | $O(n p \cdot \text{iter})$ | $O(p)$ | SAGA |
| Árbol de decisión | $O(n p \log n)$ | $O(\text{profundidad})$ | |
| Random Forest | $O(B \cdot n p \log n)$, con $B$ árboles | $O(B \cdot \text{profundidad})$ | árboles en paralelo con `n_jobs` |
| XGBoost | $O(B \cdot n p \log n)$ con el método exacto | $O(B \cdot \text{profundidad})$ | histogramas, $O(B \cdot n p)$, y parada temprana |
| SVM y SVR con núcleo | entre $O(n^2 p)$ y $O(n^3)$ | $O(n_{vs} \cdot p)$, con $n_{vs}$ vectores de soporte | `LinearSVC` o gradiente estocástico, $O(n p)$ |
""")

code(r"""
# Tiempo medido en la corrida real (n = 239 meses de entrenamiento por pliegue en promedio), por familia de modelos
tiempos = master.groupby(['task', 'model']).agg(
    ajuste_final=('final_fit_time_s', 'mean'), inferencia=('predict_time_s', 'mean'),
    busqueda=('search_time_s', 'sum'), evaluaciones=('total_evaluations', 'sum'))
tiempos['ms por evaluación'] = 1000 * tiempos['busqueda'] / tiempos['evaluaciones']
tiempos['ajuste_final'] *= 1000
tiempos['inferencia'] *= 1000
tiempos['busqueda'] /= 60
tiempos.index = pd.MultiIndex.from_tuples(
    [({'clf': 'Clasificación', 'reg': 'Regresión'}[t], A.MODEL_LABELS[m]) for t, m in tiempos.index], names=['Tarea', 'Modelo'])
tiempos = tiempos.rename(columns={'ajuste_final': 'Ajuste final (ms)', 'inferencia': 'Inferencia de 48 meses (ms)',
                                  'busqueda': 'Búsqueda total (min)', 'evaluaciones': 'Evaluaciones'})
tiempos.sort_values('Búsqueda total (min)', ascending=False).round(2)
""")

md(r"""
La búsqueda de los 140 modelos sumó 56.7 minutos, y el 82 % de ese tiempo se fue en Random Forest y XGBoost. Solo el Random Forest de clasificación tomó 25.8 minutos (484 ms por evaluación, y cada evaluación entrena tres modelos, uno por pliegue interno), seguido de XGBoost de clasificación con 13.0 minutos y de Random Forest de regresión con 5.6. Los demás cuestan 4.1 minutos o menos cada uno (la regresión logística es la más cara de ellos), y Ridge, Lasso, KNN y el árbol de regresión menos de 15 segundos. Esto es lo que anticipaba la tabla de complejidad: los métodos de conjunto multiplican el costo de un árbol por el número de árboles (hasta 300 en el espacio de búsqueda), mientras que los modelos lineales pasan pocas veces por los datos. También se ve la diferencia entre el aprendizaje perezoso y el ansioso: KNN casi no cuesta entrenarlo (2.8 ms) y predecir 48 meses le toma más (7.1 ms), mientras que en el Random Forest ocurre al revés (175 ms de ajuste y 17 ms de predicción). Con solo 239 meses de entrenamiento los tiempos absolutos son pequeños, y por eso el efecto de las variantes optimizadas solo se puede estudiar con datos sintéticos más grandes, como se hace a continuación.
""")

md(r"""
### Escalamiento con el tamaño de los datos
""")

code(r"""
escalamiento = pd.read_csv(config.EXPERIMENTS_DIR / 'complexity_scaling.csv')

fig, ejes = plt.subplots(2, 4, figsize=(18, 8))
plots.plot_scaling(escalamiento, 'KNN', x='n', y='inferencia_s', ax=ejes[0, 0], title='KNN: inferencia frente a n')
plots.plot_scaling(escalamiento, 'KNN (en p)', x='p', y='inferencia_s', ax=ejes[0, 1], title='KNN: inferencia frente a p')
plots.plot_scaling(escalamiento, 'Ridge/Lasso', x='n', y='entrenamiento_s', ax=ejes[0, 2], title='Ridge y Lasso: entrenamiento frente a n')
plots.plot_scaling(escalamiento, 'Ridge/Lasso (en p)', x='p', y='entrenamiento_s', ax=ejes[0, 3], title='Ridge y Lasso: entrenamiento frente a p')
plots.plot_scaling(escalamiento, 'Naive Bayes', x='n', y='entrenamiento_s', ax=ejes[1, 0], title='Naive Bayes: entrenamiento frente a n')
plots.plot_scaling(escalamiento, 'XGBoost', x='n', y='entrenamiento_s', ax=ejes[1, 1], title='XGBoost: entrenamiento frente a n')
plots.plot_scaling(escalamiento, 'SVM', x='n', y='entrenamiento_s', ax=ejes[1, 2], title='SVM: entrenamiento frente a n')
plots.plot_scaling(escalamiento, 'SVM', x='n', y='inferencia_s', ax=ejes[1, 3], title='SVM: inferencia frente a n')
plt.tight_layout()
plt.show()
""")

code(r"""
# Exponente empírico: pendiente de log(tiempo) frente a log(n) (o log(p)); un exponente cercano a 1 indica crecimiento lineal
filas = []
for familia, x in [('KNN', 'n'), ('KNN (en p)', 'p'), ('Ridge/Lasso', 'n'), ('Ridge/Lasso (en p)', 'p'),
                   ('Naive Bayes', 'n'), ('XGBoost', 'n'), ('SVM', 'n')]:
    sub = escalamiento[escalamiento['familia'] == familia]
    for variante, grupo in sub.groupby('variante', sort=False):
        fila = {'Familia': familia, 'Variante': variante, 'Variable': x}
        for columna, nombre in [('entrenamiento_s', 'Exponente de entrenamiento'), ('inferencia_s', 'Exponente de inferencia')]:
            valido = grupo[(grupo[columna] > 0) & (grupo[x] > 0)]
            fila[nombre] = np.polyfit(np.log(valido[x]), np.log(valido[columna]), 1)[0] if len(valido) >= 3 else np.nan
        filas.append(fila)
exponentes = pd.DataFrame(filas).set_index(['Familia', 'Variante'])
exponentes.round(2)
""")

md(r"""
La mayoría de los exponentes medidos coincide con lo que dice la teoría. En KNN, la búsqueda exhaustiva predice con un costo casi lineal en n (exponente de 0.97) y el KD-Tree crece mucho más despacio (0.25, cerca del logarítmico), a cambio de un índice que sí cuesta construir casi n log n (1.15 en entrenamiento). Aun así, con 8 variables el KD-Tree solo empieza a ganarle a la búsqueda exhaustiva a partir de unas 16 mil a 32 mil observaciones (con 32 mil predice en 0.15 s y la exhaustiva en 0.21 s; con 2 mil tarda 0.076 s contra 0.014 s). Y cuando crece la dimensión pierde su ventaja por completo: con 80 variables el KD-Tree tarda 1.9 s frente a 0.1 s de la búsqueda exhaustiva, que es la maldición de la dimensionalidad. FAISS exacto escala igual que la búsqueda exhaustiva (0.99), solo que más rápido, y la versión aproximada (IVF) crece algo menos (0.81).

Ridge y Lasso crecen de forma lineal en n y en p, como dice el costo por iteración de O(n·p) (exponentes entre 0.9 y 1.1). El solver directo (Cholesky) tiene un exponente de 0.91 en n y de 1.04 en p; la teoría dice que debería crecer con p² por el producto XᵀX, pero entre 10 y 400 variables su costo está dominado por recorrer la matriz y no por el término cuadrático, que solo se notaría con muchas más variables. XGBoost con el método exacto crece como n log n (1.15) y con histogramas bastante menos (0.60); su inferencia no crece con n (exponente cercano a cero) porque depende de los árboles y no de los datos de entrenamiento. En SVM con núcleo RBF el entrenamiento crece de forma claramente superlineal (1.67, por debajo del cuadrático teórico porque el costo depende de cuántos vectores de soporte quedan) y la inferencia también crece con n (0.73), porque cada predicción se compara con los vectores de soporte; LinearSVC (1.02) y el gradiente estocástico (0.90) son lineales y su inferencia no depende de n. En Naive Bayes, `fit` y `partial_fit` son igual de lineales (1.00 y 0.97): aprender por lotes no cambia el orden del tiempo, cambia la memoria, como se ve en la tabla siguiente.
""")

md(r"""
### Modelo estándar frente a optimizado
""")

code(r"""
comparativa = pd.read_csv(config.EXPERIMENTS_DIR / 'standard_vs_optimized.csv')
comparativa = comparativa.rename(columns={
    'modelo': 'Modelo', 'variante': 'Variante', 'complejidad_entrenamiento': 'Complejidad de entrenamiento',
    'complejidad_inferencia': 'Complejidad de inferencia', 'entrenamiento_s': 'Entrenamiento (s)', 'inferencia_s': 'Inferencia (s)',
    'memoria_pico_mb': 'Memoria pico (MB)', 'desempeño': 'Desempeño', 'aceleracion_entrenamiento': 'Aceleración del entrenamiento'})
comparativa.set_index(['Modelo', 'Variante'])[['Entrenamiento (s)', 'Aceleración del entrenamiento', 'Inferencia (s)', 'Memoria pico (MB)', 'Desempeño']].round(3)
""")

code(r"""
comparativa.set_index(['Modelo', 'Variante'])[['Complejidad de entrenamiento', 'Complejidad de inferencia']]
""")

md(r"""
La parada temprana de XGBoost se evaluó aparte: el modelo entrena hasta 1000 árboles y se detiene cuando el error sobre una validación independiente (el último 20 % de los datos de entrenamiento) no mejora en 20 rondas seguidas; la comparación es contra entrenar siempre 300 árboles.
""")

code(r"""
parada = escalamiento[escalamiento['familia'] == 'XGBoost (parada temprana)']
parada = parada[['variante', 'arboles_usados', 'entrenamiento_s', 'inferencia_s', 'desempeño']]
parada.columns = ['Variante', 'Árboles usados', 'Entrenamiento (s)', 'Inferencia (s)', 'Accuracy en prueba']
parada.set_index('Variante').round(4)
""")

md(r"""
La tabla muestra que "optimizado" no siempre significa "más rápido": depende del tamaño de los datos y de qué se sacrifica. En KNN con 16 mil observaciones y 8 variables, el KD-Tree no ayuda (predice en 0.135 s contra 0.142 s de la búsqueda exhaustiva y construir el índice cuesta 40 veces más), pero FAISS aproximado predice 6.1 veces más rápido (0.023 s) con casi la misma exactitud (0.9575 contra 0.9585): el compromiso entre exactitud y velocidad de la aproximación es aquí de una milésima de accuracy. En Ridge, SAGA es 8.4 veces más lento que Cholesky (0.186 s contra 0.022 s) con el mismo RMSE (10.2439 y 10.2440); solo compensaría con muchas más variables o con matrices dispersas, que no se probaron. En Naive Bayes, `partial_fit` tarda lo mismo pero usa 8.9 MB en lugar de 508 MB, 57 veces menos memoria, que es justo su razón de ser: aprender cuando los datos no caben en memoria. XGBoost con histogramas entrena 15.6 veces más rápido (2.5 s contra 38.7 s con 40 mil observaciones) y pierde solo 0.0005 de accuracy (de 0.9815 a 0.9810), la mejor relación de toda la tabla, a cambio de algo más de memoria (16.4 MB contra 12.4 MB) por las estructuras de los histogramas. Con parada temprana, XGBoost se detuvo en 78 de los 300 árboles y entrenó 2.4 veces más rápido (1.55 s contra 3.78 s) con el mismo accuracy (0.9430 contra 0.9435).

En SVM, LinearSVC y el gradiente estocástico entrenan 29 y 33 veces más rápido, predicen en 0.3 milisegundos en vez de 0.43 s y usan 2.7 MB y 0.02 MB en lugar de 34.9 MB, pero el accuracy cae de 0.9245 a 0.839 y a 0.816. Es el único caso donde la variante optimizada paga un precio claro en desempeño, porque el problema sintético no es linealmente separable del todo. Y Random Forest con todos los núcleos (`n_jobs=-1`) entrena 7.6 veces más rápido con exactamente el mismo resultado (0.9315). En resumen, las mejoras que no cambian el modelo (histogramas, paralelismo, parada temprana) son prácticamente gratis, las que lo aproximan (FAISS, SVM lineal) cambian velocidad por exactitud en una proporción que hay que medir caso por caso, y las que solo ahorran memoria (`partial_fit`, SGD) son útiles cuando los datos no caben, no cuando sobra memoria.
""")

md(r"""
### Paralelización

El trabajo de este proyecto se puede repartir de dos maneras: entre combinaciones (cada una de las 140 es independiente de las demás) y entre pliegues externos (los cinco pliegues de una misma combinación también lo son). Con `joblib` se midieron las dos. Para la primera se corrieron, en serie y en paralelo, 24 combinaciones baratas (KNN, Ridge y Lasso con sus cuatro optimizadores) con el backend `loky`, que reparte el trabajo en procesos, y con `threading`, que lo reparte en hilos que comparten memoria pero quedan limitados por el GIL de Python. Para la segunda se corrió una combinación pesada, Random Forest con búsqueda aleatoria, repartiendo sus cinco pliegues externos en cinco procesos. En todos los casos se comprobó que el resultado paralelo fuera idéntico al serial, porque paralelizar no puede cambiar las métricas. En estas mediciones BLAS y OpenMP se limitaron a un hilo para que lo único que se paralelice sea lo que se pide con `n_jobs`.
""")

code(r"""
paralelo = pd.read_csv(config.EXPERIMENTS_DIR / 'parallel_speedup.csv')
paralelo = paralelo.rename(columns={'parte': 'Qué se reparte', 'backend': 'Backend de joblib', 'trabajadores': 'Trabajadores',
                                    'tiempo_s': 'Tiempo (s)', 'aceleracion': 'Aceleración', 'eficiencia': 'Eficiencia',
                                    'resultado_identico': 'Resultado idéntico al serial'})
paralelo.set_index(['Qué se reparte', 'Backend de joblib', 'Trabajadores']).round(2)
""")

md(r"""
Repartir las combinaciones entre procesos acelera de forma clara pero no proporcional: con 2 procesos se gana 1.73 veces (una eficiencia de 86 %), con 4 procesos 3.10 veces (78 %), con 8 procesos 4.12 veces (52 %) y con 12 solo 4.34 veces (36 %). Pasar de 8 a 12 casi no ayuda porque el procesador tiene 6 núcleos físicos (los otros 6 hilos comparten núcleo con ellos), porque cada proceso nuevo tiene que importar las librerías y recibir los datos, y porque las 24 combinaciones no cuestan lo mismo, así que al final los trabajadores esperan a la más lenta. Repartir los cinco pliegues externos de una combinación pesada (Random Forest con búsqueda aleatoria) entre 5 procesos baja el tiempo de 93.1 s a 25.7 s (3.63 veces, 73 % de eficiencia), también con resultados idénticos a los seriales. Si la aceleración de 4.3 veces se mantuviera en la corrida completa, los 57 minutos se reducirían a unos 13.

El backend `threading`, en cambio, empeora las cosas: con 4 y 8 hilos el tiempo (83.6 s y 88.1 s) es mayor que el serial (73.4 s), es decir, aceleraciones de 0.88 y 0.83. El código de los optimizadores es Python puro y el GIL deja avanzar a un solo hilo a la vez, así que coordinar los hilos cuesta más de lo que se gana. Además los hilos cambian el resultado, como lo indica la última columna. Una prueba aparte con 16 de las combinaciones (`scripts/diagnostico_hilos.py`) mostró que solo cambian las del algoritmo genético: DEAP usa el generador aleatorio global de Python, que los hilos comparten y que cada búsqueda genética reinicia con la semilla al empezar, de modo que las secuencias de números se mezclan entre hilos. Por eso cuáles combinaciones cambian, y por cuánto, varía de una ejecución a otra (entre 0.0001 y 0.011 en las dos veces que se corrió el diagnóstico), lo típico de una condición de carrera. Con procesos (`loky`) cada trabajador tiene su propio generador y las 16 combinaciones dieron exactamente lo mismo, por lo que `loky` es el backend adecuado para este proyecto.

No se usó Dask ni Ray porque no se justifican aquí: las 140 combinaciones suman 57 minutos de cómputo, los datos son de 288 filas y `joblib` ya da una aceleración de 3.6 a 4.3 veces en un solo computador. Esas herramientas valen la pena cuando el trabajo no cabe en una máquina, con muchas más combinaciones pesadas o datos de millones de filas. Tampoco se pudo medir la aceleración por GPU de XGBoost (`device='cuda'`): el computador tiene tarjetas AMD Radeon, que no tienen CUDA, así que no hay un factor de aceleración que reportar. Con 239 observaciones por entrenamiento sería de esperar que la GPU no ayudara, porque copiar los datos a la tarjeta costaría más que el entrenamiento mismo (el ajuste final de XGBoost toma unos 118 ms), pero esa es una expectativa y no una medición.
""")

md(r"""
### Perfilamiento
""")

code(r"""
perfil = pd.read_csv(config.EXPERIMENTS_DIR / 'profile_top.csv')
total_perfil = perfil['tiempo_total_perfil_s'].iloc[0]
print(f'Tiempo total perfilado (Random Forest, SMOTE, búsqueda aleatoria): {total_perfil:.1f} s')
perfil[perfil['orden'] == 'tiempo propio'].head(8)[['funcion', 'llamadas', 'tiempo_propio_s', 'pct_propio']].rename(columns={
    'funcion': 'Función', 'llamadas': 'Llamadas', 'tiempo_propio_s': 'Tiempo propio (s)', 'pct_propio': '% del total'}).set_index('Función').round(2)
""")

code(r"""
memoria = pd.read_csv(config.EXPERIMENTS_DIR / 'profile_memory.csv')
memoria['modelo'] = memoria['modelo'].map(A.MODEL_LABELS)
memoria.rename(columns={'modelo': 'Modelo', 'tarea': 'Tarea', 'memoria_pico_mb': 'Memoria pico adicional (MB)', 'tiempo_s': 'Tiempo (s)'}).set_index('Modelo').round(1)
""")

md(r"""
El perfil de Random Forest con SMOTE y búsqueda aleatoria (160 s con cProfile, frente a 106.5 s de la misma corrida sin perfilar, porque cProfile añade su propio costo a cada llamada) confirma dónde está el costo: el 99.9 % del tiempo está dentro de la búsqueda, el 96.6 % en `_fit_predict_fold` (560 ajustes de los modelos internos) y el 91 % dentro de `RandomForest.fit`. Más de una cuarta parte del tiempo propio (26.8 %) se va en cuatro tareas que no son matemáticas: `check_random_state` (9.3 %), el control de errores de NumPy (`errstate`, 8.0 %), la inspección de firmas de funciones (5.1 %) y la búsqueda de etiquetas de los estimadores de scikit-learn (4.4 %), costos fijos por llamada que se notan porque los árboles son diminutos (239 observaciones) y se entrenan más de 100 mil. El ajuste de los árboles en sí (`_fit`) es el primero de la lista con 10.5 % del tiempo propio y 47 % del acumulado. Con datos tan pequeños, un algoritmo más rápido no ayudaría a este pipeline; lo que sí ayuda es hacer menos ajustes (menos evaluaciones por búsqueda, descartar configuraciones malas con Successive Halving) y repartirlos entre procesos.

La memoria no es un problema: el máximo adicional que usa una búsqueda anidada completa es de 0.4 MB en Ridge, 0.5 MB en KNN, 0.9 MB en Random Forest y 18 MB en XGBoost, el que más usa probablemente porque convierte los datos a su estructura interna en cada ajuste. La medida es el máximo del conjunto residente del proceso, muestreado cada 5 ms, de modo que los picos más cortos podrían escaparse.
""")
