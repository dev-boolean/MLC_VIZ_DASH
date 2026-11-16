# -*- coding: utf-8 -*-
"""Capítulo: resultados de clasificación."""

CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


md(r"""
## Resultados de clasificación

Para elegir entre los modelos se usó el F1-macro y no el accuracy. Las tres categorías no cuestan lo mismo cuando se confunden: Alto corresponde a los meses de crisis, es la categoría minoritaria y la que más interesa detectar a tiempo. El F1-macro le da el mismo peso a cada categoría sin importar cuántos meses tenga, mientras que el accuracy permitiría obtener un buen puntaje acertando siempre la categoría más frecuente e ignorando las crisis casi por completo. Por la misma razón, además del F1-macro se reporta el recall de Alto, es decir, qué proporción de los meses de crisis logró anticipar el modelo: en este problema no avisar de una crisis cuesta más que una falsa alarma.

Para cada uno de los siete modelos se muestra la mejor combinación de técnica de balanceo y optimizador, según el F1-macro promedio de los cinco pliegues externos. Todas las métricas se reportan como media ± desviación estándar entre pliegues, porque con solo cinco pliegues un número suelto esconde la variabilidad.
""")

code(r"""
master = A.load_master()
clasificacion = master[master['task'] == 'clf']
mejores_clf = A.best_per_model(master, 'clf')   # mejor combinación de cada modelo según el F1-macro promedio

cuadro_clf = pd.DataFrame({
    'Balanceo': mejores_clf['balancing'].map(A.BALANCING_LABELS),
    'Optimizador': mejores_clf['optimizer'].map(A.OPTIMIZER_LABELS),
    'Accuracy': [A.mean_std(f, 'accuracy') for _, f in mejores_clf.iterrows()],
    'Precision': [A.mean_std(f, 'precision_macro') for _, f in mejores_clf.iterrows()],
    'Recall': [A.mean_std(f, 'recall_macro') for _, f in mejores_clf.iterrows()],
    'F1-score': [A.mean_std(f, 'f1_macro') for _, f in mejores_clf.iterrows()],
    'AUC': [A.mean_std(f, 'auc_ovr') for _, f in mejores_clf.iterrows()],
})
cuadro_clf.index = [A.MODEL_LABELS[m] for m in mejores_clf.index]
cuadro_clf
""")

md(r"""
En el cuadro, el mejor F1-macro es el de **KNN con SMOTE y algoritmo genético (0.679 ± 0.211)**, seguido por Random Forest con ADASYN y optimización bayesiana (0.617 ± 0.166) y por el árbol de decisión sin balancear (0.605 ± 0.195). Los otros cuatro modelos quedan entre 0.526 y 0.573. La desviación entre pliegues es grande en todos (entre 0.14 y 0.21 en el F1-macro), y la razón es que los cinco pliegues no se parecen entre sí: el primero pronostica 2006-2009 con solo 47 meses de historia y el cuarto contiene la pandemia. Por eso diferencias de unas pocas centésimas, como las que hay entre el árbol de decisión y la regresión logística, no permiten decir que un modelo sea mejor que otro; eso se revisa con pruebas estadísticas más adelante.

El accuracy cuenta una historia más optimista que el F1-macro: en los siete modelos es bastante más alto (0.800 frente a 0.605 en el árbol de decisión, por ejemplo), porque Bajo y Medio son las categorías con más meses y pesan más en el accuracy, mientras que el F1-macro le da el mismo peso a Alto. El AUC también es alto en varios modelos (0.87 en Random Forest, 0.86 en XGBoost y 0.85 en KNN) aunque su F1-macro sea modesto, lo que significa que ordenan bastante bien los meses según su riesgo, pero escoger siempre "la categoría más probable" no es el mejor corte para la categoría rara. En seis de los siete modelos la mejor combinación incluye SMOTE o ADASYN (la excepción es el árbol de decisión) y el optimizador ganador cambia de un modelo a otro (genético en tres, Grid en tres y bayesiano en uno), una señal de que las diferencias entre optimizadores son pequeñas. Un último detalle: como cada modelo aparece con la mejor de sus 16 variantes, escogida con las mismas métricas que se reportan, estas cifras son un poco optimistas.
""")

code(r"""
# Modelos de referencia sobre los mismos pliegues: clase más frecuente, persistencia de la clase y clase del año anterior
referencias_clf = A.classification_baselines(ds)
referencias_clf.columns = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'Recall de Alto', 'Precisión de Alto']

# Métricas de los siete mejores modelos sobre las 240 predicciones fuera de muestra apiladas
oof_clf = {m: A.pooled_oof(f['run_id']) for m, f in mejores_clf.iterrows()}
agrupadas = pd.DataFrame({A.MODEL_LABELS[m]: A.pooled_classification_metrics(p) for m, p in oof_clf.items()}).T
agrupadas = agrupadas[['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'recall_alto', 'precision_alto']]
agrupadas.columns = referencias_clf.columns
pd.concat([agrupadas, referencias_clf]).sort_values('F1-score', ascending=False)
""")

md(r"""
Al apilar las 240 predicciones fuera de muestra, KNN alcanza un F1-macro de 0.815 y es el único modelo que supera a la **persistencia de la clase**, la regla que predice que el mes siguiente tendrá la misma categoría que el mes actual (0.752). Todos los demás quedan por debajo de esa regla tan simple: la regresión logística con 0.721, Random Forest con 0.674, Naive Bayes con 0.651, XGBoost con 0.644, el árbol de decisión con 0.613 y SVM con 0.557. La regla del año anterior (0.598) es peor que casi todos y la de predecir siempre la categoría más frecuente (0.203) es la peor, como era de esperar. Con la categoría Alto la diferencia es más marcada: KNN reconoce 14 de los 17 meses Alto (recall de 0.82), la regresión logística 12, la persistencia 11 y los demás modelos entre 2 y 6.

La lectura honesta es que a un mes de horizonte la categoría de hoy ya dice casi todo sobre la de mañana: la persistencia acierta en el 79.6 % de los meses. Superar una regla tan simple es difícil y solo KNN lo logra. Estos F1 agrupados no son los mismos del cuadro anterior: aquel promedia el F1 de cada pliegue y este se calcula sobre los 240 meses juntos.
""")

code(r"""
etiquetas_modelo = {m: A.MODEL_LABELS[m] for m in mejores_clf.index}
predicciones = {etiquetas_modelo[m]: (p['y_true'], p['y_pred']) for m, p in oof_clf.items()}
fig = plots.confusion_grid(predicciones)
plt.show()
""")

code(r"""
probabilidades = {etiquetas_modelo[m]: (p['y_true'], p['y_proba']) for m, p in oof_clf.items()}
fig = plots.roc_grid(probabilidades)
plt.show()
""")

md(r"""
Las matrices de confusión muestran que los errores casi siempre son entre categorías vecinas: ningún modelo confunde un mes Bajo con uno Alto (las esquinas son cero), salvo SVM, que clasificó dos meses Alto como Bajo. KNN y la regresión logística son los que mejor reconocen la categoría Alto (14 y 12 de los 17 meses), pero a distinto costo. KNN solo se equivocó en 4 meses Medio que marcó como Alto; la regresión logística generó 9 falsas alarmas y, además, clasificó 37 meses Medio como Bajo. Random Forest, XGBoost, Naive Bayes y el árbol de decisión detectan entre 2 y 6 meses Alto y mandan la mayoría de los demás hacia Medio.

En las curvas ROC (cada categoría contra las otras dos) KNN tiene la mejor curva para Alto (AUC de 0.95), seguida por Random Forest y XGBoost (0.92) y la regresión logística (0.91). La curva del árbol de decisión para Alto casi coincide con la diagonal (0.52), porque sus probabilidades toman unos pocos valores y casi no ordena los meses, y SVM es el que peor ordena esa categoría (0.67). Para Bajo todos los modelos tienen un AUC de 0.86 o más, y Medio, que es la categoría intermedia y por eso la más difícil de separar, queda entre 0.76 y 0.89.
""")

md(r"""
### Efecto del balanceo de clases
""")

code(r"""
tabla_balanceo = clasificacion.pivot_table(index='model', columns='balancing', values='f1_macro_mean')
tabla_balanceo = tabla_balanceo[['none', 'smote', 'adasyn', 'class_weight']]
tabla_balanceo.columns = [A.BALANCING_LABELS[c] for c in tabla_balanceo.columns]
tabla_balanceo.index = [A.MODEL_LABELS[m] for m in tabla_balanceo.index]
tabla_balanceo.round(3)
""")

code(r"""
# Prueba de Friedman entre las cuatro técnicas: cada bloque es un modelo con un optimizador (28 bloques)
bloques = clasificacion.pivot_table(index=['model', 'optimizer'], columns='balancing', values='f1_macro_mean')
bloques = bloques[['none', 'smote', 'adasyn', 'class_weight']]
bloques.columns = [A.BALANCING_LABELS[c] for c in bloques.columns]
friedman_balanceo = ST.friedman_test(bloques)
print(f'Friedman: chi2 = {friedman_balanceo.statistic:.2f}, gl = {friedman_balanceo.df}, p = {friedman_balanceo.p_value:.4f}')
print(friedman_balanceo.avg_ranks.round(2).to_string())

diferencias = pd.DataFrame({
    'Mejora promedio de F1-macro frente a no balancear': [(bloques[c] - bloques['Ninguno']).mean() for c in bloques.columns[1:]],
    'Delta de Cliff frente a no balancear': [ST.cliffs_delta(bloques[c], bloques['Ninguno']) for c in bloques.columns[1:]],
}, index=bloques.columns[1:])
diferencias.round(3)
""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.boxplot([bloques[c] for c in bloques.columns], labels=bloques.columns, patch_artist=True,
           boxprops=dict(facecolor=plots.BLUE, alpha=0.3, color=plots.BLUE), medianprops=dict(color=plots.RED, linewidth=1.5))
ax.set_ylabel('F1-macro promedio (28 combinaciones modelo-optimizador)')
ax.set_title('Efecto de la técnica de balanceo sobre el F1-macro')
plt.tight_layout()
plt.show()

# Resumen de los 28 bloques por técnica (lo que muestran las cajas)
pd.DataFrame({'Media': bloques.mean(), 'Mediana': bloques.median(),
              'Rango intercuartílico': bloques.quantile(0.75) - bloques.quantile(0.25), 'Máximo': bloques.max()}).round(3)
""")

md(r"""
La tabla, que promedia los cuatro optimizadores, muestra que el balanceo no ayuda por igual a todos los modelos. A KNN y a Random Forest les sirve bastante: el F1-macro de KNN sube de 0.501 sin balancear a 0.654 con SMOTE y 0.656 con ADASYN, y el de Random Forest de 0.486 a 0.556 y 0.596. A la regresión logística y a SVM les da una mejora moderada (hasta 0.510 y 0.485, frente a 0.450 y 0.415), y a Naive Bayes casi no le cambia nada, salvo SMOTE, que lo empeora (0.457). Para el árbol de decisión y XGBoost el balanceo no sirve: el árbol rinde más sin balancear (0.545 contra 0.486 con SMOTE) y a XGBoost no lo mejora ninguna técnica (0.535 sin balancear). En KNN los pesos de clase no aplican, porque el modelo no admite pesos, y por eso esa columna coincide con la de "Ninguno".

La prueba de Friedman (1937) confirma que hay diferencias entre las cuatro técnicas (χ² = 13.35, 3 grados de libertad, p = 0.0039), y ADASYN tiene el mejor rango promedio (1.75), por delante de los pesos de clase (2.64), SMOTE (2.71) y no balancear (2.89). En tamaño del efecto, ADASYN mejora el F1-macro en promedio 0.049 frente a no balancear, con un delta de Cliff (1993) de 0.40 (efecto mediano según los umbrales de Romano et al., 2006); SMOTE mejora 0.022 (delta de 0.14, despreciable) y los pesos de clase 0.011 (delta de 0.03, despreciable). El diagrama de cajas muestra que la mediana de ADASYN (0.526) es la más alta, que la de SMOTE (0.487) queda incluso por debajo de la de no balancear (0.510) —su ganancia promedio viene de unos pocos modelos, sobre todo KNN y Random Forest, que llegan hasta 0.68— y que los pesos de clase son la técnica más compacta (rango intercuartílico de 0.03) porque casi no cambian el resultado. La conclusión es que ADASYN es la técnica más segura para este problema, pero la ganancia depende del modelo.
""")

md(r"""
### Calibración de las probabilidades

Un modelo puede ordenar bien los meses por riesgo y aun así dar probabilidades poco confiables: que diga "70 % de probabilidad de Alto" no significa que siete de cada diez meses con esa predicción sean realmente Alto. Para revisarlo se usan tres herramientas sobre la categoría Alto. El diagrama de confiabilidad agrupa las predicciones en 10 intervalos de probabilidad y compara la probabilidad media de cada intervalo con la frecuencia observada de meses Alto; si el modelo está bien calibrado, los puntos quedan sobre la diagonal. El Brier score (Brier, 1950) es el error cuadrático medio entre la probabilidad predicha y el resultado observado (0 o 1), y mezcla calibración con capacidad de discriminar. El error de calibración esperado, ECE (Guo et al., 2017), es el promedio, ponderado por el número de meses de cada intervalo, de la distancia entre la probabilidad predicha y la frecuencia observada. Para corregir los modelos mal calibrados se probaron dos recalibradores: el escalado de Platt (Platt, 1999), que ajusta una regresión logística sobre la probabilidad predicha, y la regresión isotónica (Zadrozny & Elkan, 2002), que ajusta una función escalonada creciente.
""")

code(r"""
curvas, filas = {}, []
for modelo, fila in mejores_clf.iterrows():
    pooled = oof_clf[modelo]
    calibracion = A.calibration_row(pooled)
    tabla = A.reliability_bins((pooled['y_true'] == A.ALTO).astype(int), pooled['y_proba'][:, A.ALTO], n_bins=10)
    curvas[etiquetas_modelo[modelo]] = (tabla, calibracion['brier_alto'], calibracion['ece_alto'])
    filas.append({'Modelo': etiquetas_modelo[modelo], 'Brier (Alto)': calibracion['brier_alto'],
                  'ECE (Alto)': calibracion['ece_alto'], 'Brier multiclase': calibracion['brier_multiclase']})
cuadro_calibracion = pd.DataFrame(filas).set_index('Modelo').sort_values('ECE (Alto)')
frecuencia_alto = float((oof_clf[mejores_clf.index[0]]['y_true'] == A.ALTO).mean())
print(f'Frecuencia de la categoría Alto: {frecuencia_alto:.3f}. Brier de una predicción constante igual a esa frecuencia: {frecuencia_alto * (1 - frecuencia_alto):.4f}')
cuadro_calibracion.round(4)
""")

code(r"""
fig = plots.reliability_grid(curvas)
plt.show()
""")

md(r"""
La categoría Alto es rara (17 de 240 meses, el 7.1 %), así que el Brier score de esa categoría hay que compararlo con el de una predicción constante: darle siempre 7.1 % de probabilidad tendría un Brier de 0.066. Lo mejoran KNN (0.029), la regresión logística (0.047), Random Forest (0.049) y Naive Bayes (0.056); XGBoost (0.066) apenas lo iguala, y SVM (0.084) y el árbol de decisión (0.072) quedan peor que no predecir nada. En el ECE, KNN es el mejor calibrado (0.024) y XGBoost el peor (0.111), seguido de SVM (0.087), el árbol (0.082) y la regresión logística (0.077).

Los diagramas de confiabilidad hay que leerlos con cuidado. Casi todos los meses caen en el primer grupo (probabilidades cercanas a 0, la burbuja grande) y los grupos de probabilidad alta tienen unos pocos meses cada uno, así que sus puntos son muy ruidosos: un solo mes cambia la frecuencia observada de 0 a 1. Aun así se distinguen algunos patrones. Las probabilidades de XGBoost nunca pasan de 0.6, quedan comprimidas hacia la frecuencia base, algo esperable en un modelo que eligió una tasa de aprendizaje muy baja (0.007 en el último pliegue); las de SVM quedan casi siempre por debajo de la diagonal, es decir, sobreestiman el riesgo; el árbol de decisión da probabilidades de 0 o 1, por lo que su diagrama solo tiene tres puntos; y KNN y la regresión logística son los que más se acercan a la diagonal en los extremos.
""")

code(r"""
# Recalibración con escalado de Platt y con regresión isotónica: el modelo se entrena con el primer 75 % de cada
# tramo de entrenamiento y el recalibrador con el 25 % restante; la prueba solo se usa para evaluar.
filas = []
for modelo, fila in mejores_clf.iterrows():
    r = A.recalibration_experiment(modelo, fila['balancing'], fila['optimizer'], ds)
    filas.append({
        'Modelo': etiquetas_modelo[modelo],
        'Brier sin recalibrar': r['brier_sin_recalibrar'], 'Brier Platt': r['brier_platt'], 'Brier isotónica': r['brier_isotonica'],
        'ECE sin recalibrar': r['ece_sin_recalibrar'], 'ECE Platt': r['ece_platt'], 'ECE isotónica': r['ece_isotonica'],
    })
recalibracion = pd.DataFrame(filas).set_index('Modelo')
recalibracion.round(4)
""")

md(r"""
El efecto de recalibrar depende de qué tan mal calibrado estuviera el modelo. Las mejoras más claras son las de los que tenían peor calibración: XGBoost baja su ECE de 0.126 a 0.057 con Platt y a 0.054 con la isotónica (y su Brier de 0.082 a 0.060 y 0.065), SVM de 0.133 a 0.070 con Platt (Brier de 0.111 a 0.074), Random Forest de 0.106 a 0.056 y la regresión logística de 0.078 a 0.049 con Platt y 0.045 con la isotónica. Naive Bayes mejora con Platt (ECE de 0.061 a 0.029) pero empeora con la isotónica (0.082).

En cambio, a KNN y al árbol de decisión recalibrar les hace daño: el ECE de KNN pasa de 0.047 a 0.080 con Platt y 0.074 con la isotónica, y el del árbol de 0.079 a 0.099 y 0.097. Eran los modelos que ya estaban razonablemente calibrados o cuyas probabilidades solo toman unos pocos valores, y el recalibrador se ajusta con apenas el 25 % final de cada tramo de entrenamiento, que tiene muy pocos meses Alto. Con tan pocos casos de la clase rara, Platt (dos parámetros) es más estable que la regresión isotónica (muchos escalones), que se sobreajusta, como se ve en Naive Bayes. La recomendación práctica es recalibrar solo los modelos mal calibrados (XGBoost, SVM, Random Forest y regresión logística), preferir Platt cuando hay pocos datos y comprobar siempre el efecto sobre una muestra que no se usó para ajustar el recalibrador.
""")
