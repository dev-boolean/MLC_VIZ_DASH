# -*- coding: utf-8 -*-
"""Capítulo: interpretabilidad."""

CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


md(r"""
## Interpretabilidad

Un modelo que predice bien no siempre es fácil de explicar, y en un problema como este, donde un pronóstico podría alimentar decisiones de política económica, interesa saber qué variables pesan en cada predicción. Para el mejor modelo de cada tarea se usaron los valores de Shapley (SHAP; Lundberg & Lee, 2017), que reparten la predicción entre las variables a partir de la teoría de juegos cooperativos (Shapley, 1953): las contribuciones sumadas al valor promedio del modelo reproducen la predicción (eficiencia), dos variables que aportan lo mismo reciben lo mismo (simetría) y las contribuciones de modelos combinados se suman (aditividad). Para árboles y modelos lineales el cálculo es exacto; para el resto, como KNN, se aproxima.

Primero se muestra la importancia global, con cada modelo entrenado con todos los meses; luego, dos meses en detalle por tarea (uno bien pronosticado y otro con un error grande), explicados con el modelo del pliegue que realmente los pronosticó; y al final se compara SHAP con LIME (Ribeiro et al., 2016) en XGBoost y se usa el mismo análisis para responder si los indicadores de las 13 áreas metropolitanas aportan información sobre el desempleo nacional.
""")

md(r"""
### Importancia global con valores de Shapley
""")

code(r"""
import shap

# Modelo final de cada tarea entrenado con todos los meses, con los hiperparámetros del último pliegue externo
def modelo_final(tarea, modelo, fila):
    historial = A.load_history(fila['run_id'])
    parametros = historial['fold_results'][-1]['best_params']
    est, escalador, X_entrenamiento = A.fit_on_all(tarea, modelo, fila['balancing'], parametros, ds)
    return est, escalador, X_entrenamiento, parametros

mejor_clf = mejores_clf.index[0]
est_clf, esc_clf, X_clf, params_clf = modelo_final('clf', mejor_clf, mejores_clf.iloc[0])
explicador_clf, shap_clf = I.compute_shap(est_clf, esc_clf, X_clf, ds.X, 'clf')
importancia_clf = I.mean_abs_shap(shap_clf, ds.X.columns, A.ALTO)

est_reg, esc_reg, X_reg, params_reg = modelo_final('reg', mejor_reg, mejores_reg.iloc[0])
explicador_reg, shap_reg = I.compute_shap(est_reg, esc_reg, X_reg, ds.X, 'reg')
importancia_reg = I.mean_abs_shap(shap_reg, ds.X.columns)

print(f'Mejor clasificador: {A.MODEL_LABELS[mejor_clf]} con {params_clf}')
print(f'Mejor regresor: {A.MODEL_LABELS[mejor_reg]} con {params_reg}')
""")

code(r"""
shap.summary_plot(shap_clf[:, :, A.ALTO], ds.X, show=False, plot_size=(9, 6))
plt.title(f'SHAP: {A.MODEL_LABELS[mejor_clf]}, probabilidad de la clase Alto')
plt.tight_layout()
plt.show()
""")

code(r"""
shap.summary_plot(shap_reg, ds.X, show=False, plot_size=(9, 6))
plt.title(f'SHAP: {A.MODEL_LABELS[mejor_reg]}, tasa de desempleo nacional del mes siguiente')
plt.tight_layout()
plt.show()
""")

code(r"""
# Importancia global como porcentaje de la suma de los valores absolutos medios de SHAP, para poder comparar las dos tareas
relativa = pd.concat([(100 * importancia_clf / importancia_clf.sum()).rename('Clasificación: KNN, clase Alto (%)'),
                      (100 * importancia_reg / importancia_reg.sum()).rename('Regresión: Ridge (%)')], axis=1)
relativa.sort_values('Regresión: Ridge (%)', ascending=False).round(1)
""")

md(r"""
En los dos casos pesan sobre todo las variables que describen el nivel reciente del desempleo, aunque de forma distinta. En el KNN que estima la probabilidad de la categoría Alto la importancia está repartida entre muchas variables parecidas: el promedio móvil de 12 meses (10.4 % de la importancia total), el rezago de 12 meses (10.1 %), el promedio de 6 meses (9.1 %), el rezago de 6 meses (8.6 %) y la tasa de desempleo de las áreas (8.5 %) suman el 47 %. En todas ellas los valores altos (en rojo) empujan hacia arriba la probabilidad de Alto y los bajos hacia abajo, que es lo esperable: es más probable que un mes sea de crisis si el desempleo de los meses anteriores ya era alto. La estacionalidad también aparece, porque los valores altos de `mes_cos` (los meses de diciembre y enero, justo antes de los picos de comienzo de año) aumentan la probabilidad, y el índice de tendencia tiene el efecto contrario: a medida que avanza el tiempo la tasa baja.

En Ridge la variable más importante es la tasa de desempleo de las 13 áreas metropolitanas (13.7 %), seguida del rezago de 12 meses (12.3 %) y del último mes (12.2 %); el mes del año (`mes_cos` y `mes_sin`, 13.8 % entre las dos), la tasa de ocupación de las áreas (9.6 %) y la tendencia (9.6 %) completan el grupo que más pesa. Las tres variables de las áreas suman el 26.7 % de la importancia en Ridge (y el 13.9 % en el KNN), un primer indicio de que el desempleo de las ciudades principales sí aporta información sobre el nacional. Varias variables casi no cuentan: los rezagos de 2 y 3 meses y el promedio de 12 meses quedan en 0.0 %, 0.4 % y 0.1 %, porque Ridge, con una regularización fuerte (alpha de 70.9), reparte el peso entre variables muy correlacionadas y deja casi todo en las que mejor resumen el nivel y la estacionalidad.
""")

md(r"""
### Dos observaciones en detalle
""")

code(r"""
# Para cada tarea se elige un mes que el modelo acertó bien y otro con un error grande, y se explica con el modelo del pliegue
# externo que lo pronosticó (así la explicación corresponde a una predicción realmente fuera de muestra).
def explicar_mes(tarea, modelo, fila, pos_en_oof):
    pooled = A.pooled_oof(fila['run_id'])
    mes = int(pooled['idx'][pos_en_oof])
    pliegue = int(pooled['fold'][pos_en_oof])
    ajuste = A.refit_fold(tarea, modelo, fila['balancing'], fila['optimizer'], pliegue, ds)
    explicador, valores = I.compute_shap(ajuste['est'], ajuste['scaler'], ajuste['X_train'], ajuste['X_test'], tarea)
    fila_en_prueba = int(np.where(ajuste['test_idx'] == mes)[0][0])
    entrada = ajuste['X_test_t'][fila_en_prueba:fila_en_prueba + 1]
    if tarea == 'clf':
        prediccion = A.M.expand_proba(ajuste['est'].predict_proba(entrada), ajuste['est'].classes_)[0, A.ALTO]
        clase = A.ALTO
    else:
        prediccion, clase = float(ajuste['est'].predict(entrada)[0]), None
    return I.local_explanation(explicador, valores, ajuste['X_test'], fila_en_prueba, clase), mes, prediccion

# Clasificación: un mes Alto bien clasificado y un mes Alto que el modelo no reconoció
p_clf = oof_clf[mejor_clf]
alto = np.where(p_clf['y_true'] == A.ALTO)[0]
acertado = alto[np.argmax(p_clf['y_proba'][alto, A.ALTO])]
fallado = alto[np.argmin(p_clf['y_proba'][alto, A.ALTO])]
# Regresión: el mes con el error más pequeño y el de error más grande
e_reg = errores_reg[mejor_reg]['error']
mejor_mes, peor_mes = int(np.argmin(np.abs(e_reg))), int(np.argmax(np.abs(e_reg)))

casos = [('clf', mejor_clf, mejores_clf.iloc[0], acertado, 'Clasificación: mes Alto bien clasificado'),
         ('clf', mejor_clf, mejores_clf.iloc[0], fallado, 'Clasificación: mes Alto no reconocido'),
         ('reg', mejor_reg, mejores_reg.iloc[0], mejor_mes, 'Regresión: error más pequeño'),
         ('reg', mejor_reg, mejores_reg.iloc[0], peor_mes, 'Regresión: error más grande')]
explicaciones = {}
for tarea, modelo, fila, posicion, titulo in casos:
    explicacion, mes, prediccion = explicar_mes(tarea, modelo, fila, posicion)
    explicaciones[titulo] = explicacion
    fecha = fechas_objetivo.iloc[mes].strftime('%Y-%m')
    if tarea == 'clf':
        detalle = f"observado {config.CLASS_LABELS[int(ds.y_clf.iloc[mes])]}, probabilidad de Alto {prediccion:.2f}"
    else:
        detalle = f"observado {ds.y_reg.iloc[mes]:.2f} %, predicho {prediccion:.2f} %"
    print(f'{titulo} ({fecha}): {detalle}')
""")

code(r"""
for titulo, explicacion in list(explicaciones.items())[:2]:
    shap.plots.waterfall(explicacion, max_display=8, show=False)
    plt.title(titulo, fontsize=11)
    plt.gcf().set_size_inches(9, 4.5)
    plt.tight_layout()
    plt.show()
""")

code(r"""
for titulo, explicacion in list(explicaciones.items())[2:]:
    shap.plots.waterfall(explicacion, max_display=8, show=False)
    plt.title(titulo, fontsize=11)
    plt.gcf().set_size_inches(9, 4.5)
    plt.tight_layout()
    plt.show()
""")

md(r"""
En clasificación se explicó con el KNN un mes Alto bien clasificado (febrero de 2021, probabilidad de Alto de 1.00) y otro que no reconoció (enero de 2009, probabilidad de 0.00). En el primero todo empuja hacia arriba: la tasa de desempleo de las áreas (19.4 %), los promedios móviles de 3, 6 y 12 meses (entre 15.1 % y 17.0 %) y el último mes (17.6 %) están muy por encima de lo normal, y cada variable aporta entre 0.05 y 0.08 a la probabilidad. En el segundo ocurre lo contrario: el desempleo de diciembre de 2008 era de 10.8 % (y de 10.6 % el promedio de tres meses), nada en los últimos meses anunciaba una crisis, y todas las variables empujan la probabilidad hacia abajo (la de las áreas, la tendencia, el promedio de 12 meses y el rezago de 12 meses restan entre 0.05 y 0.07 cada una). El mes fue Alto por un salto estacional de enero que ningún modelo basado en la historia reciente podía prever: el modelo del primer pliegue solo había visto los años 2002-2005.

En regresión se explicó el Ridge para el mes con el error más pequeño (agosto de 2017: observado 9.52 %, predicho 9.51 %) y para el de error más grande (abril de 2020: observado 20.49 %, predicho 11.98 %). En el primero todas las variables bajan la predicción desde el promedio de 12.3 % hasta 9.5 %: el rezago de 12 meses (9.2 %) resta 0.73 puntos y el promedio móvil de 6 meses, el índice de tendencia, el último mes y el mes del año restan entre 0.25 y 0.28 cada uno. En el segundo, el Ridge sí reacciona a la caída de la tasa de ocupación de las áreas (52.6 %, que suma 0.48 puntos), pero esa señal se diluye entre el resto: el modelo ve un último mes de 13.2 % y un historial tranquilo, y como es lineal y con regularización fuerte no puede convertir una caída de la ocupación en un salto de 7 puntos, mayor que cualquiera de los que había en el entrenamiento (el máximo era de 4.7). La diferencia entre las dos explicaciones es que, cuando el error es pequeño, todas las variables apuntan en el mismo sentido y con intensidad parecida, y cuando es grande el patrón que explicaría el resultado no está en lo aprendido.
""")

md(r"""
### SHAP frente a LIME en XGBoost
""")

code(r"""
# XGBoost entrenado con todos los meses: se compara la importancia media de SHAP con la que obtiene LIME sobre 40 meses repartidos
# a lo largo de la serie (LIME ajusta un modelo lineal local alrededor de cada mes).
muestra = ds.X.iloc[np.linspace(0, len(ds.X) - 1, 40).astype(int)]
filas, importancias = [], {}
for tarea, nombre in [('clf', 'Clasificación (clase Alto)'), ('reg', 'Regresión')]:
    registro = clasificacion if tarea == 'clf' else regresion
    fila = registro[registro['model'] == 'XGBoost'].sort_values('f1_macro_mean' if tarea == 'clf' else 'rmse_mean',
                                                                 ascending=(tarea == 'reg')).iloc[0]
    parametros = A.load_history(fila['run_id'])['fold_results'][-1]['best_params']
    est, esc, X_ent = A.fit_on_all(tarea, 'XGBoost', fila['balancing'], parametros, ds)
    clase = A.ALTO if tarea == 'clf' else None
    _, valores = I.compute_shap(est, esc, X_ent, muestra, tarea)
    shap_imp = I.mean_abs_shap(valores, ds.X.columns, clase)
    lime_imp = I.lime_importance(est, esc, X_ent, muestra, tarea, class_index=clase, n_instances=len(muestra))
    acuerdo = I.agreement(shap_imp, lime_imp, top=5)
    importancias[nombre] = (shap_imp, lime_imp)
    filas.append({'Tarea': nombre, 'Correlación de Spearman': acuerdo['spearman'],
                  'Variables en común entre las 5 primeras': acuerdo['top_en_comun']})
for nombre, (shap_imp, lime_imp) in importancias.items():
    print(f'{nombre}\n  Primeras 5 según SHAP: {", ".join(shap_imp.index[:5])}\n  Primeras 5 según LIME: {", ".join(lime_imp.index[:5])}')
pd.DataFrame(filas).set_index('Tarea').round(3)
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (nombre, (shap_imp, lime_imp)) in zip(axes, importancias.items()):
    comparacion = pd.concat([(shap_imp / shap_imp.sum()).rename('SHAP'), (lime_imp / lime_imp.sum()).rename('LIME')], axis=1)
    comparacion = comparacion.loc[comparacion['SHAP'].sort_values(ascending=False).index[:8]][::-1]
    posiciones = np.arange(len(comparacion))
    ax.barh(posiciones - 0.2, comparacion['SHAP'], height=0.4, color=plots.BLUE, label='SHAP')
    ax.barh(posiciones + 0.2, comparacion['LIME'], height=0.4, color=plots.RED, label='LIME')
    ax.set_yticks(posiciones, comparacion.index, fontsize=8)
    ax.set_xlabel('Importancia relativa')
    ax.set_title(f'XGBoost: {nombre}')
    ax.legend()
plt.tight_layout()
plt.show()
""")

md(r"""
Para XGBoost, SHAP y LIME dan rankings muy parecidos: la correlación de Spearman entre las dos importancias es de 0.95 en clasificación (categoría Alto) y de 0.82 en regresión, y las cinco variables principales son las mismas con los dos métodos en las dos tareas. En clasificación encabezan la lista la tasa de desempleo de las áreas y el promedio móvil de 3 meses (con el orden intercambiado entre métodos), seguidos de los promedios y rezagos de 6 y 12 meses; en regresión, el último mes, la tasa de desempleo de las áreas, el promedio de 3 meses y el mes del año, con la tasa de participación de las áreas en quinto lugar. Las diferencias aparecen en los detalles: en regresión, LIME le da más peso a la tasa de desempleo de las áreas (0.19 frente a 0.14 de SHAP) y menos al mes del año (0.11 contra 0.14), y en clasificación le da menos a la tasa de desempleo de las áreas (0.20 contra 0.25) y más al promedio de 12 meses (0.06 contra 0.04).

Son las diferencias que cabe esperar de dos métodos distintos. SHAP reparte la predicción entre las variables con valores de Shapley, que para árboles se calculan de forma exacta y cumplen eficiencia, simetría y aditividad; LIME ajusta una regresión lineal ponderada alrededor de cada mes con observaciones perturbadas al azar, así que su explicación es local, depende del tamaño de la vecindad y del muestreo, y solo captura lo lineal. Los dos pueden divergir cuando las variables están muy correlacionadas (aquí los rezagos y los promedios móviles de la tasa lo están, y el reparto del peso entre ellas cambia de un método a otro), cuando el modelo tiene interacciones o saltos fuertes cerca de un umbral que una recta no representa, y cuando se explica una observación aislada en lugar del promedio de muchas, como se hizo aquí con 40 meses. Que coincidan en el ranking promedio es una buena señal de que la explicación no depende del método, pero no garantiza que coincidan en cada mes.
""")

md(r"""
### ¿Aportan algo los indicadores de las áreas metropolitanas?
""")

code(r"""
# Se repite el ajuste de cada pliegue externo con los mismos hiperparámetros, con y sin las tres variables de las 13 áreas
filas = []
for tarea, tabla, metrica in [('clf', mejores_clf, 'F1-macro'), ('reg', mejores_reg, 'RMSE')]:
    for modelo, fila in tabla.iterrows():
        r = A.ablation_area_features(tarea, modelo, fila['balancing'], fila['optimizer'], ds)
        filas.append({'Tarea': 'Clasificación' if tarea == 'clf' else 'Regresión', 'Modelo': A.MODEL_LABELS[modelo], 'Métrica': metrica,
                      'Con indicadores de área': r['con_areas'], 'Sin indicadores de área': r['sin_areas'],
                      'Diferencia': r['con_areas'] - r['sin_areas']})
ablacion = pd.DataFrame(filas).set_index(['Tarea', 'Modelo'])
ablacion.round(3)
""")

md(r"""
Para saber si los indicadores de las 13 áreas metropolitanas aportan información sobre el desempleo nacional, se repitió el ajuste de cada pliegue externo con los mismos hiperparámetros, quitando las tres variables de área. La diferencia es el efecto de tenerlas (en el F1-macro, positiva es mejor; en el RMSE, negativa es mejor). En clasificación mejoran cuatro de los siete modelos (la regresión logística en 0.038, KNN y Naive Bayes en 0.024 y SVM en 0.009), no cambian dos (Random Forest y XGBoost) y empeora el árbol de decisión (−0.012). En regresión, tener las variables de área reduce el RMSE en cinco de siete modelos: Ridge baja de 1.388 a 1.317, SVR de 1.406 a 1.314, XGBoost de 1.726 a 1.568, el árbol de 1.783 a 1.751 y Lasso de 1.707 a 1.692. Random Forest no cambia (1.510) y solo KNN empeora (de 1.357 a 1.373).

Esto quiere decir que las áreas ayudan, y bastante en algunos modelos: entre 5 % y 7 % menos de RMSE en Ridge y SVR y cerca de 9 % en XGBoost. Es coherente con lo que mostró SHAP, donde la tasa de desempleo de las áreas aparece entre las variables más importantes en Ridge y en XGBoost, y respalda la primera hipótesis secundaria: el desempleo de las 13 áreas es un predictor útil del desempleo nacional. La evidencia, sin embargo, es moderada: es una sola partición temporal, no se hizo una prueba formal y en clasificación las diferencias son de unas pocas centésimas.
""")
