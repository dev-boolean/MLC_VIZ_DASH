# -*- coding: utf-8 -*-
"""Capítulo: resultados de regresión."""

CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


md(r"""
## Resultados de regresión

En regresión se evalúan los siete modelos con el RMSE y el MAE, que son los errores del pronóstico en puntos porcentuales de desempleo. El RMSE castiga más los errores grandes (como los de 2020) y el MAE es el error absoluto promedio, así que mirar los dos ayuda a distinguir un modelo que falla poco en todas partes de uno que acierta casi siempre pero se equivoca mucho en unos pocos meses. Igual que en clasificación, para cada modelo se muestra su mejor optimizador según el RMSE promedio de los cinco pliegues externos, con la media ± desviación estándar entre pliegues.

Además de las métricas por pliegue, se juntaron los pronósticos de los cinco pliegues en una sola serie de 240 meses (de enero de 2006 a diciembre de 2025), todos hechos con información anterior. Con esa serie se calcularon las pruebas sobre los residuos, porque 240 observaciones dan mucha más potencia que los 48 meses de un solo pliegue. En el cuadro, la prueba de White (1980) contrasta que la varianza de los errores sea constante (se hace contra el valor predicho, porque los modelos no son lineales en las variables) y la de BDS (Brock et al., 1996) que los errores sean independientes entre sí.
""")

code(r"""
regresion = master[master['task'] == 'reg']
mejores_reg = A.best_per_model(master, 'reg')   # mejor combinación de cada modelo según el RMSE promedio

# Errores de pronóstico (observado menos predicho) de los 240 meses fuera de muestra, apilados
errores_reg = {m: A.pooled_regression_errors(f['run_id']) for m, f in mejores_reg.iterrows()}
supuestos = {m: A.residual_tests(e['error'], e['y_pred']) for m, e in errores_reg.items()}

def formato_p(valor):
    # los p-valores menores que 0.0001 se muestran como "< 0.0001" en vez de 0.0000
    return '< 0.0001' if valor < 0.0001 else f'{valor:.4f}'

cuadro_reg = pd.DataFrame({
    'Optimizador': mejores_reg['optimizer'].map(A.OPTIMIZER_LABELS),
    'RMSE': [A.mean_std(f, 'rmse') for _, f in mejores_reg.iterrows()],
    'MAE': [A.mean_std(f, 'mae') for _, f in mejores_reg.iterrows()],
    'p-valor (White)': [formato_p(supuestos[m]['white_p']) for m in mejores_reg.index],
    'p-valor (BDS)': [formato_p(supuestos[m]['bds_p']) for m in mejores_reg.index],
})
cuadro_reg.index = [A.MODEL_LABELS[m] for m in mejores_reg.index]
cuadro_reg
""")

md(r"""
El mejor RMSE promedio entre pliegues es el de **Ridge (1.196 ± 0.552)**, seguido de cerca por SVR (1.213) y KNN (1.251); XGBoost (1.266) y Random Forest (1.274) vienen después, y el árbol de decisión (1.476) y Lasso (1.510) quedan de últimos. El MAE ordena casi igual, pero con menos distancia entre los cinco primeros (entre 0.87 y 0.89 puntos porcentuales). Que el RMSE sea bastante mayor que el MAE (1.2 contra 0.9) indica que hay errores grandes concentrados en pocos meses, que son los de 2020. La desviación entre pliegues es muy alta (de 0.5 a 0.9 puntos de RMSE), de modo que las diferencias entre los primeros cinco modelos están dentro del ruido.

Que el mejor modelo sea uno lineal tiene sentido en una serie tan persistente: el desempleo del mes siguiente se parece mucho al de este mes más una pequeña corrección, y los árboles, que predicen siempre dentro del rango de valores que vieron al entrenar, no extrapolan cuando la serie sale de ese rango. En las dos últimas columnas se ve que ningún modelo cumple los supuestos: el p-valor de la prueba de White es menor que 0.05 en los siete (la varianza de los errores no es constante) y el de BDS es menor que 0.0001 en los siete (los errores no son independientes). Esto se revisa en la sección de residuos.
""")

code(r"""
# Modelos de referencia a un mes: persistencia, mismo mes del año anterior y el SARIMA(0,1,1)(0,1,1)[12] del capítulo anterior
referencias_reg = A.regression_baselines(ds)
assert (referencias_reg['idx'].values == errores_reg[mejores_reg.index[0]]['idx']).all()

filas = []
for modelo in mejores_reg.index:
    e = errores_reg[modelo]['error']
    filas.append({'Modelo': A.MODEL_LABELS[modelo], 'Tipo': 'Machine learning', 'RMSE agrupado': np.sqrt(np.mean(e ** 2)), 'MAE agrupado': np.mean(np.abs(e))})
for nombre in ['Persistencia', 'Estacional', 'SARIMA']:
    e = referencias_reg['y_true'] - referencias_reg[nombre]
    filas.append({'Modelo': nombre if nombre != 'Estacional' else 'Mismo mes del año anterior', 'Tipo': 'Referencia',
                  'RMSE agrupado': np.sqrt(np.mean(e ** 2)), 'MAE agrupado': np.mean(np.abs(e))})
comparacion_reg = pd.DataFrame(filas).set_index('Modelo').sort_values('RMSE agrupado')
comparacion_reg.round(3)
""")

code(r"""
# El choque de 2020 domina el error total, así que también se separa lo que pasó antes, durante y después de la pandemia
fechas_oof = pd.to_datetime(fechas_objetivo.iloc[errores_reg[mejores_reg.index[0]]['idx']].values)
periodos = {'Antes de la pandemia (2006-2019)': fechas_oof < '2020-01-01',
            'Pandemia (ene 2020 - jun 2021)': (fechas_oof >= '2020-01-01') & (fechas_oof <= '2021-06-30'),
            'Después (jul 2021 - 2025)': fechas_oof > '2021-06-30'}
print({nombre: int(mascara.sum()) for nombre, mascara in periodos.items()}, 'meses en cada periodo')

errores_modelos = {A.MODEL_LABELS[m]: np.asarray(e['error']) for m, e in errores_reg.items()}
for nombre in ['Persistencia', 'Estacional', 'SARIMA']:
    errores_modelos[nombre] = (referencias_reg['y_true'] - referencias_reg[nombre]).values
por_periodo = pd.DataFrame({
    nombre: {**{periodo: np.sqrt(np.mean(e[mascara] ** 2)) for periodo, mascara in periodos.items()}, 'Todo el periodo': np.sqrt(np.mean(e ** 2))}
    for nombre, e in errores_modelos.items()}).T
por_periodo.sort_values('Todo el periodo').round(3)
""")

md(r"""
Este es el resultado menos favorable para el machine learning: **ninguno de los siete modelos supera al SARIMA(0,1,1)(0,1,1)[12] del capítulo anterior**. Sobre los 240 meses, el SARIMA tiene un RMSE de 0.995 y un MAE de 0.562; el mejor modelo de machine learning, SVR, queda en 1.314 y 0.868, y la simple persistencia (pronosticar que el mes siguiente será igual al actual) llega a 1.293 de RMSE y 0.905 de MAE, a la par de los mejores modelos. La regla estacional (el mismo mes del año anterior) es la peor en el total (2.107).

El orden total, sin embargo, esconde mucho, porque los errores de 2020 pesan demasiado. Separado por periodos, el SARIMA es el mejor antes de la pandemia (168 meses, RMSE de 0.509) y después de ella (54 meses, 0.728). Entre los modelos de machine learning, los de árboles son los mejores antes de la pandemia (XGBoost con 0.831 y Random Forest con 0.874, cerca de la regla estacional, que da 0.818) y los peores durante ella (4.93 y 4.56, frente a 3.03 del SARIMA), porque predicen siempre dentro del rango de valores que vieron al entrenar y en 2020 la tasa se disparó hasta 22 %. Ridge y SVR son lo contrario: peores en los meses tranquilos (entre 1.02 y 1.05) pero más robustos al choque (3.22 y 3.10). Durante la pandemia (18 meses) el mejor es la persistencia (2.43), simplemente porque después de que la tasa salta se queda alta y repetir el último valor acierta. Después de la pandemia, KNN (0.757) y XGBoost (0.821) se acercan al SARIMA (0.728).
""")

code(r"""
# Los pronósticos fuera de muestra del mejor modelo de machine learning, del SARIMA y de la persistencia
mejor_reg = mejores_reg.index[0]
e = errores_reg[mejor_reg]
fechas_oof = fechas_objetivo.iloc[e['idx']].values

fig, ax = plt.subplots(figsize=(13, 5))
ax.axvspan(pd.Timestamp('2020-01-01'), pd.Timestamp('2021-06-01'), alpha=0.15, color='grey')
ax.text(pd.Timestamp('2020-09-01'), 20.5, 'COVID-19', size=9, color='grey')
ax.plot(fechas_oof, e['y_true'], color=plots.BLUE, linewidth=1.2, label='Observado')
ax.plot(fechas_oof, e['y_pred'], color='#e07b00', linewidth=1.0, linestyle='--', label=f'{A.MODEL_LABELS[mejor_reg]} (mejor modelo de ML)')
ax.plot(fechas_oof, referencias_reg['SARIMA'], color=plots.GREEN, linewidth=1.0, label='SARIMA(0,1,1)(0,1,1)[12]')
ax.plot(fechas_oof, referencias_reg['Persistencia'], color=plots.GREY, linewidth=0.8, alpha=0.8, label='Persistencia')
ax.set_title('Pronósticos a un mes fuera de muestra (2006-2025)')
ax.set_xlabel('Fecha')
ax.set_ylabel('Tasa de desempleo nacional (%)')
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()
""")

md(r"""
En la figura se ve por qué gana el SARIMA. La serie tiene un patrón estacional muy marcado, con picos al comienzo de cada año y valles hacia el final, montado sobre una tendencia que baja de 2006 a 2015. El SARIMA (línea verde) reproduce esos picos casi mes a mes. Ridge, el mejor modelo de machine learning (línea naranja punteada), sigue el nivel y la tendencia pero casi no oscila: es una versión suavizada de la serie que no llega a los picos estacionales, lo que se refleja en los residuos autocorrelacionados que se analizan más adelante. La persistencia (línea gris) es la misma serie corrida un mes, así que es competitiva en los tramos suaves pero siempre llega tarde a los cambios bruscos. En abril de 2020, cuando la tasa pasó de 13.2 % a 20.5 %, ninguno de los tres lo anticipó: la persistencia se pone al día un mes después, y Ridge y el SARIMA solo llegan a cerca de 17-19 % en el pico.
""")

md(r"""
### Entrenamiento, validación y prueba
""")

code(r"""
# Se reconstruye el modelo final del último pliegue externo (mismos hiperparámetros que eligió la búsqueda) y se grafican
# sus predicciones: entrenamiento (primeros 75 % del tramo), validación (último 25 %, el que usó la búsqueda interna) y prueba.
fila = mejores_reg.iloc[0]
ajuste = A.refit_fold('reg', mejor_reg, 'none', fila['optimizer'], fold=4, ds=ds)
filas_usadas = np.concatenate([np.arange(len(ajuste['X_train'])), ajuste['test_idx']])
predicho = ajuste['est'].predict(I.scale(ajuste['scaler'], ds.X.iloc[filas_usadas]))
n_entrenamiento = int(0.75 * len(ajuste['X_train']))
n_validacion = len(ajuste['X_train']) - n_entrenamiento

fig = plots.plot_train_val_test(
    fechas_objetivo.iloc[filas_usadas].values, ds.y_reg.iloc[filas_usadas].values, predicho,
    n_entrenamiento, n_validacion, f'{A.MODEL_LABELS[mejor_reg]}: observado y predicho en entrenamiento, validación y prueba')
plt.show()

for nombre, tramo in [('Entrenamiento', slice(0, n_entrenamiento)), ('Validación', slice(n_entrenamiento, n_entrenamiento + n_validacion)),
                      ('Prueba', slice(n_entrenamiento + n_validacion, None))]:
    obs, est = ds.y_reg.iloc[filas_usadas].values[tramo], predicho[tramo]
    print(f'{nombre:13s} RMSE = {np.sqrt(np.mean((obs - est) ** 2)):.3f}   MAE = {np.mean(np.abs(obs - est)):.3f}')
""")

md(r"""
La figura reconstruye el modelo Ridge del último pliegue externo, entrenado con 239 meses, y separa su entrenamiento (los primeros 179 meses), su validación (los 60 siguientes, que es aproximadamente el tramo con el que la búsqueda interna lo evaluó en su último pliegue) y la prueba (2022-2025). Las predicciones siguen bien la tendencia descendente y el nivel de cada tramo, pero suavizan los picos de comienzo de año: la línea naranja oscila mucho menos que la observada. Los errores son un RMSE de 0.918 en entrenamiento, 1.505 en validación y 0.876 en prueba. Lo que más llama la atención es que el modelo no tiene problema de sobreajuste (el error en prueba es incluso menor que en entrenamiento) y que el tramo difícil es la validación, porque contiene el choque de 2020-2021. Es decir, el error de cada tramo depende sobre todo de qué tan agitado fue ese periodo y no de si los datos se vieron durante el entrenamiento, y el desempeño de Ridge (con un alpha de 70.9, una regularización fuerte) está limitado más por sesgo que por varianza.
""")

md(r"""
### Análisis de residuos

Sobre los 240 errores de pronóstico de cada modelo se revisan los supuestos con varias herramientas. La independencia serial se mira con la función de autocorrelación (ACF) y con la prueba de Ljung-Box (Ljung & Box, 1978); la normalidad, con el histograma, el gráfico cuantil-cuantil y las pruebas de Shapiro-Wilk (Shapiro & Wilk, 1965) y de Jarque-Bera (Jarque & Bera, 1987); y la varianza constante y la independencia no lineal, con las pruebas de White y BDS que ya se vieron. Además, el gráfico de los residuos contra los valores predichos sirve para detectar formas sistemáticas. La figura es la del mejor modelo de machine learning y la tabla reúne las seis pruebas de los siete modelos.
""")

code(r"""
fig = plots.plot_residuals(
    fechas_oof, e['error'], e['y_pred'],
    {12: supuestos[mejor_reg]['ljungbox12_p'], 24: supuestos[mejor_reg]['ljungbox24_p']},
    f'Residuos de {A.MODEL_LABELS[mejor_reg]} (240 pronósticos fuera de muestra)')
plt.show()

pruebas = pd.DataFrame(supuestos).T
pruebas.columns = ['White (varianza constante)', 'BDS (independencia)', 'Ljung-Box 12 rezagos', 'Ljung-Box 24 rezagos',
                   'Shapiro-Wilk (normalidad)', 'Jarque-Bera (normalidad)']
pruebas.index = [A.MODEL_LABELS[m] for m in pruebas.index]
pruebas.apply(lambda columna: columna.map(formato_p))
""")

code(r"""
# Cifras que se usan en la lectura de los residuos de Ridge (el mejor modelo de machine learning)
from statsmodels.tsa.stattools import acf

error = np.asarray(e['error'])
prediccion = np.asarray(e['y_pred'])
fecha = pd.to_datetime(fechas_oof)
tramos = {'2006-2009': ('2006-01', '2009-12'), '2010-2013': ('2010-01', '2013-12'), '2014-2017': ('2014-01', '2017-12'),
          '2018-2021': ('2018-01', '2021-12'), '2022-2025': ('2022-01', '2025-12')}
por_tramo = pd.DataFrame({nombre: {'Error medio': error[(fecha >= a) & (fecha <= b)].mean(),
                                   'RMSE': np.sqrt(np.mean(error[(fecha >= a) & (fecha <= b)] ** 2))}
                          for nombre, (a, b) in tramos.items()}).T
print('Error de Ridge en cada pliegue externo (observado menos predicho):')
print(por_tramo.round(2).to_string())
print()
print('Meses con error mayor que 4 puntos:', [(str(f.date()), round(float(v), 2)) for f, v in zip(fecha, error) if v > 4])
despues = (fecha >= '2021-01-01') & (fecha <= '2022-12-31')
print(f'Errores de 2021-2022: media {error[despues].mean():.2f}, negativos en {int((error[despues] < 0).sum())} de {int(despues.sum())} meses')
print(f'Desviación estándar de los errores cuando se predice más de 14 %: {error[prediccion > 14].std():.2f} ({int((prediccion > 14).sum())} meses); '
      f'entre 9 % y 12 %: {error[(prediccion >= 9) & (prediccion <= 12)].std():.2f}')
autocorrelacion = acf(error, nlags=24)
print(f'ACF de los errores: rezago 1 = {autocorrelacion[1]:.2f}, rezago 12 = {autocorrelacion[12]:.2f}, rezago 24 = {autocorrelacion[24]:.2f}')
""")

md(r"""
Los residuos del mejor modelo de machine learning (Ridge) están lejos de ser ruido blanco, y las pruebas lo confirman en los siete modelos. En el tiempo se ven tres cosas. En 2006-2009 los errores son casi siempre positivos (+1.16 puntos en promedio): el modelo, entrenado solo con los primeros 47 meses, pronostica un desempleo más bajo que el real. Entre 2010 y 2017 se mantienen cerca de cero (promedios de −0.21 y 0.00 en los dos pliegues). Y en 2020 aparece el pico: cuatro meses con errores de más de 4 puntos, tres de ellos de abril a junio de 2020 (8.5, 7.1 y 4.7) y el cuarto en enero de 2009 (4.3). Después de la pandemia los errores son negativos en 22 de los 24 meses de 2021 y 2022 (−0.89 en promedio), porque el modelo sigue esperando un desempleo más alto que el que ocurre.

La varianza de los errores no es constante: en el gráfico de residuos contra valores predichos se ve que son mucho más dispersos cuando el modelo predice más de 14 % (desviación estándar de 2.66 en esos 16 meses) que cuando predice entre 9 % y 12 % (1.15), y la prueba de White rechaza la varianza constante en los siete modelos (p entre menos de 0.0001 y 0.027). El histograma tiene una cola derecha larga y el QQ-plot se despega de la recta en ese extremo; Shapiro-Wilk y Jarque-Bera rechazan la normalidad en todos los modelos (p < 0.0001). Y los errores están autocorrelacionados: la ACF empieza en 0.60 en el primer rezago y decae despacio, con un pico en el rezago 12 (0.28) que indica que queda estacionalidad sin explicar; Ljung-Box con 12 y 24 rezagos y BDS dan p < 0.0001 en los siete modelos.

Estos patrones indican que los modelos no capturan del todo la estacionalidad ni la dinámica de corto plazo, que es justo la estructura que un SARIMA sí modela: en el capítulo anterior el diagnóstico de residuos del SARIMA dio p-valores de Ljung-Box de 0.79 y 0.77 (aunque dentro de la muestra), y aquí no se logra nada parecido. Para que un modelo de machine learning aproveche esa estructura habría que enriquecer las variables (por ejemplo, más armónicos estacionales o diferencias en lugar de niveles), y mientras tanto los intervalos de pronóstico que suponen errores normales no serían confiables.
""")
