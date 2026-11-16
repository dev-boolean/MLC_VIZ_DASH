# -*- coding: utf-8 -*-
"""Capítulo: comparación estadística entre modelos."""

CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


md(r"""
## Comparación estadística entre modelos

Con 140 modelos hay 9730 pares posibles. Si se aplicara una prueba como DeLong o Diebold-Mariano a cada uno, con un nivel de significancia de 0.05, solo por azar se esperarían cerca de 487 falsos positivos. Por eso no se comparan todos los pares: primero se hace una prueba global que dice si hay alguna diferencia entre los modelos, y solo después se pasa a comparaciones dirigidas entre un grupo pequeño de candidatos, corrigiendo los p-valores por comparaciones múltiples. Las pruebas de regresión se hicieron sobre los 240 errores mensuales fuera de muestra (no sobre las cinco métricas por pliegue), y las de clasificación sobre los pliegues externos y sobre las predicciones apiladas.

La secuencia en clasificación es la prueba ómnibus de Friedman (1937), el post-hoc de Nemenyi (1963) con el diagrama de diferencia crítica (Demšar, 2006), las comparaciones dirigidas de DeLong (DeLong et al., 1988) con el ajuste de Holm (1979) o el de Benjamini y Hochberg (1995), y el tamaño del efecto con el delta de Cliff (1993). En regresión es el Model Confidence Set (Hansen et al., 2011) sobre todos los modelos, el test de Giacomini y White (2006) entre los finalistas (el de Clark y West, 2007, no aplica porque ningún modelo es un caso particular de otro), el de Diebold y Mariano (1995) con la corrección de Harvey et al. (1997) como confirmación puntual, con intervalos de bootstrap estacionario (Politis & Romano, 1994), y la d de Cohen (1988) como tamaño del efecto. Para controlar el *data snooping* se agregan la prueba de habilidad predictiva superior de Hansen (2005) y el Reality Check de White (2000).
""")

md(r"""
### Clasificación
""")

code(r"""
folds = A.load_folds()
f1_folds = folds[(folds['task'] == 'clf') & (folds['metric'] == 'f1_macro')]

# (a) Versión estricta: el mejor resultado de cada uno de los siete modelos en los cinco pliegues externos
estricto = pd.DataFrame({
    A.MODEL_LABELS[m]: folds[(folds['run_id'] == f['run_id']) & (folds['metric'] == 'f1_macro')].sort_values('fold')['value'].values
    for m, f in mejores_clf.iterrows()})
friedman_estricto = ST.friedman_test(estricto)

# (b) Versión ampliada: se usan las 16 variantes (balanceo y optimizador) de cada modelo en cada pliegue (80 bloques),
#     para no depender de haber elegido solo la mejor variante de cada modelo
ampliado = f1_folds.pivot_table(index=['balancing', 'optimizer', 'fold'], columns='model', values='value')
ampliado.columns = [A.MODEL_LABELS[m] for m in ampliado.columns]
friedman_ampliado = ST.friedman_test(ampliado)

# (c) El universo completo: los 112 modelos de clasificación en los cinco pliegues
universo = f1_folds.pivot_table(index='fold', columns='run_id', values='value')
friedman_universo = ST.friedman_test(universo)

pd.DataFrame({
    'Bloques': [len(estricto), len(ampliado), len(universo)],
    'Modelos comparados': [estricto.shape[1], ampliado.shape[1], universo.shape[1]],
    'Estadístico chi2': [friedman_estricto.statistic, friedman_ampliado.statistic, friedman_universo.statistic],
    'Grados de libertad': [friedman_estricto.df, friedman_ampliado.df, friedman_universo.df],
    'p-valor': [friedman_estricto.p_value, friedman_ampliado.p_value, friedman_universo.p_value],
}, index=['(a) Mejor variante de cada modelo', '(b) Todas las variantes de cada modelo', '(c) Los 112 modelos']).round(4)
""")

md(r"""
La prueba de Friedman se aplicó de tres maneras, porque la respuesta depende de cuántos bloques se usen. Con los siete mejores modelos y los cinco pliegues (versión estricta) no se rechaza la igualdad: χ² = 3.68 con 6 grados de libertad y p = 0.72. Pero cinco bloques dan muy poca potencia, y además solo se está comparando a los ganadores de cada modelo. Cuando se usan las 16 variantes de cada modelo en cada pliegue (80 bloques) el resultado es claro: χ² = 47.36, 6 grados de libertad y p < 0.0001, es decir, no todos los modelos rinden igual. Con los 112 modelos y los cinco pliegues el resultado es marginal (χ² = 137.0, 111 grados de libertad, p = 0.048). Se concluye que sí hay diferencias entre los modelos cuando se mira el panorama completo, pero que con cinco pliegues no hay potencia para distinguir entre los mejores. Por eso el análisis post-hoc se hace con la versión ampliada.
""")

code(r"""
# Prueba post-hoc de Nemenyi y diagrama de diferencia crítica sobre la versión ampliada
cd_clf = ST.critical_difference(ampliado.shape[1], len(ampliado))
print('Rango promedio de cada modelo (menor es mejor):')
print(friedman_ampliado.avg_ranks.round(2).to_string())
print(f'Diferencia crítica de Nemenyi (alfa = 0.05): {cd_clf:.2f}')
fig, ax = plt.subplots(figsize=(10, 3.4))
ST.plot_cd_diagram(friedman_ampliado.avg_ranks, cd_clf, ax=ax, title='Diagrama de diferencia crítica (clasificación, 80 bloques)')
plt.show()

nemenyi_clf = ST.nemenyi_posthoc(ampliado)
print('p-valores de Nemenyi entre cada par de modelos:')
nemenyi_clf.round(3)
""")

md(r"""
Con 7 modelos y 80 bloques, la diferencia crítica de Nemenyi es de 1.01 en rango promedio. Los rangos (menor es mejor) son 3.39 para Random Forest, 3.41 para KNN, 3.58 para XGBoost, 3.64 para Naive Bayes, 4.37 para el árbol de decisión, 4.57 para la regresión logística y 5.04 para SVM. Los cuatro primeros no se pueden distinguir entre sí, y el árbol de decisión tampoco se distingue de Random Forest (4.37 − 3.39 = 0.98, menos que 1.01). SVM es el único claramente peor que el primer grupo: se diferencia de Random Forest, KNN, XGBoost (p < 0.001) y Naive Bayes (p = 0.001), pero no del árbol ni de la regresión logística. La regresión logística es significativamente peor que KNN (p = 0.013) y que Random Forest (p = 0.010), aunque no que XGBoost (p = 0.059).

En otras palabras, KNN, Random Forest, XGBoost y Naive Bayes son equivalentes en rango, a pesar de que KNN tenga el mejor F1-macro en el cuadro de resultados: su ventaja en el mejor de sus 16 variantes no es consistente cuando se miran todas. Este diagrama reemplaza a las 21 comparaciones por pares entre los siete modelos, y para los 112 modelos reemplazaría a las 6216 que habría que hacer.
""")

code(r"""
# Comparaciones dirigidas con DeLong entre los tres modelos con mejor rango: AUC de la clase Alto sobre las predicciones agrupadas
mejores_rango = list(friedman_ampliado.avg_ranks.index[:3])
inverso = {A.MODEL_LABELS[m]: m for m in mejores_clf.index}
y_alto = oof_clf[inverso[mejores_rango[0]]]['y_true']
probabilidad_alto = {nombre: oof_clf[inverso[nombre]]['y_proba'][:, A.ALTO] for nombre in mejores_rango}

filas = []
for i in range(3):
    for j in range(i + 1, 3):
        a, b = mejores_rango[i], mejores_rango[j]
        r = ST.delong_test(y_alto, probabilidad_alto[a], probabilidad_alto[b], A.ALTO)
        delta = ST.cliffs_delta(estricto[a], estricto[b])
        filas.append({'Modelo A': a, 'Modelo B': b, 'AUC A': r['auc_a'], 'AUC B': r['auc_b'], 'Diferencia': r['diff'],
                      'IC 95 %': f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]", 'p crudo': r['p_value'],
                      'Delta de Cliff (F1 por pliegue)': delta})
delong = pd.DataFrame(filas)
delong['p ajustado (Holm)'] = ST.adjust_pvalues(delong['p crudo'].values, 'holm')
delong['p ajustado (BH)'] = ST.adjust_pvalues(delong['p crudo'].values, 'fdr_bh')
delong.set_index(['Modelo A', 'Modelo B']).round(3)
""")

md(r"""
Entre los tres modelos con mejor rango (Random Forest, KNN y XGBoost) se compararon los AUC de la categoría Alto con la prueba de DeLong sobre las 240 predicciones apiladas. Las diferencias son pequeñas y ninguna es significativa: KNN supera a XGBoost en 0.029 de AUC (IC 95 % de −0.017 a 0.075, p = 0.212), Random Forest queda 0.035 por debajo de KNN (IC de −0.097 a 0.028, p = 0.275) y 0.006 por debajo de XGBoost (p = 0.844). Con la corrección de Holm los p-valores ajustados son 0.635, 0.844 y 0.635 (con Benjamini-Hochberg, 0.413, 0.844 y 0.413), así que no hay evidencia de que alguno de los tres distinga mejor los meses Alto. Hay que tener presente que DeLong supone observaciones independientes y los pronósticos mensuales están autocorrelacionados, de modo que los p-valores son, si acaso, demasiado optimistas, y aun así no resultan significativos.

El delta de Cliff sobre el F1-macro por pliegue sí muestra efectos medianos o grandes (−0.24 entre Random Forest y KNN, 0.44 entre Random Forest y XGBoost y 0.48 entre KNN y XGBoost), pero con solo cinco pliegues ese tamaño de efecto es muy inestable y apenas confirma el orden que ya mostraba el cuadro: KNN, Random Forest y XGBoost.
""")

code(r"""
# Intervalos de confianza al 95 % del AUC de la clase Alto y del RMSE de cada modelo de regresión, calculados de dos maneras:
# con el bootstrap estacionario (bloques de 12 meses en promedio), que respeta la dependencia entre meses seguidos, y con el
# BCa de remuestreo independiente, que la ignora y se incluye solo para ver cuánto cambia el intervalo.
from scipy.stats import bootstrap

def auc_alto(muestra):
    # AUC de la clase Alto; si una remuestra no tiene meses Alto el AUC no está definido
    return roc_auc_score(muestra[:, 0], muestra[:, 1]) if 0 < muestra[:, 0].sum() < len(muestra) else np.nan

def bca_auc(indicadora, probabilidad):
    r = bootstrap((indicadora, probabilidad), lambda a, b: auc_alto(np.column_stack([a, b])), paired=True, vectorized=False,
                  n_resamples=2000, method='BCa', random_state=config.SEED)
    return r.confidence_interval.low, r.confidence_interval.high

def bca_rmse(cuadrados):
    r = bootstrap((cuadrados,), lambda x, axis: np.sqrt(np.mean(x, axis=axis)), n_resamples=2000, method='BCa', random_state=config.SEED)
    return r.confidence_interval.low, r.confidence_interval.high

filas = []
for modelo, p in oof_clf.items():
    indicadora, probabilidad = (p['y_true'] == A.ALTO).astype(int), p['y_proba'][:, A.ALTO]
    punto, bajo, alto = ST.bootstrap_interval(np.column_stack([indicadora, probabilidad]), auc_alto, mean_block=12, n_boot=1000)
    bca_bajo, bca_alto = bca_auc(indicadora, probabilidad)
    filas.append({'Modelo': A.MODEL_LABELS[modelo], 'Métrica': 'AUC (Alto)', 'Estimación': punto, 'Estacionario inferior': bajo,
                  'Estacionario superior': alto, 'BCa inferior': bca_bajo, 'BCa superior': bca_alto})
referencias_rmse = {A.MODEL_LABELS[m]: errores_reg[m]['error'] ** 2 for m in mejores_reg.index}
for nombre in ['Persistencia', 'Estacional', 'SARIMA']:
    referencias_rmse[nombre] = ((referencias_reg['y_true'] - referencias_reg[nombre]) ** 2).values
for nombre, cuadrados in referencias_rmse.items():
    punto, bajo, alto = ST.bootstrap_interval(cuadrados, lambda a: np.sqrt(a.mean()), mean_block=12, n_boot=2000)
    bca_bajo, bca_alto = bca_rmse(cuadrados)
    filas.append({'Modelo': nombre, 'Métrica': 'RMSE', 'Estimación': punto, 'Estacionario inferior': bajo,
                  'Estacionario superior': alto, 'BCa inferior': bca_bajo, 'BCa superior': bca_alto})
intervalos = pd.DataFrame(filas).set_index(['Métrica', 'Modelo'])
intervalos.round(3)
""")

md(r"""
Los intervalos del AUC de la categoría Alto son amplios porque solo hay 17 meses Alto: KNN tiene 0.951 con un intervalo de entre 0.77 y 0.99 (bootstrap estacionario) y los de Random Forest y XGBoost van de 0.82 a 0.97, así que los tres se solapan casi por completo, lo que concuerda con DeLong. El árbol de decisión (0.52, de 0.44 a 0.69) sí queda claramente por debajo de los tres, y SVM (0.67) también, pero con más incertidumbre. En RMSE los intervalos son todavía más anchos y se solapan todos, incluido el del SARIMA (0.995, de 0.50 a 1.57). Como el error total depende mucho de unos pocos meses de 2020, los intervalos son asimétricos y con un límite inferior muy bajo: las remuestras de bloques que no incluyen la pandemia dan errores mucho menores. Por eso comparar los intervalos de dos modelos por separado no es la forma de decidir cuál es mejor; las pruebas pareadas de la sección siguiente comparan los dos modelos mes a mes y son mucho más potentes (el intervalo de la diferencia SARIMA menos SVR, de −1.07 a −0.47, no incluye el cero aunque los intervalos individuales se solapen).

Los dos tipos de intervalo difieren de forma instructiva. El BCa de remuestreo independiente, que ignora que meses seguidos se parecen, es más estrecho (para el RMSE del SARIMA, de 0.70 a 1.56 frente a 0.50 a 1.57 del estacionario) y casi siempre tiene límites inferiores más altos. El estacionario preserva la dependencia mediante bloques de 12 meses de longitud media, es el apropiado para esta serie y por eso es el que se usa en el resto del capítulo; el BCa se muestra porque la guía lo recomienda cuando la métrica es asimétrica, y aquí coincide en lo cualitativo.
""")

md(r"""
### Regresión
""")

code(r"""
# Errores de pronóstico de cada mes para los 28 modelos de regresión y las tres referencias; de ahí salen las pérdidas
errores_todos = {}
for _, f in regresion.iterrows():
    errores_todos[f"{A.MODEL_LABELS[f['model']]} ({A.OPTIMIZER_LABELS[f['optimizer']]})"] = A.pooled_regression_errors(f['run_id'])['error']
for nombre in ['Persistencia', 'Estacional', 'SARIMA']:
    errores_todos[nombre] = (referencias_reg['y_true'] - referencias_reg[nombre]).values
errores_todos = pd.DataFrame(errores_todos)
perdidas = errores_todos ** 2   # pérdida cuadrática de cada mes

# Model Confidence Set sobre el universo completo (bootstrap por bloques de 12 meses)
mcs = ST.model_confidence_set(perdidas, alpha=0.10, n_boot=5000, block=12)
mcs.columns = ['Modelo', 'Pérdida media (MSE)', 'Rango', 'p-valor MCS', 'En el conjunto de confianza']
print(f"Modelos en el conjunto de confianza (alfa = 0.10): {int(mcs['En el conjunto de confianza'].sum())} de {len(mcs)}")
mcs.head(15).set_index('Modelo').round(4)
""")

code(r"""
# Sensibilidad del MCS: con el error absoluto, menos dominado por unos pocos meses muy malos, con un universo reducido (la mejor
# variante de cada familia y las tres referencias) y con un nivel de significancia más holgado
familias = {}
for nombre in perdidas.columns:
    familias.setdefault(nombre.split(' (')[0], []).append(nombre)
reducido = [min(columnas, key=lambda c: perdidas[c].mean()) for columnas in familias.values()]
filas = []
for nombre, tabla in [('Error cuadrático', perdidas), ('Error absoluto', errores_todos.abs()),
                      ('Error cuadrático, universo reducido (10 modelos)', perdidas[reducido])]:
    for alfa in [0.10, 0.25]:
        r = ST.model_confidence_set(tabla, alpha=alfa, n_boot=5000, block=12)
        filas.append({'Pérdida': nombre, 'Nivel de significancia': alfa, 'Modelos en el conjunto': int(r['en_conjunto'].sum()), 'De': len(r)})
pd.DataFrame(filas).set_index(['Pérdida', 'Nivel de significancia'])
""")

md(r"""
El resultado del MCS es poco informativo: no elimina a ningún modelo. Con la pérdida cuadrática y un nivel de 0.10, los 31 (los 28 de machine learning y las 3 referencias) quedan en el conjunto de confianza, incluso la regla del mismo mes del año anterior, cuyo MSE (4.44) es unas cuatro veces y media el del SARIMA (0.99). Con un nivel más holgado (0.25) sale uno, y con el error absoluto, menos dominado por los meses extremos, sale uno con 0.10 y salen ocho con 0.25. El universo reducido de 10 modelos (la mejor variante de cada familia y las tres referencias) tampoco elimina a ninguno, ni con 0.10 ni con 0.25. La razón es de potencia estadística: el MCS compara a cada modelo con el promedio de todo el grupo y tiene que superar el máximo de 31 estadísticos, y las pérdidas de estos modelos varían mucho de un mes a otro (unos pocos meses de 2020 aportan gran parte del error), de modo que el error estándar de la diferencia de pérdidas es del mismo orden que la diferencia misma.

La secuencia hay que leerla con cuidado, entonces: el MCS no permite reducir el conjunto de candidatos, y las comparaciones dirigidas se hacen entre finalistas elegidos por pérdida media. Lo que el MCS sí dice es que, con este tamaño de muestra, casi no hay evidencia para declarar a un modelo peor que el mejor. Las diferencias solo aparecen cuando se comparan pares de modelos con pruebas pareadas, que son más potentes porque cancelan la variación que comparten los dos modelos.
""")

code(r"""
# Como el MCS no elimina a ningún modelo, los finalistas se eligen por pérdida media: las dos mejores referencias y la mejor variante
# de las tres familias de machine learning con menor pérdida. Entre ellos no hay modelos anidados, así que se usa Giacomini-White
# (Clark-West no aplica) y Diebold-Mariano con la corrección de Harvey-Leybourne-Newbold, ambos sobre los errores mensuales fuera de muestra.
finalistas = ['SARIMA', 'Persistencia']
for nombre in perdidas.mean().sort_values().index:
    if nombre in ('SARIMA', 'Persistencia', 'Estacional') or nombre.split(' (')[0] in [f.split(' (')[0] for f in finalistas]:
        continue
    finalistas.append(nombre)
    if len(finalistas) == 5:
        break

filas = []
for i in range(len(finalistas)):
    for j in range(i + 1, len(finalistas)):
        a, b = finalistas[i], finalistas[j]
        gw = ST.giacomini_white_test(errores_todos[a].values, errores_todos[b].values)
        dm = ST.diebold_mariano_test(errores_todos[a].values, errores_todos[b].values)
        dif = (errores_todos[a] ** 2 - errores_todos[b] ** 2).values
        _, bajo, alto = ST.bootstrap_interval(dif, np.mean, mean_block=12, n_boot=2000)
        filas.append({'Modelo A': a, 'Modelo B': b, 'Diferencia media de pérdida': dif.mean(), 'IC 95 % (estacionario)': f'[{bajo:.3f}, {alto:.3f}]',
                      'p Diebold-Mariano (HLN)': dm['p_value'], 'p Giacomini-White': gw['p_value'], 'd de Cohen': ST.cohens_d_paired(dif)})
dirigidas = pd.DataFrame(filas)
dirigidas['p DM ajustado (Holm)'] = ST.adjust_pvalues(dirigidas['p Diebold-Mariano (HLN)'].values, 'holm')
dirigidas['p GW ajustado (Holm)'] = ST.adjust_pvalues(dirigidas['p Giacomini-White'].values, 'holm')
dirigidas.set_index(['Modelo A', 'Modelo B']).round(3)
""")

md(r"""
Como el MCS no elimina a nadie, los finalistas se escogieron por pérdida media: las dos mejores referencias (SARIMA y persistencia) y la mejor variante de las tres familias de machine learning con menor pérdida (SVR, Ridge y KNN). Con la corrección de Holm sobre las 10 comparaciones, el patrón es nítido. El SARIMA es significativamente mejor que cada uno de los tres modelos de machine learning: la diferencia media de pérdida cuadrática es de −0.73 frente a SVR (IC 95 % estacionario de −1.07 a −0.47), de −0.75 frente a Ridge y de −0.89 frente a KNN, con p-valores de Diebold-Mariano (con la corrección HLN) y de Giacomini-White menores que 0.001 incluso después del ajuste, y tamaños de efecto pequeños (d de Cohen de −0.29 a −0.37). Entre la persistencia y los tres modelos de machine learning, y entre estos tres, no hay diferencias significativas: p de DM entre 0.29 y 0.84, p de Giacomini-White entre 0.07 y 0.24 sin ajustar, todos de 0.415 o más con el ajuste, y tamaños de efecto casi nulos (|d| entre 0.01 y 0.07). Es decir, ningún modelo de machine learning se distingue de repetir el último valor.

El caso SARIMA frente a persistencia es el interesante. El DM sin ajustar da p = 0.020 (diferencia de −0.68, IC de −1.17 a 0.13, que incluye el cero) y con Holm sube a 0.139, que ya no es significativo, mientras que Giacomini-White da p < 0.001. La explicación está en la tabla por periodos: antes y después de la pandemia el SARIMA gana con claridad, pero durante ella la persistencia lo supera (2.43 contra 3.03). Como la ventaja del SARIMA cambia de signo según el momento, el promedio incondicional que mide el DM la diluye, mientras que la prueba condicional de Giacomini-White sí detecta que la diferencia de pérdidas depende del estado de la serie. Es el caso que la guía menciona para preferir esta prueba.
""")

code(r"""
# Prueba SPA: ¿algún modelo de machine learning supera de verdad a la referencia, o es solo producto de haber probado 28?
candidatos = perdidas.drop(columns=['Persistencia', 'Estacional', 'SARIMA'])
filas = []
for referencia in ['SARIMA', 'Persistencia']:
    r = ST.spa_test(perdidas[referencia].values, candidatos, n_boot=5000, mean_block=12)
    filas.append({'Referencia': referencia, 'Mejor candidato': r['best_candidate'], 'Ganancia media de pérdida (MSE)': r['best_mean_loss_gain'],
                  'p SPA': r['p_consistent'], 'p Reality Check': r['p_upper']})
pd.DataFrame(filas).set_index('Referencia').round(4)
""")

md(r"""
La prueba SPA responde la pregunta de fondo: ¿algún modelo de machine learning supera de verdad a las referencias o lo parecería solo por haber probado 28? Contra el SARIMA, el mejor candidato (SVR con Grid) tiene una pérdida mayor en 0.73, y los p-valores (0.60 del SPA y 0.90 del Reality Check) están lejos de la significancia: ningún modelo supera al SARIMA. Contra la persistencia, el mejor candidato (otra vez SVR con Grid) pierde por 0.05 y el p-valor es de 0.77 en las dos pruebas: tampoco ningún modelo la supera. Las dos pruebas coinciden cuando los candidatos están cerca de la referencia, como con la persistencia (0.77 en ambas), y se separan cuando casi todos son claramente peores, como con el SARIMA, donde el Reality Check, más conservador, da 0.90 y el SPA 0.60; ese es justo el sesgo que corrige Hansen.
""")
