# Guion para explicar Metodología y Resultados con el paper proyectado

**Cómo usar este guion:** se va recorriendo el paper en pantalla, sección por sección. Cada bloque dice **qué mostrar** (📌), **qué decir** (en cita) y, cuando hace falta, una **idea sencilla** (💡) para que el concepto se entienda sin fórmulas. Los números de sección, tabla y figura son los del PDF final.

**Duración aproximada:** 12 a 15 minutos (unos 6–7 de metodología y 6–7 de resultados).

**Idea que debe quedar al final:** el modelo se evaluó de forma justa y sin hacer trampa, gana con claridad a todos los modelos del curso en nueve países, y sabemos *por qué* gana.

---

# PARTE 1 · MATERIALES Y MÉTODOS (Sección 2)

## 2.1 · Formulación del problema

📌 *Mostrar la definición de la tarea y la fórmula del error cuadrático medio.*

> Lo primero es decir con precisión qué queremos aprender. Con lo que se sabe hoy, al cierre de un mes, queremos predecir la tasa de desempleo del **mes siguiente**. Eso es la función f que va de X, la información de hoy, a Y, el desempleo de mañana.
>
> Para saber si un pronóstico es bueno o malo usamos el **error cuadrático medio**: tomamos la diferencia entre lo que pasó y lo que pronosticamos, la elevamos al cuadrado y promediamos.

💡 *Elevar al cuadrado castiga mucho más los errores grandes. Equivocarse por 4 puntos en un mes pesa como equivocarse por 1 punto en 16 meses. Para el desempleo eso tiene sentido: lo grave es fallar en las crisis.*

> Al final de la sección está la notación, para que todos los símbolos del paper signifiquen siempre lo mismo.

---

## 2.2 · Datos y preprocesamiento

📌 *Mostrar el párrafo de Colombia y luego el de conjuntos adicionales.*

> El problema principal es **Colombia**: datos mensuales del DANE, publicados por el Banco de la República, de 2001 a 2025. Con ellos construimos las variables de entrada: los valores de los meses anteriores (rezagos), promedios y variabilidad de los últimos 3, 6 y 12 meses, el mes del año y los indicadores de las 13 áreas metropolitanas.
>
> Un modelo que solo funciona en un país puede haber tenido suerte. Por eso lo probamos en **ocho países más**: Estados Unidos, España, México, Canadá, Australia, Japón, Alemania y Corea del Sur. Son nueve conjuntos de datos en total. Todos los países se eligieron **antes** de ver cómo le iba al modelo en ellos.

📌 *Señalar el párrafo de Licencia.*

> Todos los datos son públicos y se pueden reutilizar citando la fuente.

📌 *Proyectar la Tabla 1.*

> Esta tabla describe la variable que queremos predecir en cada país: cuánto vale en promedio, cuánto varía, su mínimo y su máximo. La última columna cuenta los **cambios atípicos**, es decir, meses en que el desempleo saltó de forma anormal. El caso más claro es la pandemia de 2020.
>
> Decidimos **no eliminar** esos atípicos. Un modelo de pronóstico tiene que enfrentar justamente esas crisis; si las borramos, el modelo se vería mejor de lo que es.

📌 *Señalar el párrafo de Escalamiento y fuga de información.*

> Esta es la regla más importante de la metodología: **evitar la fuga de información**. Ningún dato del futuro puede colarse en el entrenamiento. Por eso:
> - todas las variables usan solo meses pasados;
> - cuando normalizamos los datos (restar la media y dividir por la desviación), esos valores se calculan **solo con los datos de entrenamiento** y después se aplican a los de prueba.

💡 *Es como un examen: si el estudiante ve las respuestas antes, su nota no dice nada. Calcular la media con los datos de prueba es ver un poquito las respuestas.*

---

## 2.3 · Modelo propuesto: ALEE

📌 *Leer el párrafo "Idea central" y proyectar la Figura 1.*

> ALEE significa **agregación local entrópica de expertos**. La mejor forma de entenderlo es con el diagrama.
>
> **Arriba** hay tres "expertos", tres formas sencillas de pronosticar:
> 1. **Persistencia:** el mes siguiente será igual al de hoy.
> 2. **Núcleo de cambio:** busca en la historia momentos parecidos a hoy y mira cuánto cambió el desempleo después.
> 3. **Estacional:** el valor de hoy más el cambio típico de este mes del año. Por ejemplo, en enero el desempleo suele subir.
>
> El pronóstico final es un **promedio ponderado** de los tres. La pregunta es cuánto pesa cada uno.
>
> **Abajo** está cómo se deciden esos pesos. El modelo mira los meses pasados que se parecen al mes actual (por ejemplo, con volatilidad parecida) y revisa cómo les fue a los tres expertos en esos meses. Al que acertó más le da más peso.

💡 *Es como pedir consejo a tres personas. No les haces el mismo caso siempre: le haces más caso a la que acertó en situaciones parecidas a la que estás viviendo.*

📌 *Mostrar las ecuaciones (1) y (2).*

> La ecuación 1 mide qué tan parecido es cada mes pasado al actual y da más peso a los más parecidos y a los más recientes.
>
> La ecuación 2 elige los pesos con dos objetivos a la vez:
> - **equivocarse poco** en los meses parecidos;
> - **no alejarse demasiado** de una combinación general y estable. Eso es el término de Kullback-Leibler.
>
> Si hay pocos meses parecidos (por ejemplo, en una crisis nunca vista), el modelo desconfía de esa poca evidencia y se queda cerca de la combinación general. Eso evita decisiones extremas con pocos datos.

📌 *Mostrar el Algoritmo 1.*

> El algoritmo muestra el proceso paso a paso:
> 1. Primero registra cómo se equivocó cada experto en el pasado, siempre usando solo información anterior.
> 2. Después calcula los pesos para el mes actual.
> 3. Encuentra los pesos óptimos con un método iterativo que se detiene cuando ya no cambian.

📌 *Mostrar las Proposiciones 1, 2 y 3.*

> Las tres proposiciones dan **garantías matemáticas**:
> 1. El problema siempre tiene **una única respuesta**.
> 2. El algoritmo **mejora en cada paso**: nunca empeora.
> 3. El error del pronóstico **nunca supera al del peor experto**.
>
> No son teoremas nuevos; se derivan de resultados conocidos de optimización. Pero garantizan que el modelo se comporta bien.

📌 *Señalar "Qué es nuevo" e "Implementación y complejidad".*

> Lo novedoso es la forma de elegir los pesos: un problema matemático con solución única que se ajusta solo cuando hay poca información. Todo está programado por nosotros en NumPy, sin modelos de librería. También indicamos cuánto tarda en función del tamaño de los datos.

---

## 2.4 · Modelos de comparación

> Para saber si ALEE es bueno hay que compararlo con algo. Lo comparamos con los **ocho modelos de regresión vistos en clase**: KNN, Ridge, Lasso, regresión bayesiana, árbol de decisión, Random Forest, XGBoost y SVR. Además agregamos **LightGBM**, un método reciente y muy usado.

💡 *Regla de oro de la comparación: **todos** los modelos usan exactamente los mismos datos, las mismas particiones y el mismo esfuerzo de ajuste. Si nuestro modelo tuviera más intentos de ajuste que los demás, ganar no significaría nada.*

---

## 2.5 · Entrenamiento y optimización

📌 *Señalar la validación anidada de origen móvil.*

> En series de tiempo no se pueden mezclar los datos al azar. Siempre se entrena con el pasado y se prueba con el futuro. Usamos **validación anidada de origen móvil**:
> - Partimos la serie en 5 bloques de prueba consecutivos. Para cada uno, el modelo entrena **solo con los meses anteriores**.
> - Dentro de ese entrenamiento hay otra división en 3 partes, que sirve **únicamente** para elegir los hiperparámetros.
> - Los meses de prueba nunca se usan para tomar decisiones.

💡 *Es como evaluar a un pronosticador del clima: solo vale si predijo antes de que pasara. Y los ajustes del instrumento se hacen con días de práctica, nunca con los días del examen.*

> Los hiperparámetros se buscan con **optimización bayesiana**, 40 intentos por modelo; ese es el mismo presupuesto para todos. También reportamos la semilla, el computador usado y las versiones de las librerías, que están en las referencias, para que cualquiera pueda repetir el experimento.

---

## 2.6 · Protocolo de evaluación

> Medimos el error con cinco métricas:
> - **RMSE:** el error típico, en puntos porcentuales.
> - **MAE:** el error promedio en valor absoluto.
> - **MAPE:** el error en porcentaje.
> - **R²:** qué parte de la variación logramos explicar.
> - **MASE:** el error comparado con un pronóstico ingenuo. Menor que 1 significa que somos mejores que repetir lo del año anterior.
>
> Pero un número menor puede ser **suerte**. Para descartarlo usamos **pruebas estadísticas**:
> - **Diebold-Mariano:** compara mes a mes los errores de dos modelos y pregunta si la diferencia es real o casualidad.
> - **Giacomini-White:** pregunta lo mismo, pero considerando que la ventaja puede cambiar con el tiempo.
> - **Prueba bayesiana con ROPE:** en vez de un sí o un no, da la probabilidad de que ALEE sea mejor, igual o peor. Si la diferencia es menor al 1 %, la consideramos "prácticamente igual".

💡 *Como comparamos ALEE con nueve modelos, hacemos muchas pruebas a la vez, y alguna podría salir significativa por azar. La **corrección de Holm** sube la exigencia para evitarlo.*

> Entre países usamos **Friedman**, que pregunta si los modelos son distintos en general considerando los nueve países a la vez, y **Wilcoxon**, que compara ALEE con cada modelo, país por país.

---

## 2.7 · Diseño del estudio de ablación

> La **ablación** consiste en quitarle piezas al modelo, una a la vez, para ver cuál es importante. Antes de ver los resultados, planteamos cinco hipótesis:
> - **H1:** combinar con pesos aprendidos es mejor que un promedio simple.
> - **H2:** usar los meses parecidos (localización) ayuda.
> - **H3:** la regularización que se adapta a la cantidad de evidencia ayuda.
> - **H4:** acercarse a la combinación global es mejor que acercarse a pesos iguales.
> - **H5:** combinar los tres expertos es mejor que usar cualquiera de ellos solo.

💡 *Es como un carro que anda rápido: para saber qué lo hace rápido, le quitas el turbo y miras, luego le quitas las llantas especiales y miras, y así con cada pieza.*

---

## 2.8 · Pre-registro y desviaciones

📌 *Proyectar la Tabla 2.*

> El protocolo se escribió y se congeló **antes** de ver los resultados finales. Esta tabla separa con honestidad qué estaba fijado desde el principio y qué se agregó después, con el motivo de cada cambio. Por ejemplo, la regresión bayesiana y la prueba con ROPE se agregaron porque la guía del curso las pide.
>
> Declaramos también que en Colombia probamos diez versiones del diseño. Por eso la prueba más confiable son los **ocho países nuevos**, donde el modelo se evaluó **una sola vez**, ya terminado, sin cambiarle nada.

---

# PARTE 2 · RESULTADOS (Sección 3)

## 3.1 · Resumen del montaje

> Para recordar: diez modelos (ALEE, los ocho del curso y LightGBM), nueve países y 4 260 meses de prueba en total. Todos con las mismas reglas.

---

## 3.2 · Desempeño global

📌 *Proyectar la Tabla 3.*

> Esta es la tabla principal. Cada columna es un país y cada fila un modelo. Los números son el error típico (RMSE): **mientras más bajo, mejor**.
>
> La primera fila es ALEE, y está **en negrita en los nueve países**: es el mejor en todos. La cruz (†) junto a cada modelo indica que ALEE es mejor que ese modelo y que la diferencia es estadísticamente significativa.
>
> En números concretos, frente al mejor modelo del curso en cada país, ALEE reduce el error entre **21 % (Japón) y 33 % (Colombia)**.
>
> Algo interesante: entre los modelos del curso no hay un ganador fijo. En algunos países gana la regresión bayesiana, en otros Lasso, Ridge o SVR. ALEE gana en todos.

📌 *Pasar rápido por la Tabla 4.*

> Esta tabla es el detalle con las cinco métricas. ALEE es el mejor en **las cinco y en los nueve países**, así que el resultado no depende de cómo midamos el error.

📌 *Proyectar la Figura 2.*

> Aquí se ve el pronóstico en Colombia. La línea negra es la realidad, la azul es ALEE y la naranja es Ridge, el mejor modelo del curso en Colombia. ALEE sigue los vaivenes estacionales mes a mes, mientras que Ridge los suaviza y se equivoca más. Los dos fallan en el salto de 2020: nadie podía anticipar la pandemia.

📌 *Figuras 3 y 4.*

> En la Figura 3, mientras más cerca estén los puntos de la diagonal, mejor es el pronóstico. En la Figura 4, los errores de ALEE (en azul) están más concentrados alrededor de cero.

📌 *Figura 5.*

> Aquí se ve cada uno de los 45 bloques de prueba (5 por país). ALEE es el mejor en **37 de los 45**, lo que muestra que no gana por un solo periodo afortunado.

---

## 3.3 · Pruebas de significancia

📌 *Proyectar la Tabla 5.*

> Esta tabla responde a la pregunta "¿es suerte?". Para cada país y cada modelo muestra la diferencia de error con su intervalo de confianza, el resultado de Diebold-Mariano y de Giacomini-White, la probabilidad bayesiana y el tamaño del efecto.
>
> El resultado es contundente: en las **81 comparaciones** (nueve modelos por nueve países), las dos pruebas dicen que la diferencia es **real**. Incluso el caso más ajustado tiene un valor p de 0.0074, muy por debajo de 0.05. Y **todos** los intervalos de confianza quedan del lado favorable a ALEE.

💡 *El tamaño del efecto (d de Cohen y A12) dice **cuánto** mejor es ALEE, no solo si lo es. Un A12 de 0.70, por ejemplo, significa que en el 70 % de los meses ALEE se equivoca menos que el otro modelo.*

> La prueba bayesiana es más prudente porque trabaja con menos datos, solo cinco bloques por país. Aun así da entre 66 % y 99.9 % de probabilidad de que ALEE sea mejor, y casi cero de que sean equivalentes.

📌 *Proyectar la Tabla 6 y la Figura 6.*

> Ahora miramos los nueve países juntos. El **rango medio** es como una posición en el campeonato: ALEE queda primero en los nueve países, con rango 1.00. Le siguen la regresión bayesiana, Lasso y Ridge.
>
> El Wilcoxon confirma que ALEE es mejor que **cada uno** de los nueve modelos. El Conover lo confirma frente a ocho; frente a la regresión bayesiana, el segundo mejor, no alcanza la significancia (p = 0.071), porque con nueve países la prueba tiene un poder limitado. Lo reportamos tal cual.

📌 *En la Figura 6, señalar ALEE a la izquierda.*

> En el diagrama de diferencia crítica, los mejores modelos están a la izquierda. ALEE está solo en la primera posición.

---

## 3.4 · Ablación

📌 *Proyectar la Tabla 7 (y mencionar que la Tabla 8 tiene las cinco métricas).*

> Aquí vemos qué piezas del modelo importan:
> - **H1, combinar:** sí importa. El promedio simple es peor en 7 de 9 países.
> - **H2, localización:** aporta menos de lo esperado. Solo es significativa en México y Japón.
> - **H3, regularización adaptativa:** ayuda en Canadá y Japón.
> - **H4, previo global:** no hace diferencia.
> - **H5, cada experto solo:** la persistencia y el núcleo, por separado, son claramente peores. El estacional solo es peor en Canadá, Australia y Japón.
>
> La conclusión es honesta: **la mayor parte de la ventaja viene de los expertos**, que ya incorporan cómo se comporta el desempleo (persistencia y estacionalidad), y de combinarlos. La localización aporta menos de lo que esperábamos, y lo decimos.

📌 *Figura 7: mostrar que la barra azul (ALEE completo) es de las más cortas en cada país.*

---

## 3.5 · Sensibilidad y robustez

📌 *Tabla 9.*

> Repetimos el ajuste con distintas semillas aleatorias. ALEE casi no cambia: su variación es menor al 1 %. Random Forest, XGBoost y LightGBM varían mucho más, y ni en su mejor semilla se acercan a ALEE.

📌 *Figura 8.*

> El mapa de colores muestra el error de ALEE con distintas combinaciones de hiperparámetros. En casi toda la zona el error es bajo y estable: el modelo no depende de una configuración exacta.

📌 *Figura 9.*

> La curva de aprendizaje muestra cuántos datos necesita cada modelo. ALEE alcanza su mejor error con solo 60 meses de historia. Ridge necesita mucha más, y aun con toda la serie queda por encima de ALEE.

---

## 3.6 · Costo computacional

📌 *Tabla 10.*

> ALEE es más lento que los modelos lineales: menos de un segundo para entrenar y medio segundo para pronosticar 48 meses. Para un pronóstico que se hace una vez al mes, ese costo es irrelevante frente a reducir el error entre 21 % y 33 %.

---

## 3.7 · Interpretabilidad y análisis de errores

📌 *Figura 10.*

> ¿En qué se fija el modelo? La importancia por permutación lo responde: si desordenamos una variable y el error sube mucho, esa variable es importante. Lo que más importa es el **último valor del desempleo** y el **mes del año**. Tiene sentido: el desempleo del próximo mes depende sobre todo de cómo está hoy y de la época del año.

📌 *Figura 11.*

> Estos son los pesos de cada experto en el tiempo, en Colombia. Al principio, con poca historia, cambian bastante. Desde 2014 el modelo confía casi siempre en el experto estacional.

📌 *Figura 12.*

> Separamos los meses en volatilidad baja, media y alta. ALEE es el mejor en los tres grupos en los nueve países: 27 de 27. Su peor momento es la llegada de la pandemia, que llegó tras meses tranquilos y nada permitía anticiparla.

---

## Sección 4 · Discusión, limitaciones y cierre

📌 *Señalar los párrafos de limitaciones.*

> También decimos qué **no** podemos afirmar:
> - Nueve países dan un poder estadístico moderado.
> - En Colombia probamos varias versiones del diseño, así que ahí puede haber algo de optimismo.
> - Algunos datos se revisan después de publicados, y eso puede introducir una pequeña fuga.
> - Los países adicionales usan 14 variables en vez de 17.
>
> Declarar esto no debilita el trabajo: le dice al lector hasta dónde llega la evidencia.

> En resumen:
> 1. **La metodología es justa y verificable:** las mismas reglas para todos los modelos, sin fuga de información y con todo declarado.
> 2. **Los resultados son claros:** ALEE gana a los ocho modelos del curso y a LightGBM en los nueve países, en las cinco métricas, y las pruebas estadísticas confirman que no es suerte.
> 3. **Sabemos por qué gana:** sus expertos incorporan cómo se comporta el desempleo, y combinarlos bien agrega valor.
>
> Muchas gracias.

---

# Preguntas probables

**¿La comparación es justa?**
> Sí. Todos los modelos usaron los mismos datos, las mismas particiones y el mismo presupuesto de 40 intentos de ajuste.

**¿Cómo saben que no hay trampa con datos del futuro?**
> Las variables usan solo meses pasados, la normalización se calcula solo con el entrenamiento y los hiperparámetros se eligen sin tocar los meses de prueba (Sección 2.2 y 2.5).

**Si la localización aporta poco, ¿por qué gana ALEE?**
> Porque sus expertos ya incorporan la persistencia y la estacionalidad, algo que los modelos genéricos tienen que aprender de pocos datos, y porque la combinación con pesos aprendidos supera al promedio simple en 7 de 9 países.

**¿No sobreajustaron probando muchas versiones?**
> En Colombia puede haber algo de optimismo, y lo declaramos. Por eso la evidencia principal son los ocho países nuevos, donde el modelo se evaluó una sola vez y gana igual.

**¿Por qué el Conover no es significativo frente a la regresión bayesiana?**
> Es el segundo mejor modelo y, con nueve países, esa prueba tiene poco poder. El Wilcoxon sí es significativo, y en cada país Diebold-Mariano también.

**¿Qué es lo original del modelo?**
> Elegir los pesos de los expertos resolviendo un problema matemático con solución única que se vuelve más prudente cuando hay poca evidencia, aplicado al pronóstico del desempleo. Está programado por nosotros, sin modelos de librería.
