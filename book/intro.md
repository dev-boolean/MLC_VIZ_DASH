# Mercado Laboral Colombiano: pronóstico del desempleo con machine learning

**Andrés Parejo, Santiago Hurtado y Juan Marín** · Pregrado en Ciencia de Datos · Machine Learning, tercer entregable

## Resumen

Se pronosticó la tasa de desempleo nacional de Colombia del mes siguiente con la información disponible al cierre de cada mes, entre 2002 y 2025. Sobre 288 meses y 17 variables se compararon 140 combinaciones de siete modelos, cuatro técnicas de balanceo (en clasificación) y cuatro optimizadores de hiperparámetros, con validación anidada de ventana expansiva. En regresión, la mejor combinación por RMSE medio fue Ridge (1.196 ± 0.552); en clasificación, KNN (F1-macro 0.679 ± 0.211), y ninguna de las dos se distingue estadísticamente de las demás.

Como modelo original se propuso ALEE (agregación local entrópica de expertos). Combina tres expertos programados desde cero (persistencia, regresión de Nadaraya-Watson del cambio y cambio estacional medio) con pesos que dependen del estado del mercado laboral, se restringen al símplex y se regularizan con divergencia de Kullback-Leibler hacia una combinación global. El problema es convexo, tiene solución única y se resuelve por descenso de espejo con descenso monótono.

ALEE se comparó con los ocho modelos de regresión del curso y con LightGBM en Colombia y en ocho países más. Tiene el menor error en los nueve conjuntos y en las cinco métricas, con reducciones del RMSE de 21 % a 33 % frente al mejor modelo de comparación. Diebold-Mariano-HLN y Giacomini-White con corrección de Holm rechazan la igualdad en las 81 comparaciones, y el Wilcoxon con Holm entre conjuntos da p = 0.035 frente a cada modelo. La ablación muestra que la ventaja viene sobre todo de los expertos: la localización por estado solo aporta de forma significativa en dos conjuntos.

## Problema y motivación

La tasa de desempleo es la cifra del mercado laboral que más se sigue en el país. El DANE la publica con un mes de rezago, y anticiparla sirve para orientar decisiones de política. La serie es difícil de pronosticar: es muy persistente, tiene una estacionalidad marcada (sube cada enero) y sufrió un choque extremo en 2020, cuando llegó a 21.97 %.

## Contribuciones

1. Una comparación de 140 combinaciones de modelos, técnicas de balanceo y optimizadores, con validación anidada que respeta el orden del tiempo y una batería estadística jerárquica.
2. La evidencia de que el mejor modelo de machine learning cambia con el régimen de volatilidad del mercado laboral (capítulo 5).
3. ALEE, una regla de combinación de pronósticos localizada por estado, con cuatro propiedades demostradas y verificadas con pruebas unitarias.
4. Una evaluación en nueve conjuntos de datos con protocolo pre-registrado, ablación y estudios de robustez (capítulos 6 y 7).
5. Un dashboard en Dash con los modelos, los resultados y un asistente conversacional (capítulo 8).
6. Un paquete de reproducibilidad: artefactos verificados, registro de iteraciones de diseño y pruebas automáticas.

## Objetivos


- Elegir con criterios estadísticos la mejor combinación de cada modelo y ponerla en producción (capítulos 3 y 8).
- Diseñar, implementar y validar un modelo original frente a los modelos del curso (capítulos 4 a 7).

## Mapa del libro

| Capítulo | Contenido |
|---|---|
| 2 | Datos: ETL, EDA y auditoría de fuga de información |
| 3 | Modelos benchmark y mejores combinaciones |
| 4 | Revisión de literatura y novedad |
| 5 | Modelo original (ALEE) |
| 6 | Protocolo y validación estadística |
| 7 | Evaluación ampliada y robustez |
| 8 | Dashboard y despliegue |
| 9 | Conclusiones y limitaciones |
| 10 | Reproducibilidad y bibliografía |

**Dashboard:** pendiente del despliegue en Render. Para verlo localmente: `python app.py` y abrir <http://127.0.0.1:8050>.
