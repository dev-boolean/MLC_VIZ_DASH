---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python (MLC ML)
  language: python
  name: mlc-ml
---

# Conclusiones y limitaciones

```{code-cell} ipython3
:tags: [remove-cell]
import os, sys
if os.path.basename(os.getcwd()) == "book":
    os.chdir("..")
sys.path.insert(0, os.getcwd())
import warnings
warnings.filterwarnings("ignore")  # avisos de versiones de librerías que no afectan los resultados
from src import dash_data as D
```

## Hallazgos

1. **Entre los modelos del segundo entregable ninguno se distingue de los demás.** Las mejores combinaciones fueron Ridge en regresión (RMSE 1.196 ± 0.552) y KNN en clasificación (F1-macro 0.679 ± 0.211), pero Friedman da p = 0.72 en clasificación y el MCS conserva a todos los modelos de regresión.
2. **El mejor modelo genérico depende del régimen.** El modelo de machine learning con menor error cambia según la volatilidad reciente del mercado laboral. Esa observación motivó al modelo original.
3. **ALEE supera a los modelos del curso en los nueve conjuntos.**

```{code-cell} ipython3
:tags: [remove-input]
t = D.paper_table()
otros = t.drop(columns="ALEE")
red = 1 - t["ALEE"] / otros.min(axis=1)
pr = D.paper_tests()
print(f"ALEE tiene el menor RMSE en {(t['ALEE'] < otros.min(axis=1)).sum()} de {len(t)} conjuntos, con entre "
      f"{red.min():.0%} y {red.max():.0%} menos error que el mejor modelo de comparación. Diebold-Mariano-HLN y "
      f"Giacomini-White con Holm rechazan la igualdad en {(pr['dm_p_holm'] < 0.05).sum()} y "
      f"{(pr['gw_p_holm'] < 0.05).sum()} de {len(pr)} comparaciones.")
```

4. **La ventaja viene sobre todo de los expertos.** El experto estacional incorpora directamente la estructura que los modelos genéricos tienen que aprender a partir de pocas observaciones: el nivel actual más el cambio típico del mes. La combinación aprendida supera al promedio simple en siete de los nueve conjuntos, pero la localización por estado, que era la idea central del modelo, solo aporta de forma significativa en México y Japón. Coincide con lo observado en comparaciones a gran escala, donde los métodos que codifican la estructura temporal suelen superar a los modelos genéricos en series univariadas {cite}`makridakis2018,makridakis2020`.

Para el pronóstico del desempleo a un mes, incorporar explícitamente la estructura temporal de la serie pesa más que la flexibilidad de un modelo genérico.

## Veredicto sobre las condiciones del modelo original

| Condición | Estado |
|---|---|
| R1. Novedad en la aplicación | Cumplida según la revisión en Scopus (capítulo 4): 90 registros, ningún trabajo reúne las cuatro piezas del mecanismo. Los más cercanos son los *local prediction pools* (2024) y Dang (2027) |
| R2. Implementación algorítmica propia | Cumplida: expertos, localización y solucionador programados con NumPy, con pruebas unitarias. El equipo debe poder explicar y modificar cada línea |
| R3. Superioridad estadísticamente verificada | Cumplida frente a los ocho modelos del curso y LightGBM: más de tres pruebas rechazan con Holm (Diebold-Mariano-HLN, Giacomini-White, Wilcoxon y Conover), y la d de Cohen supera 0.2 en valor absoluto en 77 de las 81 comparaciones (capítulo 6) |

## Limitaciones

- Nueve conjuntos dan una potencia moderada a las pruebas entre conjuntos: la guía recomienda diez o más para Friedman. El Wilcoxon alcanza su mínimo posible, pero el Conover no distingue a ALEE de la regresión bayesiana.
- En Colombia se evaluaron diez iteraciones de diseño sobre los mismos 240 meses, lo que puede volver optimista el resultado en ese conjunto. Los ocho conjuntos adicionales, evaluados una sola vez con el modelo congelado, son la evidencia más confiable.
- Varias piezas del análisis se agregaron después de conocer los resultados (capítulo 6), aunque ninguna modifica las predicciones de ALEE.
- La prueba bayesiana con ROPE supone particiones intercambiables y aquí se aplica sobre pliegues temporales dependientes: sus probabilidades son orientativas.
- Aunque todas las variables se construyen solo con información pasada, puede quedar fuga residual por revisiones de las series: los datos son la versión vigente y no la publicada en tiempo real cada mes.
- Los conjuntos adicionales no tienen indicadores de área: en ellos todos los modelos usan 14 variables y no 17.
- Las categorías de clasificación del segundo entregable son una decisión del equipo.

## Riesgos

El pronóstico es una herramienta académica. Usarlo para decisiones de política sin la incertidumbre que lo acompaña, o sin el criterio de expertos, podría llevar a reaccionar ante cambios que no ocurren.

## Trabajo futuro

- Variables de estado que reaccionen antes a los saltos bruscos: hoy la volatilidad de tres meses y la brecha frente a la media de doce se basan en meses ya observados, y ALEE reacciona tarde al inicio de una crisis.
- Pronósticos a más de un mes.
- Más conjuntos de datos, para dar más potencia a las pruebas entre conjuntos.

## Declaración de autoría y uso de inteligencia artificial

Este trabajo fue realizado por Andrés Parejo, Santiago Hurtado y Juan Marín, estudiantes del Pregrado en Ciencia de Datos de la Universidad del Norte, en el curso de Machine Learning.

**Herramienta usada.** Se usó **Claude Code (Anthropic)**, un asistente de programación con inteligencia artificial, durante el desarrollo del tercer entregable.

**Para qué se usó:**

- reorganizar el repositorio con la estructura de la guía y dividir el libro en capítulos;
- programar y probar el módulo de artefactos, el modelo original (ALEE) y sus pruebas unitarias, los experimentos de la validación confirmatoria y de los nueve conjuntos, el dashboard y los scripts de precálculo;
- redactar borradores del texto del libro, del paper, del protocolo, de la ficha del modelo y de la declaración de los datos;
- ejecutar, con el acceso institucional del equipo, la búsqueda en Scopus de la revisión de literatura, y registrar sus resultados;
- revisar la coherencia entre las cifras del texto y las de los archivos de resultados.

**Para qué no se usó.** La herramienta no inventó datos, resultados ni referencias: todas las cifras del libro, del paper y del dashboard se calculan con el código del repositorio a partir de datos públicos citados, y cada referencia se verificó en su fuente. Las decisiones de diseño del modelo, la elección de los conjuntos de datos y de las pruebas, la decisión de qué modelos comparar y la interpretación de los resultados fueron del equipo.

**Responsabilidad.** El equipo revisó el código y los textos generados con ayuda de la herramienta, los corrigió cuando fue necesario y asume la responsabilidad completa sobre el contenido. Cada integrante puede explicar y defender el código y los resultados.

Andrés Parejo · Santiago Hurtado · Juan Marín

Barranquilla, noviembre de 2026
