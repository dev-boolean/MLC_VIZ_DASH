# Declaración de los datos

## Procedencia

- **Archivo:** `data/MLC.xlsx`, hojas «Series de datos» y «Metadatos», descargado del sitio del Banco de la República (grupo de series «Mercado laboral, indicadores del mercado laboral»).
- **Productor:** Departamento Administrativo Nacional de Estadística (DANE), Gran Encuesta Integrada de Hogares (GEIH).
- **Series:** tasa global de participación, tasa de desempleo y tasa de ocupación, para el total nacional y para las 13 ciudades y áreas metropolitanas. Mensuales, en porcentaje, de enero de 2001 a diciembre de 2025 (300 meses, sin valores faltantes).

## Licencia

Son datos públicos. **Por verificar:** los términos de uso y la forma de citación que exige el Banco de la República. Mientras tanto se cita como «DANE, GEIH; series del Banco de la República».

## Preprocesamiento

`src/data.py` construye el problema de pronóstico a un mes: rezagos (1, 2, 3, 6 y 12 meses), medias y desviaciones móviles (3, 6 y 12 meses), el mes codificado con seno y coseno, un índice de tendencia y los tres indicadores de las 13 áreas. Quedan 288 filas con 17 variables. La ocupación y la participación nacionales del mismo mes se excluyen por la identidad contable TD = 1 − TO/TGP.

## Sesgos y limitaciones conocidos

- **Encuesta:** las cifras son estimaciones de una encuesta por muestreo, con error muestral, y el DANE ha actualizado su metodología a lo largo del periodo. Hay que revisar en la documentación del DANE si hubo cambios de marco o de factores de expansión y cómo afectan la comparabilidad.
- **Cobertura:** el total nacional promedia realidades urbanas y rurales muy distintas, y las 13 áreas solo representan a las ciudades principales.
- **Choque de 2020:** la pandemia produce valores extremos (21.97 % en mayo de 2020) que dominan las métricas de error.
- **Categorías:** los cortes de clasificación (10 % y 14 %) son una decisión del equipo.

## Usos inadecuados

No deben sacarse de estos datos conclusiones sobre grupos poblacionales (sexo, edad, región), porque las series son agregadas.
