# MLC VIZ DASH

# Visualizacion del mercado laboral colombiano.

El mercado laboral colombiano presenta dinamicas complejas que varian significativamente entre regiones, grupos poblacionales y ciclos economicos. Comprender estas dinamicas es esencial para el diseño de politicas publicas efectivas, la toma de decisiones empresariales y la investigacion academica en economia laboral.

## Contexto.

Colombia cuenta con una de las encuestas de hogares mas completas de America Latina: la Gran Encuesta Integrada de Hogares (GEIH), administrada por el Departamento Administrativo Nacional de Estadistica (DANE). Esta encuesta produce mensualmente indicadores clave del mercado de trabajo a nivel nacional y por areas metropolitanas, con series que se remontan al ano 2001.

A pesar de la riqueza de esta informacion, su exploracion requiere procesamiento estadistico especializado y herramientas que permitan visualizar tendencias, comparar ciudades y analizar el comportamiento de indicadores en el tiempo, incluyendo periodos de choque como la pandemia de COVID-19 (2020).

Este proyecto construye un dashboard interactivo que centraliza el analisis exploratorio de los principales indicadores del mercado laboral colombiano, con capacidad de modelacion de series de tiempo mediante modelos predictivos para la tasa de desempleo nacional.

## Indicadores cubiertos.

- Tasa Global de Participación – 13 áreas (TGP área).

- Tasa Global de Participación – Total nacional (TGP nacional).

- Tasa de Desempleo – 13 áreas.

- Tasa de Desempleo – Total nacional.

- Tasa de Ocupación – 13 áreas.

- Tasa de Ocupación – Total nacional.

## Ejecutar la aplicacion

Todos los archivos necesarios para correr la aplicacion localmente se encuentran en la raiz del repositorio. El dashboard fue desarrollado en **Python** utilizando la libreria **Dash** de Plotly. Es necesario tener Python instalado en el computador para que la aplicacion funcione.

**0. (Opcional)** Se recomienda crear un entorno virtual para evitar conflictos con otras librerias instaladas en el sistema. [Guia de entornos virtuales en Python](https://towardsdatascience.com/virtual-environments-104c62d48c54)

```
python -m venv venv
```

En Windows:

```
venv\Scripts\activate
```

En macOS / Linux:

```
source venv/bin/activate
```

**1.** Clonar el repositorio:

```
git clone https://github.com/dev-boolean/MLC_VIZ_DASH.git
cd MLC_VIZ_DASH
```

**2.** Instalar las dependencias listadas en `requirements.txt`:

```
pip install -r requirements.txt
```

**3.** Una vez completada la instalacion, iniciar la aplicacion:

```
python app.py
```

**4.** Acceder a la aplicacion en el navegador mediante el siguiente enlace:

```
http://127.0.0.1:8050/
```

## Estructura del repositorio

```
MLC_VIZ_DASH/
|
|-- assets/             # Estilos CSS y recursos estaticos
|-- data/               # Series procesadas del GEIH/DANE (.xlsx)
|-- app.py              # Punto de entrada de la aplicacion
|-- layout.py           # Estructura visual y componentes UI
|-- callbacks.py        # Logica de interactividad (inputs / outputs)
|-- charts.py           # Funciones de visualizacion con Plotly
|-- data_loader.py      # Carga y procesamiento de datos
|-- stats.py            # Calculos estadisticos y modelacion SARIMA
|-- utils.py            # Funciones auxiliares
|
|-- requirements.txt    # Dependencias Python
|-- runtime.txt         # Version de Python para despliegue
|-- Procfile            # Configuracion para Render
```

## Dependencias

| Paquete | Uso |
|---|---|
| `dash` | Framework principal del dashboard |
| `dash-bootstrap-components` | Componentes UI con Bootstrap |
| `plotly` | Graficos interactivos |
| `pandas` | Manipulacion y transformacion de datos |
| `openpyxl` | Lectura de archivos `.xlsx` |
| `statsmodels` | Modelacion de series de tiempo |
| `gunicorn` | Servidor WSGI para entornos de produccion |

## Datos

Los datos se encuentran en la carpeta `data/` y corresponden a series historicas procesadas del GEIH, con cobertura nacional y por principales ciudades y areas metropolitanas de Colombia.

**Fuente:** Departamento Administrativo Nacional de Estadistica (DANE) — Gran Encuesta Integrada de Hogares (GEIH), 2001–2025.

## Implementacion en R (Shiny)

Los archivos de la aplicacion Shiny se encuentran comprimidos en MLC_READY.zip en la raiz del repositorio. Es necesario tener R instalado en el computador para que la aplicacion funcione. Se recomienda usar RStudio.

```
MLC_READY/
|-- global.R
|-- ui.R
|-- server.R
```
Cambiar la ruta en el archivo de global.R.

```
MLC <- read_excel(ruta del archivo)
```
Ahora instala las librerias necesarias desde la consola.
```
install.packages(c(
  "shiny",
  "shinydashboard",
  "tidyverse",
  "lubridate",
  "readxl",
  "corrplot",
  "forecast",
  "tseries",
  "Amelia",
  "DT",
  "plotly"
))
```
Por ultimo correr el proyecto donde dice Run App, arriba del panel del código.

## Equipo.

Este proyecto fue desarrollado por:

- Santiago Hurtado.

- Andres Parejo.

