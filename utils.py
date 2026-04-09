import pandas as pd

DATA_PATH = "data/MLC.xlsx"
UMBRAL_ALERTA = 12


def cargar_datos():
    df = pd.read_excel(DATA_PATH, sheet_name="Series de datos")
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.sort_values("fecha").reset_index(drop=True)
    return df


def filtrar_por_fecha(df, start_date=None, end_date=None):
    dff = df.copy()
    if start_date:
        dff = dff[dff["fecha"] >= pd.to_datetime(start_date)]
    if end_date:
        dff = dff[dff["fecha"] <= pd.to_datetime(end_date)]
    return dff.reset_index(drop=True)


def transformar_a_largo(df):
    df_long = df.melt(
        id_vars=["fecha"],
        var_name="serie",
        value_name="valor"
    )

    df_long["cobertura"] = df_long["serie"].apply(
        lambda x: "Nacional" if "nacional" in x else "13 áreas"
    )

    df_long["indicador"] = df_long["serie"].apply(extraer_indicador)
    return df_long.sort_values("fecha").reset_index(drop=True)


def extraer_indicador(nombre):
    if "desempleo" in nombre:
        return "Desempleo"
    elif "ocupacion" in nombre:
        return "Ocupación"
    elif "participacion" in nombre:
        return "Participación"
    return "Otro"


def obtener_kpis(df):
    ultima = df.iloc[-1]
    return {
        "media_desempleo": round(df["tasa_desempleo_nacional"].mean(), 2),
        "min_desempleo": round(df["tasa_desempleo_nacional"].min(), 2),
        "min_fecha": df.loc[df["tasa_desempleo_nacional"].idxmin(), "fecha"].strftime("%b %Y"),
        "max_desempleo": round(df["tasa_desempleo_nacional"].max(), 2),
        "max_fecha": df.loc[df["tasa_desempleo_nacional"].idxmax(), "fecha"].strftime("%b %Y"),
        "obs": len(df),
        "ultimo_desempleo": round(ultima["tasa_desempleo_nacional"], 2),
        "ultima_fecha": ultima["fecha"].strftime("%b %Y")
    }


def generar_alerta(df):
    ult = df["tasa_desempleo_nacional"].iloc[-1]
    fecha = df["fecha"].iloc[-1].strftime("%b %Y")

    if ult > UMBRAL_ALERTA + 3:
        return {
            "clase": "danger",
            "icono": "🚨",
            "titulo": f"ALERTA CRÍTICA — Desempleo en {ult:.1f}%",
            "texto": f"El último dato disponible ({fecha}) supera en más de 3 puntos el umbral de alerta ({UMBRAL_ALERTA}%)."
        }
    elif ult > UMBRAL_ALERTA:
        return {
            "clase": "warning",
            "icono": "⚠️",
            "titulo": f"ATENCIÓN — Desempleo en {ult:.1f}%",
            "texto": f"El último dato disponible ({fecha}) supera el umbral de alerta del {UMBRAL_ALERTA}%."
        }
    else:
        return {
            "clase": "ok",
            "icono": "✅",
            "titulo": f"BAJO CONTROL — Desempleo en {ult:.1f}%",
            "texto": f"El último dato disponible ({fecha}) se encuentra por debajo del umbral de alerta ({UMBRAL_ALERTA}%)."
        }