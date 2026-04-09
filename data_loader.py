import pandas as pd


def cargar_datos(path):
    df = pd.read_excel(path)

    # =====================================
    # LIMPIAR NOMBRES DE COLUMNAS
    # =====================================
    df.columns = [
        str(c).strip().lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(".", "")
        for c in df.columns
    ]

    print("Columnas detectadas en el Excel:")
    print(df.columns.tolist())

    # =====================================
    # CREAR / IDENTIFICAR COLUMNA FECHA
    # =====================================

    # Caso 1: ya existe una columna tipo fecha
    posibles_fecha = ["fecha", "date", "periodo", "period", "mes_fecha"]

    for col in posibles_fecha:
        if col in df.columns:
            df["fecha"] = pd.to_datetime(df[col], errors="coerce")
            break

    # Caso 2: si NO existe fecha, intentar construirla con año + mes
    if "fecha" not in df.columns or df["fecha"].isna().all():

        # posibles nombres de año y mes
        posibles_anio = ["anio", "año", "year"]
        posibles_mes = ["mes", "month"]

        col_anio = next((c for c in posibles_anio if c in df.columns), None)
        col_mes = next((c for c in posibles_mes if c in df.columns), None)

        if col_anio and col_mes:
            df["fecha"] = pd.to_datetime(
                df[col_anio].astype(str) + "-" + df[col_mes].astype(str) + "-01",
                errors="coerce"
            )

    # =====================================
    # VALIDACIÓN FINAL
    # =====================================
    if "fecha" not in df.columns or df["fecha"].isna().all():
        raise ValueError(
            "No se pudo encontrar ni construir la columna 'fecha'. "
            "Revisa cómo se llama en tu Excel."
        )

    # =====================================
    # ORDENAR Y LIMPIAR
    # =====================================
    df = df.sort_values("fecha").reset_index(drop=True)

    return df