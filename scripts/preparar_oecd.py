"""Filtra las series de la OCDE (DSD_LFS@DF_IALFS_UNE_M) según los criterios de inclusión del protocolo §16.4.

Uso: python scripts/preparar_oecd.py CAN AUS JPN DEU KOR
Para cada país lee data/adicionales/oecd_<ISO>_crudo.csv y escribe data/adicionales/oecd_<ISO>.csv
(columnas fecha, valor) con: ajuste N (sin desestacionalizar), sexo total, edad 15+ (o 15-74, o 15-64),
frecuencia mensual, y el tramo sin huecos más largo que termina en el último mes disponible
(un mes faltante aislado se interpola). Solo usa metadatos de la serie, nunca resultados de modelos.
"""
import sys
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "adicionales"


def preparar(iso):
    d = pd.read_csv(DATA / f"oecd_{iso}_crudo.csv")
    d = d[(d.ADJUSTMENT == "N") & (d.SEX == "_T") & (d.FREQ == "M") & (d.MEASURE == "UNE_LF_M")]
    for edad in ["Y_GE15", "Y15T74", "Y15T64"]:
        s = d[d.AGE == edad]
        if len(s):
            break
    else:
        return {"pais": iso, "incluido": False, "motivo": "sin serie mensual sin ajuste"}
    s = s.assign(fecha=pd.to_datetime(s.TIME_PERIOD)).sort_values("fecha").set_index("fecha")["OBS_VALUE"].astype(float)
    s = s[~s.index.duplicated()]
    full = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    s = s.reindex(full)
    # tramo sin huecos (salvo meses aislados) que termina en el último mes
    falta = s.isna().to_numpy()
    inicio = 0
    for i in range(len(s) - 1, 0, -1):
        if falta[i] and (falta[i - 1] or (i + 1 < len(s) and falta[i + 1])):
            inicio = i + 1
            break
    s = s.iloc[inicio:]
    n_interp = int(s.isna().sum())
    s = s.interpolate(limit=1)
    ok = len(s) >= 240 and s.notna().all()
    if ok:
        pd.DataFrame({"fecha": s.index.strftime("%Y-%m"), "valor": s.values}).to_csv(DATA / f"oecd_{iso}.csv", index=False)
    return {"pais": iso, "edad": edad, "desde": s.index.min().strftime("%Y-%m"), "hasta": s.index.max().strftime("%Y-%m"),
            "meses": len(s), "meses_interpolados": n_interp, "incluido": ok}


if __name__ == "__main__":
    print(pd.DataFrame([preparar(c) for c in sys.argv[1:]]).to_string(index=False))
