"""Descarga las tres series de los conjuntos adicionales a data/adicionales/.

Uso: python scripts/descargar_conjuntos_adicionales.py

Fuentes (sin desestacionalizar), consultadas el 2026-11-08:
* EE.UU.: FRED, serie UNRATENSA.
* España: Eurostat, une_rt_m (M.NSA.TOTAL.PC_ACT.T.ES), formato SDMX-CSV.
* México: API SDMX de la OCDE, DSD_LFS@DF_IALFS_UNE_M, filtrada a ajuste N, sexo total,
  edad 15+ y frecuencia mensual (redistribuye la ENE/ENOE de INEGI).
Las series se revisan con el tiempo; las copias usadas en el paper están versionadas en data/adicionales/.
"""
import csv
import io
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "adicionales"
FUENTES = {
    "fred_UNRATENSA.csv": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATENSA",
    "es_une_rt_m.csv": "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/une_rt_m/M.NSA.TOTAL.PC_ACT.T.ES?format=SDMX-CSV",
    "oecd_mex_crudo.csv": "https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,DSD_LFS@DF_IALFS_UNE_M,1.0/MEX..........?startPeriod=2000-01",
    # protocolo §19 (filtrar después con scripts/preparar_oecd.py)
    **{f"oecd_{iso}_crudo.csv": f"https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,DSD_LFS@DF_IALFS_UNE_M,1.0/{iso}..........?startPeriod=1950-01"
       for iso in ["CAN", "AUS", "JPN", "DEU", "KOR"]},
}


def bajar(url, accept=None):
    req = urllib.request.Request(url, headers={"Accept": accept} if accept else {})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read().decode("utf-8")


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    for nombre, url in FUENTES.items():
        accept = "application/vnd.sdmx.data+csv; charset=utf-8" if "oecd" in url else None
        (DATA / nombre).write_text(bajar(url, accept), encoding="utf-8")
        print("descargado", nombre)
    # México: total, 15+ años, sin ajuste, mensual, ordenado por fecha (mismo formato que mex_nsa.csv)
    filas = list(csv.reader(io.StringIO((DATA / "oecd_mex_crudo.csv").read_text(encoding="utf-8"))))
    sel = [f for f in filas[1:] if len(f) > 11 and f[5] == "N" and f[6] == "_T" and f[7] == "Y_GE15" and f[9] == "M"]
    sel.sort(key=lambda f: f[10])
    with open(DATA / "mex_nsa.csv", "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(sel)
    print("mex_nsa.csv:", len(sel), "meses")


if __name__ == "__main__":
    main()
