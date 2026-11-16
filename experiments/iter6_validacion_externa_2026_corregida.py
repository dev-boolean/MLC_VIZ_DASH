"""Repite la validación externa de 2026 (`iter6_validacion_externa_2026.py`)
pero con ALEE corregido: el experto nucleo_cambio excluye AREA_COLS de su
kernel (ver `iter6_nucleo_sin_area.py`). Como el kernel ya no usa esas
columnas, se rellenan con el último valor real conocido (noviembre de 2025)
en vez de ceros, solo por prolijidad -- no deberían influir en nada.

Hiperparámetros de ALEE: los del pliegue 4 (más datos de entrenamiento) de
`artifacts/confirmatorio/iter6/rmse_nucleo_sin_area.csv`, ya elegidos en la
Tarea 2 (nested CV honesta sobre los 240 meses) -- NO se reoptimizan aquí
mirando los datos de 2026.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.data import load_raw, assemble_dataset, AREA_COLS, LAGS, ROLLING_WINDOWS
from experiments.iter6_nucleo_sin_area import ModeloNucleoSinArea

OUT = config.ROOT_DIR / "artifacts" / "confirmatorio" / "externa_2026"
OUT.mkdir(parents=True, exist_ok=True)

raw = load_raw()
ext2026 = pd.read_csv(config.ROOT_DIR / "artifacts" / "confirmatorio" / "colombia_2026_tasa_nacional_verificada.csv")
ext2026["fecha"] = pd.to_datetime(ext2026["mes"] + "-01") + pd.offsets.MonthEnd(0)

serie = pd.concat(
    [raw[["fecha", "tasa_desempleo_nacional"]], ext2026[["fecha", "tasa_desempleo_nacional"]]],
    ignore_index=True,
).sort_values("fecha").reset_index(drop=True)

y_full = serie["tasa_desempleo_nacional"].values
fechas_full = serie["fecha"].values
n_hist = len(raw)
n_2026 = len(ext2026)
first_2026 = n_hist

# últimos valores reales conocidos de área (nov-2025) -- no influyen en el kernel corregido,
# se usan solo para no rellenar con un valor artificial sin sentido (p. ej. 0).
ultimos_area = raw[AREA_COLS].iloc[-1].to_dict()

FEATURE_ORDER = (
    [f"td_nacional_lag{lag}" for lag in LAGS]
    + [f"td_nacional_roll{stat}{w}" for w in ROLLING_WINDOWS for stat in ("mean", "std")]
    + ["mes_sin", "mes_cos", "trend_idx"]
    + AREA_COLS
)


def build_X_row(y_hist, fecha):
    row = {}
    for lag in LAGS:
        row[f"td_nacional_lag{lag}"] = y_hist[-lag]
    for w in ROLLING_WINDOWS:
        window = y_hist[-w:]
        row[f"td_nacional_rollmean{w}"] = np.mean(window)
        row[f"td_nacional_rollstd{w}"] = np.std(window, ddof=1)
    row["mes_sin"] = np.sin(2 * np.pi * fecha.month / 12)
    row["mes_cos"] = np.cos(2 * np.pi * fecha.month / 12)
    row["trend_idx"] = len(y_hist) - 1
    for c in AREA_COLS:
        row[c] = ultimos_area[c]
    return np.array([row[c] for c in FEATURE_ORDER], dtype=float)


ds = assemble_dataset()
hp = pd.read_csv(config.ROOT_DIR / "artifacts" / "confirmatorio" / "iter6" / "rmse_nucleo_sin_area.csv")
params_final = hp.iloc[4][["bandwidth", "half_life", "lam"]].to_dict()
print("Hiperparámetros ALEE (núcleo sin área) usados, ya elegidos en la Tarea 2:", params_final)

alee = ModeloNucleoSinArea(**params_final).fit(ds.X.values, ds.y_reg.values)

# ---- reusa los resultados ya calculados de persistencia/estacional/SARIMA (no cambian) ----
anterior = pd.read_csv(OUT / "predicciones_2026.csv")

y_hist_list = list(y_full[:n_hist])
alee_preds = []
for i in range(n_2026):
    fecha_obj = pd.Timestamp(fechas_full[first_2026 + i])
    x_row = build_X_row(np.array(y_hist_list), fecha_obj).reshape(1, -1)
    alee_preds.append(float(alee.predict(x_row)[0]))
    y_hist_list.append(y_full[first_2026 + i])

res = anterior.copy()
res["ALEE_3_nucleo_sin_area"] = alee_preds
res.to_csv(OUT / "predicciones_2026_corregida.csv", index=False)
print(res.round(3))

modelos = ["Persistencia", "Estacional_aislado", "SARIMA", "ALEE_3", "ALEE_3_nucleo_sin_area"]
resumen = []
for m in modelos:
    err = res["y_true"] - res[m]
    resumen.append({"modelo": m, "rmse_2026": np.sqrt(np.mean(err ** 2)), "mae_2026": np.mean(np.abs(err))})
resumen = pd.DataFrame(resumen).sort_values("rmse_2026")
resumen.to_csv(OUT / "rmse_mae_2026_corregida.csv", index=False)
print("\n=== RMSE/MAE sobre los 9 meses reales de 2026 (n=9, SIN potencia estadística) ===")
print(resumen.to_string(index=False))
print("\nErrores mes a mes de ALEE corregido (y_true - pred):", (res["y_true"] - res["ALEE_3_nucleo_sin_area"]).round(3).tolist())
print("Errores mes a mes de ALEE original (y_true - pred):   ", (res["y_true"] - res["ALEE_3"]).round(3).tolist())
