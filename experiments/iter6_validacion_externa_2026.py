"""Validación externa real: 9 meses de 2026 (ene-sep), nunca vistos por ningún diseño.

Evalúa persistencia, estacional aislado, SARIMA y ALEE-3 (iteración 4) contra
los valores reales verificados en `colombia_2026_tasa_nacional_verificada.csv`.
Los hiperparámetros de ALEE NO se reoptimizan aquí: se usan los de la última
fila (pliegue 4) de `artifacts/confirmatorio/iter4/rmse_K3_sin_ridge.csv`, ya
elegidos antes de ver estos 9 meses.

Solo se evalúan los modelos cuyas variables dependen exclusivamente de la
serie nacional (persistencia, estacional, SARIMA, ALEE-3): los indicadores de
las 13 áreas metropolitanas para 2026 no están disponibles, así que Ridge,
XGBoost, etc. quedan fuera de esta validación.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from statsmodels.tsa.statespace.sarimax import SARIMAX

from src import config
from src.data import load_raw, AREA_COLS, LAGS, ROLLING_WINDOWS
from src.original_model import ModeloOriginal, _month_of

OUT = config.ROOT_DIR / "artifacts" / "confirmatorio" / "externa_2026"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 1. serie extendida
raw = load_raw()
ext2026 = pd.read_csv(config.ROOT_DIR / "artifacts" / "confirmatorio" / "colombia_2026_tasa_nacional_verificada.csv")
ext2026["fecha"] = pd.to_datetime(ext2026["mes"] + "-01") + pd.offsets.MonthEnd(0)

serie = pd.concat(
    [raw[["fecha", "tasa_desempleo_nacional"]],
     ext2026[["fecha", "tasa_desempleo_nacional"]]],
    ignore_index=True,
).sort_values("fecha").reset_index(drop=True)

# chequeo de continuidad mensual sin huecos
deltas = serie["fecha"].diff().dropna()
assert all(d.days in (28, 29, 30, 31) for d in deltas), "hay huecos de mes en la serie extendida"
print(f"Serie extendida: {serie['fecha'].min().date()} -> {serie['fecha'].max().date()}  ({len(serie)} meses)")

y_full = serie["tasa_desempleo_nacional"].values
fechas_full = serie["fecha"].values
n_hist = len(raw)  # filas hasta dic-2025 (incluida)
n_2026 = len(ext2026)  # 9 meses, ene-sep 2026
assert len(y_full) == n_hist + n_2026

# índice (0-based) del primer mes de 2026 en la serie extendida
first_2026 = n_hist  # dic-2025 es n_hist-1

# ---------------------------------------------------------------- 2. features nacionales
def build_national_features_row(y_hist, fecha):
    """Rezagos/medias/desv./seno-coseno/tendencia a partir de la serie nacional
    ``y_hist`` (todo lo conocido hasta el mes anterior al que se pronostica, inclusive),
    en el mismo orden de columnas que ``DEFAULT_FEATURES`` (sin las 3 de área)."""
    row = {}
    for lag in LAGS:
        row[f"td_nacional_lag{lag}"] = y_hist[-lag]
    for w in ROLLING_WINDOWS:
        window = y_hist[-w:]
        row[f"td_nacional_rollmean{w}"] = np.mean(window)
        row[f"td_nacional_rollstd{w}"] = np.std(window, ddof=1)
    row["mes_sin"] = np.sin(2 * np.pi * fecha.month / 12)
    row["mes_cos"] = np.cos(2 * np.pi * fecha.month / 12)
    row["trend_idx"] = len(y_hist) - 1  # mismo criterio que build_features (arange)
    return row


FEATURE_ORDER = (
    [f"td_nacional_lag{lag}" for lag in LAGS]
    + [f"td_nacional_roll{stat}{w}" for w in ROLLING_WINDOWS for stat in ("mean", "std")]
    + ["mes_sin", "mes_cos", "trend_idx"]
    + AREA_COLS
)


def build_X_row(y_hist, fecha):
    row = build_national_features_row(y_hist, fecha)
    for c in AREA_COLS:
        row[c] = 0.0  # ALEE no usa estas columnas; se incluyen solo para el shape (17,)
    return np.array([row[c] for c in FEATURE_ORDER], dtype=float)


# ---------------------------------------------------------------- 3. modelo final (hasta nov-2025, como el proyecto)
# El dataset oficial (ds.X/ds.y_reg) descarta las primeras 12 filas (sin lag12) y
# la última (sin mes siguiente): fila r predice con info hasta dic/raw[r], target
# en raw[r+1]. Reconstruyo X_hist/y_hist con el mismo criterio para entrenar ALEE
# con TODA la información hasta nov-2025 (target dic-2025), igual que el modelo vigente.
from src.data import assemble_dataset

ds = assemble_dataset()
assert ds.fechas.max() == raw["fecha"].iloc[-2]  # última fila predice dic-2025 con info de nov-2025

hp = pd.read_csv(config.ROOT_DIR / "artifacts" / "confirmatorio" / "iter4" / "rmse_K3_sin_ridge.csv")
params_final = hp.iloc[4][["bandwidth", "half_life", "lam"]].to_dict()  # pliegue 4: más datos de entrenamiento
print("Hiperparámetros ALEE-3 usados (ya elegidos en iteración 4, NO reoptimizados aquí):", params_final)

alee = ModeloOriginal(**params_final).fit(ds.X.values, ds.y_reg.values)

# ---------------------------------------------------------------- 4. SARIMA: fit en todo el histórico, extiende con refit=False
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sarima_fit = SARIMAX(y_full[:n_hist], order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False, maxiter=200)
sarima_ext = sarima_fit.append(y_full[n_hist:], refit=False)
sarima_preds_2026 = np.asarray(
    sarima_ext.get_prediction(start=n_hist, end=n_hist + n_2026 - 1, dynamic=False).predicted_mean
)

# ---------------------------------------------------------------- 5. walk-forward para persistencia/estacional/ALEE
# Para el experto estacional "aislado" (fuera de ALEE) reproducimos exactamente la
# regla: lag1 + cambio medio histórico del mismo mes (con la misma shrinkage +1).
def estacional_aislado_pred(y_hist_vals, fechas_hist, fecha_obj):
    lag1 = y_hist_vals[-1]
    dy = np.diff(y_hist_vals)  # dy[i] = y[i+1]-y[i], asociado al mes de fechas_hist[i+1]
    meses_dy = pd.DatetimeIndex(fechas_hist[1:]).month
    sel = meses_dy == fecha_obj.month
    mc = dy[sel].sum() / (sel.sum() + 1.0)
    return lag1 + mc


rows = []
fechas_hist_list = list(pd.DatetimeIndex(fechas_full[:n_hist]))
y_hist_list = list(y_full[:n_hist])
for i in range(n_2026):
    fecha_obj = pd.Timestamp(fechas_full[first_2026 + i])
    y_true = y_full[first_2026 + i]

    pers_pred = y_hist_list[-1]
    est_pred = estacional_aislado_pred(np.array(y_hist_list), np.array(fechas_hist_list), fecha_obj)
    sarima_pred = sarima_preds_2026[i]

    x_row = build_X_row(np.array(y_hist_list), fecha_obj).reshape(1, -1)
    alee_pred = float(alee.predict(x_row)[0])

    rows.append({
        "mes": fecha_obj.strftime("%Y-%m"), "y_true": y_true,
        "Persistencia": pers_pred, "Estacional_aislado": est_pred,
        "SARIMA": sarima_pred, "ALEE_3": alee_pred,
    })

    # walk-forward: ahora sí se conoce el valor real de este mes
    fechas_hist_list.append(fecha_obj)
    y_hist_list.append(y_true)

res = pd.DataFrame(rows)
res.to_csv(OUT / "predicciones_2026.csv", index=False)
print(res.round(3))

modelos = ["Persistencia", "Estacional_aislado", "SARIMA", "ALEE_3"]
resumen = []
for m in modelos:
    err = res["y_true"] - res[m]
    rmse = np.sqrt(np.mean(err ** 2))
    mae = np.mean(np.abs(err))
    resumen.append({"modelo": m, "rmse_2026": rmse, "mae_2026": mae})
resumen = pd.DataFrame(resumen).sort_values("rmse_2026")
resumen.to_csv(OUT / "rmse_mae_2026.csv", index=False)
print("\n=== RMSE/MAE sobre los 9 meses reales de 2026 (n=9, SIN potencia estadística) ===")
print(resumen.to_string(index=False))

print("\nErrores mes a mes (y_true - pred):")
for m in modelos:
    print(f"  {m:20s}", (res["y_true"] - res[m]).round(3).tolist())
