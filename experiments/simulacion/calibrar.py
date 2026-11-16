import sys, numpy as np, pandas as pd
df = pd.read_excel(sys.argv[1], sheet_name="Series de datos").sort_values("fecha").reset_index(drop=True)
y = df["tasa_desempleo_nacional"].values; m = df["fecha"].dt.month.values
print("n", len(y), df.fecha.min(), df.fecha.max(), "mean", y.mean().round(2), "min/max", y.min().round(2), y.max().round(2))
dy = np.diff(y); md = m[1:]
seas = np.array([dy[md==k].mean() for k in range(1,13)])
print("cambio medio por mes:", seas.round(3))
res = dy - seas[md-1]
print("std resid total", res.std().round(3))
yrs = df.fecha.dt.year.values[1:]
mask = (yrs<2020)|(yrs>2021)
print("std resid sin 2020-21", res[mask].std().round(3), " std en 2020-21", res[~mask].std().round(3))
print("autocorr resid lag1 (sin covid)", np.corrcoef(res[mask][1:], res[mask][:-1])[0,1].round(3))
# nivel desestacionalizado: tendencia
lvl = pd.Series(y).rolling(12, center=True).mean()
print("nivel suavizado cada 2 anios:", lvl.iloc[::24].round(2).tolist())
print("2020:", y[(df.fecha.dt.year==2020).values].round(1))
