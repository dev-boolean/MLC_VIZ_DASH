"""Uso: python -P analizar.py <raiz_proyecto> <csv1> [<csv2> ...]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1])
from src.stats_tests import diebold_mariano_test

df = pd.concat([pd.read_csv(p) for p in sys.argv[2:]], ignore_index=True)
rmse = lambda e: float(np.sqrt(np.mean(np.square(e))))
rows = []
for (sc, sd), g in df.groupby(["escenario", "semilla"]):
    eA, eE, eP = g.y - g.ALEE, g.y - g.Estacional, g.y - g.Persistencia
    fA = g.groupby("pliegue").apply(lambda h: rmse(h.y - h.ALEE)); fE = g.groupby("pliegue").apply(lambda h: rmse(h.y - h.Estacional))
    dm = diebold_mariano_test(eA, eE)
    r = dict(escenario=sc, semilla=sd, n_test=len(g), RMSE_ALEE=rmse(eA), RMSE_Estacional=rmse(eE), RMSE_Persistencia=rmse(eP),
             pliegue_ALEE=f"{fA.mean():.3f}±{fA.std():.3f}", pliegue_Est=f"{fE.mean():.3f}±{fE.std():.3f}",
             DM_p=dm["p_value"], dif_perdida=dm["mean_loss_diff"])
    for flag, name in [(False, "normal"), (True, "crisis")]:
        h = g[g.crisis == flag]
        if len(h):
            r[f"n_{name}"] = len(h); r[f"ALEE_{name}"] = rmse(h.y - h.ALEE); r[f"Est_{name}"] = rmse(h.y - h.Estacional); r[f"Pers_{name}"] = rmse(h.y - h.Persistencia)
    rows.append(r)
t = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(t.round(4).to_string(index=False))
print("\nResumen por escenario (todas las semillas juntas):")
for sc, g in df.groupby("escenario"):
    eA, eE = g.y - g.ALEE, g.y - g.Estacional
    wins = (t[t.escenario == sc].RMSE_ALEE < t[t.escenario == sc].RMSE_Estacional).sum()
    sig = ((t[t.escenario == sc].DM_p < 0.05) & (t[t.escenario == sc].RMSE_ALEE < t[t.escenario == sc].RMSE_Estacional)).sum()
    print(f"  {sc}: RMSE ALEE={rmse(eA):.3f}  Estacional={rmse(eE):.3f}  mejora={100*(1-rmse(eA)/rmse(eE)):+.1f}%  "
          f"ALEE gana en {wins}/{t[t.escenario==sc].shape[0]} semillas, con p<0.05 en {sig}")
print("\nHiperparámetros elegidos (frecuencia):")
print(df.drop_duplicates(["escenario", "semilla", "pliegue"]).groupby("escenario")[["hp_bandwidth", "hp_half_life", "hp_lam"]].agg(lambda s: s.value_counts().to_dict()))
