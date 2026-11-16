"""Simulación: ALEE vs experto estacional con ~900 meses.

Uso: python -P simular.py <raiz_proyecto> <salida_dir> [semillas]

Dos procesos generadores calibrados con la serie real de Colombia (2001-2025):
  A "estable":   y_t = L_t + S_mes + u_t (patrón estacional fijo).
  B "crisis":    igual, más un régimen de crisis de Markov (entrada 1/120 por mes,
                 duración media 12 meses): salto del nivel (+6 pts, decae),
                 patrón estacional atenuado al 20 % y ruido x2.7.
Validación: TimeSeriesSplit 5 externos / 3 internos, gap=1 (como el proyecto).
Hiperparámetros de ALEE por rejilla en los pliegues internos (no hay optuna en
este entorno; el proyecto usa 40 evaluaciones bayesianas).
"""
import sys, time, itertools, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 2, 3, 4, 5]
sys.path.insert(0, str(ROOT))
from sklearn.model_selection import TimeSeriesSplit
from src.original_model import ModeloOriginal, _Experts
from src.stats_tests import diebold_mariano_test

# ---------------------------------------------------------- calibración (real)
SEAS_CHANGE = np.array([2.998, -1.029, -1.126, 0.253, -0.155, -0.28, 0.416, -0.532, -0.367, -0.524, -0.317, 0.437])
SEAS_CHANGE = SEAS_CHANGE - SEAS_CHANGE.mean()          # sin tendencia
S_LEVEL = np.cumsum(np.roll(SEAS_CHANGE, 0)); S_LEVEL -= S_LEVEL.mean()   # S[m] = nivel estacional del mes m (ene=índice 0)
NOISE_SD = 0.657 / np.sqrt(2)                              # la autocorr -0.47 de las diferencias indica ruido blanco en nivel
N_MONTHS = 900 + 13                                        # 13 se pierden por rezagos/objetivo -> ~900 filas


def simulate(scenario, seed):
    rng = np.random.default_rng(seed)
    n = N_MONTHS
    L = np.empty(n); L[0] = 11.0
    crisis = np.zeros(n, bool); in_c = False
    for t in range(1, n):
        if not in_c and scenario == "crisis" and rng.random() < 1 / 120:
            in_c = True
        elif in_c and rng.random() < 1 / 12:
            in_c = False
        crisis[t] = in_c
        L[t] = 11.0 + 0.995 * (L[t - 1] - 11.0) + rng.normal(0, 0.10)
    month = np.arange(n) % 12                              # 0 = enero
    seas_amp = np.where(crisis, 0.2, 1.0)
    noise = rng.normal(0, NOISE_SD, n) * np.where(crisis, 2.7, 1.0)
    # el salto se acumula mes a mes mientras dura la crisis
    J = np.zeros(n); j = 0.0
    for t in range(1, n):
        if crisis[t] and not crisis[t - 1]:
            j += 6.0
        j *= 0.9
        J[t] = j
    y = L + seas_amp * S_LEVEL[month] + J + noise
    return y, month + 1, crisis


def features(y, month1):
    s = pd.Series(y)
    f = pd.DataFrame({f"td_nacional_lag{k}": s.shift(k - 1) for k in [1, 2, 3, 6, 12]})
    for w in [3, 6, 12]:
        f[f"td_nacional_rollmean{w}"] = s.rolling(w).mean()
        f[f"td_nacional_rollstd{w}"] = s.rolling(w).std()
    f["mes_sin"] = np.sin(2 * np.pi * month1 / 12); f["mes_cos"] = np.cos(2 * np.pi * month1 / 12)
    f["trend_idx"] = np.arange(len(y))
    target = s.shift(-1)
    ok = f.notna().all(axis=1) & target.notna()
    return f[ok].reset_index(drop=True), target[ok].to_numpy(), np.flatnonzero(ok.to_numpy())


# --------------------------------------------- ALEE con errores precalculados
class Pre:
    """Errores fuera de muestra de los expertos, calculados una vez para toda la serie.

    El error del mes s solo depende de los meses 0..s-1 (ventana expansiva), así
    que el ajuste de ModeloOriginal sobre el prefijo 0..n'-1 usa exactamente las
    primeras filas de esta tabla. Se verifica contra .fit() en `check_equivalence`.
    """
    def __init__(self, X, y, cols, m0=36):
        self.m0 = m0
        self.E = np.array([y[s] - _Experts(X[:s], y[:s], cols).predict(X[s:s + 1])[0] for s in range(m0, len(y))])


def fit_fast(params, X, y, n_train, pre, feat_names):
    m = ModeloOriginal(feature_names=tuple(feat_names), **params)
    m._validate_params()
    m.cols_ = {"lag1": m._col(m.lag_col), "mes_sin": m._col("mes_sin"), "mes_cos": m._col("mes_cos")}
    m0 = int(min(m.min_train, max(2, n_train // 2))); assert m0 == pre.m0
    Xt = X[:n_train]
    m.errors_ = pre.E[: n_train - m0]; m.oof_rows_ = np.arange(m0, n_train)
    st = m._state(Xt); m.state_mu_, m.state_sd_ = st.mean(0), st.std(0); m.state_sd_[m.state_sd_ < 1e-12] = 1.0
    m.state_ = (st[m.oof_rows_] - m.state_mu_) / m.state_sd_
    g = 0.0 if np.isinf(m.half_life) else np.log(0.5) / m.half_life
    m.log_decay_ = g * (n_train - 1 - m.oof_rows_)
    m.experts_ = _Experts(Xt, y[:n_train], m.cols_)
    m.n_features_in_ = X.shape[1]; m.expert_names_ = ("persistencia", "nucleo_cambio", "estacional")
    m.prior_ = m._global_prior()
    return m


GRID = [dict(bandwidth=b, half_life=h, lam=l) for b, h, l in
        itertools.product([0.2, 0.6, 2.0], [24.0, 120.0], [0.01, 0.3, 10.0])]


def check_equivalence(X, y, pre, feat_names):
    p = GRID[4]; n_train = 150
    a = fit_fast(p, X, y, n_train, pre, feat_names).predict(X[n_train:n_train + 20])
    b = ModeloOriginal(feature_names=tuple(feat_names), **p).fit(X[:n_train], y[:n_train]).predict(X[n_train:n_train + 20])
    return float(np.abs(a - b).max())


def run(scenario, seed):
    yraw, month1, crisis_raw = simulate(scenario, seed)
    F, y, rows = features(yraw, month1)
    crisis = crisis_raw[rows + 1]                          # régimen del mes que se pronostica
    X = F.to_numpy(float); names = list(F.columns)
    cols = {"lag1": names.index("td_nacional_lag1"), "mes_sin": names.index("mes_sin"), "mes_cos": names.index("mes_cos")}
    t0 = time.time(); pre = Pre(X, y, cols); t_pre = time.time() - t0
    diff = check_equivalence(X, y, pre, names)
    outer = TimeSeriesSplit(n_splits=5, gap=1)
    recs = []
    for k, (tr, te) in enumerate(outer.split(X)):
        n_tr = len(tr)
        # --- elegir hiperparámetros solo con pliegues internos
        inner = TimeSeriesSplit(n_splits=3, gap=1)
        scores = []
        for p in GRID:
            se = []
            for itr, iva in inner.split(X[:n_tr]):
                if len(itr) <= 72:   # m0 cambia con prefijos cortos; se exige al menos 2*m0 meses
                    continue
                m = fit_fast(p, X, y, len(itr), pre, names)
                se.append(np.sqrt(np.mean((y[iva] - m.predict(X[iva])) ** 2)))
            scores.append(np.mean(se))
        best = GRID[int(np.argmin(scores))]
        m = fit_fast(best, X, y, n_tr, pre, names)
        alee = m.predict(X[te]); ex = m.expert_predictions(X[te])
        for i, idx in enumerate(te):
            recs.append(dict(escenario=scenario, semilla=seed, pliegue=k, idx=idx, y=y[idx], crisis=bool(crisis[idx]),
                             ALEE=alee[i], Estacional=ex[i, 2], Persistencia=ex[i, 0], **{f"hp_{a}": v for a, v in best.items()}))
    df = pd.DataFrame(recs)
    print(f"  {scenario} semilla {seed}: n={len(y)} meses, test={len(df)}, meses en crisis={int(crisis.sum())}, "
          f"precálculo {t_pre:.0f}s, dif. vs .fit()={diff:.1e}", flush=True)
    return df


if __name__ == "__main__":
    allr = []
    for sc in ["estable", "crisis"]:
        for sd in SEEDS:
            allr.append(run(sc, sd))
    res = pd.concat(allr, ignore_index=True)
    res.to_csv(OUT / "predicciones_simulacion.csv", index=False)
    print("guardado", OUT / "predicciones_simulacion.csv")
