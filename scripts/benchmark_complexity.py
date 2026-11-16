"""Escalamiento empírico y comparación "estándar vs optimizado" (Sección 4).

Mide, sobre datos sintéticos de tamaño controlado, cómo crece el tiempo de
entrenamiento e inferencia con el número de observaciones (n) y de variables
(p), y compara la variante estándar de cada modelo con la optimizada que
sugiere la guía:

* KNN: búsqueda exhaustiva frente a KD-Tree, Ball-Tree y FAISS (aproximada).
* Ridge: solver directo (Cholesky) frente a SAGA; Lasso por descenso de
  coordenadas.
* Naive Bayes: ``fit`` con todo el conjunto frente a ``partial_fit`` por lotes.
* XGBoost: método exacto frente a histogramas, y parada temprana.
* SVM: núcleo RBF completo frente a ``LinearSVC`` y gradiente estocástico.

Cada medición se repite ``REPEATS`` veces y se reporta la mediana. La memoria
es el máximo del conjunto residente del proceso durante la llamada.

Salidas (en experiments/): complexity_scaling.csv y standard_vs_optimized.csv.

Uso::

    python scripts/benchmark_complexity.py
"""
import os
import sys
import threading
import time
import warnings
from pathlib import Path

# Los tiempos de este script deben medir el algoritmo y no la contención entre hilos de BLAS/OpenMP (con 12 hilos de
# OpenBLAS compitiendo con otros procesos, un Lasso pasó de 0.01 s a 1 s): se fijan a un hilo antes de importar NumPy.
# El único paralelismo que se mide es el que se pide de forma explícita (n_jobs, joblib).
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import Lasso, Ridge, SGDRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier

from src import config

if os.environ.get("EXP_DIR"):  # permite redirigir la salida (por ejemplo, para pruebas)
    config.set_experiments_dir(os.environ["EXP_DIR"])

QUICK = os.environ.get("BENCH_QUICK") == "1"  # versión reducida, solo para probar que el script corre
REPEATS = 1 if QUICK else 3
SEED = config.SEED
OUT = config.EXPERIMENTS_DIR


def pick(values):
    """Reduce las listas de tamaños en la versión rápida."""
    return values[:2] if QUICK else values


def size(n):
    return max(n // 20, 300) if QUICK else n


def timed(fn, repeats: int = REPEATS):
    """Mediana del tiempo de ``fn`` y su último resultado."""
    times, out = [], None
    repeats = min(repeats, REPEATS)
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), out


def peak_memory_mb(fn) -> float:
    """Máximo de memoria residente adicional (MB) mientras corre ``fn``."""
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = [baseline]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], process.memory_info().rss)
            time.sleep(0.005)

    thread = threading.Thread(target=sample)
    thread.start()
    try:
        fn()
    finally:
        stop.set()
        thread.join()
    return (peak[0] - baseline) / 2 ** 20


def classification_data(n, p, n_test=2000):
    X, y = make_classification(
        n_samples=n + n_test, n_features=p, n_informative=min(p, 8), n_redundant=0,
        n_classes=3, n_clusters_per_class=1, class_sep=1.0, random_state=SEED,
    )
    X = StandardScaler().fit_transform(X)
    return X[:n], y[:n], X[n:], y[n:]


def regression_data(n, p, n_test=2000):
    X, y = make_regression(n_samples=n + n_test, n_features=p, n_informative=min(p, 20), noise=10, random_state=SEED)
    return X[:n], y[:n], X[n:], y[n:]


rows = []


def record(family, variant, n, p, fit_s, predict_s, score=np.nan, extra=None):
    rows.append({"familia": family, "variante": variante_label(variant), "n": n, "p": p,
                 "entrenamiento_s": fit_s, "inferencia_s": predict_s, "desempeño": score, **(extra or {})})


def variante_label(v):
    return v


# ------------------------------------------------------------------ KNN
def faiss_knn(Xtr, ytr, Xte, k=5, approximate=False):
    import faiss

    d = Xtr.shape[1]
    Xtr32, Xte32 = np.ascontiguousarray(Xtr, dtype="float32"), np.ascontiguousarray(Xte, dtype="float32")
    if approximate:
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, 32)
        index.train(Xtr32)
        index.add(Xtr32)
        index.nprobe = 4
    else:
        index = faiss.IndexFlatL2(d)
        index.add(Xtr32)
    return index, Xte32


def knn_experiments():
    print("KNN: escalamiento en n", flush=True)
    for n in pick([2000, 4000, 8000, 16000, 32000]):
        Xtr, ytr, Xte, yte = classification_data(n, 8)
        brute_pred = None
        for algorithm in ["brute", "kd_tree", "ball_tree"]:
            model = KNeighborsClassifier(n_neighbors=5, algorithm=algorithm)
            fit_s, _ = timed(lambda: model.fit(Xtr, ytr))
            pred_s, pred = timed(lambda: model.predict(Xte))
            if algorithm == "brute":
                brute_pred = pred
            record("KNN", f"{algorithm}", n, 8, fit_s, pred_s, accuracy_score(yte, pred),
                   {"coincide_con_exacto": float((pred == brute_pred).mean())})
        for approx, label in [(False, "FAISS exacto"), (True, "FAISS aproximado (IVF)")]:
            def build():
                return faiss_knn(Xtr, ytr, Xte, approximate=approx)
            fit_s, (index, Xq) = timed(build)
            pred_s, (_, neighbors) = timed(lambda: index.search(Xq, 5))
            votes = ytr[neighbors]
            pred = np.array([np.bincount(v, minlength=3).argmax() for v in votes])
            record("KNN", label, n, 8, fit_s, pred_s, accuracy_score(yte, pred),
                   {"coincide_con_exacto": float((pred == brute_pred).mean())})
    print("KNN: escalamiento en p", flush=True)
    for p in pick([2, 5, 10, 20, 40, 80]):
        Xtr, ytr, Xte, yte = classification_data(size(8000), p)
        for algorithm in ["brute", "kd_tree", "ball_tree"]:
            model = KNeighborsClassifier(n_neighbors=5, algorithm=algorithm)
            fit_s, _ = timed(lambda: model.fit(Xtr, ytr))
            pred_s, pred = timed(lambda: model.predict(Xte))
            record("KNN (en p)", algorithm, size(8000), p, fit_s, pred_s, accuracy_score(yte, pred))


# ------------------------------------------------------------- Ridge y Lasso
def linear_experiments():
    print("Ridge/Lasso", flush=True)
    for n in pick([5000, 10000, 20000, 40000, 80000]):
        Xtr, ytr, Xte, yte = regression_data(n, 50)
        for label, make in [
            ("Ridge cholesky", lambda: Ridge(alpha=1.0, solver="cholesky")),
            ("Ridge saga", lambda: Ridge(alpha=1.0, solver="saga", max_iter=50, tol=1e-3, random_state=SEED)),
            ("Lasso (coordenadas)", lambda: Lasso(alpha=0.1, max_iter=2000, random_state=SEED)),
        ]:
            model = make()
            fit_s, _ = timed(lambda: model.fit(Xtr, ytr))
            pred_s, pred = timed(lambda: model.predict(Xte))
            record("Ridge/Lasso", label, n, 50, fit_s, pred_s, float(np.sqrt(mean_squared_error(yte, pred))))
    for p in pick([10, 50, 100, 200, 400]):
        Xtr, ytr, Xte, yte = regression_data(size(20000), p)
        for label, make in [
            ("Ridge cholesky", lambda: Ridge(alpha=1.0, solver="cholesky")),
            ("Ridge saga", lambda: Ridge(alpha=1.0, solver="saga", max_iter=50, tol=1e-3, random_state=SEED)),
            ("Lasso (coordenadas)", lambda: Lasso(alpha=0.1, max_iter=2000, random_state=SEED)),
        ]:
            model = make()
            fit_s, _ = timed(lambda: model.fit(Xtr, ytr))
            pred_s, pred = timed(lambda: model.predict(Xte))
            record("Ridge/Lasso (en p)", label, size(20000), p, fit_s, pred_s, float(np.sqrt(mean_squared_error(yte, pred))))


# -------------------------------------------------------------- Naive Bayes
def nb_stream(n, p, batch, incremental):
    rng = np.random.RandomState(SEED)
    means = rng.normal(0, 1.5, size=(3, p))
    model = GaussianNB()

    def chunk(m):
        y = rng.randint(0, 3, size=m)
        return rng.normal(size=(m, p)) + means[y], y

    if incremental:
        for start in range(0, n, batch):
            Xb, yb = chunk(min(batch, n - start))
            model.partial_fit(Xb, yb, classes=[0, 1, 2])
    else:
        X, y = chunk(n)
        model.fit(X, y)
    return model


def naive_bayes_experiments():
    print("Naive Bayes", flush=True)
    for n in pick([50000, 100000, 200000, 400000]):
        for incremental in (False, True):
            label = "partial_fit (lotes de 5 000)" if incremental else "fit (todo en memoria)"
            fit_s, _ = timed(lambda: nb_stream(n, 60, 5000, incremental))
            memory = peak_memory_mb(lambda: nb_stream(n, 60, 5000, incremental))
            record("Naive Bayes", label, n, 60, fit_s, np.nan, extra={"memoria_pico_mb": memory})


# ------------------------------------------------------------------ XGBoost
def xgboost_experiments():
    print("XGBoost", flush=True)
    for n in pick([2000, 5000, 10000, 20000, 40000]):
        Xtr, ytr, Xte, yte = classification_data(n, 20)
        for method in ["exact", "hist"]:
            model = XGBClassifier(n_estimators=100, max_depth=6, tree_method=method, n_jobs=1, verbosity=0, random_state=SEED)
            fit_s, _ = timed(lambda: XGBClassifier(**model.get_params()).fit(Xtr, ytr), repeats=2)
            fitted = XGBClassifier(**model.get_params()).fit(Xtr, ytr)
            pred_s, pred = timed(lambda: fitted.predict(Xte))
            record("XGBoost", f"método {method}", n, 20, fit_s, pred_s, accuracy_score(yte, pred))
    Xtr, ytr, Xte, yte = classification_data(size(20000), 20)
    cut = int(len(Xtr) * 0.8)
    for label, params in [
        ("300 árboles fijos", dict(n_estimators=300)),
        ("parada temprana (20 rondas)", dict(n_estimators=1000, early_stopping_rounds=20)),
    ]:
        def fit():
            m = XGBClassifier(max_depth=6, tree_method="hist", n_jobs=1, verbosity=0, random_state=SEED, **params)
            if "early_stopping_rounds" in params:
                m.fit(Xtr[:cut], ytr[:cut], eval_set=[(Xtr[cut:], ytr[cut:])], verbose=False)
            else:
                m.fit(Xtr[:cut], ytr[:cut])
            return m
        fit_s, fitted = timed(fit, repeats=2)
        pred_s, pred = timed(lambda: fitted.predict(Xte))
        trees = getattr(fitted, "best_iteration", None)
        record("XGBoost (parada temprana)", label, size(20000), 20, fit_s, pred_s, accuracy_score(yte, pred),
               {"arboles_usados": (trees + 1) if trees is not None else params["n_estimators"]})


# ---------------------------------------------------------------------- SVM
def svm_experiments():
    print("SVM", flush=True)
    for n in pick([500, 1000, 2000, 4000, 8000]):
        Xtr, ytr, Xte, yte = classification_data(n, 20)
        for label, make in [
            ("SVC núcleo RBF", lambda: SVC(kernel="rbf", C=1.0, gamma="scale")),
            ("LinearSVC", lambda: LinearSVC(C=1.0, dual=False, max_iter=5000)),
            ("SGD (hinge)", lambda: SGDClassifier(loss="hinge", alpha=1e-4, max_iter=20, tol=None, random_state=SEED)),
        ]:
            model = make()
            fit_s, _ = timed(lambda: make().fit(Xtr, ytr), repeats=2)
            model.fit(Xtr, ytr)
            pred_s, pred = timed(lambda: model.predict(Xte))
            record("SVM", label, n, 20, fit_s, pred_s, accuracy_score(yte, pred))


# ------------------------------------------------------ estándar vs optimizado
THEORY = {
    "KNN": ("O(1) (perezoso)", "O(n·p) por consulta con búsqueda exhaustiva; O(p·log n) con índices en árbol (baja dimensión)"),
    "Ridge": ("O(n·p² + p³) con solver directo; O(n·p·iter) con SAGA", "O(p)"),
    "Naive Bayes": ("O(n·p)", "O(p)"),
    "XGBoost": ("O(B·n·p·log n) exacto; O(B·n·p) con histogramas", "O(B·profundidad)"),
    "SVM": ("O(n²·p) a O(n³) con núcleo; O(n·p) con LinearSVC o SGD", "O(n_vs·p) con núcleo; O(p) lineal"),
    "Random Forest": ("O(B·n·p·log n)", "O(B·profundidad)"),
}


def standard_vs_optimized():
    print("Tabla estándar vs optimizado", flush=True)
    out = []

    def add(family, variant, fit_fn, predict_fn, score_fn, standard_score=None, standard_time=None):
        fit_s, model = timed(fit_fn, repeats=2)
        memory = peak_memory_mb(fit_fn)
        pred_s, pred = timed(lambda: predict_fn(model))
        score = score_fn(pred)
        out.append({
            "modelo": family, "variante": variant, "complejidad_entrenamiento": THEORY[family][0],
            "complejidad_inferencia": THEORY[family][1], "entrenamiento_s": fit_s, "inferencia_s": pred_s,
            "memoria_pico_mb": memory, "desempeño": score,
        })

    n = size(16000)
    Xtr, ytr, Xte, yte = classification_data(n, 8)
    acc = lambda pred: accuracy_score(yte, pred)
    knn = lambda alg: (lambda: KNeighborsClassifier(n_neighbors=5, algorithm=alg).fit(Xtr, ytr))
    add("KNN", "estándar (exhaustivo)", knn("brute"), lambda m: m.predict(Xte), acc)
    add("KNN", "optimizado (KD-Tree)", knn("kd_tree"), lambda m: m.predict(Xte), acc)

    def faiss_fit():
        return faiss_knn(Xtr, ytr, Xte, approximate=True)

    def faiss_predict(m):
        index, Xq = m
        _, nb = index.search(Xq, 5)
        return np.array([np.bincount(v, minlength=3).argmax() for v in ytr[nb]])

    add("KNN", "optimizado (FAISS aproximado)", faiss_fit, faiss_predict, acc)

    Xr, yr, Xrt, yrt = regression_data(size(40000), 50)
    rmse = lambda pred: float(np.sqrt(mean_squared_error(yrt, pred)))
    add("Ridge", "estándar (Cholesky)", lambda: Ridge(alpha=1.0, solver="cholesky").fit(Xr, yr), lambda m: m.predict(Xrt), rmse)
    add("Ridge", "optimizado (SAGA)", lambda: Ridge(alpha=1.0, solver="saga", max_iter=50, tol=1e-3, random_state=SEED).fit(Xr, yr),
        lambda m: m.predict(Xrt), rmse)

    n_nb = size(400000)
    add("Naive Bayes", "estándar (fit completo)", lambda: nb_stream(n_nb, 60, 5000, False), lambda m: m.predict(np.zeros((1, 60))), lambda p: np.nan)
    add("Naive Bayes", "optimizado (partial_fit)", lambda: nb_stream(n_nb, 60, 5000, True), lambda m: m.predict(np.zeros((1, 60))), lambda p: np.nan)

    Xx, yx, Xxt, yxt = classification_data(size(40000), 20)
    accx = lambda pred: accuracy_score(yxt, pred)
    xgb = lambda method: (lambda: XGBClassifier(n_estimators=100, max_depth=6, tree_method=method, n_jobs=1, verbosity=0, random_state=SEED).fit(Xx, yx))
    add("XGBoost", "estándar (exacto)", xgb("exact"), lambda m: m.predict(Xxt), accx)
    add("XGBoost", "optimizado (histogramas)", xgb("hist"), lambda m: m.predict(Xxt), accx)

    Xs, ys, Xst, yst = classification_data(size(8000), 20)
    accs = lambda pred: accuracy_score(yst, pred)
    add("SVM", "estándar (núcleo RBF)", lambda: SVC(kernel="rbf").fit(Xs, ys), lambda m: m.predict(Xst), accs)
    add("SVM", "optimizado (LinearSVC)", lambda: LinearSVC(dual=False, max_iter=5000).fit(Xs, ys), lambda m: m.predict(Xst), accs)
    add("SVM", "optimizado (SGD hinge)", lambda: SGDClassifier(loss="hinge", max_iter=20, tol=None, random_state=SEED).fit(Xs, ys),
        lambda m: m.predict(Xst), accs)

    from sklearn.ensemble import RandomForestClassifier

    Xf, yf, Xft, yft = classification_data(size(16000), 20)
    accf = lambda pred: accuracy_score(yft, pred)
    rf = lambda jobs: (lambda: RandomForestClassifier(n_estimators=100, n_jobs=jobs, random_state=SEED).fit(Xf, yf))
    add("Random Forest", "estándar (1 núcleo)", rf(1), lambda m: m.predict(Xft), accf)
    add("Random Forest", "optimizado (n_jobs=-1)", rf(-1), lambda m: m.predict(Xft), accf)

    df = pd.DataFrame(out)
    df["aceleracion_entrenamiento"] = np.nan
    for family, group in df.groupby("modelo"):
        base = group.iloc[0]["entrenamiento_s"]
        df.loc[group.index, "aceleracion_entrenamiento"] = base / group["entrenamiento_s"]
    df.to_csv(OUT / "standard_vs_optimized.csv", index=False)
    return df


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    knn_experiments()
    linear_experiments()
    naive_bayes_experiments()
    xgboost_experiments()
    svm_experiments()
    pd.DataFrame(rows).to_csv(OUT / "complexity_scaling.csv", index=False)
    standard_vs_optimized()
    print(f"Listo en {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
