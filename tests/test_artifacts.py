"""Pruebas de los artefactos de las mejores combinaciones (``artifacts/``).

Verifican lo que el dashboard da por hecho: que cada modelo serializado carga,
predice exactamente lo mismo que el modelo en memoria que lo generó (prueba de
ida y vuelta), que su ``metadata.json`` está completo y que el esquema de
entrada coincide con las variables del problema.
"""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.artifacts import ARTIFACTS_DIR, MAX_ARTIFACT_MB, SELECTION_PATH, sha256
from src.data import assemble_dataset

ARTIFACT_DIRS = sorted(p for p in ARTIFACTS_DIR.glob("*_*") if (p / "model.joblib").exists())
REQUIRED_KEYS = {
    "task", "model", "balancing", "optimizer", "seed", "hyperparameters", "input_schema",
    "outer_cv_metrics", "libraries", "roundtrip_test", "file", "selection",
}


@pytest.fixture(scope="module")
def dataset():
    return assemble_dataset()


def test_hay_un_artefacto_por_modelo_base():
    selection = pd.read_csv(SELECTION_PATH)
    assert len(selection) == 14
    assert selection.groupby("task").size().to_dict() == {"clf": 7, "reg": 7}
    # Además de las 14 mejores combinaciones están el modelo original y los modelos que agregó el paper.
    paper = {"reg_ALEE", "reg_BayesianRidge", "reg_SVR_RBF", "reg_LightGBM"}
    assert {p.name for p in ARTIFACT_DIRS} - paper == set(selection["artifact"])


def test_una_sola_mejor_global_por_tarea():
    selection = pd.read_csv(SELECTION_PATH)
    assert selection.groupby("task")["is_global_best"].sum().to_dict() == {"clf": 1, "reg": 1}


@pytest.mark.parametrize("art_dir", ARTIFACT_DIRS, ids=lambda p: p.name)
def test_ida_y_vuelta_exacta(art_dir):
    reference = np.load(art_dir / "roundtrip.npz", allow_pickle=False)
    X = pd.DataFrame(reference["X"], columns=list(reference["columns"]))
    pipe = joblib.load(art_dir / "model.joblib")
    np.testing.assert_array_equal(pipe.predict(X), reference["y_pred"])
    if "y_proba" in reference.files:
        np.testing.assert_array_equal(pipe.predict_proba(X), reference["y_proba"])


@pytest.mark.parametrize("art_dir", ARTIFACT_DIRS, ids=lambda p: p.name)
def test_carga_en_un_proceso_nuevo(art_dir):
    # El dashboard carga los modelos desde otro programa: si algún objeto quedó
    # referido a ``__main__`` (el script que los generó), la carga falla.
    import subprocess
    import sys

    code = f"import joblib; joblib.load(r'{art_dir / 'model.joblib'}')"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ARTIFACTS_DIR.parent)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("art_dir", ARTIFACT_DIRS, ids=lambda p: p.name)
def test_metadata_completa(art_dir, dataset):
    meta = json.loads((art_dir / "metadata.json").read_text(encoding="utf-8"))
    assert REQUIRED_KEYS <= set(meta)
    assert [c["name"] for c in meta["input_schema"]] == dataset.feature_names
    assert meta["roundtrip_test"]["passed"] is True
    assert meta["file"]["sha256"] == sha256(art_dir / "model.joblib")
    assert meta["file"]["size_mb"] < MAX_ARTIFACT_MB


@pytest.mark.parametrize("art_dir", [p for p in ARTIFACT_DIRS if p.name.startswith("clf_")], ids=lambda p: p.name)
def test_probabilidades_validas(art_dir, dataset):
    pipe = joblib.load(art_dir / "model.joblib")
    proba = pipe.predict_proba(dataset.X)
    assert proba.shape == (len(dataset.X), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert set(np.unique(pipe.predict(dataset.X))) <= {0, 1, 2}
