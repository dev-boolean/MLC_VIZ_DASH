"""Analiza los resultados de run_confirmatorio.py: tabla de RMSE, bateria de
pruebas estadisticas (MCS, Giacomini-White, DM-HLN, SPA) y chequeo del
criterio de superioridad del protocolo congelado."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.stats_tests import (
    model_confidence_set, giacomini_white_test, diebold_mariano_test,
    spa_test, adjust_pvalues, cliffs_delta, cohens_d_paired, effect_label,
)

OUT_DIR = ROOT / "artifacts" / "confirmatorio"

MODEL_FILES = {
    "ALEE": "oof_ALEE.csv",
    "Ridge": "oof_Ridge.csv",
    "Lasso": "oof_Lasso.csv",
    "KNN": "oof_KNN.csv",
    "DecisionTree": "oof_DecisionTree.csv",
    "RandomForest": "oof_RandomForest.csv",
    "XGBoost": "oof_XGBoost.csv",
    "SVR": "oof_SVR.csv",
    "LightGBM": "oof_LightGBM.csv",
    "Estacional(ALEE)": "oof_EstacionalALEE.csv",
}

def load_pred(name_file):
    df = pd.read_csv(OUT_DIR / name_file).sort_values("idx").reset_index(drop=True)
    return df


def main():
    preds = {name: load_pred(f) for name, f in MODEL_FILES.items()}
    baselines = pd.read_csv(OUT_DIR / "baselines_series_tiempo.csv").sort_values("idx").reset_index(drop=True)

    idx_ref = preds["ALEE"]["idx"].values
    assert all((preds[n]["idx"].values == idx_ref).all() for n in preds), "Los pliegues externos no coinciden entre modelos"
    assert (baselines["idx"].values == idx_ref).all(), "Las referencias de series de tiempo no coinciden en indices"
    y_true = preds["ALEE"]["y_true"].values

    # ---- Tabla de RMSE / MAE -------------------------------------------
    rows = []
    errors = {}
    for name, df in preds.items():
        e = df["y_true"].values - df["y_pred"].values
        errors[name] = e
        rows.append({"modelo": name, "rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e)))})
    for col in ["Persistencia", "Estacional", "SARIMA"]:
        e = baselines["y_true"].values - baselines[col].values
        errors[col] = e
        rows.append({"modelo": col, "rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e)))})
    tabla = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    tabla.to_csv(OUT_DIR / "tabla_rmse_confirmatorio.csv", index=False)
    print("=== Tabla RMSE/MAE (pliegues externos, 240 meses) ===")
    print(tabla.to_string(index=False))

    best_ref_name = tabla[tabla["modelo"] != "ALEE"].iloc[0]["modelo"]
    best_ref_rmse = tabla[tabla["modelo"] != "ALEE"].iloc[0]["rmse"]
    alee_rmse = tabla[tabla["modelo"] == "ALEE"].iloc[0]["rmse"]
    print(f"\nMejor referencia: {best_ref_name} (RMSE={best_ref_rmse:.4f}) vs ALEE (RMSE={alee_rmse:.4f})")
    cond_a = alee_rmse < best_ref_rmse and all(
        alee_rmse < tabla[tabla["modelo"] == m].iloc[0]["rmse"]
        for m in ["Ridge", "Lasso", "KNN", "DecisionTree", "RandomForest", "XGBoost", "SVR"]
    )

    # ---- Model Confidence Set (todos los candidatos, ALEE incluido) ----
    all_names = list(preds.keys()) + ["Persistencia", "Estacional", "SARIMA"]
    losses = pd.DataFrame({n: errors[n] ** 2 for n in all_names})
    mcs = model_confidence_set(losses, alpha=0.10, block=12, seed=42)
    mcs.to_csv(OUT_DIR / "mcs_confirmatorio.csv", index=False)
    print("\n=== Model Confidence Set (alpha=0.10) ===")
    print(mcs.to_string(index=False))
    finalistas = mcs[mcs["en_conjunto"]]["modelo"].tolist()
    alee_en_mcs = "ALEE" in finalistas

    # ---- Giacomini-White: ALEE vs cada finalista del MCS ----------------
    gw_rows = []
    for m in finalistas:
        if m == "ALEE":
            continue
        r = giacomini_white_test(errors["ALEE"], errors[m], conditional=True, power=2)
        gw_rows.append({"comparacion": f"ALEE vs {m}", **r})
    gw_df = pd.DataFrame(gw_rows)
    if len(gw_df):
        gw_df["p_holm"] = adjust_pvalues(gw_df["p_value"].values, method="holm")
        gw_df["rechaza_igualdad_0.05"] = gw_df["p_holm"] < 0.05
    gw_df.to_csv(OUT_DIR / "giacomini_white_confirmatorio.csv", index=False)
    print("\n=== Giacomini-White condicional: ALEE vs finalistas MCS (Holm) ===")
    print(gw_df.to_string(index=False) if len(gw_df) else "(ALEE es el unico finalista o no hay otros con quien comparar)")

    # ---- Diebold-Mariano HLN: ALEE vs mejor referencia -------------------
    dm = diebold_mariano_test(errors["ALEE"], errors[best_ref_name], h=1, power=2)
    print(f"\n=== Diebold-Mariano (HLN) ALEE vs {best_ref_name} ===")
    print(dm)
    cond_b_dm = dm["p_value"] < 0.05 and dm["mean_loss_diff"] < 0

    # ---- SPA de Hansen: universo completo vs mejor referencia -----------
    cand_cols = [n for n in all_names if n != best_ref_name]
    cand_df = pd.DataFrame({n: errors[n] ** 2 for n in cand_cols})
    spa = spa_test(errors[best_ref_name] ** 2, cand_df, seed=42)
    print(f"\n=== SPA de Hansen (referencia={best_ref_name}) ===")
    print(spa)

    # ---- Tamanio del efecto: ALEE vs mejor referencia --------------------
    diffs = np.abs(errors["ALEE"]) - np.abs(errors[best_ref_name])
    d_cohen = cohens_d_paired(diffs)
    delta_cliff = cliffs_delta(np.abs(errors["ALEE"]), np.abs(errors[best_ref_name]))
    print(f"\nTamanio del efecto ALEE vs {best_ref_name}: d de Cohen={d_cohen:.3f} ({effect_label(d_cohen,'cohen')}), "
          f"delta de Cliff={delta_cliff:.3f} ({effect_label(delta_cliff,'cliff')})")

    n_rechazos = int(alee_en_mcs is False) + int(cond_b_dm) + int(gw_df["rechaza_igualdad_0.05"].sum() if len(gw_df) else 0)
    # condicion b del protocolo: al menos 3 pruebas rechazan con p ajustado < alpha a favor de ALEE
    cond_b = (dm["p_value"] < 0.05 and dm["mean_loss_diff"] < 0) + (bool(len(gw_df)) and gw_df["rechaza_igualdad_0.05"].any()) + (spa["p_consistent"] < 0.05 and spa["best_candidate"] == "ALEE")
    cond_c = abs(delta_cliff) >= 0.147 or abs(d_cohen) >= 0.2

    resumen = {
        "condicion_a_mejor_metrica": bool(cond_a),
        "mejor_referencia": best_ref_name,
        "alee_rmse": float(alee_rmse),
        "mejor_referencia_rmse": float(best_ref_rmse),
        "alee_en_mcs": bool(alee_en_mcs),
        "dm_hln_p_value": float(dm["p_value"]),
        "dm_hln_favorece_alee": bool(cond_b_dm),
        "spa_p_consistent": float(spa["p_consistent"]),
        "spa_mejor_candidato": spa["best_candidate"],
        "condicion_b_al_menos_3_rechazos": bool(cond_b >= 3),
        "pruebas_que_rechazan_a_favor_de_alee": int(cond_b),
        "d_cohen": float(d_cohen),
        "delta_cliff": float(delta_cliff),
        "condicion_c_efecto_no_despreciable": bool(cond_c),
        "condicion_d_datasets_adicionales": "NO EVALUADO: datasets adicionales (EE.UU./Espana/Mexico) no se pudieron ensamblar en esta corrida (ver limitaciones declaradas)",
        "criterio_superioridad_completo_abcd": False,  # (d) no evaluable -> el criterio conjunto no puede darse por cumplido
    }
    pd.Series(resumen).to_csv(OUT_DIR / "criterio_superioridad.csv")
    print("\n=== Veredicto del criterio de superioridad (protocolo, seccion 6) ===")
    for k, v in resumen.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
