"""Punto de entrada único: reproduce todas las tablas, pruebas y figuras del paper.

Uso (desde la raíz del proyecto, con el entorno de requirements-paper.txt):
    python experiments/reproducir_todo.py            # usa las predicciones ya guardadas cuando existen
    python experiments/reproducir_todo.py --desde-cero  # borra artifacts/resultados y recalcula esa parte

Orden:
 1. Colombia: modelos del segundo entregable, LightGBM y referencias (experiments/run_confirmatorio.py).
 2. Conjuntos adicionales: experiments/run_conjuntos_adicionales.py {eeuu, espana, mexico}.
    (Datos en data/adicionales/; para volver a bajarlos: scripts/descargar_conjuntos_adicionales.py.)
 3. experiments/resultados_finales.py: modelos que faltaban, ablación, sensibilidad, ALEE canónico,
    curva de aprendizaje, costo, análisis, figuras y tablas LaTeX.
 4. Cinco conjuntos nuevos de la OCDE (protocolo §19; descargar con scripts/descargar_conjuntos_adicionales.py
    y los crudos oecd_<ISO>_crudo.csv, filtrar con scripts/preparar_oecd.py).
 5. experiments/resultados_curso.py: tablas, pruebas y figuras del paper (ALEE frente a los ocho modelos
    del curso en los nueve conjuntos).

Nota sobre ALEE en Colombia: sus predicciones canónicas (iteración 4) se reconstruyen con los
hiperparámetros guardados en artifacts/confirmatorio/iter4/rmse_K3_sin_ridge.csv (etapa "extras",
diferencia máxima 1.8e-14). La búsqueda original de la iteración 4 incluía un hiperparámetro inerte
(ridge_alpha) que el modelo actual ya no acepta, por lo que esa búsqueda no se vuelve a ejecutar.
Todo el proceso tarda varias horas desde cero; cada etapa reutiliza lo que ya está calculado.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
CONJ = ["colombia", "eeuu", "espana", "mexico"]
NUEVOS = ["canada", "australia", "japon", "alemania", "corea"]  # protocolo §19
VARIANTES = ["sin_localizacion", "sin_lambda_adaptativa", "previo_uniforme"]


def run(*args):
    print(">>", " ".join(args), flush=True)
    subprocess.run([PY, *args], cwd=ROOT, check=True)


def main():
    if "--desde-cero" in sys.argv:
        shutil.rmtree(ROOT / "artifacts" / "resultados", ignore_errors=True)
    run("experiments/run_confirmatorio.py")
    for c in CONJ[1:]:
        run("experiments/run_conjuntos_adicionales.py", c)
    R = "experiments/resultados_finales.py"
    for c in CONJ:
        run(R, "fijos", c)
        run(R, "modelos", c)
        for v in VARIANTES:
            run(R, "alee", c, v)
        for s in ["43", "44", "45", "46"]:
            run(R, "alee", c, "completo", s)
    run(R, "alee", "colombia", "completo")
    run(R, "extras")
    run(R, "costo_ajustado")
    run(R, "analisis")
    run(R, "latex")
    # 4. Cinco conjuntos nuevos (protocolo §19): datos, ALEE congelado, modelos del curso y ablación
    run("scripts/preparar_oecd.py", "CAN", "AUS", "JPN", "DEU", "KOR")
    for c in NUEVOS:
        run("experiments/run_conjuntos_adicionales.py", c)
        run(R, "fijos", c)
        run(R, "modelos", c)
        for v in VARIANTES:
            run(R, "alee", c, v)
    # 5. Tablas y figuras del paper: ALEE frente a los ocho modelos del curso en los nueve conjuntos
    run("experiments/resultados_curso.py")
    print("Listo: artifacts/resultados/solo_curso/ (tablas en latex/, figuras en figuras/)")


if __name__ == "__main__":
    main()
