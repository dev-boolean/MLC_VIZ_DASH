"""Divide el cuaderno único del segundo entregable en los capítulos del tercero.

Corrección de la observación 4 del docente: el trabajo se presentaba como un
solo cuaderno. Las celdas se copian sin cambiar sus salidas a:

* ``book/02_datos.ipynb``: introducción del primer entregable y EDA.
* ``book/03_benchmarks.ipynb``: SARIMA y los 140 modelos de machine learning.
* ``book/_referencias_e2.md``: la lista de referencias del cuaderno, que se
  incluye en el capítulo 10.

En cada capítulo los encabezados bajan un nivel (``#`` pasa a ``##``) para
que cada archivo tenga un único título. La carga de datos deja de usar la
ruta absoluta ``C:/Users/boolean/Downloads/MLC.xlsx`` (hallazgo P1 de la
trazabilidad) y lee ``../data/MLC.xlsx``, que es el mismo archivo (huella
SHA-256 idéntica). El cuaderno original se conserva en
``book/capitulo_ml/MLC_entregable2.ipynb``.

Uso::

    python scripts/split_book.py
"""
import copy
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOOK = ROOT / "book"
SOURCE = BOOK / "capitulo_ml" / "MLC_entregable2.ipynb"
KERNEL = {"display_name": "Python (MLC ML)", "language": "python", "name": "mlc-ml"}
OLD_PATH = "C:/Users/boolean/Downloads/MLC.xlsx"
NEW_PATH = "../data/MLC.xlsx"


def demote(cells):
    """Baja un nivel los encabezados y quita los saltos de nivel (por ejemplo, de ## a ####).

    Los cuadernos originales pasaban de ``#`` a ``###``; Sphinx advierte por cada salto.
    """
    out_cells, prev = [], 1  # el título del capítulo, que se agrega aparte, es nivel 1
    for cell in cells:
        if cell["cell_type"] != "markdown":
            out_cells.append(cell)
            continue
        out, in_code = [], False
        for line in "".join(cell["source"]).split("\n"):
            if line.strip().startswith("```"):
                in_code = not in_code
            m = None if in_code else re.match(r"^(#{1,5}) ", line)
            if m:
                level = min(len(m.group(1)) + 1, prev + 1)
                line = "#" * level + line[len(m.group(1)):]
                prev = level
            out.append(line)
        cell = copy.deepcopy(cell)
        cell["source"] = "\n".join(out)
        out_cells.append(cell)
    return out_cells


def markdown(text, tags=None):
    cell = {"cell_type": "markdown", "metadata": {}, "source": text}
    if tags:
        cell["metadata"]["tags"] = tags
    return cell


def code(text, tags=None):
    return {"cell_type": "code", "execution_count": None, "metadata": {"tags": tags or []}, "outputs": [], "source": text}


def write(path, cells, nb):
    out = copy.deepcopy(nb)
    out["cells"] = cells
    out["metadata"]["kernelspec"] = KERNEL
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


def main():
    nb = json.load(open(SOURCE, encoding="utf-8"))
    cells = nb["cells"]
    starts = {"".join(c["source"]).splitlines()[0]: i for i, c in enumerate(cells)
              if c["cell_type"] == "markdown" and "".join(c["source"]).startswith("# ")}
    i_sarima = starts["# Modelado Predictivo: SARIMA"]
    i_refs = starts["# Referencias"]

    datos = []
    for c in cells[:i_sarima]:
        c = copy.deepcopy(c)
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            if OLD_PATH in src:
                c["source"] = src.replace(OLD_PATH, NEW_PATH)
        datos.append(c)
    datos = demote(datos)
    intro = ("# Datos: ETL y EDA\n\nEste capítulo reúne la introducción, el ETL y el análisis exploratorio de los "
             "entregables anteriores, sin cambios en sus resultados. Las celdas leen los datos desde "
             "`data/MLC.xlsx` del repositorio. Las correcciones y la auditoría de fuga de información están en "
             "{doc}`02b_auditoria`.")
    write(BOOK / "02_datos.ipynb", [markdown(intro)] + datos, nb)

    bench_intro = ("# Modelos benchmark\n\nEl SARIMA del primer entregable y las 140 combinaciones de machine "
                   "learning del segundo, con sus resultados originales. La selección de la mejor combinación de "
                   "cada modelo y su reentrenamiento final están en {doc}`03b_mejores_combinaciones`.")
    # Si se vuelve a ejecutar el capítulo, `import src` y las rutas a experiments/ necesitan la raíz del repositorio.
    setup = code("import os\nif os.path.basename(os.getcwd()) == 'book':\n    os.chdir('..')", tags=["remove-cell"])
    write(BOOK / "03_benchmarks.ipynb",
          [markdown(bench_intro), setup] + demote(cells[i_sarima:i_refs]), nb)

    refs = "".join(cells[i_refs]["source"]).split("\n", 1)[1].strip()
    (BOOK / "_referencias_e2.md").write_text(refs + "\n", encoding="utf-8")
    print(f"02_datos: {i_sarima} celdas; 03_benchmarks: {i_refs - i_sarima} celdas; referencias: {len(refs.splitlines())} líneas")


if __name__ == "__main__":
    main()
