# -*- coding: utf-8 -*-
"""Arma el capítulo nuevo, lo ejecuta y lo inserta en una copia de MLC.ipynb.

Uso:  python build_chapter.py <partes separadas por coma> <salida.ipynb> [--no-exec | --reuse <capitulo_ejecutado.ipynb>]

* Al ejecutar, también guarda el capítulo ejecutado solo, como <salida>_capitulo.ipynb.
* --reuse reutiliza las salidas de un capítulo ya ejecutado y toma de las partes el texto de las celdas
  Markdown (sirve para corregir la redacción sin volver a correr todo; si cambia una celda de código, falla).

* Lee las celdas de chapter/<parte>.py (listas CELLS).
* Ejecuta el capítulo en el kernel del entorno del proyecto, con el directorio
  de trabajo en la raíz del repositorio (para que funcionen `import src` y las
  rutas a experiments/).
* Inserta las celdas ya ejecutadas justo antes de "# Referencias" en una copia
  del MLC.ipynb original (sin tocar sus celdas) y agrega las referencias nuevas.
"""
import importlib.util
import re
import sys
import unicodedata
import uuid
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # raíz del repositorio (donde están src/ y experiments/); el libro vive en book/
ORIGINAL = HERE.parent / "MLC_original.ipynb"  # cuaderno original del libro, sin el capítulo de machine learning


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chapter_notebook(parts):
    cells = []
    for name in parts:
        cells += load_module(name).CELLS
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python (Entregable2 ML)", "language": "python", "name": "book_py_entregable2"}
    for kind, text in cells:
        nb.cells.append(nbformat.v4.new_markdown_cell(text) if kind == "markdown" else nbformat.v4.new_code_cell(text))
    return nb


def execute(nb):
    client = NotebookClient(
        nb, timeout=3600, kernel_name="book_py_entregable2", resources={"metadata": {"path": str(REPO)}},
        allow_errors=False,
    )
    client.execute()
    return nb


def reuse_outputs(nb, executed_path):
    """Copia a ``nb`` las salidas de un capítulo ya ejecutado; el texto Markdown es el de ``nb``."""
    executed = nbformat.read(executed_path, as_version=4)
    if len(executed.cells) != len(nb.cells):
        raise SystemExit(f"El capítulo ejecutado tiene {len(executed.cells)} celdas y el nuevo {len(nb.cells)}: hay que volver a ejecutar.")
    for new, old in zip(nb.cells, executed.cells):
        if new.cell_type != old.cell_type:
            raise SystemExit("Cambió el orden de las celdas: hay que volver a ejecutar.")
        if new.cell_type == "code":
            if new.source != old.source:
                resumen = " ".join(new.source[:60].split())
                raise SystemExit(f"Cambió una celda de código ({resumen}...): hay que volver a ejecutar.")
            new["outputs"], new["execution_count"] = old["outputs"], old["execution_count"]
    return nb


def norm_key(paragraph):
    text = unicodedata.normalize("NFKD", paragraph).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", "", text)


def merge_references(original_text, new_refs):
    """Une las referencias originales con las nuevas, en orden alfabético."""
    head, *rest = original_text.split("\n\n", 1)
    entries = [e.strip() for e in (rest[0] if rest else "").split("\n\n") if e.strip()]
    for ref in new_refs:
        if ref.strip() not in entries:
            entries.append(ref.strip())
    entries.sort(key=norm_key)
    return head + "\n\n" + "\n\n".join(entries) + "\n"


def splice(chapter, output):
    original = nbformat.read(ORIGINAL, as_version=4)
    idx = next(i for i, c in enumerate(original.cells) if c.cell_type == "markdown" and c.source.startswith("# Referencias"))
    for cell in chapter.cells:
        cell["id"] = uuid.uuid4().hex[:8]
    original.cells[idx:idx] = chapter.cells
    ref_cell = original.cells[idx + len(chapter.cells)]
    refs = load_module("refs")
    texto = chr(10).join(c.source for c in chapter.cells if c.cell_type == "markdown")
    new_refs, sin_usar, sin_forma = refs.select(texto)
    print(f"Referencias nuevas citadas: {len(new_refs)}; sin citar en el texto: {sin_usar}; sin forma registrada: {sin_forma}")
    ref_cell.source = merge_references(ref_cell.source, new_refs)
    nbformat.write(original, output)


if __name__ == "__main__":
    parts = sys.argv[1].split(",")
    output = Path(sys.argv[2])
    nb = chapter_notebook(parts)
    if "--reuse" in sys.argv:
        nb = reuse_outputs(nb, sys.argv[sys.argv.index("--reuse") + 1])
    elif "--no-exec" not in sys.argv:
        execute(nb)
        nbformat.write(nb, output.with_name(output.stem + "_capitulo.ipynb"))
    splice(nb, output)
    print(f"Capítulo con {len(nb.cells)} celdas insertado en {output}")
