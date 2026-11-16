from __future__ import annotations

from pathlib import Path

import nbformat


NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / "main.ipynb"


def test_notebook_is_valid_nbformat_and_has_no_outputs():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nbformat.validate(nb)

    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            assert cell.get("outputs", []) == [], f"Celda {idx} tiene salidas guardadas"
            assert cell.get("execution_count") is None, f"Celda {idx} tiene execution_count"


def test_notebook_has_no_zip_upload():
    content = NOTEBOOK_PATH.read_text(encoding="utf-8")
    assert "files.upload" not in content
    assert "REPO_URL" in content
    assert "RAMA_O_ETIQUETA" in content
    assert "NGROK_DOMAIN" in content
    assert "APP_URL" in content
