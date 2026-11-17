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


def test_notebook_handles_active_job_restart_without_traceback():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    server_cells = [c for c in nb.cells if c.cell_type == "code" and "start_server_colab" in c.source]
    assert len(server_cells) == 1
    src = server_cells[0].source
    assert "ActiveJobRestartError" in src
    assert "except ActiveJobRestartError" in src
    assert "print(" in src
