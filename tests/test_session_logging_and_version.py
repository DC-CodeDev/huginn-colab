from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from typing import List

import nibabel as nib
import numpy as np
import pytest

from colab import server
from colab.memory import StageMemorySampler
from colab.pipeline import run_segmentation_worker
from colab.session_log import (
    append_log_line,
    format_huginn_line,
    get_job_log_path,
    get_session_log_path,
    monitor_session_logs,
    sanitize_log_text,
)
from colab.version import (
    LOADED_COMMIT_SHORT,
    SERVICE_VERSION,
    STALE_CODE_WARNING,
    check_code_staleness,
    format_version_banner,
)


def test_version_banner_and_staleness_detection(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@localhost"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)

    f = repo / "mod.py"
    f.write_text("v = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "mod.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "c1"], cwd=repo, check=True)
    c1 = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()

    assert check_code_staleness(repo_dir=str(repo), loaded_hash=c1) is None

    f.write_text("v = 2\n", encoding="utf-8")
    subprocess.run(["git", "add", "mod.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "c2"], cwd=repo, check=True)
    c2 = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()

    warning = check_code_staleness(repo_dir=str(repo), loaded_hash=c1)
    assert warning is not None
    assert STALE_CODE_WARNING in warning
    assert c1[:7] in warning
    assert c2[:7] in warning

    banner = format_version_banner()
    assert SERVICE_VERSION in banner
    assert LOADED_COMMIT_SHORT in banner
    assert banner.startswith("[huginn]")
    assert "HGN1-" not in banner


def test_worker_log_relay_separates_mem_and_summary(tmp_path: Path, monkeypatch):
    logs_dir = tmp_path / "logs"
    monkeypatch.setenv("HUGINN_LOGS_DIR", str(logs_dir))

    job_id = "12345678-aaaa-bbbb-cccc-ddddeeeeffff"
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    input_path = job_dir / "input.nii.gz"
    vol = np.zeros((16, 16, 8), dtype=np.int16)
    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(input_path))

    def fake_totalseg(input, output, task="total", roi_subset=None, device="cpu", **_kwargs):
        out_p = Path(output)
        out_p.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.zeros((16, 16, 8), dtype=np.uint8), np.eye(4)), str(out_p / "vertebrae_L3.nii.gz"))

    fake_pkg = types.ModuleType("totalsegmentator")
    fake_api = types.ModuleType("totalsegmentator.python_api")
    fake_api.totalsegmentator = fake_totalseg
    monkeypatch.setitem(sys.modules, "totalsegmentator", fake_pkg)
    monkeypatch.setitem(sys.modules, "totalsegmentator.python_api", fake_api)

    status_q: queue.Queue = queue.Queue()
    run_segmentation_worker(
        job_id=job_id,
        input_path=str(input_path),
        output_dir=str(job_dir),
        license_valid=False,
        status_queue=status_q,
        created_at="2026-01-01T00:00:00Z",
        started_at="2026-01-01T00:00:01Z",
        device="cpu",
    )

    job_log = Path(get_job_log_path(job_id, str(logs_dir)))
    assert job_log.exists()
    job_log_text = job_log.read_text(encoding="utf-8")
    assert "[mem] job=12345678" in job_log_text
    assert "[mem-summary] job=12345678" in job_log_text
    assert "etapa: validando volumen" in job_log_text
    assert "etapa: localizando L3" in job_log_text
    assert "etapa: empaquetando resultado" in job_log_text

    # Simular el drenaje en el proceso padre hacia session.log
    session_log = Path(get_session_log_path(str(logs_dir)))
    record = server.JobRecord(job_id=job_id, job_dir=str(job_dir), created_at="2026-01-01T00:00:00Z")
    last_stage_ref = [None]
    server._drain_worker_status_queue(record, status_q, last_stage_ref, str(session_log))

    session_text = session_log.read_text(encoding="utf-8")
    assert "[huginn] job=12345678 etapa: validando volumen" in session_text
    assert "[huginn] job=12345678 etapa: localizando L3" in session_text
    assert "[huginn] [mem-summary] job=12345678" in session_text
    # Las líneas [mem] de cada segundo NO van al monitor de la celda
    assert "[mem] " not in session_text

    # Verificar que el monitor lee y emite esas líneas
    emitted: List[str] = []
    stop_ev = threading.Event()

    def _on_line(line: str) -> None:
        emitted.append(line)
        if "[mem-summary]" in line:
            stop_ev.set()

    monitor_session_logs(
        log_path=str(session_log),
        poll_interval=0.01,
        start_offset=0,
        stop_event=stop_ev,
        out=_on_line,
    )
    assert any("[huginn] [mem-summary] job=12345678" in line for line in emitted)


def test_monitor_keyboard_interrupt_keeps_server_thread_alive(tmp_path: Path):
    session_log = tmp_path / "session.log"
    append_log_line(str(session_log), "[huginn] job=abcd1234 iniciado")

    stop_bg = threading.Event()
    bg_thread = threading.Thread(target=lambda: stop_bg.wait(5.0), daemon=True)
    bg_thread.start()

    emitted: List[str] = []

    def raise_interrupt(_line: str) -> None:
        if "Monitor detenido" not in _line:
            raise KeyboardInterrupt()
        emitted.append(_line)

    try:
        monitor_session_logs(
            log_path=str(session_log),
            poll_interval=0.01,
            start_offset=0,
            out=raise_interrupt,
        )
        assert bg_thread.is_alive()
        assert emitted == ["[huginn] Monitor detenido; el servicio sigue activo"]
    finally:
        stop_bg.set()
        bg_thread.join(timeout=1.0)


def test_sanitization_removes_secrets_urls_and_patient_filenames(tmp_path: Path, monkeypatch):
    logs_dir = tmp_path / "logs"
    monkeypatch.setenv("HUGINN_LOGS_DIR", str(logs_dir))

    sample_secret = "tok_secreto_123456789"
    conn_code = "HGN1-eyJ1cmwiOiJodHRwczovL2ZvbzphYmNAZXhhbXBsZS5jb20ifQ"
    cred_url = "https://x-access-token:gh_secreto_999@localhost/org/repo.git"
    patient_file = "Juan_Perez_Abdomen_2026.nii.gz"

    raw = (
        f"fallo con {sample_secret} codigo={conn_code} "
        f"repo={cred_url} Bearer mi_bearer_xyz archivo={patient_file}"
    )

    cleaned = sanitize_log_text(raw, extra_secrets=[sample_secret])
    assert sample_secret not in cleaned
    assert conn_code not in cleaned
    assert "gh_secreto_999" not in cleaned
    assert "mi_bearer_xyz" not in cleaned
    assert patient_file not in cleaned
    assert "HGN1-***" in cleaned
    assert "://***@" in cleaned
    assert "Bearer ***" in cleaned
    assert "<archivo>" in cleaned

    # También al escribir en session.log y en sampler
    session_log = get_session_log_path(str(logs_dir))
    append_log_line(session_log, format_huginn_line(raw, extra_secrets=[sample_secret]), extra_secrets=[sample_secret])
    written = Path(session_log).read_text(encoding="utf-8")
    assert sample_secret not in written
    assert conn_code not in written
    assert "gh_secreto_999" not in written
    assert patient_file not in written
