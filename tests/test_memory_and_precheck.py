from __future__ import annotations

import os
import tempfile
import time
from typing import Dict

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from colab import server
from colab.memory import (
    DEFAULT_MEMORY_ESTIMATION_K,
    StageMemorySampler,
    estimate_required_memory_bytes,
)


@pytest.fixture
def auth_header() -> Dict[str, str]:
    return {"Authorization": f"Bearer {server.API_TOKEN}"}


@pytest.fixture
def client() -> TestClient:
    return TestClient(server.app)


def _create_synthetic_nifti_bytes(shape=(20, 20, 10), spacing=(1.0, 1.0, 1.0)) -> bytes:
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    data = np.zeros(shape, dtype=np.int16)
    img = nib.Nifti1Image(data, affine)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp_name = tmp.name
    try:
        nib.save(img, tmp_name)
        with open(tmp_name, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def test_stage_memory_sampler_collects_samples_and_summary(capsys):
    sampler = StageMemorySampler(job_id="abcdef12-0000-0000", stage="segmenting_tissues", interval_seconds=0.05)
    sampler.start()
    time.sleep(0.14)
    summary = sampler.stop()

    assert summary["samples"] >= 2
    assert summary["duration_seconds"] >= 0.0
    assert summary["peak_rss_mb"] is not None and summary["peak_rss_mb"] > 0
    assert summary["peak_sys_used_mb"] is not None and summary["peak_sys_used_mb"] > 0
    assert summary["min_sys_avail_mb"] is not None and summary["min_sys_avail_mb"] > 0

    out = capsys.readouterr().out
    assert "[mem] job=abcdef12 stage=segmenting_tissues" in out
    assert "[mem-summary] job=abcdef12 stage=segmenting_tissues" in out


def test_stage_memory_sampler_flushes_mem_and_summary_lines(monkeypatch):
    import builtins

    flushed_lines = []
    real_print = builtins.print

    def spy_print(*args, **kwargs):
        text = " ".join(str(a) for a in args)
        if text.startswith("[mem]") or text.startswith("[mem-summary]"):
            flushed_lines.append((text, kwargs.get("flush", False)))
        return real_print(*args, **kwargs)

    monkeypatch.setattr(builtins, "print", spy_print)
    sampler = StageMemorySampler(job_id="flush1234-0000", stage="locating_l3", interval_seconds=0.05)
    sampler.start()
    time.sleep(0.08)
    sampler.stop()

    mem_lines = [f for t, f in flushed_lines if t.startswith("[mem] ")]
    summary_lines = [f for t, f in flushed_lines if t.startswith("[mem-summary] ")]
    assert len(mem_lines) >= 1 and all(mem_lines)
    assert len(summary_lines) == 1 and all(summary_lines)


def test_health_reports_system_ram(client: TestClient, auth_header: Dict[str, str]):
    res = client.get("/health", headers=auth_header)
    assert res.status_code == 200
    data = res.json()
    assert data["contract_version"] == "2.0"
    assert isinstance(data["ram_total_mb"], float) and data["ram_total_mb"] > 0
    assert isinstance(data["ram_available_mb"], float) and data["ram_available_mb"] > 0


def test_default_memory_estimation_k_is_100():
    assert DEFAULT_MEMORY_ESTIMATION_K == 100.0


def test_estimate_required_memory_bytes_scales_with_dimensions_and_k():
    est_k10 = estimate_required_memory_bytes((100, 100, 50), k=10.0)
    est_k20 = estimate_required_memory_bytes((100, 100, 50), k=20.0)
    assert est_k10 == 100 * 100 * 50 * 4 * 10
    assert est_k20 == 2 * est_k10


def test_estimate_required_memory_bytes_resampled_thin_vs_thick_slices():
    # Caso de cortes finos: muchos cortes de 0,5 mm (512x512x600 a 0.75x0.75x0.5 mm)
    # Remuestreado a 1,5 mm: (256, 256, 200) -> 13.1M vóxeles (vs 157.3M brutos)
    est_thin = estimate_required_memory_bytes(
        (512, 512, 600),
        spacing=(0.75, 0.75, 0.5),
        k=100.0,
    )
    expected_thin = int((512 * 0.75 / 1.5) * (512 * 0.75 / 1.5) * (600 * 0.5 / 1.5) * 4 * 100.0)
    assert est_thin == expected_thin
    assert est_thin < 6 * 1024**3  # Cabe en runtime gratis (~12 GB) pese a tener 600 cortes

    # Caso de cortes gruesos: 100 cortes de 5,0 mm (512x512x100 a 0.75x0.75x5.0 mm)
    # Remuestreado a 1,5 mm: (256, 256, 333.33) -> 21.8M vóxeles (mayor que 256x256x100)
    est_thick = estimate_required_memory_bytes(
        (512, 512, 100),
        spacing=(0.75, 0.75, 5.0),
        k=100.0,
    )
    expected_thick = int((512 * 0.75 / 1.5) * (512 * 0.75 / 1.5) * (100 * 5.0 / 1.5) * 4 * 100.0)
    assert est_thick == expected_thick
    assert est_thick > est_thin

    # Escenario de calibración con k=100: ~500 cortes (0.9x0.9x1.0 mm) excede ~12 GB;
    # 107 cortes (0.7x0.7x2.5 mm) y 212 cortes (0.7x0.7x1.5 mm) caben en ~12 GB.
    free_colab_avail = 11.5 * 1024**3
    est_500 = estimate_required_memory_bytes((512, 512, 500), spacing=(0.9, 0.9, 1.0), k=100.0)
    est_107 = estimate_required_memory_bytes((512, 512, 107), spacing=(0.7, 0.7, 2.5), k=100.0)
    est_212 = estimate_required_memory_bytes((512, 512, 212), spacing=(0.7, 0.7, 1.5), k=100.0)
    assert est_500 > free_colab_avail
    assert est_107 < free_colab_avail
    assert est_212 < free_colab_avail


def test_create_job_precheck_uses_header_spacing_resampled_at_1_5mm(
    client: TestClient,
    auth_header: Dict[str, str],
    monkeypatch,
):
    # 40x40x60 con cortes finos (0.75x0.75x0.5 mm):
    # Bruto: 96,000 vóxeles * 4 * 100 = 38.4 MB
    # Remuestreado 1.5mm: 20 * 20 * 20 = 8,000 vóxeles * 4 * 100 = 3.2 MB
    # Con 10 MB libres: bruto fallaría, remuestreado debe aceptarse.
    monkeypatch.setattr(server, "get_system_memory_bytes", lambda: (1024**3, 10 * 1024 * 1024))
    monkeypatch.setattr(server, "MEMORY_ESTIMATION_K", 100.0)

    thin_bytes = _create_synthetic_nifti_bytes(shape=(40, 40, 60), spacing=(0.75, 0.75, 0.5))
    res_thin = client.post(
        "/jobs",
        headers=auth_header,
        files={"file": ("thin.nii.gz", thin_bytes, "application/gzip")},
    )
    assert res_thin.status_code == 201
    client.delete(f"/jobs/{res_thin.json()['job_id']}", headers=auth_header)

    # 20x20x20 con cortes gruesos (3.0x3.0x4.5 mm):
    # Bruto: 8,000 vóxeles * 4 * 100 = 3.2 MB
    # Remuestreado 1.5mm: 40 * 40 * 60 = 96,000 vóxeles * 4 * 100 = 38.4 MB
    # Con 10 MB libres: bruto pasaría, remuestreado debe rechazarse (422).
    thick_bytes = _create_synthetic_nifti_bytes(shape=(20, 20, 20), spacing=(3.0, 3.0, 4.5))
    res_thick = client.post(
        "/jobs",
        headers=auth_header,
        files={"file": ("thick.nii.gz", thick_bytes, "application/gzip")},
    )
    assert res_thick.status_code == 422
    assert res_thick.json()["error_code"] == "ERR_INSUFFICIENT_MEMORY"


def test_create_job_rejects_insufficient_memory_before_processing(
    client: TestClient,
    auth_header: Dict[str, str],
    monkeypatch,
):
    # Simular que solo quedan 100 KB disponibles y que el volumen requiere más
    monkeypatch.setattr(server, "get_system_memory_bytes", lambda: (1024 * 1024 * 1024, 100 * 1024))
    monkeypatch.setattr(server, "MEMORY_ESTIMATION_K", 100.0)

    before_jobs = set(server.jobs_db.keys())
    nii_bytes = _create_synthetic_nifti_bytes(shape=(20, 20, 10))
    res = client.post(
        "/jobs",
        headers=auth_header,
        files={"file": ("study.nii.gz", nii_bytes, "application/gzip")},
    )

    assert res.status_code == 422
    payload = res.json()
    assert payload["error_code"] == "ERR_INSUFFICIENT_MEMORY"
    assert payload["stage"] == "validating"
    assert "Memoria insuficiente" in payload["detail"]
    # No debe haberse encolado ningún trabajo
    assert set(server.jobs_db.keys()) == before_jobs


def test_create_job_accepts_when_memory_sufficient(
    client: TestClient,
    auth_header: Dict[str, str],
    monkeypatch,
):
    monkeypatch.setattr(server, "get_system_memory_bytes", lambda: (16 * 1024**3, 12 * 1024**3))
    monkeypatch.setattr(server, "MEMORY_ESTIMATION_K", 100.0)

    nii_bytes = _create_synthetic_nifti_bytes(shape=(20, 20, 10))
    res = client.post(
        "/jobs",
        headers=auth_header,
        files={"file": ("study.nii.gz", nii_bytes, "application/gzip")},
    )
    assert res.status_code == 201
    job_id = res.json()["job_id"]
    client.delete(f"/jobs/{job_id}", headers=auth_header)


def test_job_status_returns_diagnostics_when_present(client: TestClient, auth_header: Dict[str, str]):
    nii_bytes = _create_synthetic_nifti_bytes(shape=(20, 20, 10))
    res = client.post(
        "/jobs",
        headers=auth_header,
        files={"file": ("study.nii.gz", nii_bytes, "application/gzip")},
    )
    assert res.status_code == 201
    job_id = res.json()["job_id"]

    server.jobs_db[job_id].diagnostics = {
        "stages": {
            "locating_l3": {"duration_seconds": 1.2, "samples": 2, "peak_rss_mb": 250.0}
        }
    }
    status_res = client.get(f"/jobs/{job_id}", headers=auth_header)
    assert status_res.status_code == 200
    body = status_res.json()
    assert body["diagnostics"]["stages"]["locating_l3"]["peak_rss_mb"] == 250.0

    client.delete(f"/jobs/{job_id}", headers=auth_header)
