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
    DEFAULT_MEMORY_BASE_MB,
    DEFAULT_MEMORY_ESTIMATION_K,
    StageMemorySampler,
    compute_volume_geometry_stats,
    estimate_required_memory_bytes,
    get_memory_base_mb,
    get_memory_estimation_k,
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


def test_default_memory_constants_and_env_config(monkeypatch):
    assert DEFAULT_MEMORY_BASE_MB == 4000.0
    assert DEFAULT_MEMORY_ESTIMATION_K == 150.0

    monkeypatch.setenv("HUGINN_MEMORY_BASE_MB", "3500")
    monkeypatch.setenv("HUGINN_MEMORY_ESTIMATION_K", "180")
    assert get_memory_base_mb() == 3500.0
    assert get_memory_estimation_k() == 180.0


def test_estimate_required_memory_bytes_linear_model():
    # estimado = BASE_MB + k * resampled_voxels (k en bytes por vóxel)
    est_k100 = estimate_required_memory_bytes((100, 100, 50), k=100.0, base_mb=4000.0)
    est_k200 = estimate_required_memory_bytes((100, 100, 50), k=200.0, base_mb=4000.0)
    base_bytes = 4000 * 1024 * 1024
    voxels = 100 * 100 * 50
    assert est_k100 == base_bytes + voxels * 100
    assert est_k200 == base_bytes + voxels * 200


def test_estimate_required_memory_bytes_resampled_thin_vs_thick_slices():
    # Caso de cortes finos: 600 cortes de 0,5 mm (512x512x600 a 0.75x0.75x0.5 mm)
    # Remuestreado a 1,5 mm: (256, 256, 200) -> ~13.1M vóxeles
    est_thin = estimate_required_memory_bytes(
        (512, 512, 600),
        spacing=(0.75, 0.75, 0.5),
        k=150.0,
        base_mb=4000.0,
    )
    thin_voxels = round((512 * 0.75 / 1.5) * (512 * 0.75 / 1.5) * (600 * 0.5 / 1.5))
    assert est_thin == int(4000 * 1024**2 + thin_voxels * 150.0)
    assert est_thin < 8 * 1024**3  # Entra en runtime gratis (~11,5 GB disponibles)

    # Caso de cortes gruesos: 100 cortes de 5,0 mm (512x512x100 a 0.75x0.75x5.0 mm)
    # Remuestreado a 1,5 mm: (256, 256, 333.33) -> ~21.8M vóxeles
    est_thick = estimate_required_memory_bytes(
        (512, 512, 100),
        spacing=(0.75, 0.75, 5.0),
        k=150.0,
        base_mb=4000.0,
    )
    thick_voxels = round((512 * 0.75 / 1.5) * (512 * 0.75 / 1.5) * (100 * 5.0 / 1.5))
    assert est_thick == int(4000 * 1024**2 + thick_voxels * 150.0)
    assert est_thick > est_thin

    # Punto de calibración real: 421 cortes de 1,3 mm (~0.7x0.7x1.3) entra en ~11.5 GB libres
    # Estudio mucho más largo (900 cortes a 1,5 mm, 512x512 a 0.9x0.9x1.5) excede ~11.5 GB libres
    free_colab_avail = 11.5 * 1024**3
    est_421 = estimate_required_memory_bytes((512, 512, 421), spacing=(0.7, 0.7, 1.3), k=150.0, base_mb=4000.0)
    est_900 = estimate_required_memory_bytes((512, 512, 900), spacing=(0.9, 0.9, 1.5), k=150.0, base_mb=4000.0)
    assert est_421 < free_colab_avail
    assert est_900 > free_colab_avail


def test_create_job_precheck_uses_header_spacing_resampled_at_1_5mm(
    client: TestClient,
    auth_header: Dict[str, str],
    monkeypatch,
):
    # Con BASE_MB=4000 y k=150:
    # 40x40x60 con cortes finos (0.75x0.75x0.5 mm) -> 8,000 vóxeles remuestreados -> ~4001.1 MB
    # 20x20x20 con cortes gruesos (3.0x3.0x4.5 mm) -> 96,000 vóxeles remuestreados -> ~4013.7 MB
    # Con 4005 MB disponibles: cortes finos entra (201), cortes gruesos se rechaza (422).
    monkeypatch.setattr(server, "get_system_memory_bytes", lambda: (12 * 1024**3, 4005 * 1024 * 1024))
    monkeypatch.setattr(server, "MEMORY_BASE_MB", 4000.0)
    monkeypatch.setattr(server, "MEMORY_ESTIMATION_K", 150.0)

    thin_bytes = _create_synthetic_nifti_bytes(shape=(40, 40, 60), spacing=(0.75, 0.75, 0.5))
    res_thin = client.post(
        "/jobs",
        headers=auth_header,
        files={"file": ("thin.nii.gz", thin_bytes, "application/gzip")},
    )
    assert res_thin.status_code == 201
    client.delete(f"/jobs/{res_thin.json()['job_id']}", headers=auth_header)

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
    # Simular que solo quedan 3500 MB disponibles (< BASE_MB=4000)
    monkeypatch.setattr(server, "get_system_memory_bytes", lambda: (12 * 1024**3, 3500 * 1024 * 1024))
    monkeypatch.setattr(server, "MEMORY_BASE_MB", 4000.0)
    monkeypatch.setattr(server, "MEMORY_ESTIMATION_K", 150.0)

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
    assert "3500.0 MB" in payload["detail"]
    assert "más RAM" in payload["detail"]
    assert "menos cortes" in payload["detail"]
    # No debe haberse encolado ningún trabajo
    assert set(server.jobs_db.keys()) == before_jobs


def test_create_job_accepts_when_memory_sufficient(
    client: TestClient,
    auth_header: Dict[str, str],
    monkeypatch,
):
    monkeypatch.setattr(server, "get_system_memory_bytes", lambda: (16 * 1024**3, 12 * 1024**3))
    monkeypatch.setattr(server, "MEMORY_BASE_MB", 4000.0)
    monkeypatch.setattr(server, "MEMORY_ESTIMATION_K", 150.0)

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
