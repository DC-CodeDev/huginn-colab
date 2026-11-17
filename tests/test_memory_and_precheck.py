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
from colab.memory import StageMemorySampler, estimate_required_memory_bytes


@pytest.fixture
def auth_header() -> Dict[str, str]:
    return {"Authorization": f"Bearer {server.API_TOKEN}"}


@pytest.fixture
def client() -> TestClient:
    return TestClient(server.app)


def _create_synthetic_nifti_bytes(shape=(20, 20, 10)) -> bytes:
    affine = np.eye(4)
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


def test_health_reports_system_ram(client: TestClient, auth_header: Dict[str, str]):
    res = client.get("/health", headers=auth_header)
    assert res.status_code == 200
    data = res.json()
    assert data["contract_version"] == "2.0"
    assert isinstance(data["ram_total_mb"], float) and data["ram_total_mb"] > 0
    assert isinstance(data["ram_available_mb"], float) and data["ram_available_mb"] > 0


def test_estimate_required_memory_bytes_scales_with_dimensions_and_k():
    est_k10 = estimate_required_memory_bytes((100, 100, 50), k=10.0)
    est_k20 = estimate_required_memory_bytes((100, 100, 50), k=20.0)
    assert est_k10 == 100 * 100 * 50 * 4 * 10
    assert est_k20 == 2 * est_k10


def test_create_job_rejects_insufficient_memory_before_processing(
    client: TestClient,
    auth_header: Dict[str, str],
    monkeypatch,
):
    # Simular que solo quedan 100 KB disponibles y que el volumen requiere más
    monkeypatch.setattr(server, "get_system_memory_bytes", lambda: (1024 * 1024 * 1024, 100 * 1024))
    monkeypatch.setattr(server, "MEMORY_ESTIMATION_K", 20.0)

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
    monkeypatch.setattr(server, "MEMORY_ESTIMATION_K", 20.0)

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
