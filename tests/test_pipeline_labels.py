import gzip
import os
import queue
import struct
import sys
import types
import zipfile

import nibabel as nib
from nibabel.nifti1 import Nifti1Extension
import numpy as np
import pytest

from colab.pipeline import run_segmentation_worker


@pytest.mark.parametrize("input_dtype", [np.int16, np.float32])
def test_worker_saves_labels_as_uint8_in_input_grid(tmp_path, monkeypatch, input_dtype):
    """
    Verifica que run_segmentation_worker guarde labels.nii.gz con dtype uint8,
    bitpix 8, escala neutra, sin campos PII ni extensiones, y con la misma forma
    y matriz affine del volumen de entrada, incluso cuando input.nii.gz es int16 o float32.
    """
    shape = (8, 8, 6)
    affine = np.array(
        [
            [-1.0, 0.0, 0.0, 10.0],
            [0.0, -1.0, 0.0, 20.0],
            [0.0, 0.0, 2.5, -5.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    input_data = np.zeros(shape, dtype=input_dtype)
    input_img = nib.Nifti1Image(input_data, affine)
    input_img.header["descrip"] = b"Paciente Prueba"
    input_img.header["aux_file"] = b"aux_privado"
    input_img.header["db_name"] = b"db_privado"
    input_img.header["intent_name"] = b"intent_priv"
    input_img.header.set_slope_inter(2.0, -1024.0)
    input_img.header.extensions.append(Nifti1Extension(0, b"ext_privada"))
    input_path = str(tmp_path / "input.nii.gz")
    nib.save(input_img, input_path)

    def fake_totalsegmentator(input, output, task, roi_subset=None, device="gpu"):
        os.makedirs(output, exist_ok=True)
        if task == "total":
            l3 = np.zeros(shape, dtype=np.uint8)
            l3[2:5, 2:5, 2:4] = 1
            nib.save(nib.Nifti1Image(l3, affine), os.path.join(output, "vertebrae_L3.nii.gz"))
        elif task == "tissue_4_types":
            muscle = np.zeros(shape, dtype=np.uint8)
            muscle[1:4, 1:4, 1:5] = 1
            nib.save(nib.Nifti1Image(muscle, affine), os.path.join(output, "skeletal_muscle.nii.gz"))

    fake_pkg = types.ModuleType("totalsegmentator")
    fake_api = types.ModuleType("totalsegmentator.python_api")
    fake_api.totalsegmentator = fake_totalsegmentator
    monkeypatch.setitem(sys.modules, "totalsegmentator", fake_pkg)
    monkeypatch.setitem(sys.modules, "totalsegmentator.python_api", fake_api)

    output_dir = str(tmp_path / "out")
    os.makedirs(output_dir, exist_ok=True)
    status_q = queue.Queue()

    run_segmentation_worker(
        job_id="test-job",
        input_path=input_path,
        output_dir=output_dir,
        license_valid=True,
        status_queue=status_q,
        created_at="2026-01-01T00:00:00Z",
        started_at="2026-01-01T00:00:01Z",
        device="cpu",
    )

    result_zip = os.path.join(output_dir, "result.zip")
    assert os.path.exists(result_zip)

    extract_dir = str(tmp_path / "extracted")
    with zipfile.ZipFile(result_zip, "r") as zf:
        zf.extract("labels.nii.gz", extract_dir)

    labels_path = os.path.join(extract_dir, "labels.nii.gz")
    labels_img = nib.load(labels_path)
    hdr = labels_img.header

    assert labels_img.get_data_dtype() == np.dtype(np.uint8)
    assert int(hdr["datatype"]) == 2
    assert int(hdr["bitpix"]) == 8

    with gzip.open(labels_path, "rb") as gz:
        raw_hdr = gz.read(352)
    raw_slope, raw_inter = struct.unpack("<ff", raw_hdr[112:120])
    assert np.isnan(raw_slope) or raw_slope == 1.0
    assert raw_inter == 0.0

    for field in ("descrip", "aux_file", "db_name", "intent_name"):
        assert bytes(hdr[field]).rstrip(b"\x00") == b""
    assert len(hdr.extensions) == 0

    assert labels_img.shape == shape
    assert np.allclose(labels_img.affine, affine, atol=1e-4)


def test_worker_labels_identical_voxel_by_voxel_and_emits_diagnostics(tmp_path, monkeypatch, capsys):
    """
    Verifica que las reducciones de memoria producen labels.nii.gz idénticas
    vóxel a vóxel respecto a build_labels_volume, con dtype uint8 y encabezado neutro,
    y que se registran métricas de memoria por etapa sin datos de paciente.
    """
    from colab.geometry import build_labels_volume

    shape = (10, 10, 8)
    affine = np.diag([-1.0, -1.0, 2.0, 1.0])
    input_data = np.zeros(shape, dtype=np.int16)
    input_img = nib.Nifti1Image(input_data, affine)
    input_img.header["descrip"] = b"Paciente Secreto"
    input_path = str(tmp_path / "input.nii.gz")
    nib.save(input_img, input_path)

    l3 = np.zeros(shape, dtype=np.uint8)
    l3[3:7, 3:7, 3:5] = 1

    subcut = np.zeros(shape, dtype=np.uint8)
    subcut[0:4, 0:4, 1:6] = 1
    torso = np.zeros(shape, dtype=np.uint8)
    torso[2:6, 2:6, 1:6] = 1  # solapa con subcut -> debe ganar 2
    muscle = np.zeros(shape, dtype=np.uint8)
    muscle[4:8, 4:8, 2:7] = 1  # solapa con torso -> debe ganar 3
    interm = np.zeros(shape, dtype=np.uint8)
    interm[5:7, 5:7, 3:6] = 1  # solapa con muscle -> debe ganar 4

    baseline_labels = build_labels_volume(
        subcutaneous_data=subcut,
        torso_data=torso,
        muscle_data=muscle,
        intermuscular_data=interm,
        shape=shape,
    )

    def fake_totalsegmentator(input, output, task, roi_subset=None, device="gpu", nr_thr_saving=1, nr_thr_resamp=1):
        assert nr_thr_saving == 1
        assert nr_thr_resamp == 1
        os.makedirs(output, exist_ok=True)
        if task == "total":
            nib.save(nib.Nifti1Image(l3, affine), os.path.join(output, "vertebrae_L3.nii.gz"))
        elif task == "tissue_4_types":
            nib.save(nib.Nifti1Image(subcut, affine), os.path.join(output, "subcutaneous_fat.nii.gz"))
            nib.save(nib.Nifti1Image(torso, affine), os.path.join(output, "torso_fat.nii.gz"))
            nib.save(nib.Nifti1Image(muscle, affine), os.path.join(output, "skeletal_muscle.nii.gz"))
            nib.save(nib.Nifti1Image(interm, affine), os.path.join(output, "intermuscular_fat.nii.gz"))

    fake_pkg = types.ModuleType("totalsegmentator")
    fake_api = types.ModuleType("totalsegmentator.python_api")
    fake_api.totalsegmentator = fake_totalsegmentator
    monkeypatch.setitem(sys.modules, "totalsegmentator", fake_pkg)
    monkeypatch.setitem(sys.modules, "totalsegmentator.python_api", fake_api)

    output_dir = str(tmp_path / "out")
    os.makedirs(output_dir, exist_ok=True)
    status_q = queue.Queue()

    run_segmentation_worker(
        job_id="12345678-test-job",
        input_path=input_path,
        output_dir=output_dir,
        license_valid=True,
        status_queue=status_q,
        created_at="2026-01-01T00:00:00Z",
        started_at="2026-01-01T00:00:01Z",
        device="cpu",
    )

    extract_dir = str(tmp_path / "extracted")
    with zipfile.ZipFile(os.path.join(output_dir, "result.zip"), "r") as zf:
        zf.extract("labels.nii.gz", extract_dir)

    labels_img = nib.load(os.path.join(extract_dir, "labels.nii.gz"))
    assert labels_img.get_data_dtype() == np.dtype(np.uint8)
    worker_labels = np.asanyarray(labels_img.dataobj)
    assert worker_labels.dtype == np.uint8
    assert np.array_equal(worker_labels, baseline_labels)

    # Verificar diagnóstico emitido en status_queue
    messages = []
    while not status_q.empty():
        messages.append(status_q.get_nowait())
    completed_msgs = [m for m in messages if m.get("status") == "completed"]
    assert len(completed_msgs) == 1
    diag = completed_msgs[0]["diagnostics"]
    for stage in ("validating", "locating_l3", "segmenting_tissues", "packaging"):
        assert stage in diag["stages"]
        assert diag["stages"][stage]["samples"] >= 1
        assert diag["stages"][stage]["peak_rss_mb"] is not None

    captured = capsys.readouterr()
    assert "[mem]" in captured.out
    assert "[mem-summary]" in captured.out
    assert "Paciente Secreto" not in captured.out


def test_call_totalsegmentator_introspection_and_logging(caplog):
    import logging
    from colab.pipeline import _call_totalsegmentator

    received = {}

    def fn_with_thr(input, output, task, nr_thr_saving=None, nr_thr_resamp=None):
        received["with_thr"] = (nr_thr_saving, nr_thr_resamp)

    def fn_without_thr(input, output, task):
        received["without_thr"] = True

    def fn_with_kwargs(**kwargs):
        received["with_kwargs"] = (kwargs.get("nr_thr_saving"), kwargs.get("nr_thr_resamp"))

    with caplog.at_level(logging.INFO, logger="huginn.pipeline"):
        _call_totalsegmentator(fn_with_thr, input="a", output="b", task="total")
        assert received["with_thr"] == (1, 1)
        assert not caplog.records

        _call_totalsegmentator(fn_with_kwargs, input="a", output="b", task="total")
        assert received["with_kwargs"] == (1, 1)
        assert not caplog.records

        _call_totalsegmentator(fn_without_thr, input="a", output="b", task="total")
        assert received["without_thr"] is True
        assert any(
            "nr_thr_saving" in rec.message and "nr_thr_resamp" in rec.message
            for rec in caplog.records
        )
