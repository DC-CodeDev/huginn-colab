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
