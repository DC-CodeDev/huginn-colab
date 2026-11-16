import os
import queue
import sys
import types
import zipfile

import nibabel as nib
import numpy as np
import pytest

from colab.pipeline import run_segmentation_worker


@pytest.mark.parametrize("input_dtype", [np.int16, np.float32])
def test_worker_saves_labels_as_uint8_in_input_grid(tmp_path, monkeypatch, input_dtype):
    """
    Verifica que run_segmentation_worker guarde labels.nii.gz con dtype uint8
    y la misma forma y matriz affine del volumen de entrada, incluso cuando
    input.nii.gz es int16 o float32.
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
    input_path = str(tmp_path / "input.nii.gz")
    nib.save(nib.Nifti1Image(input_data, affine), input_path)

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

    labels_img = nib.load(os.path.join(extract_dir, "labels.nii.gz"))
    assert labels_img.get_data_dtype() == np.dtype(np.uint8)
    assert labels_img.shape == shape
    assert np.allclose(labels_img.affine, affine, atol=1e-4)
