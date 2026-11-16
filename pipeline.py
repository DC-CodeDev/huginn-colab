from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import shutil
import time
import zipfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import fastapi
import nibabel as nib
import numpy as np

from colab.geometry import analyze_l3, build_labels_volume, build_manifest

log = logging.getLogger("huginn.pipeline")

# Versiones fijadas de paquetes requeridos en el manifiesto
TOTALSEGMENTATOR_VERSION = "2.18.0"
FASTAPI_VERSION = getattr(fastapi, "__version__", "0.141.1")
NIBABEL_VERSION = getattr(nib, "__version__", "5.5.0")
NUMPY_VERSION = getattr(np, "__version__", "2.5.2")


def run_segmentation_worker(
    job_id: str,
    input_path: str,
    output_dir: str,
    license_valid: bool,
    status_queue: Any,
    created_at: str,
    started_at: str,
    device: str = "gpu",
) -> None:
    """
    Función de entrada del subproceso (spawn).
    Ejecuta las pasadas de TotalSegmentator, analiza L3, empaqueta result.zip y
    comunica el avance mediante status_queue.
    """
    try:
        t0 = time.time()
        status_queue.put({"stage": "validating", "progress": 0.05})

        input_img = nib.load(input_path)
        input_shape = input_img.shape
        affine = input_img.affine

        tasks_executed: List[str] = []

        # -------------------------------------------------------------
        # 1. Pasada de vértebras (Localización de L3)
        # -------------------------------------------------------------
        status_queue.put({"stage": "locating_l3", "progress": 0.15})
        tasks_executed.append("total:roi_subset=vertebrae_L3")

        l3_work_dir = os.path.join(output_dir, "work_l3")
        os.makedirs(l3_work_dir, exist_ok=True)

        from totalsegmentator.python_api import totalsegmentator

        totalsegmentator(
            input=input_path,
            output=l3_work_dir,
            task="total",
            roi_subset=["vertebrae_L3"],
            device=device,
        )

        l3_file = os.path.join(l3_work_dir, "vertebrae_L3.nii.gz")
        if os.path.exists(l3_file):
            l3_img = nib.load(l3_file)
            l3_data = np.asanyarray(l3_img.dataobj)
        else:
            l3_data = None

        l3_analysis = analyze_l3(l3_data, affine)

        # -------------------------------------------------------------
        # 2. Pasada de tejidos (tissue_4_types)
        # -------------------------------------------------------------
        tissue_segmentation_skipped = not license_valid
        labels_nii_path: Optional[str] = None

        if license_valid:
            status_queue.put({"stage": "segmenting_tissues", "progress": 0.50})
            tasks_executed.append("tissue_4_types")

            tissues_work_dir = os.path.join(output_dir, "work_tissues")
            os.makedirs(tissues_work_dir, exist_ok=True)

            totalsegmentator(
                input=input_path,
                output=tissues_work_dir,
                task="tissue_4_types",
                device=device,
            )

            def _load_mask(name: str) -> Optional[np.ndarray]:
                p = os.path.join(tissues_work_dir, name)
                if os.path.exists(p):
                    return np.asanyarray(nib.load(p).dataobj)
                return None

            subcutaneous = _load_mask("subcutaneous_fat.nii.gz")
            torso = _load_mask("torso_fat.nii.gz")
            muscle = _load_mask("skeletal_muscle.nii.gz")
            intermuscular = _load_mask("intermuscular_fat.nii.gz")

            labels_data = build_labels_volume(
                subcutaneous_data=subcutaneous,
                torso_data=torso,
                muscle_data=muscle,
                intermuscular_data=intermuscular,
                shape=input_shape[:3],
            )

            # Ajuste 1: solo se guarda labels.nii.gz si hay licencia válida
            # Contrato 2.0: labels.nii.gz debe guardarse siempre en uint8 sobre la grilla de entrada
            labels_nii_path = os.path.join(output_dir, "labels.nii.gz")
            labels_img = nib.Nifti1Image(np.asarray(labels_data, dtype=np.uint8), affine)
            labels_img.set_data_dtype(np.uint8)
            nib.save(labels_img, labels_nii_path)

        # -------------------------------------------------------------
        # 3. Empaquetado de resultados
        # -------------------------------------------------------------
        status_queue.put({"stage": "packaging", "progress": 0.90})
        t1 = time.time()
        completed_at = datetime.now(timezone.utc).isoformat()
        duration_seconds = t1 - t0

        gpu_name = None
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass

        manifest_data = build_manifest(
            job_id=job_id,
            created_at=created_at,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration_seconds,
            totalsegmentator_version=TOTALSEGMENTATOR_VERSION,
            fastapi_version=FASTAPI_VERSION,
            nibabel_version=NIBABEL_VERSION,
            numpy_version=NUMPY_VERSION,
            tasks_executed=tasks_executed,
            license_valid=license_valid,
            tissue_segmentation_skipped=tissue_segmentation_skipped,
            device=device,
            gpu_name=gpu_name,
            input_shape=input_shape,
            affine=affine,
            l3_analysis=l3_analysis,
        )

        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        # Generar result.zip
        result_zip_path = os.path.join(output_dir, "result.zip")
        with zipfile.ZipFile(result_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(manifest_path, arcname="manifest.json")
            # Ajuste 1: Si fue omitido, NO se incluye labels.nii.gz
            if not tissue_segmentation_skipped and labels_nii_path and os.path.exists(labels_nii_path):
                zf.write(labels_nii_path, arcname="labels.nii.gz")

        # Limpieza de carpetas intermedias de máscaras para ahorrar espacio
        shutil.rmtree(l3_work_dir, ignore_errors=True)
        if license_valid and os.path.exists(tissues_work_dir):
            shutil.rmtree(tissues_work_dir, ignore_errors=True)

        status_queue.put({
            "stage": "completed",
            "progress": 1.0,
            "status": "completed",
            "completed_at": completed_at,
        })

    except Exception as exc:
        log.exception("Error durante la segmentación del trabajo %s", job_id)
        status_queue.put({
            "stage": "failed",
            "status": "failed",
            "error_code": "ERR_PIPELINE_FAILED",
            "detail": str(exc),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })


def pack_result_from_arrays(
    output_dir: str,
    job_id: str,
    input_shape: Tuple[int, ...],
    affine: np.ndarray,
    l3_mask_data: Optional[np.ndarray],
    tissue_masks: Optional[Dict[str, np.ndarray]],
    license_valid: bool,
    created_at: str,
    started_at: str,
    device: str = "cpu",
    gpu_name: Optional[str] = None,
) -> str:
    """
    Función utilitaria pura (usada también para pruebas sin requerir GPU) que ejecuta
    el análisis geométrico y el empaquetado final de result.zip y manifest.json.
    """
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    l3_analysis = analyze_l3(l3_mask_data, affine)
    tissue_segmentation_skipped = not license_valid
    labels_nii_path: Optional[str] = None

    tasks_executed = ["total:roi_subset=vertebrae_L3"]
    if license_valid:
        tasks_executed.append("tissue_4_types")
        subcutaneous = tissue_masks.get("subcutaneous_fat") if tissue_masks else None
        torso = tissue_masks.get("torso_fat") if tissue_masks else None
        muscle = tissue_masks.get("skeletal_muscle") if tissue_masks else None
        intermuscular = tissue_masks.get("intermuscular_fat") if tissue_masks else None

        labels_data = build_labels_volume(
            subcutaneous_data=subcutaneous,
            torso_data=torso,
            muscle_data=muscle,
            intermuscular_data=intermuscular,
            shape=input_shape[:3],
        )
        labels_nii_path = os.path.join(output_dir, "labels.nii.gz")
        labels_img = nib.Nifti1Image(labels_data, affine)
        nib.save(labels_img, labels_nii_path)

    completed_at = datetime.now(timezone.utc).isoformat()
    duration_seconds = time.time() - t0

    manifest_data = build_manifest(
        job_id=job_id,
        created_at=created_at,
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=duration_seconds,
        totalsegmentator_version=TOTALSEGMENTATOR_VERSION,
        fastapi_version=FASTAPI_VERSION,
        nibabel_version=NIBABEL_VERSION,
        numpy_version=NUMPY_VERSION,
        tasks_executed=tasks_executed,
        license_valid=license_valid,
        tissue_segmentation_skipped=tissue_segmentation_skipped,
        device=device,
        gpu_name=gpu_name,
        input_shape=input_shape,
        affine=affine,
        l3_analysis=l3_analysis,
    )

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    result_zip_path = os.path.join(output_dir, "result.zip")
    with zipfile.ZipFile(result_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(manifest_path, arcname="manifest.json")
        # Ajuste 1: Si fue omitido, NO se incluye labels.nii.gz
        if not tissue_segmentation_skipped and labels_nii_path and os.path.exists(labels_nii_path):
            zf.write(labels_nii_path, arcname="labels.nii.gz")

    return result_zip_path
