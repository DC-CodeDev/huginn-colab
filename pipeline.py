from __future__ import annotations

import gc
import inspect
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

from colab.geometry import (
    analyze_l3,
    build_labels_volume,
    build_labels_volume_from_files,
    build_manifest,
)
from colab.memory import (
    StageMemorySampler,
    compute_volume_geometry_stats,
    estimate_required_memory_bytes,
)
from colab.session_log import (
    append_log_line,
    format_stage_label,
    get_job_log_path,
)

log = logging.getLogger("huginn.pipeline")

# Versiones fijadas de paquetes requeridos en el manifiesto
TOTALSEGMENTATOR_VERSION = "2.18.0"
FASTAPI_VERSION = getattr(fastapi, "__version__", "0.141.1")
NIBABEL_VERSION = getattr(nib, "__version__", "5.5.0")
NUMPY_VERSION = getattr(np, "__version__", "2.5.2")


def _release_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _call_totalsegmentator(totalsegmentator_fn: Any, **kwargs: Any) -> Any:
    """
    Invoca totalsegmentator añadiendo nr_thr_saving=1 y nr_thr_resamp=1
    si la firma de la función (o **kwargs) los acepta, para reducir uso de RAM.
    Si la firma no los acepta, registra una línea informándolo.
    """
    try:
        params = inspect.signature(totalsegmentator_fn).parameters
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        unsupported: List[str] = []
        if accepts_kwargs or "nr_thr_saving" in params:
            kwargs.setdefault("nr_thr_saving", 1)
        else:
            unsupported.append("nr_thr_saving")
        if accepts_kwargs or "nr_thr_resamp" in params:
            kwargs.setdefault("nr_thr_resamp", 1)
        else:
            unsupported.append("nr_thr_resamp")
        if unsupported:
            log.info(
                "totalsegmentator no acepta %s en su firma; se omite(n).",
                ", ".join(unsupported),
            )
    except Exception as exc:
        log.info("No se pudo inspeccionar la firma de totalsegmentator: %s", exc)
    return totalsegmentator_fn(**kwargs)


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
    diagnostics: Dict[str, Any] = {"stages": {}}
    active_sampler: Optional[StageMemorySampler] = None
    job_log_path = get_job_log_path(job_id)

    def _begin_stage(stage: str, progress: float) -> StageMemorySampler:
        append_log_line(job_log_path, f"job={job_id[:8]} {format_stage_label(stage)}")
        status_queue.put({"stage": stage, "progress": progress})
        return StageMemorySampler(job_id=job_id, stage=stage, log_path=job_log_path).start()

    def _end_stage(sampler: StageMemorySampler, push_diagnostics: bool = True) -> None:
        diagnostics["stages"][sampler.stage] = sampler.stop()
        if sampler.last_summary_line:
            status_queue.put({"mem_summary": sampler.last_summary_line})
        if push_diagnostics:
            status_queue.put({"diagnostics": diagnostics})

    try:
        t0 = time.time()
        active_sampler = _begin_stage("validating", 0.05)

        input_img = nib.load(input_path)
        input_shape = input_img.shape
        affine = input_img.affine
        zooms = tuple(float(z) for z in input_img.header.get_zooms()[:3])
        del input_img

        vol_stats = compute_volume_geometry_stats(input_shape, spacing=zooms)
        est_mb = round(estimate_required_memory_bytes(input_shape, spacing=zooms) / (1024.0 * 1024.0), 1)
        vol_stats["estimated_ram_mb"] = est_mb
        diagnostics["volume"] = vol_stats

        volume_line = (
            f"[volume] job={job_id[:8]} "
            f"shape={tuple(vol_stats['shape'])} "
            f"spacing_mm={tuple(vol_stats['spacing_mm']) if vol_stats['spacing_mm'] else None} "
            f"orig_voxels={vol_stats['original_voxels']} "
            f"resampled_1_5mm_voxels={vol_stats['resampled_voxels']} "
            f"est_ram_mb={est_mb}"
        )
        print(volume_line, flush=True)
        append_log_line(job_log_path, volume_line)
        status_queue.put({"log_line": volume_line})

        tasks_executed: List[str] = []
        _end_stage(active_sampler)
        active_sampler = None

        # -------------------------------------------------------------
        # 1. Pasada de vértebras (Localización de L3)
        # -------------------------------------------------------------
        active_sampler = _begin_stage("locating_l3", 0.15)
        tasks_executed.append("total:roi_subset=vertebrae_L3")

        l3_work_dir = os.path.join(output_dir, "work_l3")
        os.makedirs(l3_work_dir, exist_ok=True)

        from totalsegmentator.python_api import totalsegmentator

        _call_totalsegmentator(
            totalsegmentator,
            input=input_path,
            output=l3_work_dir,
            task="total",
            roi_subset=["vertebrae_L3"],
            device=device,
        )

        l3_file = os.path.join(l3_work_dir, "vertebrae_L3.nii.gz")
        l3_img = None
        if os.path.exists(l3_file):
            l3_img = nib.load(l3_file)
            l3_data = np.asanyarray(l3_img.dataobj)
        else:
            l3_data = None

        l3_analysis = analyze_l3(l3_data, affine)
        del l3_data, l3_img
        gc.collect()
        _release_cuda_cache()

        _end_stage(active_sampler)
        active_sampler = None

        # -------------------------------------------------------------
        # 2. Pasada de tejidos (tissue_4_types)
        # -------------------------------------------------------------
        tissue_segmentation_skipped = not license_valid
        labels_nii_path: Optional[str] = None
        tissues_work_dir = os.path.join(output_dir, "work_tissues")

        if license_valid:
            active_sampler = _begin_stage("segmenting_tissues", 0.50)
            tasks_executed.append("tissue_4_types")

            os.makedirs(tissues_work_dir, exist_ok=True)

            _call_totalsegmentator(
                totalsegmentator,
                input=input_path,
                output=tissues_work_dir,
                task="tissue_4_types",
                device=device,
            )

            labels_data = build_labels_volume_from_files(
                mask_paths=[
                    (os.path.join(tissues_work_dir, "subcutaneous_fat.nii.gz"), 1),
                    (os.path.join(tissues_work_dir, "torso_fat.nii.gz"), 2),
                    (os.path.join(tissues_work_dir, "skeletal_muscle.nii.gz"), 3),
                    (os.path.join(tissues_work_dir, "intermuscular_fat.nii.gz"), 4),
                ],
                shape=input_shape[:3],
            )

            # Ajuste 1: solo se guarda labels.nii.gz si hay licencia válida
            # Contrato 2.0: labels.nii.gz debe guardarse siempre en uint8 con cabecera limpia sobre la grilla de entrada
            labels_nii_path = os.path.join(output_dir, "labels.nii.gz")
            labels_img = nib.Nifti1Image(labels_data, affine)
            labels_img.set_data_dtype(np.uint8)
            labels_img.header.set_slope_inter(1.0, 0.0)
            nib.save(labels_img, labels_nii_path)
            del labels_img, labels_data
            gc.collect()
            _release_cuda_cache()

            _end_stage(active_sampler)
            active_sampler = None

        # -------------------------------------------------------------
        # 3. Empaquetado de resultados
        # -------------------------------------------------------------
        active_sampler = _begin_stage("packaging", 0.90)
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

        _end_stage(active_sampler, push_diagnostics=False)
        active_sampler = None

        append_log_line(job_log_path, f"job={job_id[:8]} {format_stage_label('completed')}")
        status_queue.put({
            "stage": "completed",
            "progress": 1.0,
            "status": "completed",
            "completed_at": completed_at,
            "diagnostics": diagnostics,
        })

    except Exception as exc:
        if active_sampler is not None:
            try:
                _end_stage(active_sampler, push_diagnostics=False)
            except Exception:
                pass
        append_log_line(job_log_path, f"job={job_id[:8]} {format_stage_label('failed')} (ERR_PIPELINE_FAILED)")
        log.exception("Error durante la segmentación del trabajo %s", job_id)
        status_queue.put({
            "stage": "failed",
            "status": "failed",
            "error_code": "ERR_PIPELINE_FAILED",
            "detail": str(exc),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "diagnostics": diagnostics,
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
        labels_img = nib.Nifti1Image(np.asarray(labels_data, dtype=np.uint8), affine)
        labels_img.set_data_dtype(np.uint8)
        labels_img.header.set_slope_inter(1.0, 0.0)
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
