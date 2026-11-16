from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import nibabel as nib
import numpy as np


def analyze_l3(mask_l3_data: np.ndarray, affine: np.ndarray) -> Dict[str, Any]:
    """
    Analiza la máscara de la vértebra L3 para determinar centroide, corte axial sugerido
    y estado ('found', 'edge', 'not_found').

    Criterio de estado 'edge' (Ajuste 4):
    - Criterio exclusivamente en milímetros (distancia <= 10.0 mm a cualquier borde Z)
      MÁS condición de máscara tocando el primer corte (Z=0) o el último corte (Z=depth-1).
    """
    if mask_l3_data is None or not np.any(mask_l3_data > 0):
        return {
            "status": "not_found",
            "slice_index": None,
            "slice_position_z_mm": None,
            "vertebra_centroid_voxel": None,
            "vertebra_centroid_world_mm": None,
            "z_range_slices": None,
            "distance_to_volume_edge_mm": None,
            "touches_first_or_last_slice": False,
        }

    indices = np.argwhere(mask_l3_data > 0)  # Shape (N, 3): [x, y, z]
    mean_voxel = np.mean(indices, axis=0)
    centroid_voxel = [float(mean_voxel[0]), float(mean_voxel[1]), float(mean_voxel[2])]

    # Coordenadas físicas en mm aplicando la matriz affine
    world_coord = nib.affines.apply_affine(affine, centroid_voxel)
    centroid_world_mm = [float(world_coord[0]), float(world_coord[1]), float(world_coord[2])]

    # Corte sugerido: entero Z más cercano al centroide
    slice_index = int(round(mean_voxel[2]))
    slice_pos_world = nib.affines.apply_affine(affine, [mean_voxel[0], mean_voxel[1], slice_index])
    slice_position_z_mm = float(slice_pos_world[2])

    min_z = int(np.min(indices[:, 2]))
    max_z = int(np.max(indices[:, 2]))
    depth = mask_l3_data.shape[2]

    # Verificar si toca el primer (0) o último (depth - 1) corte
    touches_first_or_last_slice = bool((min_z == 0) or (max_z == depth - 1))

    # Distancia física en mm al borde axial
    voxel_sizes = nib.affines.voxel_sizes(affine)
    spacing_z = float(abs(voxel_sizes[2])) if len(voxel_sizes) > 2 else 1.0

    dist_to_bottom_mm = float(min_z * spacing_z)
    dist_to_top_mm = float((depth - 1 - max_z) * spacing_z)
    distance_to_volume_edge_mm = float(min(dist_to_bottom_mm, dist_to_top_mm))

    # Criterio estricto: distancia física <= 10.0 mm o tocar los cortes extremos
    is_edge = touches_first_or_last_slice or (distance_to_volume_edge_mm <= 10.0)
    status = "edge" if is_edge else "found"

    return {
        "status": status,
        "slice_index": slice_index,
        "slice_position_z_mm": slice_position_z_mm,
        "vertebra_centroid_voxel": centroid_voxel,
        "vertebra_centroid_world_mm": centroid_world_mm,
        "z_range_slices": [min_z, max_z],
        "distance_to_volume_edge_mm": distance_to_volume_edge_mm,
        "touches_first_or_last_slice": touches_first_or_last_slice,
    }


def build_labels_volume(
    subcutaneous_data: Optional[np.ndarray],
    torso_data: Optional[np.ndarray],
    muscle_data: Optional[np.ndarray],
    intermuscular_data: Optional[np.ndarray],
    shape: Tuple[int, int, int],
) -> np.ndarray:
    """
    Consolida las 4 máscaras anatómicas en un único volumen uint8 crudo (sin rangos HU).
    Mapeo de etiquetas:
      1: subcutaneous_fat
      2: torso_fat
      3: skeletal_muscle
      4: intermuscular_fat
    """
    labels = np.zeros(shape, dtype=np.uint8)

    if subcutaneous_data is not None:
        labels[subcutaneous_data > 0] = 1
    if torso_data is not None:
        labels[torso_data > 0] = 2
    if muscle_data is not None:
        labels[muscle_data > 0] = 3
    if intermuscular_data is not None:
        labels[intermuscular_data > 0] = 4

    return labels


def build_manifest(
    job_id: str,
    created_at: str,
    started_at: Optional[str],
    completed_at: str,
    duration_seconds: float,
    totalsegmentator_version: str,
    fastapi_version: str,
    nibabel_version: str,
    numpy_version: str,
    tasks_executed: List[str],
    license_valid: bool,
    tissue_segmentation_skipped: bool,
    device: str,
    gpu_name: Optional[str],
    input_shape: Tuple[int, ...],
    affine: np.ndarray,
    l3_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Construye el contenido JSON de manifest.json declarando explícitamente el espacio de
    coordenadas, metadatos del entorno y la detección anatómica de L3.
    """
    voxel_sizes = nib.affines.voxel_sizes(affine)
    orientation = "".join(nib.aff2axcodes(affine))

    labels_legend = None
    if not tissue_segmentation_skipped:
        labels_legend = {
            "1": "subcutaneous_fat",
            "2": "torso_fat",
            "3": "skeletal_muscle",
            "4": "intermuscular_fat",
        }

    return {
        "contract_version": "2.0",
        "job_id": job_id,
        "timestamps": {
            "created_at": created_at,
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_seconds": round(duration_seconds, 2),
        },
        "environment": {
            "totalsegmentator_version": totalsegmentator_version,
            "fastapi_version": fastapi_version,
            "nibabel_version": nibabel_version,
            "numpy_version": numpy_version,
            "tasks_executed": tasks_executed,
            "license_valid": license_valid,
            "tissue_segmentation_skipped": tissue_segmentation_skipped,
            "device": device,
            "gpu_name": gpu_name,
        },
        "coordinate_space": {
            "dim": [int(d) for d in input_shape],
            "pixdim_mm": [float(round(s, 4)) for s in voxel_sizes],
            "slice_axis": 2,  # Eje Z axial
            "affine": [[float(round(val, 6)) for val in row] for row in affine.tolist()],
            "orientation": orientation,
        },
        "l3": l3_analysis,
        "labels_legend": labels_legend,
    }
