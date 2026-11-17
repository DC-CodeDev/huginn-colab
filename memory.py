from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

# Coeficiente k provisional (múltiplo sobre el tamaño float32 del volumen remuestreado a 1,5 mm)
from colab.session_log import append_log_line

# Valor inicial provisional: a calibrar con mediciones reales de [mem-summary] en Colab.
DEFAULT_MEMORY_ESTIMATION_K = 100.0
TARGET_RESAMPLE_SPACING_MM = 1.5


def get_memory_estimation_k() -> float:
    """Obtiene el coeficiente k configurable para estimar RAM requerida."""
    raw = os.environ.get("HUGINN_MEMORY_ESTIMATION_K")
    if raw is not None:
        try:
            val = float(raw)
            if val > 0:
                return val
        except ValueError:
            pass
    return DEFAULT_MEMORY_ESTIMATION_K


def _bytes_to_mb(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return round(value / (1024.0 * 1024.0), 1)


def get_system_memory_bytes() -> Tuple[Optional[int], Optional[int]]:
    """Retorna (total_bytes, available_bytes) de RAM del sistema usando psutil o /proc/meminfo."""
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except Exception:
        pass

    try:
        total_kb: Optional[int] = None
        avail_kb: Optional[int] = None
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
                if total_kb is not None and avail_kb is not None:
                    break
        if total_kb is not None and avail_kb is not None:
            return total_kb * 1024, avail_kb * 1024
    except Exception:
        pass

    return None, None


def get_process_rss_bytes(pid: Optional[int] = None) -> Optional[int]:
    """Retorna RSS del proceso en bytes usando psutil o /proc/self/status."""
    try:
        import psutil  # type: ignore

        proc = psutil.Process(pid or os.getpid())
        return int(proc.memory_info().rss)
    except Exception:
        pass

    if pid is None or pid == os.getpid():
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            pass

    return None


def get_gpu_memory_bytes() -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Retorna (vram_used_bytes, vram_total_bytes, max_memory_allocated_bytes)
    si CUDA está disponible, o (None, None, None) en caso contrario.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None, None, None
        free_b, total_b = torch.cuda.mem_get_info()
        used_b = int(total_b) - int(free_b)
        max_alloc_b = int(torch.cuda.max_memory_allocated())
        return used_b, int(total_b), max_alloc_b
    except Exception:
        return None, None, None


def estimate_required_memory_bytes(
    shape: Tuple[int, ...],
    k: Optional[float] = None,
    spacing: Optional[Tuple[float, ...]] = None,
    target_spacing_mm: float = TARGET_RESAMPLE_SPACING_MM,
) -> int:
    """
    Estima la RAM de CPU necesaria para segmentar un volumen 3D a partir del volumen
    remuestreado a 1,5 mm por eje: (forma * espaciado_header / 1,5) por eje.
    Base: resampled_voxels * 4 bytes (float32) * k.
    Si no se provee espaciado válido, usa las dimensiones brutas como fallback.
    """
    if len(shape) < 3:
        return 0
    coeff = k if (k is not None and k > 0) else get_memory_estimation_k()
    dims = []
    for i in range(3):
        dim = float(shape[i])
        if (
            spacing is not None
            and len(spacing) > i
            and spacing[i] is not None
            and float(spacing[i]) > 0
            and target_spacing_mm > 0
        ):
            dim = dim * (float(spacing[i]) / float(target_spacing_mm))
        dims.append(max(0.0, dim))
    resampled_voxels = dims[0] * dims[1] * dims[2]
    return int(resampled_voxels * 4 * coeff)


class StageMemorySampler:
    """
    Hilo muestreador de memoria por etapa (~1 s) para registrar RSS, RAM del sistema
    y VRAM de GPU en la salida del notebook sin exponer datos de paciente.
    """

    def __init__(
        self,
        job_id: str,
        stage: str,
        interval_seconds: float = 1.0,
        log_path: Optional[str] = None,
    ) -> None:
        self.job_id = job_id
        self.stage = stage
        self.interval_seconds = interval_seconds
        self.log_path = log_path
        self.last_summary_line: Optional[str] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0
        self._lock = threading.Lock()
        self._samples_count = 0
        self._peak_rss_b: Optional[int] = None
        self._peak_sys_used_b: Optional[int] = None
        self._min_sys_avail_b: Optional[int] = None
        self._peak_vram_used_b: Optional[int] = None
        self._peak_vram_alloc_b: Optional[int] = None

    def _sample_once(self) -> None:
        rss_b = get_process_rss_bytes()
        sys_total_b, sys_avail_b = get_system_memory_bytes()
        sys_used_b = (
            (sys_total_b - sys_avail_b)
            if (sys_total_b is not None and sys_avail_b is not None)
            else None
        )
        vram_used_b, vram_total_b, vram_max_alloc_b = get_gpu_memory_bytes()

        with self._lock:
            self._samples_count += 1
            if rss_b is not None:
                self._peak_rss_b = rss_b if self._peak_rss_b is None else max(self._peak_rss_b, rss_b)
            if sys_used_b is not None:
                self._peak_sys_used_b = (
                    sys_used_b if self._peak_sys_used_b is None else max(self._peak_sys_used_b, sys_used_b)
                )
            if sys_avail_b is not None:
                self._min_sys_avail_b = (
                    sys_avail_b if self._min_sys_avail_b is None else min(self._min_sys_avail_b, sys_avail_b)
                )
            if vram_used_b is not None:
                self._peak_vram_used_b = (
                    vram_used_b if self._peak_vram_used_b is None else max(self._peak_vram_used_b, vram_used_b)
                )
            if vram_max_alloc_b is not None:
                self._peak_vram_alloc_b = (
                    vram_max_alloc_b
                    if self._peak_vram_alloc_b is None
                    else max(self._peak_vram_alloc_b, vram_max_alloc_b)
                )

        elapsed = round(time.time() - self._t0, 1)
        parts = [
            f"[mem] job={self.job_id[:8]}",
            f"stage={self.stage}",
            f"t={elapsed}s",
            f"rss={_bytes_to_mb(rss_b)}MB",
            f"sys_used={_bytes_to_mb(sys_used_b)}MB",
            f"sys_avail={_bytes_to_mb(sys_avail_b)}MB",
        ]
        if vram_used_b is not None and vram_total_b is not None:
            parts.append(f"vram_used={_bytes_to_mb(vram_used_b)}/{_bytes_to_mb(vram_total_b)}MB")
            parts.append(f"vram_max_alloc={_bytes_to_mb(vram_max_alloc_b)}MB")
        line = " ".join(parts)
        print(line, flush=True)
        if self.log_path:
            append_log_line(self.log_path, line)

    def _run(self) -> None:
        self._sample_once()
        while not self._stop_event.wait(self.interval_seconds):
            self._sample_once()

    def start(self) -> "StageMemorySampler":
        self._t0 = time.time()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"mem-sampler-{self.stage}",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> Dict[str, Any]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        # Muestra final para capturar el estado al cierre de la etapa
        self._sample_once()
        duration = round(time.time() - self._t0, 2)

        with self._lock:
            summary: Dict[str, Any] = {
                "duration_seconds": duration,
                "samples": self._samples_count,
                "peak_rss_mb": _bytes_to_mb(self._peak_rss_b),
                "peak_sys_used_mb": _bytes_to_mb(self._peak_sys_used_b),
                "min_sys_avail_mb": _bytes_to_mb(self._min_sys_avail_b),
                "peak_vram_used_mb": _bytes_to_mb(self._peak_vram_used_b),
                "peak_vram_allocated_mb": _bytes_to_mb(self._peak_vram_alloc_b),
            }

        summary_line = (
            f"[mem-summary] job={self.job_id[:8]} stage={self.stage} "
            f"duration={duration}s "
            f"peak_rss={summary['peak_rss_mb']}MB "
            f"peak_vram_used={summary['peak_vram_used_mb']}MB "
            f"peak_vram_alloc={summary['peak_vram_allocated_mb']}MB"
        )
        self.last_summary_line = summary_line
        print(summary_line, flush=True)
        if self.log_path:
            append_log_line(self.log_path, summary_line)
        return summary
