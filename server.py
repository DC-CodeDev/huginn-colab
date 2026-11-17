from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing as mp
import os
import secrets
import shutil
import socket
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import fastapi
from fastapi import FastAPI, File, Header, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import nibabel as nib

from colab.connection_code import build_connect_link, encode
from colab.memory import (
    DEFAULT_MEMORY_BASE_MB,
    DEFAULT_MEMORY_ESTIMATION_K,
    estimate_required_memory_bytes,
    get_memory_base_mb,
    get_memory_estimation_k,
    get_system_memory_bytes,
)
from colab.pipeline import (
    FASTAPI_VERSION,
    NIBABEL_VERSION,
    NUMPY_VERSION,
    TOTALSEGMENTATOR_VERSION,
    run_segmentation_worker,
)
from colab.schemas import (
    ErrorResponse,
    HealthResponse,
    JobCreatedResponse,
    JobStatusResponse,
    PurgeResponse,
)
from colab.session_log import (
    append_log_line,
    format_huginn_line,
    format_stage_label,
    get_session_log_path,
    monitor_session_logs,
)
from colab.version import (
    SERVICE_VERSION,
    check_code_staleness,
    format_version_banner,
)

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("huginn.server")

# ---------------------------------------------------------------------------
# Configuración del Entorno y Secretos
# ---------------------------------------------------------------------------
def _get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Obtiene un secreto desde google.colab.userdata si existe, o del entorno."""
    try:
        from google.colab import userdata  # type: ignore

        val = userdata.get(key)
        if val:
            return str(val)
    except Exception:
        pass
    return os.environ.get(key, default)


# Puerto local
PORT = int(os.environ.get("HUGINN_PORT", "8000"))

# Token de autenticación obligatorio
_configured_token = _get_secret("HUGINN_API_TOKEN")
if not _configured_token:
    API_TOKEN = secrets.token_urlsafe(24)
    print("AVISO: HUGINN_API_TOKEN no configurado en Secretos ni en entorno.")
    print("Se generó un token temporal para esta sesión (se incluirá únicamente en el código de conexión).")
else:
    API_TOKEN = _configured_token
    log.info("HUGINN_API_TOKEN cargado exitosamente desde configuración/secretos.")

# Licencia académica de TotalSegmentator
def setup_license(license_str: Optional[str]) -> bool:
    """Configura la licencia de TotalSegmentator sin exponer la clave en logs ni excepciones."""
    if not license_str:
        log.info("HUGINN_TOTALSEG_LICENSE no configurado. Iniciando en modo degradado (solo L3).")
        return False
    try:
        import subprocess

        res = subprocess.run(
            ["totalseg_set_license", "-l", license_str],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            log.info("Licencia académica de TotalSegmentator activada correctamente.")
            return True
        log.warning("totalseg_set_license falló (código de salida: %s).", res.returncode)
        return False
    except Exception:
        log.warning("totalseg_set_license falló al ejecutarse.")
        return False

TOTALSEG_LICENSE = _get_secret("HUGINN_TOTALSEG_LICENSE")
LICENSE_VALID = setup_license(TOTALSEG_LICENSE)

# Origen permitido para CORS (Ajuste 7)
ALLOWED_ORIGIN = os.environ.get("HUGINN_ALLOWED_ORIGIN", "*")
if ALLOWED_ORIGIN == "*":
    print("WARNING: HUGINN_ALLOWED_ORIGIN is set to '*'. For production, restrict this to your PWA origin.")

# Tamaño máximo de subida (Ajuste 6: 500 MB por defecto)
MAX_UPLOAD_SIZE_BYTES = int(os.environ.get("HUGINN_MAX_UPLOAD_SIZE", str(500 * 1024 * 1024)))

# Constantes provisionales para prechequeo de RAM (calibrados con 1 punto, revisar)
MEMORY_BASE_MB = get_memory_base_mb()
MEMORY_ESTIMATION_K = get_memory_estimation_k()


class ActiveJobRestartError(RuntimeError):
    """Error interno cuando se rechaza el reinicio por haber un job en procesamiento."""

# Directorio base de trabajos
JOBS_BASE_DIR = os.environ.get("HUGINN_JOBS_DIR", "/tmp/huginn/jobs")
os.makedirs(JOBS_BASE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Estado en Memoria
# ---------------------------------------------------------------------------
class JobRecord:
    def __init__(self, job_id: str, job_dir: str, created_at: str):
        self.job_id = job_id
        self.job_dir = job_dir
        self.status = "queued"
        self.stage = "queued"
        self.progress = 0.0
        self.error_code: Optional[str] = None
        self.detail: Optional[str] = None
        self.created_at = created_at
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.diagnostics: Optional[Dict[str, Any]] = None
        self.process: Optional[mp.Process] = None
        self.status_queue: Optional[Any] = None


jobs_db: Dict[str, JobRecord] = {}
job_queue: asyncio.Queue[str] = asyncio.Queue()
is_ready: bool = False
server_start_time: float = time.time()


# ---------------------------------------------------------------------------
# Funciones de Soporte y Cola de Trabajos
# ---------------------------------------------------------------------------
def _verify_token(authorization: Optional[str]) -> None:
    """Verifica el token Bearer usando secrets.compare_digest sin exponer el token."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization scheme. Expected 'Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = parts[1].strip()
    if not secrets.compare_digest(token, API_TOKEN):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _active_secrets() -> tuple[str, ...]:
    """Secretos activos a enmascarar en cualquier salida de log de sesión."""
    candidates = [
        API_TOKEN,
        TOTALSEG_LICENSE or "",
        os.environ.get("NGROK_AUTHTOKEN", ""),
        os.environ.get("GITHUB_TOKEN", ""),
    ]
    return tuple(c for c in candidates if c)


def _drain_worker_status_queue(
    job: JobRecord,
    status_queue: Any,
    last_stage_ref: list[Optional[str]],
    session_log_path: str,
) -> None:
    """Lee mensajes del subproceso worker, actualiza JobRecord y reenvía eventos al log de sesión."""
    secrets_mask = _active_secrets()
    short_id = job.job_id[:8]
    while not status_queue.empty():
        msg = status_queue.get_nowait()
        if "stage" in msg:
            job.stage = msg["stage"]
        if "progress" in msg:
            job.progress = msg["progress"]
        if "status" in msg:
            job.status = msg["status"]
        if "error_code" in msg:
            job.error_code = msg["error_code"]
        if "detail" in msg:
            job.detail = msg["detail"]
        if "completed_at" in msg:
            job.completed_at = msg["completed_at"]
        if "diagnostics" in msg:
            job.diagnostics = msg["diagnostics"]

        if "stage" in msg and msg["stage"] != last_stage_ref[0]:
            last_stage_ref[0] = msg["stage"]
            label = format_stage_label(msg["stage"])
            if msg["stage"] == "failed" and msg.get("error_code"):
                label = f"{label} ({msg['error_code']})"
            append_log_line(
                session_log_path,
                format_huginn_line(f"job={short_id} {label}", extra_secrets=secrets_mask),
                extra_secrets=secrets_mask,
            )

        if "mem_summary" in msg and msg["mem_summary"]:
            append_log_line(
                session_log_path,
                format_huginn_line(msg["mem_summary"], extra_secrets=secrets_mask),
                extra_secrets=secrets_mask,
            )

        if "log_line" in msg and msg["log_line"]:
            append_log_line(
                session_log_path,
                format_huginn_line(msg["log_line"], extra_secrets=secrets_mask),
                extra_secrets=secrets_mask,
            )


async def single_worker_loop() -> None:
    """Worker secuencial: procesa 1 trabajo a la vez para no sobrecargar la GPU."""
    spawn_ctx = mp.get_context("spawn")

    while True:
        job_id = await job_queue.get()
        job = jobs_db.get(job_id)

        if not job or job.status != "queued":
            job_queue.task_done()
            continue

        try:
            job.status = "processing"
            job.stage = "validating"
            job.started_at = datetime.now(timezone.utc).isoformat()
            input_path = os.path.join(job.job_dir, "input.nii.gz")

            session_log_path = get_session_log_path()
            secrets_mask = _active_secrets()
            append_log_line(
                session_log_path,
                format_huginn_line(f"job={job.job_id[:8]} iniciado"),
                extra_secrets=secrets_mask,
            )
            stale_msg = check_code_staleness()
            if stale_msg:
                log.warning(stale_msg)
                append_log_line(
                    session_log_path,
                    format_huginn_line(f"AVISO: {stale_msg}"),
                    extra_secrets=secrets_mask,
                )

            status_queue = spawn_ctx.Queue()
            job.status_queue = status_queue
            last_stage_ref: list[Optional[str]] = [None]

            device = "gpu"
            try:
                import torch

                if not torch.cuda.is_available():
                    device = "cpu"
            except Exception:
                device = "cpu"

            proc = spawn_ctx.Process(
                target=run_segmentation_worker,
                kwargs={
                    "job_id": job.job_id,
                    "input_path": input_path,
                    "output_dir": job.job_dir,
                    "license_valid": LICENSE_VALID,
                    "status_queue": status_queue,
                    "created_at": job.created_at,
                    "started_at": job.started_at,
                    "device": device,
                },
            )
            job.process = proc
            proc.start()

            # Monitoreo asíncrono sin bloquear el event loop (Ajuste 2 y 5)
            # Timeout de 15 minutos (900 s) desde started_at
            timeout_seconds = 900.0
            start_proc_time = time.time()

            while proc.is_alive():
                # Leer actualizaciones del proceso hijo si las hay
                _drain_worker_status_queue(job, status_queue, last_stage_ref, session_log_path)

                # Verificar timeout de 15 minutos
                if time.time() - start_proc_time > timeout_seconds:
                    log.error("Timeout de procesamiento de 15 minutos excedido para %s", job_id)
                    proc.terminate()
                    await asyncio.sleep(0.5)
                    if proc.is_alive():
                        proc.kill()
                    job.status = "failed"
                    job.error_code = "ERR_PROCESSING_TIMEOUT"
                    job.detail = "El trabajo excedió el tiempo límite de procesamiento de 15 minutos."
                    job.completed_at = datetime.now(timezone.utc).isoformat()
                    append_log_line(
                        session_log_path,
                        format_huginn_line(f"job={job_id[:8]} {format_stage_label('failed')} (ERR_PROCESSING_TIMEOUT)"),
                        extra_secrets=secrets_mask,
                    )
                    break

                await asyncio.sleep(0.5)

            # Esperar a que el subproceso termine de cerrar
            proc.join(timeout=2.0)

            # Vaciar mensajes finales de la cola
            _drain_worker_status_queue(job, status_queue, last_stage_ref, session_log_path)

            if job.status == "processing":
                # Si el proceso murió sin mensaje de éxito
                if proc.exitcode != 0 and job.status != "failed":
                    job.status = "failed"
                    job.error_code = "ERR_SUBPROCESS_CRASHED"
                    job.detail = f"El subproceso de segmentación terminó inesperadamente con código {proc.exitcode}."
                    job.completed_at = datetime.now(timezone.utc).isoformat()
                elif os.path.exists(os.path.join(job.job_dir, "result.zip")):
                    job.status = "completed"
                    job.stage = "completed"
                    job.progress = 1.0
                    if not job.completed_at:
                        job.completed_at = datetime.now(timezone.utc).isoformat()

        except Exception as exc:
            log.exception("Falla inesperada en worker para el trabajo %s", job_id)
            job.status = "failed"
            job.error_code = "ERR_INTERNAL"
            job.detail = str(exc)
            job.completed_at = datetime.now(timezone.utc).isoformat()
        finally:
            job_queue.task_done()


async def cleanup_loop() -> None:
    """
    Tarea periódica de recolección de basura (Ajuste 5):
    - Trabajos en cola o procesando: expiran a los 30 minutos desde created_at.
    - Trabajos terminados (completed/failed): expiran a los 30 minutos desde completed_at.
    """
    while True:
        await asyncio.sleep(30.0)
        now = datetime.now(timezone.utc)

        to_purge = []
        for job_id, job in list(jobs_db.items()):
            try:
                if job.status in ("queued", "processing"):
                    created_dt = datetime.fromisoformat(job.created_at)
                    if (now - created_dt).total_seconds() > 1800.0:  # 30 minutos
                        log.info("Expirando trabajo abandonado %s (30m desde created_at)", job_id)
                        to_purge.append(job_id)
                elif job.status in ("completed", "failed") and job.completed_at:
                    completed_dt = datetime.fromisoformat(job.completed_at)
                    if (now - completed_dt).total_seconds() > 1800.0:  # 30 minutos
                        log.info("Purgando trabajo completado/fallido %s (30m desde completed_at)", job_id)
                        to_purge.append(job_id)
            except Exception as err:
                log.warning("Error calculando expiración para %s: %s", job_id, err)

        for jid in to_purge:
            _purge_job_internal(jid)


def _purge_job_internal(job_id: str) -> None:
    """Mata el proceso si está vivo y borra archivos del trabajo."""
    job = jobs_db.pop(job_id, None)
    if not job:
        return

    # Ajuste 3: matar subproceso sin tocar torch.cuda.empty_cache() en proceso principal
    if job.process and job.process.is_alive():
        log.info("Terminando subproceso para trabajo purgado %s", job_id)
        job.process.terminate()
        job.process.join(timeout=3.0)
        if job.process.is_alive():
            job.process.kill()
            job.process.join(timeout=1.0)

    if os.path.exists(job.job_dir):
        shutil.rmtree(job.job_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Aplicación FastAPI y Ciclo de Vida
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global is_ready, job_queue
    job_queue = asyncio.Queue()
    # Iniciar workers de cola y limpieza
    worker_task = asyncio.create_task(single_worker_loop())
    cleaner_task = asyncio.create_task(cleanup_loop())

    # Precalentamiento simulado / verificación
    is_ready = True
    log.info("Servidor Huginn listo para recibir peticiones.")

    yield

    worker_task.cancel()
    cleaner_task.cancel()


app = FastAPI(
    title="Huginn TotalSegmentator Colab Service",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN] if ALLOWED_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "ngrok-skip-browser-warning"],
)


# ---------------------------------------------------------------------------
# Manejador de Excepciones Estructuradas
# ---------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # Si detail ya es un dict estructurado, se devuelve intacto
    if isinstance(exc.detail, dict) and "error_code" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)

    # Si es 401
    if exc.status_code == status.HTTP_401_UNAUTHORIZED:
        error_code = "ERR_UNAUTHORIZED"
    elif exc.status_code == status.HTTP_404_NOT_FOUND:
        error_code = "ERR_NOT_FOUND"
    elif exc.status_code == 413:
        error_code = "ERR_FILE_TOO_LARGE"
    elif exc.status_code == 422:
        error_code = "ERR_UNPROCESSABLE_ENTITY"
    else:
        error_code = f"ERR_HTTP_{exc.status_code}"

    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": str(exc.detail), "error_code": error_code},
        headers=getattr(exc, "headers", None),
    )


# ---------------------------------------------------------------------------
# Endpoints del Contrato
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health_endpoint(authorization: Optional[str] = Header(None)):
    """Verifica estado, versión del contrato y compatibilidad."""
    _verify_token(authorization)

    gpu_available = False
    gpu_name = None
    try:
        import torch

        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    tasks_available = ["vertebrae_L3"]
    if LICENSE_VALID:
        tasks_available.append("tissue_4_types")

    ram_total_b, ram_avail_b = get_system_memory_bytes()
    ram_total_mb = round(ram_total_b / (1024.0 * 1024.0), 1) if ram_total_b is not None else None
    ram_available_mb = round(ram_avail_b / (1024.0 * 1024.0), 1) if ram_avail_b is not None else None

    return HealthResponse(
        status="healthy",
        ready=is_ready,
        contract_version="2.0",
        totalsegmentator_version=TOTALSEGMENTATOR_VERSION,
        license_valid=LICENSE_VALID,
        tasks_available=tasks_available,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        queue_len=job_queue.qsize(),
        uptime_seconds=round(time.time() - server_start_time, 2),
        ram_total_mb=ram_total_mb,
        ram_available_mb=ram_available_mb,
    )


@app.post("/jobs", response_model=JobCreatedResponse, status_code=status.HTTP_201_CREATED)
async def create_job_endpoint(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None),
):
    """
    Crea un nuevo trabajo de segmentación.
    Valida tamaño máximo (Ajuste 6: 500 MB) y que sea un volumen NIfTI 3D legible.
    """
    _verify_token(authorization)

    job_id = str(uuid.uuid4())
    job_dir = os.path.join(JOBS_BASE_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    temp_nii_path = os.path.join(job_dir, "input.nii.gz")

    bytes_read = 0
    try:
        with open(temp_nii_path, "wb") as f_out:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                bytes_read += len(chunk)
                if bytes_read > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail={
                            "detail": f"El archivo excede el tamaño máximo permitido de {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB.",
                            "error_code": "ERR_FILE_TOO_LARGE",
                        },
                    )
                f_out.write(chunk)
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"detail": f"Error guardando el archivo: {exc}", "error_code": "ERR_UPLOAD_FAILED"},
        )

    # Validar que sea un archivo NIfTI legible
    try:
        nii = nib.load(temp_nii_path)
        shape = nii.shape
        zooms = tuple(float(z) for z in nii.header.get_zooms()[:3])
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "detail": f"El archivo enviado no es un NIfTI válido o legible: {exc}",
                "error_code": "ERR_INVALID_NIFTI",
            },
        )

    # Validar que sea un volumen 3D
    if len(shape) != 3:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(
            status_code=422,
            detail={
                "detail": f"Se esperaba un volumen NIfTI 3D, pero se recibió dimensión {shape}.",
                "error_code": "ERR_INVALID_DIMENSIONS",
            },
        )

    # Prechequeo de RAM estimada (BASE_MB + k * vóxeles remuestreados a 1,5 mm) vs RAM disponible
    estimated_bytes = estimate_required_memory_bytes(
        shape,
        k=MEMORY_ESTIMATION_K,
        spacing=zooms,
        base_mb=MEMORY_BASE_MB,
    )
    _, avail_bytes = get_system_memory_bytes()
    if avail_bytes is not None and estimated_bytes > avail_bytes:
        est_mb = round(estimated_bytes / (1024.0 * 1024.0), 1)
        avail_mb = round(avail_bytes / (1024.0 * 1024.0), 1)
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(
            status_code=422,
            detail={
                "detail": (
                    f"Memoria insuficiente para procesar el volumen {tuple(shape)}: "
                    f"se estiman ~{est_mb} MB y hay ~{avail_mb} MB disponibles. "
                    "Usá un runtime con más RAM (p. ej. High-RAM en Colab) o un estudio con menos cortes."
                ),
                "error_code": "ERR_INSUFFICIENT_MEMORY",
                "stage": "validating",
            },
        )

    created_at = datetime.now(timezone.utc).isoformat()
    record = JobRecord(job_id=job_id, job_dir=job_dir, created_at=created_at)
    jobs_db[job_id] = record
    await job_queue.put(job_id)

    log.info("Trabajo %s encolado exitosamente (%d bytes).", job_id, bytes_read)

    return JobCreatedResponse(
        job_id=job_id,
        status="queued",
        created_at=created_at,
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status_endpoint(job_id: str, authorization: Optional[str] = Header(None)):
    """Consulta el estado del trabajo."""
    _verify_token(authorization)

    job = jobs_db.get(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"detail": f"Trabajo {job_id} no encontrado.", "error_code": "ERR_JOB_NOT_FOUND"},
        )

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        stage=job.stage,
        progress=job.progress,
        error_code=job.error_code,
        detail=job.detail,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        diagnostics=job.diagnostics,
    )


@app.get("/jobs/{job_id}/result")
async def get_job_result_endpoint(job_id: str, authorization: Optional[str] = Header(None)):
    """Descarga el paquete result.zip."""
    _verify_token(authorization)

    job = jobs_db.get(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"detail": f"Trabajo {job_id} no encontrado.", "error_code": "ERR_JOB_NOT_FOUND"},
        )

    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "detail": f"El trabajo {job_id} no está terminado (estado actual: {job.status}).",
                "error_code": "ERR_JOB_NOT_READY",
            },
        )

    result_zip = os.path.join(job.job_dir, "result.zip")
    if not os.path.exists(result_zip):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "detail": "El archivo result.zip no se encuentra en el servidor.",
                "error_code": "ERR_RESULT_FILE_MISSING",
            },
        )

    return FileResponse(
        path=result_zip,
        media_type="application/zip",
        filename=f"huginn_{job_id}_result.zip",
    )


@app.delete("/jobs/{job_id}", response_model=PurgeResponse)
async def delete_job_endpoint(job_id: str, authorization: Optional[str] = Header(None)):
    """
    Purga los datos de un trabajo o cancela un trabajo en curso matando su subproceso.
    """
    _verify_token(authorization)

    if job_id not in jobs_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"detail": f"Trabajo {job_id} no encontrado.", "error_code": "ERR_JOB_NOT_FOUND"},
        )

    _purge_job_internal(job_id)
    log.info("Trabajo %s purgado/cancelado.", job_id)

    return PurgeResponse(
        detail=f"Trabajo {job_id} purgado y recursos liberados.",
        job_id=job_id,
    )


# ---------------------------------------------------------------------------
# Lanzador de Servicio con Ngrok para Colab
# ---------------------------------------------------------------------------
# Entrada Colab
# ---------------------------------------------------------------------------
@dataclass
class _ServerRuntimeState:
    """Registro a nivel de módulo del servidor Uvicorn y túnel ngrok activos."""

    server: Optional[Any] = None
    thread: Optional[threading.Thread] = None
    tunnel: Optional[Any] = None
    public_url: Optional[str] = None
    port: Optional[int] = None
    commit_hash: Optional[str] = None
    startup_error: Optional[BaseException] = None


_runtime_lock = threading.Lock()
_runtime_state = _ServerRuntimeState()

STALE_RESTART_BANNER = (
    "Hay una versión más nueva del servicio. "
    "Para cargarla: Entorno de ejecución > Reiniciar sesión y ejecutar todo"
)


def format_ngrok_url(domain: str) -> str:
    """
    Construye la URL completa con https:// a partir de NGROK_DOMAIN,
    sin duplicar el esquema si el usuario ya lo incluyó.
    """
    cleaned = domain.strip()
    if cleaned.startswith("https://") or cleaned.startswith("http://"):
        return cleaned.rstrip("/")
    return f"https://{cleaned.rstrip('/')}"


def setup_ngrok_auth(token: str) -> None:
    """
    Configura el authtoken de ngrok de forma idempotente.
    Tolera el caso en que el token ya esté configurado en el archivo de configuración.
    Garantiza que el token nunca aparezca en logs ni en mensajes de error.
    """
    from pyngrok import conf, ngrok
    from pyngrok.exception import PyngrokError

    token_clean = token.strip()
    try:
        ngrok.set_auth_token(token_clean)
    except (PyngrokError, Exception) as exc:
        err_msg = str(exc)
        if "Authtoken already set" in err_msg:
            log.info("Authtoken de ngrok ya configurado en el archivo de configuración.")
        else:
            safe_err = err_msg.replace(token_clean, "***").replace(token, "***")
            log.error("Error al configurar authtoken de ngrok: %s", safe_err)
            raise RuntimeError(f"Error al configurar authtoken de ngrok: {safe_err}") from None

    # Sincronizar en la configuración en memoria de pyngrok
    conf.get_default().auth_token = token_clean


def _sanitize_runtime_message(message: str) -> str:
    """Enmascara secretos activos y códigos de conexión en mensajes impresos o errores."""
    from colab.session_log import sanitize_log_text

    return sanitize_log_text(message, extra_secrets=_active_secrets())


def _find_active_processing_jobs() -> List[JobRecord]:
    """Devuelve los trabajos que están actualmente en procesamiento o con subproceso vivo."""
    active: List[JobRecord] = []
    for job in list(jobs_db.values()):
        proc_alive = False
        if job.process is not None:
            try:
                proc_alive = bool(job.process.is_alive())
            except Exception:
                proc_alive = False
        if job.status == "processing" or proc_alive:
            active.append(job)
    return active


def _probe_health_url(url: str, timeout: float = 3.0) -> Tuple[bool, str]:
    """Consulta GET /health con Bearer token y devuelve (ok, detalle)."""
    req = urllib.request.Request(
        f"{url.rstrip('/')}/health",
        headers={
            "Authorization": f"Bearer {API_TOKEN}",
            "ngrok-skip-browser-warning": "true",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status_code = getattr(resp, "status", resp.getcode())
            body = resp.read().decode("utf-8", errors="replace")
            if status_code != 200:
                return False, f"HTTP {status_code}"
            data = json.loads(body)
            if data.get("status") != "healthy":
                return False, f"estado inesperado: {data.get('status')}"
            return True, "ok"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return False, _sanitize_runtime_message(str(exc.reason or exc))
    except Exception as exc:
        return False, _sanitize_runtime_message(str(exc))


def _is_ngrok_tunnel_active(public_url: Optional[str]) -> bool:
    """Verifica si el túnel ngrok registrado sigue activo en pyngrok."""
    if not public_url:
        return False
    try:
        from pyngrok import ngrok

        tunnels = ngrok.get_tunnels()
        if isinstance(tunnels, (list, tuple)):
            return any(str(getattr(t, "public_url", "")) == public_url for t in tunnels)
        return True
    except Exception:
        return False


def _is_own_server_healthy() -> bool:
    """Devuelve True si el servidor propio de esta sesión está vivo, sano y con túnel activo."""
    srv = _runtime_state.server
    thr = _runtime_state.thread
    if srv is None or thr is None or not thr.is_alive():
        return False
    if not getattr(srv, "started", False) or getattr(srv, "should_exit", False):
        return False
    port = _runtime_state.port or PORT
    local_ok, _ = _probe_health_url(f"http://127.0.0.1:{port}", timeout=3.0)
    if not local_ok:
        return False
    return _is_ngrok_tunnel_active(_runtime_state.public_url)


def _wait_for_port_free(port: int, timeout: float = 5.0) -> bool:
    """Espera brevemente hasta que el puerto local deje de aceptar conexiones."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return True
        time.sleep(0.1)
    return False


def _shutdown_existing_service(timeout: float = 5.0) -> None:
    """Cierra de forma limpia el uvicorn propio, túneles ngrok, agente ngrok y espera el puerto."""
    srv = _runtime_state.server
    thr = _runtime_state.thread
    pub_url = _runtime_state.public_url
    port = _runtime_state.port or PORT

    if srv is not None:
        srv.should_exit = True
    if thr is not None and thr.is_alive():
        thr.join(timeout=timeout)
        if thr.is_alive() and srv is not None:
            srv.force_exit = True
            thr.join(timeout=2.0)

    try:
        from pyngrok import ngrok

        if pub_url:
            try:
                ngrok.disconnect(pub_url)
            except Exception:
                pass
        try:
            tunnels = ngrok.get_tunnels()
            if isinstance(tunnels, (list, tuple)):
                for t in tunnels:
                    t_url = getattr(t, "public_url", None)
                    if t_url:
                        try:
                            ngrok.disconnect(t_url)
                        except Exception:
                            pass
        except Exception:
            pass
        try:
            ngrok.kill()
        except Exception:
            pass
    except Exception:
        pass

    if srv is not None or (thr is not None):
        _wait_for_port_free(port, timeout=timeout)

    _runtime_state.server = None
    _runtime_state.thread = None
    _runtime_state.tunnel = None
    _runtime_state.public_url = None
    _runtime_state.port = None
    _runtime_state.commit_hash = None
    _runtime_state.startup_error = None


def _is_ngrok_endpoint_online_error(exc: BaseException) -> bool:
    """Detecta ERR_NGROK_334 (dominio/endpoint fijo ya en línea en otra sesión)."""
    msg = str(exc)
    msg_lower = msg.lower()
    return "ERR_NGROK_334" in msg or ("already online" in msg_lower and "endpoint" in msg_lower)


def _format_ngrok_error_message(exc: BaseException, has_fixed_domain: bool) -> str:
    """Traduce errores de ngrok a mensajes accionables sin exponer credenciales."""
    safe_err = _sanitize_runtime_message(str(exc))
    if _is_ngrok_endpoint_online_error(exc):
        return (
            "El dominio fijo de ngrok sigue ocupado por una sesión anterior (ERR_NGROK_334) "
            "y no se liberó dentro del tiempo de espera. "
            "Qué hacer: 1) Cerrá el agente anterior en https://dashboard.ngrok.com/agents "
            "o esperá unos minutos a que expire la sesión previa; "
            "2) O dejá NGROK_DOMAIN vacío en la Celda 1 para usar una URL aleatoria."
        )
    if has_fixed_domain:
        return f"No se pudo abrir el túnel ngrok con dominio fijo: {safe_err}"
    return f"No se pudo abrir el túnel ngrok: {safe_err}"


def _connect_ngrok_with_retry(
    port: int,
    fixed_url: Optional[str],
    retry_interval: float = 5.0,
    retry_timeout: float = 120.0,
) -> Any:
    """Abre el túnel ngrok; ante ERR_NGROK_334 reintenta con espera acotada."""
    from pyngrok import ngrok

    deadline = time.monotonic() + max(0.0, retry_timeout)
    start_ts = time.monotonic()

    while True:
        try:
            if fixed_url:
                return ngrok.connect(port, url=fixed_url)
            return ngrok.connect(port)
        except Exception as exc:
            if fixed_url and _is_ngrok_endpoint_online_error(exc):
                now = time.monotonic()
                remaining = deadline - now
                if remaining > 0:
                    wait_s = min(retry_interval, remaining)
                    elapsed = now - start_ts
                    print(
                        f"[huginn] Dominio fijo de ngrok ocupado por una sesión anterior (ERR_NGROK_334). "
                        f"Reintentando en {wait_s:.0f}s ({elapsed:.0f}s / {retry_timeout:.0f}s)...",
                        flush=True,
                    )
                    time.sleep(wait_s)
                    continue
            raise RuntimeError(_format_ngrok_error_message(exc, bool(fixed_url))) from exc


def _resolve_app_url(app_url: Optional[str]) -> str:
    return (
        app_url
        or os.environ.get("HUGINN_APP_URL")
        or _get_secret("APP_URL")
        or ""
    ).strip().rstrip("/")


def _print_connection_banner(public_url: str, resolved_app_url: str) -> None:
    """Imprime el banner de conexión en la salida de la celda (nunca en archivos de log)."""
    connection_code = encode(public_url, API_TOKEN)
    print("\n========================================================")
    print(f"HUGINN SERVICE V2 ONLINE: {public_url}")
    if resolved_app_url:
        connect_link = build_connect_link(resolved_app_url, connection_code)
        try:
            from IPython.display import HTML, display  # type: ignore

            display(HTML(f'<p><strong>Enlace de conexión:</strong> <a href="{connect_link}" target="_blank">{connect_link}</a></p>'))
        except Exception:
            pass
        print(f"ENLACE DE CONEXION: {connect_link}")
    print(f"CODIGO DE CONEXION: {connection_code}")
    print("Este código contiene tu token. No compartas el notebook con la salida guardada: Editar > Borrar todas las salidas antes de compartir.")
    print("========================================================\n", flush=True)


def _run_monitor_loop() -> None:
    session_log_path = get_session_log_path()
    start_offset = os.path.getsize(session_log_path) if os.path.exists(session_log_path) else 0
    print(
        "[huginn] Monitoreando actividad del servidor. "
        "Para volver a ejecutar esta celda, primero detenela con el botón Stop "
        "(detenerla corta solo el monitor y no apaga el servicio).",
        flush=True,
    )
    monitor_session_logs(log_path=session_log_path, start_offset=start_offset)


def start_server_colab(
    ngrok_domain: Optional[str] = None,
    app_url: Optional[str] = None,
    monitor: bool = True,
    ngrok_retry_interval: float = 5.0,
    ngrok_retry_timeout: float = 120.0,
) -> str:
    """
    Inicia Uvicorn en un hilo daemon de forma idempotente, abre el túnel ngrok,
    verifica /health por la URL pública y monitorea los logs de sesión.
    """
    import uvicorn

    from colab.version import LOADED_COMMIT_HASH

    print(format_version_banner(), flush=True)
    resolved_app_url = _resolve_app_url(app_url)

    # 1. Commit distinto en disco: no reiniciar en el mismo kernel
    stale_msg = check_code_staleness()
    if stale_msg:
        print("\n========================================================")
        print(f"⚠️  {STALE_RESTART_BANNER}")
        print(f"Detalle: {stale_msg}")
        print("========================================================\n", flush=True)
        return _runtime_state.public_url or ""

    # 2. Servidor propio vivo, sano y del mismo commit: reenganchar monitor sin reiniciar
    with _runtime_lock:
        if _is_own_server_healthy():
            public_url = str(_runtime_state.public_url)
            print("[huginn] Servicio ya activo, reconectando monitor", flush=True)
            _print_connection_banner(public_url, resolved_app_url)
            reuse_monitor = True
        else:
            reuse_monitor = False
            # 3. No matar un job en curso antes de intentar cualquier cierre/rearranque
            active_jobs = _find_active_processing_jobs()
            if active_jobs:
                first = active_jobs[0]
                if len(active_jobs) == 1:
                    msg = (
                        f"No se reinicia el servicio: el job {first.job_id[:8]} está en etapa {first.stage}. "
                        "Esperá a que termine o cancelalo desde la app"
                    )
                else:
                    job_desc = ", ".join(f"{j.job_id[:8]} ({j.stage})" for j in active_jobs)
                    msg = (
                        f"No se reinicia el servicio: hay jobs en procesamiento [{job_desc}]. "
                        "Esperá a que terminen o cancelalos desde la app"
                    )
                raise ActiveJobRestartError(msg)

            # 4. Cierre limpio de cualquier instancia/túnel previo roto o zombi
            _shutdown_existing_service()

            ngrok_token = _get_secret("NGROK_AUTHTOKEN")
            if not ngrok_token or not ngrok_token.strip():
                raise ValueError("NGROK_AUTHTOKEN requerido en Secretos o variables de entorno.")

            setup_ngrok_auth(ngrok_token)

            # Iniciar Uvicorn con instancia Server registrada
            config = uvicorn.Config(app, host="0.0.0.0", port=PORT, log_level="warning")
            uv_server = uvicorn.Server(config)
            _runtime_state.server = uv_server
            _runtime_state.port = PORT
            _runtime_state.commit_hash = LOADED_COMMIT_HASH
            _runtime_state.startup_error = None

            def _serve_target() -> None:
                try:
                    uv_server.run()
                except BaseException as exc:
                    _runtime_state.startup_error = exc

            thread = threading.Thread(target=_serve_target, daemon=True)
            _runtime_state.thread = thread
            thread.start()

            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                if _runtime_state.startup_error is not None:
                    err = _runtime_state.startup_error
                    _shutdown_existing_service()
                    safe_err = _sanitize_runtime_message(str(err))
                    raise RuntimeError(
                        f"Error al iniciar el servidor en el puerto {PORT} (¿puerto ocupado?): {safe_err}"
                    ) from err
                if getattr(uv_server, "started", False):
                    break
                if not thread.is_alive():
                    _shutdown_existing_service()
                    raise RuntimeError(
                        f"El servidor uvicorn terminó inesperadamente al iniciar en el puerto {PORT}."
                    )
                time.sleep(0.05)

            if not getattr(uv_server, "started", False):
                _shutdown_existing_service()
                raise RuntimeError(
                    f"Timeout esperando que uvicorn inicie en el puerto {PORT}."
                )

            # Abrir túnel ngrok con dominio opcional y reintento acotado ante ERR_NGROK_334
            resolved_domain = (
                ngrok_domain
                or os.environ.get("NGROK_DOMAIN")
                or _get_secret("NGROK_DOMAIN")
                or ""
            ).strip()
            fixed_url: Optional[str] = None
            if resolved_domain:
                fixed_url = format_ngrok_url(resolved_domain)
                log.info("Abriendo túnel ngrok con URL fija: %s", fixed_url)
            else:
                log.info("Abriendo túnel ngrok con URL aleatoria.")
                print("Nota: NGROK_DOMAIN no está configurado; la URL de ngrok cambiará en cada sesión.")

            try:
                tunnel = _connect_ngrok_with_retry(
                    PORT,
                    fixed_url=fixed_url,
                    retry_interval=ngrok_retry_interval,
                    retry_timeout=ngrok_retry_timeout,
                )
            except Exception:
                _shutdown_existing_service()
                raise

            public_url = str(tunnel.public_url)
            _runtime_state.tunnel = tunnel
            _runtime_state.public_url = public_url

            _print_connection_banner(public_url, resolved_app_url)

            # Autochequeo de /health por la URL pública
            pub_ok, pub_detail = _probe_health_url(public_url, timeout=10.0)
            if pub_ok:
                print("[huginn] Servicio accesible desde internet", flush=True)
            else:
                print(
                    f"[huginn] AVISO: No se pudo verificar /health por la URL pública ({pub_detail}).",
                    flush=True,
                )

    if monitor:
        _run_monitor_loop()

    return public_url


if __name__ == "__main__":
    start_server_colab()
