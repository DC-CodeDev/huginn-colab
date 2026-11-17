from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
import secrets
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import fastapi
from fastapi import FastAPI, File, Header, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import nibabel as nib

from colab.connection_code import build_connect_link, encode
from colab.memory import (
    DEFAULT_MEMORY_ESTIMATION_K,
    estimate_required_memory_bytes,
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

# Coeficiente k conservador para prechequeo de RAM — a calibrar con mediciones reales
MEMORY_ESTIMATION_K = get_memory_estimation_k()

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

            status_queue = spawn_ctx.Queue()
            job.status_queue = status_queue

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
                    break

                await asyncio.sleep(0.5)

            # Esperar a que el subproceso termine de cerrar
            proc.join(timeout=2.0)

            # Vaciar mensajes finales de la cola
            while not status_queue.empty():
                msg = status_queue.get_nowait()
                if "status" in msg:
                    job.status = msg["status"]
                if "stage" in msg:
                    job.stage = msg["stage"]
                if "progress" in msg:
                    job.progress = msg["progress"]
                if "error_code" in msg:
                    job.error_code = msg["error_code"]
                if "detail" in msg:
                    job.detail = msg["detail"]
                if "completed_at" in msg:
                    job.completed_at = msg["completed_at"]
                if "diagnostics" in msg:
                    job.diagnostics = msg["diagnostics"]

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
    version="2.0.0",
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

    # Prechequeo de RAM estimada vs RAM disponible antes de encolar/procesar
    estimated_bytes = estimate_required_memory_bytes(shape, k=MEMORY_ESTIMATION_K)
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
                    f"se estiman ~{est_mb} MB y hay ~{avail_mb} MB disponibles."
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


def start_server_colab(
    ngrok_domain: Optional[str] = None,
    app_url: Optional[str] = None,
) -> str:
    """
    Inicia Uvicorn en un hilo daemon y abre el túnel ngrok.
    Soporta NGROK_DOMAIN y APP_URL opcionales.
    """
    import threading
    import uvicorn
    from pyngrok import ngrok

    ngrok_token = _get_secret("NGROK_AUTHTOKEN")
    if not ngrok_token or not ngrok_token.strip():
        raise ValueError("NGROK_AUTHTOKEN requerido en Secretos o variables de entorno.")

    setup_ngrok_auth(ngrok_token)

    # Iniciar Uvicorn
    thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning"),
        daemon=True,
    )
    thread.start()
    time.sleep(1.0)

    # Abrir túnel ngrok con dominio opcional
    resolved_domain = (
        ngrok_domain
        or os.environ.get("NGROK_DOMAIN")
        or _get_secret("NGROK_DOMAIN")
        or ""
    ).strip()
    if resolved_domain:
        fixed_url = format_ngrok_url(resolved_domain)
        log.info("Abriendo túnel ngrok con URL fija: %s", fixed_url)
        tunnel = ngrok.connect(PORT, url=fixed_url)
    else:
        log.info("Abriendo túnel ngrok con URL aleatoria.")
        print("Nota: NGROK_DOMAIN no está configurado; la URL de ngrok cambiará en cada sesión.")
        tunnel = ngrok.connect(PORT)

    public_url = str(tunnel.public_url)
    connection_code = encode(public_url, API_TOKEN)
    resolved_app_url = (
        app_url
        or os.environ.get("HUGINN_APP_URL")
        or _get_secret("APP_URL")
        or ""
    ).strip().rstrip("/")

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
    print("========================================================\n")
    return public_url


if __name__ == "__main__":
    start_server_colab()
