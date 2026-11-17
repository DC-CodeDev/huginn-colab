from __future__ import annotations

import os
import re
import threading
import time
from typing import Callable, Iterable, Optional

DEFAULT_LOGS_DIR = "/tmp/huginn/logs"

STAGE_LABELS = {
    "validating": "etapa: validando volumen",
    "locating_l3": "etapa: localizando L3",
    "segmenting_tissues": "etapa: segmentando tejidos",
    "packaging": "etapa: empaquetando resultado",
    "completed": "etapa: completado",
    "failed": "etapa: error",
}

_CONN_CODE_RE = re.compile(r"HGN1-[A-Za-z0-9_\-=]+")
_URL_CREDS_RE = re.compile(r"://[^/@\s]+@")
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/=]+")
_PATIENT_FILE_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?!(?:input\.nii\.gz|result\.zip)\b)[^\s/\\]+\.(?:nii(?:\.gz)?|dcm|dicom)\b",
    re.IGNORECASE,
)

_SESSION_LOG_LOCK = threading.Lock()


def get_logs_dir(logs_dir: Optional[str] = None) -> str:
    """Devuelve el directorio de logs fuera del repositorio y lo crea si no existe."""
    path = logs_dir or os.environ.get("HUGINN_LOGS_DIR", DEFAULT_LOGS_DIR)
    os.makedirs(path, exist_ok=True)
    return path


def get_job_log_path(job_id: str, logs_dir: Optional[str] = None) -> str:
    """Devuelve la ruta del archivo de log completo para un job (sin PII en el nombre)."""
    base = get_logs_dir(logs_dir)
    short_id = re.sub(r"[^A-Za-z0-9_-]", "", job_id[:16]) or "unknown"
    return os.path.join(base, f"job_{short_id[:8]}.log")


def get_session_log_path(logs_dir: Optional[str] = None) -> str:
    """Devuelve la ruta del archivo de log resumido de la sesión para el monitor."""
    base = get_logs_dir(logs_dir)
    return os.path.join(base, "session.log")


def format_stage_label(stage: str) -> str:
    """Traduce el nombre interno de etapa a una etiqueta corta legible."""
    return STAGE_LABELS.get(stage, f"etapa: {stage}")


def sanitize_log_text(
    text: str,
    extra_secrets: Optional[Iterable[str]] = None,
) -> str:
    """
    Elimina tokens, códigos de conexión, credenciales en URL y nombres de archivo
    de paciente antes de escribir en logs o mostrar en celda.
    """
    cleaned = str(text)
    if extra_secrets:
        for secret in extra_secrets:
            if secret and len(secret) >= 4:
                cleaned = cleaned.replace(secret, "***")
    cleaned = _CONN_CODE_RE.sub("HGN1-***", cleaned)
    cleaned = _URL_CREDS_RE.sub("://***@", cleaned)
    cleaned = _BEARER_RE.sub("Bearer ***", cleaned)
    cleaned = _PATIENT_FILE_RE.sub("<archivo>", cleaned)
    return cleaned


def format_huginn_line(
    message: str,
    extra_secrets: Optional[Iterable[str]] = None,
) -> str:
    """Formatea una línea corta con prefijo [huginn] y sanitizada."""
    safe = sanitize_log_text(message, extra_secrets=extra_secrets).strip()
    if safe.startswith("[huginn]"):
        return safe
    return f"[huginn] {safe}"


def append_log_line(
    log_path: str,
    line: str,
    extra_secrets: Optional[Iterable[str]] = None,
) -> None:
    """Agrega una línea sanitizada al archivo de log con flush inmediato."""
    safe_line = sanitize_log_text(line, extra_secrets=extra_secrets).rstrip("\r\n")
    parent = os.path.dirname(log_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with _SESSION_LOG_LOCK:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(safe_line + "\n")
            fh.flush()


def monitor_session_logs(
    log_path: Optional[str] = None,
    poll_interval: float = 0.5,
    start_offset: Optional[int] = None,
    stop_event: Optional[threading.Event] = None,
    out: Optional[Callable[[str], None]] = None,
) -> int:
    """
    Monitorea el archivo de log de sesión e imprime líneas nuevas en la celda.
    Al recibir KeyboardInterrupt detiene solo el monitoreo sin afectar el servidor.
    Retorna el offset final leído.
    """
    target_path = log_path or get_session_log_path()
    emit = out or (lambda msg: print(msg, flush=True))

    if start_offset is not None:
        offset = max(0, start_offset)
    elif os.path.exists(target_path):
        offset = os.path.getsize(target_path)
    else:
        offset = 0

    try:
        while stop_event is None or not stop_event.is_set():
            if os.path.exists(target_path):
                size = os.path.getsize(target_path)
                if size < offset:
                    offset = 0
                if size > offset:
                    with open(target_path, "r", encoding="utf-8", errors="replace") as fh:
                        fh.seek(offset)
                        chunk = fh.read()
                        offset = fh.tell()
                    for line in chunk.splitlines():
                        if line.strip():
                            emit(line)
            if stop_event is not None:
                if stop_event.wait(poll_interval):
                    break
            else:
                time.sleep(poll_interval)
    except KeyboardInterrupt:
        emit("[huginn] Monitor detenido; el servicio sigue activo")

    return offset
