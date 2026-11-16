from __future__ import annotations

import importlib.metadata
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

PRESERVED_COLAB_PACKAGES = ("torch", "torchvision", "torchaudio", "numpy")
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_MAX_ATTEMPTS = 2
DEFAULT_TAIL_LINES = 25

_REQ_PIN_RE = re.compile(r"^([A-Za-z0-9_.-]+)(?:\[[^\]]+\])?==([^\s;#]+)$")


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_timestamp()}] {msg}", flush=True)


def normalize_pkg_name(name: str) -> str:
    """Normaliza el nombre del paquete para consultas de metadatos."""
    base = name.split("[", 1)[0].strip().lower()
    return re.sub(r"[-_.]+", "-", base)


def parse_requirements(requirements_path: str | Path) -> List[Tuple[str, str]]:
    """Lee un archivo requirements.txt con versiones fijadas (==)."""
    path = Path(requirements_path)
    items: List[Tuple[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _REQ_PIN_RE.match(line)
        if not match:
            raise ValueError(f"Requerimiento sin versión exacta (==): {raw_line!r}")
        spec_name = raw_line.split("==", 1)[0].strip()
        version = match.group(2).strip()
        items.append((spec_name, version))
    return items


def get_installed_version(package_name: str) -> Optional[str]:
    """Devuelve la versión instalada de un paquete o None si no está instalado."""
    try:
        return importlib.metadata.version(normalize_pkg_name(package_name))
    except importlib.metadata.PackageNotFoundError:
        return None


def find_missing_or_mismatched(
    requirements: Sequence[Tuple[str, str]],
    installed_fn: Callable[[str], Optional[str]] = get_installed_version,
) -> List[Tuple[str, str]]:
    """Devuelve los paquetes faltantes o cuya versión instalada difiere del lock."""
    pending: List[Tuple[str, str]] = []
    for spec_name, expected_version in requirements:
        current = installed_fn(spec_name)
        if current != expected_version:
            pending.append((spec_name, expected_version))
    return pending


def build_preserved_constraints(
    packages: Iterable[str] = PRESERVED_COLAB_PACKAGES,
    installed_fn: Callable[[str], Optional[str]] = get_installed_version,
) -> List[str]:
    """Genera constraints para no reinstalar paquetes pesados que Colab ya provee."""
    constraints: List[str] = []
    for pkg in packages:
        ver = installed_fn(pkg)
        if ver:
            constraints.append(f"{normalize_pkg_name(pkg)}=={ver}")
    return constraints


def ensure_uv(
    which_fn: Callable[[str], Optional[str]] = shutil.which,
    run_fn: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> bool:
    """Verifica si uv está disponible o intenta instalarlo con pip."""
    if which_fn("uv"):
        return True
    _log("uv no encontrado; instalando uv con pip para acelerar la resolución...")
    try:
        res = run_fn(
            [sys.executable, "-m", "pip", "install", "-q", "uv"],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        if res.returncode == 0 and which_fn("uv"):
            _log("uv instalado correctamente.")
            return True
    except Exception as exc:
        _log(f"No se pudo instalar uv ({exc}); se usará pip.")
    return False


def _stream_process(
    cmd: Sequence[str],
    timeout_seconds: int,
    tail_lines: int = DEFAULT_TAIL_LINES,
    popen_fn: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> Tuple[int, bool, List[str]]:
    """Ejecuta un comando con salida con marcas de tiempo y tope de tiempo."""
    proc = popen_fn(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    line_queue: queue.Queue[Optional[str]] = queue.Queue()

    def _reader() -> None:
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                line_queue.put(line.rstrip("\r\n"))
        finally:
            line_queue.put(None)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    tail: deque[str] = deque(maxlen=tail_lines)
    deadline = time.monotonic() + timeout_seconds
    timed_out = False
    eof = False

    while not eof:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            proc.kill()
            break
        try:
            item = line_queue.get(timeout=min(1.0, remaining))
        except queue.Empty:
            if proc.poll() is not None and line_queue.empty():
                break
            continue
        if item is None:
            eof = True
            break
        if item.strip():
            tail.append(item)
            _log(item)

    try:
        returncode = proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        returncode = proc.wait(timeout=5)
        timed_out = True

    reader_thread.join(timeout=1)
    return returncode, timed_out, list(tail)


def run_with_timeout_and_retry(
    cmd: Sequence[str],
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    tail_lines: int = DEFAULT_TAIL_LINES,
    stream_fn: Callable[..., Tuple[int, bool, List[str]]] = _stream_process,
) -> None:
    """Ejecuta un comando de instalación con tope de tiempo y un único reintento."""
    last_tail: List[str] = []
    for attempt in range(1, max_attempts + 1):
        _log(f"Iniciando instalación (intento {attempt}/{max_attempts}, tope {timeout_seconds}s)...")
        returncode, timed_out, tail = stream_fn(cmd, timeout_seconds, tail_lines)
        last_tail = tail
        if not timed_out and returncode == 0:
            _log("Instalación completada.")
            return
        if timed_out:
            _log(f"Tiempo límite ({timeout_seconds}s) excedido en intento {attempt}/{max_attempts}.")
            if attempt < max_attempts:
                _log("Reintentando instalación una única vez...")
                continue
            tail_text = "\n".join(last_tail) if last_tail else "(sin salida capturada)"
            raise RuntimeError(
                "La instalación se colgó: Entorno > Reiniciar sesión y ejecutar todo\n"
                f"Última salida útil:\n{tail_text}"
            )
        tail_text = "\n".join(last_tail) if last_tail else "(sin salida capturada)"
        raise RuntimeError(
            f"La instalación falló con código de salida {returncode}.\n"
            f"Última salida útil:\n{tail_text}"
        )


def install_requirements(
    requirements_path: str | Path,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    installed_fn: Callable[[str], Optional[str]] = get_installed_version,
) -> bool:
    """Instala dependencias de requirements.txt de forma idempotente y robusta."""
    reqs = parse_requirements(requirements_path)
    pending = find_missing_or_mismatched(reqs, installed_fn=installed_fn)
    if not pending:
        _log("Todas las dependencias ya están instaladas en la versión exacta. Saltando instalación.")
        return False

    pending_summary = ", ".join(f"{name}=={ver}" for name, ver in pending)
    _log(f"Paquetes por instalar/ajustar ({len(pending)}): {pending_summary}")

    constraints = build_preserved_constraints(installed_fn=installed_fn)
    use_uv = ensure_uv()

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        tmp_path = tmp.name
        if constraints:
            tmp.write("\n".join(constraints) + "\n")
            _log(f"Preservando versiones del entorno: {', '.join(constraints)}")

    try:
        if use_uv:
            cmd = [
                "uv",
                "pip",
                "install",
                "--system",
                "--no-cache",
                "-r",
                str(requirements_path),
            ]
        else:
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "-r",
                str(requirements_path),
            ]
        if constraints:
            cmd.extend(["-c", tmp_path])

        run_with_timeout_and_retry(
            cmd,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
        )
        return True
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    default_req = Path(__file__).resolve().parent / "requirements.txt"
    install_requirements(default_req)
