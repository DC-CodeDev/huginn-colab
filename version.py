from __future__ import annotations

import os
import subprocess
from typing import Optional

SERVICE_VERSION = "2.0.0"

STALE_CODE_WARNING = (
    "El código en disco es más nuevo que el servidor en ejecución. "
    "Reiniciá el entorno y ejecutá todo desde cero."
)


def get_git_commit_hash(repo_dir: Optional[str] = None, short: bool = False) -> Optional[str]:
    """Obtiene el hash del commit HEAD del repositorio que contiene a `repo_dir`."""
    target = repo_dir or os.path.dirname(os.path.abspath(__file__))
    cmd = ["git", "-C", target, "rev-parse"]
    if short:
        cmd.append("--short")
    cmd.append("HEAD")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=5)
        if res.returncode == 0:
            val = res.stdout.strip()
            return val or None
    except Exception:
        pass
    return None


# Capturado una única vez al importar el paquete en memoria.
LOADED_COMMIT_HASH: Optional[str] = get_git_commit_hash(short=False)
LOADED_COMMIT_SHORT: str = LOADED_COMMIT_HASH[:7] if LOADED_COMMIT_HASH else "desconocido"


def check_code_staleness(
    repo_dir: Optional[str] = None,
    loaded_hash: Optional[str] = LOADED_COMMIT_HASH,
) -> Optional[str]:
    """
    Compara el commit en disco contra el cargado en memoria.
    Retorna un mensaje de aviso si difieren, o None si coinciden o no hay git.
    """
    if not loaded_hash:
        return None
    disk_hash = get_git_commit_hash(repo_dir=repo_dir, short=False)
    if not disk_hash or disk_hash == loaded_hash:
        return None
    return (
        f"{STALE_CODE_WARNING} "
        f"(cargado={loaded_hash[:7]}, disco={disk_hash[:7]})"
    )


def format_version_banner() -> str:
    """Devuelve la línea corta de versión y commit cargado en memoria."""
    return f"[huginn] servicio v{SERVICE_VERSION} | commit={LOADED_COMMIT_SHORT}"
