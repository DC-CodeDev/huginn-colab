from __future__ import annotations

import base64
import json
import zlib
from typing import Optional
from urllib.parse import urlparse


def _validate_ascii_no_spaces(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} debe ser una cadena de texto.")
    if not value.isascii():
        raise ValueError(f"{field_name} solo puede contener caracteres ASCII.")
    if any(ch.isspace() for ch in value):
        raise ValueError(f"{field_name} no puede contener espacios ni saltos de línea.")


def _validate_https_url(url: str, field_name: str, allow_empty: bool = False) -> str:
    _validate_ascii_no_spaces(url, field_name)
    if url == "":
        if allow_empty:
            return ""
        raise ValueError(f"{field_name} no puede estar vacía.")
    if not url.startswith("https://"):
        raise ValueError(f"{field_name} debe comenzar con https://.")
    parsed = urlparse(url)
    if not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(f"{field_name} debe ser una URL HTTPS válida sin credenciales, query ni fragmento.")
    return url.rstrip("/")


def encode(url: str, token: str, notebook_url: Optional[str] = None) -> str:
    """Codifica un código de conexión HGN1 idéntico a app/docs/codigo-de-conexion.md."""
    clean_url = _validate_https_url(url, "url", allow_empty=True)
    _validate_ascii_no_spaces(token, "token")
    if not token:
        raise ValueError("token no puede estar vacío.")

    payload = {"u": clean_url, "t": token}
    if notebook_url is not None and notebook_url != "":
        payload["c"] = _validate_https_url(notebook_url, "notebook_url", allow_empty=False)

    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    b64 = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
    body = f"HGN1-{b64}"
    crc = zlib.crc32(body.encode("utf-8")) & 0xFFFFFFFF
    return f"{body}.{crc:08x}"


def build_connect_link(app_url: str, code: str) -> str:
    """Construye el enlace directo <APP_URL>/#/conectar/<código>."""
    clean_app = app_url.strip().rstrip("/")
    if not clean_app:
        raise ValueError("app_url no puede estar vacía.")
    return f"{clean_app}/#/conectar/{code}"
