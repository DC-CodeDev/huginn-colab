from __future__ import annotations

import os
import socket
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from colab import server
from colab.session_log import get_job_log_path, get_session_log_path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _install_fake_pyngrok(monkeypatch, public_url: str = "https://fake-huginn.ngrok-free.app"):
    tunnels = []
    calls = {"connect": 0, "disconnect": [], "kill": 0}

    fake_pyngrok = types.ModuleType("pyngrok")
    fake_conf = types.ModuleType("pyngrok.conf")
    fake_ngrok = types.ModuleType("pyngrok.ngrok")
    fake_exc = types.ModuleType("pyngrok.exception")

    class PyngrokError(Exception):
        pass

    fake_exc.PyngrokError = PyngrokError
    default_conf = types.SimpleNamespace(auth_token=None)
    fake_conf.get_default = lambda: default_conf

    def set_auth_token(tok: str) -> None:
        default_conf.auth_token = tok

    def connect(port: int, url: str | None = None):
        calls["connect"] += 1
        t = types.SimpleNamespace(public_url=url or public_url)
        tunnels.clear()
        tunnels.append(t)
        return t

    def disconnect(url: str) -> None:
        calls["disconnect"].append(url)
        tunnels[:] = [t for t in tunnels if t.public_url != url]

    def kill() -> None:
        calls["kill"] += 1
        tunnels.clear()

    def get_tunnels():
        return list(tunnels)

    fake_ngrok.set_auth_token = set_auth_token
    fake_ngrok.connect = connect
    fake_ngrok.disconnect = disconnect
    fake_ngrok.kill = kill
    fake_ngrok.get_tunnels = get_tunnels

    fake_pyngrok.conf = fake_conf
    fake_pyngrok.ngrok = fake_ngrok
    fake_pyngrok.exception = fake_exc

    monkeypatch.setitem(sys.modules, "pyngrok", fake_pyngrok)
    monkeypatch.setitem(sys.modules, "pyngrok.conf", fake_conf)
    monkeypatch.setitem(sys.modules, "pyngrok.ngrok", fake_ngrok)
    monkeypatch.setitem(sys.modules, "pyngrok.exception", fake_exc)
    return fake_ngrok, tunnels, calls


@pytest.fixture(autouse=True)
def _clean_runtime_state(monkeypatch, tmp_path):
    monkeypatch.setenv("HUGINN_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("NGROK_AUTHTOKEN", "fake-ngrok-secret-token-12345")
    yield
    server._shutdown_existing_service(timeout=3.0)
    server.jobs_db.clear()


def _route_public_probe_to_local(monkeypatch, public_url: str, port: int) -> None:
    original_probe = server._probe_health_url

    def routed_probe(url: str, timeout: float = 3.0):
        if url.rstrip("/") == public_url.rstrip("/"):
            return original_probe(f"http://127.0.0.1:{port}", timeout=timeout)
        return original_probe(url, timeout=timeout)

    monkeypatch.setattr(server, "_probe_health_url", routed_probe)


def test_double_start_reattaches_monitor_without_restarting(monkeypatch, capsys):
    port = _free_port()
    monkeypatch.setattr(server, "PORT", port)
    monkeypatch.setattr(server, "check_code_staleness", lambda *a, **kw: None)

    public_url = "https://fake-huginn.ngrok-free.app"
    _, _, calls = _install_fake_pyngrok(monkeypatch, public_url=public_url)
    _route_public_probe_to_local(monkeypatch, public_url, port)

    monitor_calls = []
    monkeypatch.setattr(
        server,
        "monitor_session_logs",
        lambda **kw: monitor_calls.append(kw) or 0,
    )

    url1 = server.start_server_colab(monitor=True)
    assert url1 == public_url
    assert calls["connect"] == 1
    assert len(monitor_calls) == 1
    first_server = server._runtime_state.server
    first_thread = server._runtime_state.thread
    assert first_server is not None and first_thread is not None and first_thread.is_alive()

    out1 = capsys.readouterr().out
    assert out1.splitlines()[0] == "[huginn] Servicio nuevo: arrancando."
    assert "Servicio accesible desde internet" in out1

    # Segunda ejecución en el mismo proceso: no reinicia, reengancha monitor
    url2 = server.start_server_colab(monitor=True)
    assert url2 == public_url
    assert calls["connect"] == 1
    assert len(monitor_calls) == 2
    assert server._runtime_state.server is first_server
    assert server._runtime_state.thread is first_thread

    out2 = capsys.readouterr().out
    assert out2.splitlines()[0] == "[huginn] Servicio ya activo: reconectando el monitor (sin reiniciar)."
    assert "CODIGO DE CONEXION: HGN1-" in out2

    # Verificar que ni API_TOKEN ni HGN1- aparecen en session.log ni job logs
    logs_dir = Path(os.environ["HUGINN_LOGS_DIR"])
    session_log = Path(get_session_log_path(str(logs_dir)))
    if session_log.exists():
        text = session_log.read_text(encoding="utf-8")
        assert server.API_TOKEN not in text
        assert "HGN1-" not in text


def test_double_start_restarts_cleanly_when_tunnel_died_without_zombie(monkeypatch, capsys):
    port = _free_port()
    monkeypatch.setattr(server, "PORT", port)
    monkeypatch.setattr(server, "check_code_staleness", lambda *a, **kw: None)

    public_url = "https://fake-huginn.ngrok-free.app"
    _, tunnels, calls = _install_fake_pyngrok(monkeypatch, public_url=public_url)
    _route_public_probe_to_local(monkeypatch, public_url, port)

    url1 = server.start_server_colab(monitor=False)
    assert url1 == public_url
    first_server = server._runtime_state.server
    first_thread = server._runtime_state.thread
    capsys.readouterr()

    # Simular que el túnel ngrok cayó aunque uvicorn seguía vivo
    tunnels.clear()

    url2 = server.start_server_colab(monitor=False)
    out2 = capsys.readouterr().out
    assert out2.splitlines()[0] == "[huginn] Servicio anterior con problemas: se cierra y se arranca de nuevo."
    assert url2 == public_url
    assert calls["connect"] == 2
    assert calls["kill"] >= 1
    assert not first_thread.is_alive()
    assert server._runtime_state.server is not first_server
    assert server._runtime_state.thread is not None and server._runtime_state.thread.is_alive()


def test_different_commit_prints_restart_banner_and_does_not_restart(monkeypatch, capsys):
    port = _free_port()
    monkeypatch.setattr(server, "PORT", port)

    # Arranque inicial sin staleness
    monkeypatch.setattr(server, "check_code_staleness", lambda *a, **kw: None)
    public_url = "https://fake-huginn.ngrok-free.app"
    _, _, calls = _install_fake_pyngrok(monkeypatch, public_url=public_url)
    _route_public_probe_to_local(monkeypatch, public_url, port)
    server.start_server_colab(monitor=False)
    first_server = server._runtime_state.server
    kill_before = calls["kill"]
    capsys.readouterr()

    # Ahora simular commit distinto en disco
    monkeypatch.setattr(
        server,
        "check_code_staleness",
        lambda *a, **kw: "El código en disco es más nuevo (cargado=aaaa111, disco=bbbb222)",
    )
    res = server.start_server_colab(monitor=False)
    assert res == public_url
    assert calls["connect"] == 1
    assert calls["kill"] == kill_before
    assert server._runtime_state.server is first_server
    assert not first_server.should_exit

    out = capsys.readouterr().out
    assert out.splitlines()[0] == "[huginn] Hay una versión más nueva del servicio: reiniciá el entorno."
    assert "Hay una versión más nueva del servicio. Para cargarla: Entorno de ejecución > Reiniciar sesión y ejecutar todo" in out


def test_active_job_blocks_restart_without_killing_server_or_subprocess(monkeypatch, capsys):
    port = _free_port()
    monkeypatch.setattr(server, "PORT", port)
    monkeypatch.setattr(server, "check_code_staleness", lambda *a, **kw: None)

    public_url = "https://fake-huginn.ngrok-free.app"
    fake_ngrok, tunnels, calls = _install_fake_pyngrok(monkeypatch, public_url=public_url)
    _route_public_probe_to_local(monkeypatch, public_url, port)
    server.start_server_colab(monitor=False)
    kill_before = calls["kill"]
    capsys.readouterr()

    # Romper túnel para forzar camino de reinicio, pero con un job activo en curso
    tunnels.clear()
    fake_proc = MagicMock()
    fake_proc.is_alive.return_value = True
    job = server.JobRecord(job_id="abcdef12-3456-7890-abcd-ef1234567890", job_dir="/tmp", created_at="2026-01-01T00:00:00Z")
    job.status = "processing"
    job.stage = "segmenting_tissues"
    job.process = fake_proc
    server.jobs_db[job.job_id] = job

    srv = server._runtime_state.server
    with pytest.raises(server.ActiveJobRestartError) as exc_info:
        server.start_server_colab(monitor=False)

    out_active = capsys.readouterr().out
    assert out_active.splitlines()[0] == "[huginn] No se reinicia: hay un trabajo en procesamiento."
    assert isinstance(exc_info.value, RuntimeError)
    assert str(exc_info.value) == (
        "No se reinicia el servicio: el job abcdef12 está en etapa segmenting_tissues. "
        "Esperá a que termine o cancelalo desde la app"
    )
    assert not srv.should_exit
    assert calls["kill"] == kill_before
    fake_proc.terminate.assert_not_called()
    fake_proc.kill.assert_not_called()


def test_ngrok_err_334_retries_and_translates_to_clear_message(monkeypatch, capsys):
    port = _free_port()
    monkeypatch.setattr(server, "PORT", port)
    monkeypatch.setattr(server, "check_code_staleness", lambda *a, **kw: None)
    fake_ngrok, _, calls = _install_fake_pyngrok(monkeypatch)

    def always_334(port_arg: int, url: str | None = None):
        calls["connect"] += 1
        raise RuntimeError("ERR_NGROK_334: The endpoint 'https://mi.ngrok-free.app' is already online.")

    fake_ngrok.connect = always_334

    with pytest.raises(RuntimeError) as exc_info:
        server.start_server_colab(
            ngrok_domain="mi.ngrok-free.app",
            monitor=False,
            ngrok_retry_interval=0.02,
            ngrok_retry_timeout=0.05,
        )

    msg = str(exc_info.value)
    assert "ERR_NGROK_334" in msg
    assert "dashboard.ngrok.com/agents" in msg
    assert "NGROK_DOMAIN" in msg
    assert "fake-ngrok-secret-token-12345" not in msg
    assert calls["connect"] >= 2
    # El servidor uvicorn debe haberse limpiado tras el fallo de ngrok
    assert server._runtime_state.server is None


def test_port_bind_failure_in_thread_propagates_to_caller(monkeypatch):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("0.0.0.0", 0))
    sock.listen(1)
    occupied_port = int(sock.getsockname()[1])

    try:
        monkeypatch.setattr(server, "PORT", occupied_port)
        monkeypatch.setattr(server, "check_code_staleness", lambda *a, **kw: None)
        _install_fake_pyngrok(monkeypatch)

        with pytest.raises(RuntimeError) as exc_info:
            server.start_server_colab(monitor=False)

        assert str(occupied_port) in str(exc_info.value)
        assert server._runtime_state.server is None
    finally:
        sock.close()


def test_public_health_check_failure_reports_reason_without_secrets(monkeypatch, capsys):
    port = _free_port()
    monkeypatch.setattr(server, "PORT", port)
    monkeypatch.setattr(server, "check_code_staleness", lambda *a, **kw: None)
    _install_fake_pyngrok(monkeypatch, public_url="https://unreachable.invalid")

    original_probe = server._probe_health_url

    def fake_probe(url: str, timeout: float = 3.0):
        if "unreachable.invalid" in url:
            return False, "connection refused"
        return original_probe(url, timeout=timeout)

    monkeypatch.setattr(server, "_probe_health_url", fake_probe)

    server.start_server_colab(monitor=False)
    out = capsys.readouterr().out
    assert "No se pudo verificar /health por la URL pública (connection refused)" in out
    assert "fake-ngrok-secret-token-12345" not in out
    assert f"Bearer {server.API_TOKEN}" not in out


def test_cors_allowed_origins_restricts_wildcard_and_keeps_local_dev(monkeypatch):
    monkeypatch.delenv("HUGINN_ALLOWED_ORIGIN", raising=False)
    monkeypatch.delenv("HUGINN_APP_URL", raising=False)
    monkeypatch.delenv("HUGINN_ALLOW_LOCAL_DEV", raising=False)
    monkeypatch.setattr(server, "_get_secret", lambda _name: "")

    default_origins = server._resolve_allowed_origins()
    assert "*" not in default_origins
    assert "https://visor-dicom-l3.vercel.app" in default_origins
    assert "https://visor-dicom-l3.bylgjalabs.workers.dev" not in default_origins
    assert "http://localhost:5174" in default_origins
    assert "http://127.0.0.1:5174" in default_origins

    monkeypatch.setenv("HUGINN_APP_URL", "https://visor-dicom-l3.vercel.app/subruta/")
    with_app_url = server._resolve_allowed_origins()
    assert "https://visor-dicom-l3.vercel.app" in with_app_url
    assert "http://localhost:5174" in with_app_url
    assert "*" not in with_app_url

    monkeypatch.setenv("HUGINN_ALLOWED_ORIGIN", "https://custom.vercel.app, *")
    with_explicit = server._resolve_allowed_origins()
    assert "https://custom.vercel.app" in with_explicit
    assert "*" not in with_explicit


def test_cors_middleware_accepts_prod_and_local_dev_and_rejects_other():
    from fastapi.testclient import TestClient

    client = TestClient(server.app)

    res_prod = client.options(
        "/health",
        headers={
            "Origin": "https://visor-dicom-l3.vercel.app",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert res_prod.status_code == 200
    assert res_prod.headers.get("access-control-allow-origin") == "https://visor-dicom-l3.vercel.app"

    res_local = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:5174",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert res_local.status_code == 200
    assert res_local.headers.get("access-control-allow-origin") == "http://localhost:5174"

    res_workers = client.options(
        "/health",
        headers={
            "Origin": "https://visor-dicom-l3.bylgjalabs.workers.dev",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert res_workers.headers.get("access-control-allow-origin") is None

    res_evil = client.options(
        "/health",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert res_evil.headers.get("access-control-allow-origin") is None
