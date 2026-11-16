from __future__ import annotations

from pathlib import Path

import pytest

from colab.install_deps import (
    build_preserved_constraints,
    find_missing_or_mismatched,
    install_requirements,
    parse_requirements,
    run_with_timeout_and_retry,
)


def test_parse_requirements_reads_exact_pins(tmp_path: Path):
    req_file = tmp_path / "requirements.txt"
    req_file.write_text(
        "# comentario\n"
        "fastapi==0.141.1\n"
        "uvicorn[standard]==0.49.0\n\n",
        encoding="utf-8",
    )
    assert parse_requirements(req_file) == [
        ("fastapi", "0.141.1"),
        ("uvicorn[standard]", "0.49.0"),
    ]


def test_parse_requirements_rejects_unpinned(tmp_path: Path):
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("numpy<2.1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_requirements(req_file)


def test_idempotent_skips_when_all_installed(tmp_path: Path):
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("fastapi==0.141.1\npyngrok==9.1.3\n", encoding="utf-8")

    versions = {"fastapi": "0.141.1", "pyngrok": "9.1.3"}
    ran = install_requirements(req_file, installed_fn=lambda name: versions.get(name))
    assert ran is False


def test_find_missing_or_mismatched():
    reqs = [("fastapi", "0.141.1"), ("pyngrok", "9.1.3"), ("nibabel", "5.5.0")]
    versions = {"fastapi": "0.141.1", "pyngrok": "9.0.0"}
    assert find_missing_or_mismatched(reqs, installed_fn=lambda n: versions.get(n)) == [
        ("pyngrok", "9.1.3"),
        ("nibabel", "5.5.0"),
    ]


def test_build_preserved_constraints():
    versions = {"torch": "2.6.0+cu124", "numpy": "2.0.2"}
    constraints = build_preserved_constraints(
        packages=("torch", "torchvision", "numpy"),
        installed_fn=lambda n: versions.get(n),
    )
    assert constraints == ["torch==2.6.0+cu124", "numpy==2.0.2"]


def test_timeout_retries_once_and_fails_with_clear_message():
    calls = 0

    def fake_stream(_cmd, _timeout, _tail):
        nonlocal calls
        calls += 1
        return 1, True, [f"linea-{calls}-a", f"linea-{calls}-b"]

    with pytest.raises(RuntimeError) as exc_info:
        run_with_timeout_and_retry(["uv", "pip"], timeout_seconds=10, max_attempts=2, stream_fn=fake_stream)

    assert calls == 2
    msg = str(exc_info.value)
    assert "La instalación se colgó: Entorno > Reiniciar sesión y ejecutar todo" in msg
    assert "linea-2-b" in msg


def test_timeout_succeeds_on_second_attempt():
    attempts = []

    def fake_stream(_cmd, _timeout, _tail):
        attempts.append(len(attempts) + 1)
        if len(attempts) == 1:
            return 1, True, ["descargando..."]
        return 0, False, ["listo"]

    run_with_timeout_and_retry(["uv", "pip"], timeout_seconds=10, max_attempts=2, stream_fn=fake_stream)
    assert attempts == [1, 2]
