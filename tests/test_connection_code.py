from __future__ import annotations

import pytest

from colab.connection_code import build_connect_link, encode

VECTOR_1_URL = "https://mi-tunel.ngrok-free.app"
VECTOR_1_TOKEN = "tok-prueba-123"
VECTOR_1_CODE = "HGN1-eyJ1IjoiaHR0cHM6Ly9taS10dW5lbC5uZ3Jvay1mcmVlLmFwcCIsInQiOiJ0b2stcHJ1ZWJhLTEyMyJ9.9e4b16b8"

VECTOR_2_URL = "https://otro-tunel.ngrok-free.app"
VECTOR_2_TOKEN = "tok-abc-999"
VECTOR_2_NOTEBOOK = "https://colab.research.google.com/drive/1abc"
VECTOR_2_CODE = "HGN1-eyJ1IjoiaHR0cHM6Ly9vdHJvLXR1bmVsLm5ncm9rLWZyZWUuYXBwIiwidCI6InRvay1hYmMtOTk5IiwiYyI6Imh0dHBzOi8vY29sYWIucmVzZWFyY2guZ29vZ2xlLmNvbS9kcml2ZS8xYWJjIn0.7b9fab59"

VECTOR_3_CODE = "HGN1-eyJ1IjoiIiwidCI6InRvay1zb2xvLWRlZmF1bHQifQ.25f78085"


def test_fixed_vectors_match_spec():
    assert encode(VECTOR_1_URL, VECTOR_1_TOKEN) == VECTOR_1_CODE
    assert encode(VECTOR_2_URL, VECTOR_2_TOKEN, VECTOR_2_NOTEBOOK) == VECTOR_2_CODE
    assert encode("", "tok-solo-default") == VECTOR_3_CODE


def test_trailing_slash_normalized():
    assert encode(VECTOR_1_URL + "/", VECTOR_1_TOKEN) == VECTOR_1_CODE


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://mi-tunel.ngrok-free.app",
        "mi-tunel.ngrok-free.app",
        "ftp://mi-tunel.ngrok-free.app",
        "https://mi-tunel.ngrok-free.app?foo=bar",
        "https://mi-tunel.ngrok-free.app#frag",
        "https://user:pass@mi-tunel.ngrok-free.app",
    ],
)
def test_rejects_non_https_or_invalid_url(bad_url: str):
    with pytest.raises(ValueError):
        encode(bad_url, VECTOR_1_TOKEN)


@pytest.mark.parametrize("bad_token", ["", "   ", "tok con espacios", " tok", "tok\n"])
def test_rejects_empty_or_whitespace_token(bad_token: str):
    with pytest.raises(ValueError):
        encode(VECTOR_1_URL, bad_token)


@pytest.mark.parametrize(
    "bad_url",
    [
        " https://mi-tunel.ngrok-free.app",
        "https://mi tunel.ngrok-free.app",
        "https://mi-tunel.ngrok-free.app\n",
    ],
)
def test_rejects_whitespace_in_url(bad_url: str):
    with pytest.raises(ValueError):
        encode(bad_url, VECTOR_1_TOKEN)


@pytest.mark.parametrize(
    ("url", "token"),
    [
        ("https://túnel.ngrok-free.app", VECTOR_1_TOKEN),
        (VECTOR_1_URL, "tok-ñ"),
        (VECTOR_1_URL, "tok-é"),
    ],
)
def test_rejects_non_ascii(url: str, token: str):
    with pytest.raises(ValueError):
        encode(url, token)


def test_build_connect_link():
    assert (
        build_connect_link("https://huginn.example.com/", VECTOR_1_CODE)
        == f"https://huginn.example.com/#/conectar/{VECTOR_1_CODE}"
    )
