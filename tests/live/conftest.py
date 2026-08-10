"""Shared setup for live provider tests.

Every module under `tests/live/` hits a real provider endpoint, so each one
carries `pytestmark = pytest.mark.live` and the root conftest skips it unless
`--run-live` is passed. Tests also skip themselves when the provider's API key
is missing, so an incomplete `.env` never fails the suite.
"""

import base64
import os
from pathlib import Path

import httpx
import pytest
from dotenv import load_dotenv

load_dotenv()

ASSETS = Path(__file__).parent / "assets"

OLLAMA_DEFAULT_API_BASE = "http://localhost:11434"


@pytest.fixture(scope="session")
def require_api_key():
    def _require(env_key_name: str) -> None:
        if not os.getenv(env_key_name):
            pytest.skip(f"{env_key_name} is not set")

    return _require


def _ollama_api_base() -> str:
    return (os.getenv("OLLAMA_API_BASE") or OLLAMA_DEFAULT_API_BASE).rstrip("/")


@pytest.fixture(scope="session")
def require_ollama():
    """The local-backend counterpart to `require_api_key`.

    Ollama is served locally, so there is no key to guard on — what can be
    missing instead is the daemon itself or the specific model. Lives here
    rather than in `ollama/conftest.py` because the root-level pipeline test
    needs the same guard, and two copies could disagree about the host.
    """

    def _require(model_id: str | None = None) -> None:
        """Skip unless the daemon answers and, when named, the model is pulled."""
        base = _ollama_api_base()
        try:
            httpx.get(f"{base}/api/version", timeout=5.0).raise_for_status()
        except httpx.HTTPError as error:
            pytest.skip(f"no Ollama daemon at {base} ({error.__class__.__name__})")

        if model_id is None:
            return

        response = httpx.post(
            f"{base}/api/show", json={"model": model_id}, timeout=10.0
        )
        if response.is_error:
            pytest.skip(
                f"Ollama model {model_id} is not pulled — `ollama pull {model_id}`"
            )

    return _require


def _data_uri(name: str, media_type: str) -> str:
    data = base64.standard_b64encode((ASSETS / name).read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{data}"


@pytest.fixture(scope="session")
def image_asset():
    """A solid red 64x64 PNG, as a base64 data URI. The expected answer is "red"."""
    return _data_uri("square.png", "image/png")


@pytest.fixture(scope="session")
def document_path():
    """The same PDF as `document_asset`, as a path — for served models whose file
    input goes through an upload API rather than an inline data URI."""
    return ASSETS / "passphrase.pdf"


@pytest.fixture(scope="session")
def document_asset():
    """A one-page PDF, as a base64 data URI. Its only text is the passphrase
    "ZEBRAFISH-42"."""
    return _data_uri("passphrase.pdf", "application/pdf")
