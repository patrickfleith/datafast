"""Shared setup for the live Ollama suite.

Ollama is served locally, so there is no API key to guard on — `no_api_key` on
both profiles says as much. What can be missing instead is the daemon itself or
the specific model, so `require_ollama` checks reachability and then that the
model is pulled, skipping with a message that names whichever is absent.
"""

import os

import httpx
import pytest

from datafast import ollama

# qwen3:0.6b is the smallest thinking-capable model, which makes it the only one
# that can carry both the reasoning tests and the rest of the suite at a usable
# speed. gemma4:12b is the vision model — the profiles declare Modality.IMAGE for
# every Ollama model, but only some can honour it.
MODEL_ID = "qwen3:0.6b"
VISION_MODEL_ID = "gemma4:12b"

DEFAULT_API_BASE = "http://localhost:11434"


def _api_base() -> str:
    return (os.getenv("OLLAMA_API_BASE") or DEFAULT_API_BASE).rstrip("/")


@pytest.fixture(scope="session")
def require_ollama():
    def _require(model_id: str | None = None) -> None:
        """Skip unless the daemon answers and, when named, the model is pulled."""
        base = _api_base()
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
            pytest.skip(f"Ollama model {model_id} is not pulled — `ollama pull {model_id}`")

    return _require


@pytest.fixture
def served_model(require_ollama):
    """Factory for a live Ollama served model; overrides merge over the defaults.

    `thinking=False` is a default rather than a per-test argument because qwen3
    thinks by default: without it every test in the suite would spend its token
    budget on a trace before answering. `test_reasoning.py` overrides it.
    """
    require_ollama(MODEL_ID)

    def _make(**overrides):
        params = {"temperature": 0.0, "max_completion_tokens": 300, "thinking": False}
        params.update(overrides)
        return ollama(MODEL_ID, **params)

    return _make


@pytest.fixture
def vision_served_model(require_ollama):
    """Factory for the vision model. The timeout is generous because a 12B model
    that is not resident yet has to be loaded before it can answer."""
    require_ollama(VISION_MODEL_ID)

    def _make(**overrides):
        params = {
            "temperature": 0.0,
            "max_completion_tokens": 300,
            "thinking": False,
            "timeout": 300.0,
        }
        params.update(overrides)
        return ollama(VISION_MODEL_ID, **params)

    return _make
