"""Shared setup for the live Ollama suite.

Ollama is served locally, so there is no API key to guard on — `no_api_key` on
both profiles says as much. What can be missing instead is the daemon itself or
the specific model, which is what `require_ollama` checks; it lives in the
parent conftest because the root-level pipeline test guards on it too.
"""

import pytest

from datafast import ollama

# qwen3:0.6b is the smallest thinking-capable model, which makes it the only one
# that can carry both the reasoning tests and the rest of the suite at a usable
# speed. gemma4:12b is the vision model — the profiles declare Modality.IMAGE for
# every Ollama model, but only some can honour it.
MODEL_ID = "qwen3:0.6b"
VISION_MODEL_ID = "gemma4:12b"


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
