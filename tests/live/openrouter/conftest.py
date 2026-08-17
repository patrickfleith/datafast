import pytest

from datafast import openrouter

# google/gemma-4-31b-it is the cheapest catalog entry that still covers what
# OPENROUTER_CHAT declares: it takes image input, and its endpoints support
# structured outputs. The docs' `z-ai/glm-4.6` is text-only and could not
# exercise the profile's IMAGE modality; the factory default
# `openai/gpt-5.4-mini` costs ~7x more and only retreads the OpenAI suite.
MODEL_ID = "google/gemma-4-31b-it"

# OpenRouter routes one model id across 19 endpoints, and they do not agree
# about what they can do. Unpinned — or worse, throughput-sorted via a `:nitro`
# suffix — a test would pass or fail depending on where the call landed, which
# measures routing luck rather than datafast. So every request pins one
# endpoint and forbids fallbacks, and test_provider_routing.py proves the pin
# is real.
#
# novita/bf16 ($0.14/$0.40 per M) is the cheapest endpoint that serves the
# whole suite. The choice is measured, not read off the catalog: the cheaper
# deepinfra/turbo ($0.09/$0.34) rejects json_schema with "response format is
# not supported for model" — the catalog advertises `structured_outputs` for it
# and is wrong — and no DeepInfra variant of this model accepts an image
# content part, both answering 405. One endpoint for the whole suite is worth
# more than the $0.05/M, since it keeps every module testing the same server.
PROVIDER_TAG = "novita/bf16"
PROVIDER_NAME = "Novita"
PROVIDER_PIN = {"only": [PROVIDER_TAG], "allow_fallbacks": False}


@pytest.fixture
def served_model(require_api_key):
    """Factory for a live OpenRouter served model; overrides merge over the defaults."""
    require_api_key("OPENROUTER_API_KEY")

    def _make(**overrides):
        params = {
            "temperature": 0.0,
            "max_completion_tokens": 300,
            "provider_params": {"provider": PROVIDER_PIN},
        }
        params.update(overrides)
        return openrouter(model_id=MODEL_ID, **params)

    return _make
