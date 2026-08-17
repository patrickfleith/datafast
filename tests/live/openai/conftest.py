import pytest

from datafast import openai

MODEL_ID = "gpt-5.4-mini"
# gpt-4o-mini is the only way to reach OPENAI_CHAT: every openai id in the
# catalog resolves to OPENAI_RESPONSES.
CHAT_MODEL_ID = "gpt-4o-mini"


@pytest.fixture
def served_model(require_api_key):
    """Factory for a live OpenAI served model; overrides merge over the defaults.

    No temperature default, unlike the Anthropic fixture: RESPONSES_PARAMS omits
    temperature, so setting one would warn on every single call.
    """
    require_api_key("OPENAI_API_KEY")

    def _make(**overrides):
        params = {"max_completion_tokens": 300}
        params.update(overrides)
        return openai(model_id=MODEL_ID, **params)

    return _make


@pytest.fixture
def chat_served_model(require_api_key):
    """Factory for a served model on the OPENAI_CHAT profile, where temperature
    is supported and batching is native."""
    require_api_key("OPENAI_API_KEY")

    def _make(**overrides):
        params = {"temperature": 0.0, "max_completion_tokens": 300}
        params.update(overrides)
        return openai(model_id=CHAT_MODEL_ID, **params)

    return _make
