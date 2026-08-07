import pytest

from datafast import anthropic

MODEL_ID = "claude-haiku-4-5"


@pytest.fixture
def served_model(require_api_key):
    """Factory for a live Anthropic served model; overrides merge over the defaults."""
    require_api_key("ANTHROPIC_API_KEY")

    def _make(**overrides):
        params = {"temperature": 0.0, "max_completion_tokens": 300}
        params.update(overrides)
        return anthropic(model_id=MODEL_ID, **params)

    return _make
