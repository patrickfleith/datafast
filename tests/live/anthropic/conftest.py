import pytest

from datafast import anthropic

MODEL_ID = "claude-haiku-5-5"


@pytest.fixture
def served_model(require_api_key):
    """Factory for a live Anthropic served model; overrides merge over the defaults.

    Reasoning is off because thinking shares max_completion_tokens with the
    answer. Tests about reasoning turn it back on and raise the cap.
    """
    require_api_key("ANTHROPIC_API_KEY")

    def _make(model_id=MODEL_ID, **overrides):
        params = {"thinking": False, "max_completion_tokens": 300}
        params.update(overrides)
        return anthropic(model_id=model_id, **params)

    return _make
