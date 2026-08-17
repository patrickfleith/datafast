import pytest

from datafast import gemini

# The suite runs on two models because Gemini's reasoning floor is per-model,
# not per-provider. gemini-3.5-flash-lite is the cheap workhorse and defaults
# to minimal thinking, so the generation, structured-output and multimodal
# modules do not pay for reasoning they never assert on.
MODEL_ID = "gemini-3.5-flash-lite"
# Unlike the other suites, no temperature is pinned. Gemini 3 warns that any
# value below 1.0 risks infinite loops and degraded reasoning, and datafast
# sends nothing unless a caller asks, so the provider default stands. Every
# assertion here keys on a short factual answer that survives it.
# gemini-3.7-flash is the current Flash line and the reason
# GEMINI_NO_MINIMAL_CHAT exists: it rejects 'minimal', so thinking=False has no
# value to send. Only the reasoning module uses it.
FLASH_MODEL_ID = "gemini-3.7-flash"


@pytest.fixture
def served_model(require_api_key):
    """Factory for a live Gemini served model; overrides merge over the defaults."""
    require_api_key("GEMINI_API_KEY")

    def _make(**overrides):
        params = {"max_completion_tokens": 300}
        params.update(overrides)
        return gemini(model_id=MODEL_ID, **params)

    return _make


@pytest.fixture
def flash_served_model(require_api_key):
    """The same factory on gemini-3.7-flash, whose reasoning floor is 'low'."""
    require_api_key("GEMINI_API_KEY")

    def _make(**overrides):
        params = {"max_completion_tokens": 300}
        params.update(overrides)
        return gemini(model_id=FLASH_MODEL_ID, **params)

    return _make
