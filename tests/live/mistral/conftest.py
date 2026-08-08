import pytest

from datafast import mistral

# mistral-small-2603 is the factory default and the cheapest catalog entry that
# resolves to MISTRAL_REASONING_CHAT — the profile that needs the allowlist
# escape hatch to get reasoning_effort past LiteLLM's per-model param filter.
MODEL_ID = "mistral-small-2603"


@pytest.fixture
def served_model(require_api_key):
    """Factory for a live Mistral served model; overrides merge over the defaults."""
    require_api_key("MISTRAL_API_KEY")

    def _make(**overrides):
        params = {"temperature": 0.0, "max_completion_tokens": 300}
        params.update(overrides)
        return mistral(model_id=MODEL_ID, **params)

    return _make
