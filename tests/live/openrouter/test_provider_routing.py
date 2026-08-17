"""Live provider routing against OpenRouter.

OpenRouter is the one provider where a model id does not determine who serves
the request, which is what makes `provider_params` load-bearing here rather
than an escape hatch. The rest of the suite assumes the pin in `conftest.py`
holds; these two tests are what turn that assumption into a measurement.
"""

import pytest

from .conftest import PROVIDER_NAME

pytestmark = [pytest.mark.live, pytest.mark.openrouter]


def test_provider_pin_is_honoured(served_model):
    """`provider_params` is applied last and unchecked, so the only proof it
    reached OpenRouter is the routed provider coming back in the response."""
    response = served_model().generate_response(
        prompt="What is the capital of France? Answer in one word."
    )

    assert "Paris" in response.text
    assert getattr(response.raw, "provider", None) == PROVIDER_NAME


def test_unroutable_pin_fails_instead_of_falling_back(served_model):
    """`allow_fallbacks: False` is the half of the pin that matters: without it
    OpenRouter would quietly serve an unavailable endpoint from somewhere else,
    and the suite would go on testing a provider it did not choose."""
    model = served_model(
        provider_params={
            "provider": {"only": ["nonexistent/provider"], "allow_fallbacks": False}
        }
    )

    with pytest.raises(RuntimeError):
        model.generate(prompt="What is the capital of France? Answer in one word.")
