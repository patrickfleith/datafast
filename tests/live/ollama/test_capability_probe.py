"""Live capability probe against the Ollama daemon.

`probe_capabilities()` exists because an Ollama model's capabilities are a
property of the machine rather than of the id, so `resolve_capabilities` can only
guess from the name. The mocked tests pin the request shape; these check the
guess against the answer, on two models chosen to disagree with it.

The disagreement is the finding worth keeping under test: both Ollama profiles
declare `Modality.IMAGE` unconditionally, so datafast will accept an image part
for a model that cannot see. Asserting it here means a future fix — deriving
modalities from the probe — fails this test loudly instead of passing silently.
"""

import httpx
import pytest

from datafast import ollama
from datafast.llm.types import Modality

pytestmark = [pytest.mark.live, pytest.mark.ollama]


def test_probe_confirms_the_resolved_reasoning_support(served_model):
    """qwen3 matches the reasoning heuristic by name; the daemon agrees."""
    model = served_model()

    assert "thinking" in model.probe_capabilities()
    assert model.capabilities.supports_reasoning is True


def test_probe_contradicts_the_declared_image_modality(served_model):
    """The default model has no vision, yet the profile says images are fine."""
    model = served_model()

    assert "vision" not in model.probe_capabilities()
    assert Modality.IMAGE in model.capabilities.modalities


def test_probe_reports_vision_for_the_vision_model(vision_served_model):
    assert "vision" in vision_served_model().probe_capabilities()


def test_probe_fails_loudly_for_a_model_that_is_not_pulled(require_ollama):
    """A missing model is an HTTP error from the daemon, not an empty capability
    set — otherwise a typo would look like a model that can simply do nothing."""
    require_ollama()

    with pytest.raises(httpx.HTTPStatusError):
        ollama("definitely-not-a-real-model:0b").probe_capabilities()
