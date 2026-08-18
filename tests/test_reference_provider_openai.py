"""The OpenAI provider page, pinned against the catalog and its capability profiles.

Template for the six remaining provider pages. A provider page makes claims a reader
cannot check — which models exist, which transport they use, which parameters they
accept — so every one of them is asserted here against `capabilities.py`.
"""

import inspect
import re
from pathlib import Path

import pytest

from datafast.llm.capabilities import (
    OPENAI_CHAT,
    OPENAI_RESPONSES,
    _SERVED_MODEL_CATALOG,
    resolve_capabilities,
)
from datafast.llm.served_model import openai
from datafast.llm.types import BatchMode, EndpointMode, Modality, StructuredOutputMode

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "reference" / "providers" / "openai.md"

CATALOGUED = sorted(m for (p, m) in _SERVED_MODEL_CATALOG if p == "openai")
REASONING_PREFIXES = ["gpt-5", "o1", "o3", "o4"]


def _page() -> str:
    return PAGE.read_text()


def test_the_documented_default_model_is_the_factory_default():
    default = inspect.signature(openai).parameters["model_id"].default
    assert default == "gpt-5.5"
    assert f"`{default}`" in _page()


def test_the_model_table_lists_every_catalogued_openai_model():
    assert CATALOGUED, "guard against an empty catalog scan"
    missing = [m for m in CATALOGUED if f"`{m}`" not in _page()]
    assert not missing, f"catalogued but absent from the model table: {missing}"


def test_the_page_claims_no_models_the_catalog_lacks():
    """A table row for a model that does not resolve as claimed is worse than none."""
    rows = re.findall(r"^\|\s*`(gpt-[\w.\-]+)`\s*\|\s*`(OPENAI_\w+)`", _page(), re.M)
    assert rows, "the model table should have rows"
    for model_id, profile in rows:
        expected = {"OPENAI_RESPONSES": OPENAI_RESPONSES, "OPENAI_CHAT": OPENAI_CHAT}[profile]
        assert resolve_capabilities("openai", model_id) is expected, (
            f"the page puts {model_id} on {profile}; the resolver disagrees"
        )


@pytest.mark.parametrize("prefix", REASONING_PREFIXES)
def test_documented_reasoning_prefixes_really_resolve_to_responses(prefix):
    assert f"`{prefix}`" in _page()
    capabilities = resolve_capabilities("openai", f"{prefix}-something-unreleased")
    assert capabilities is OPENAI_RESPONSES


@pytest.mark.parametrize("model_id", ["gpt-4o", "gpt-4.1"])
def test_the_models_the_page_calls_chat_models_are_chat_models(model_id):
    assert f"`{model_id}`" in _page()
    assert resolve_capabilities("openai", model_id) is OPENAI_CHAT


def test_the_profile_table_matches_the_real_supported_params():
    """The page's central claim: reasoning models reject sampling controls."""
    assert "temperature" not in OPENAI_RESPONSES.supported_params
    assert "top_p" not in OPENAI_RESPONSES.supported_params
    assert "temperature" in OPENAI_CHAT.supported_params
    for parameter in OPENAI_RESPONSES.supported_params | OPENAI_CHAT.supported_params:
        assert f"`{parameter}`" in _page(), f"{parameter} is accepted but undocumented"


def test_the_documented_transports_are_the_profile_defaults():
    assert OPENAI_RESPONSES.default_endpoint_mode is EndpointMode.RESPONSES
    assert OPENAI_CHAT.default_endpoint_mode is EndpointMode.CHAT
    # The page says both profiles support both transports.
    for profile in (OPENAI_RESPONSES, OPENAI_CHAT):
        assert profile.endpoint_modes == frozenset({EndpointMode.CHAT, EndpointMode.RESPONSES})


def test_the_documented_batch_modes_are_the_real_ones():
    assert OPENAI_CHAT.batch_mode is BatchMode.LITELLM_BATCH
    assert OPENAI_RESPONSES.batch_mode is BatchMode.FALLBACK_CONCURRENCY
    assert "native LiteLLM batch" in _page() and "fallback concurrency" in _page()


def test_the_documented_reasoning_off_value_is_the_one_that_is_sent():
    """The trap the page is built around: off is explicit, not omission."""
    assert OPENAI_RESPONSES.reasoning_off_param == ("reasoning_effort", "none")
    assert 'reasoning_effort="none"' in _page()
    assert OPENAI_CHAT.supports_reasoning is False


def test_the_documented_structured_output_mode_matches_both_profiles():
    for profile in (OPENAI_RESPONSES, OPENAI_CHAT):
        assert profile.structured_output is StructuredOutputMode.JSON_SCHEMA
    assert "`json_schema`" in _page()


def test_the_documented_modalities_match_both_profiles():
    expected = frozenset({Modality.TEXT, Modality.IMAGE, Modality.FILE})
    for profile in (OPENAI_RESPONSES, OPENAI_CHAT):
        assert profile.modalities == expected
    assert "text, image, file" in _page()


def test_the_per_model_default_effort_claim_comes_from_the_profile_notes():
    """The page states this as fact; the profile is where the fact is recorded."""
    notes = " ".join(OPENAI_RESPONSES.notes)
    assert "medium" in notes and "gpt-5.5" in notes
    assert "medium" in _page() and "gpt-5.4" in _page()


@pytest.mark.parametrize("block", re.findall(r"```python\n(.*?)```", PAGE.read_text(), re.DOTALL))
def test_every_example_constructs(block, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
    exec(compile(block, str(PAGE), "exec"), {})


def test_every_page_linked_to_exists():
    links = re.findall(r"\]\((?!https?:)([^)#]+\.md)", _page())
    missing = sorted(link for link in links if not (PAGE.parent / link).resolve().exists())
    assert not missing, f"links to pages that do not exist: {missing}"
