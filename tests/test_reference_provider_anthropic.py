"""The Anthropic provider page, pinned against the catalog and its capability profiles.

Built on tests/test_reference_provider_openai.py. Anthropic ships two profiles that
differ in ways a reader cannot check by eye — which parameters they accept, how
reasoning is turned off, which efforts are allowed — so every claim on the page is
asserted here against `capabilities.py`.
"""

import inspect
import re
from pathlib import Path

import pytest

from datafast.llm.capabilities import (
    ANTHROPIC_ADAPTIVE_CHAT,
    ANTHROPIC_CHAT,
    _SERVED_MODEL_CATALOG,
    resolve_capabilities,
)
from datafast.llm.served_model import anthropic
from datafast.llm.types import BatchMode, EndpointMode, Modality, StructuredOutputMode

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "reference" / "providers" / "anthropic.md"

CATALOGUED = sorted(m for (p, m) in _SERVED_MODEL_CATALOG if p == "anthropic")
PROFILES = {
    "ANTHROPIC_CHAT": ANTHROPIC_CHAT,
    "ANTHROPIC_ADAPTIVE_CHAT": ANTHROPIC_ADAPTIVE_CHAT,
}


def _page() -> str:
    return PAGE.read_text()


def _model_table_rows() -> list[tuple[str, str]]:
    rows = re.findall(r"^\|\s*`(claude-[\w.\-]+)`\s*\|\s*`(ANTHROPIC_\w+)`", _page(), re.M)
    assert rows, "the model table should have rows"
    return rows


def test_the_documented_default_model_is_the_factory_default():
    default = inspect.signature(anthropic).parameters["model_id"].default
    assert default == "claude-haiku-4-5"
    assert f"`{default}`" in _page()


def test_the_model_table_lists_every_catalogued_anthropic_model():
    assert CATALOGUED, "guard against an empty catalog scan"
    documented = {model_id for model_id, _ in _model_table_rows()}
    assert documented == set(CATALOGUED), (
        f"catalogued: {CATALOGUED}; documented in the model table: {sorted(documented)}"
    )


def test_every_model_the_table_names_resolves_to_the_profile_it_claims():
    for model_id, profile in _model_table_rows():
        assert resolve_capabilities("anthropic", model_id) is PROFILES[profile], (
            f"the page puts {model_id} on {profile}; the resolver disagrees"
        )


def test_the_two_profiles_really_are_different_objects():
    """The whole page rests on claude-sonnet-5 not sharing the provider default."""
    assert ANTHROPIC_ADAPTIVE_CHAT is not ANTHROPIC_CHAT
    assert resolve_capabilities("anthropic", "claude-sonnet-5") is ANTHROPIC_ADAPTIVE_CHAT
    for model_id in ("claude-haiku-4-5", "claude-sonnet-4-6"):
        assert resolve_capabilities("anthropic", model_id) is ANTHROPIC_CHAT


def test_an_uncatalogued_model_falls_back_to_the_provider_default():
    assert ("anthropic", "claude-opus-4-9") not in _SERVED_MODEL_CATALOG
    assert resolve_capabilities("anthropic", "claude-opus-4-9") is ANTHROPIC_CHAT
    assert "provider default" in _page()


def test_every_parameter_each_profile_accepts_is_documented():
    accepted = ANTHROPIC_CHAT.supported_params | ANTHROPIC_ADAPTIVE_CHAT.supported_params
    assert accepted, "guard against an empty parameter scan"
    for parameter in accepted:
        assert f"`{parameter}`" in _page(), f"{parameter} is accepted but undocumented"


def test_the_temperature_rule_differs_between_the_profiles():
    """The page's central claim: locked while reasoning on one, unsupported on the other."""
    assert "temperature" in ANTHROPIC_CHAT.supported_params
    assert ANTHROPIC_CHAT.reasoning_locks_temperature is True
    assert "temperature" not in ANTHROPIC_ADAPTIVE_CHAT.supported_params
    assert ANTHROPIC_ADAPTIVE_CHAT.reasoning_locks_temperature is False
    for profile in PROFILES.values():
        assert "top_p" not in profile.supported_params
        assert "frequency_penalty" not in profile.supported_params
    assert "`top_p` and `frequency_penalty` are on neither profile" in _page()


def test_the_documented_reasoning_off_values_are_the_ones_that_are_sent():
    """The trap the page is built around: off is explicit on sonnet-5 only."""
    assert ANTHROPIC_CHAT.reasoning_off_param is None
    assert ANTHROPIC_CHAT.reasoning_always_on is False
    assert ANTHROPIC_ADAPTIVE_CHAT.reasoning_off_param == ("thinking", {"type": "disabled"})
    assert 'thinking={"type": "disabled"}' in _page()
    assert 'reasoning_effort="none"' in _page()


def test_the_documented_accepted_efforts_match_both_profiles():
    assert ANTHROPIC_CHAT.reasoning_efforts is None  # any value, forwarded unchecked
    assert "any value, forwarded unchecked" in _page()

    efforts = ANTHROPIC_ADAPTIVE_CHAT.reasoning_efforts
    assert efforts == {"low", "medium", "high", "xhigh", "max"}
    for effort in efforts:
        assert f"`{effort}`" in _page(), f"{effort} is accepted but undocumented"
    for refused in ("none", "minimal"):
        assert refused not in efforts
        assert f"`{refused}`" in _page(), f"{refused} is refused but undocumented"


def test_thinking_true_asks_for_the_documented_effort_on_both_profiles():
    for profile in PROFILES.values():
        assert profile.supports_reasoning is True
        assert profile.reasoning_effort_on == "low"
    assert "effort `low` on both profiles" in _page()


def test_the_documented_transport_is_the_only_one_both_profiles_have():
    for profile in PROFILES.values():
        assert profile.default_endpoint_mode is EndpointMode.CHAT
        assert profile.endpoint_modes == frozenset({EndpointMode.CHAT})
    assert "Chat Completions, the only one" in _page()


def test_the_documented_batch_mode_is_the_real_one():
    for profile in PROFILES.values():
        assert profile.batch_mode is BatchMode.LITELLM_BATCH
    assert "native LiteLLM batch" in _page()


def test_the_documented_structured_output_mode_matches_both_profiles():
    for profile in PROFILES.values():
        assert profile.structured_output is StructuredOutputMode.JSON_SCHEMA
    assert "`json_schema`" in _page()


def test_the_documented_modalities_match_both_profiles():
    expected = frozenset({Modality.TEXT, Modality.IMAGE, Modality.FILE})
    for profile in PROFILES.values():
        assert profile.modalities == expected
    assert "text, image, file" in _page()


def test_the_empty_trace_claim_comes_from_the_profile_notes():
    """The page states this as fact; the profile is where the fact is recorded."""
    notes = " ".join(ANTHROPIC_ADAPTIVE_CHAT.notes)
    assert "thinking_blocks" in notes and "reasoning_content" in notes
    assert "thinking_blocks" in _page() and "reasoning_content" in _page()
    assert "max_completion_tokens" in notes and "10000" in notes
    assert "10000" in _page()


def test_the_documented_api_key_variable_is_the_one_the_factory_reads(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    model = anthropic()
    assert model.config.env_key_name == "ANTHROPIC_API_KEY"
    assert "`ANTHROPIC_API_KEY`" in _page()


@pytest.mark.parametrize("block", re.findall(r"```python\n(.*?)```", PAGE.read_text(), re.DOTALL))
def test_every_example_constructs(block, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    exec(compile(block, str(PAGE), "exec"), {})


def test_every_page_linked_to_exists():
    links = re.findall(r"\]\((?!https?:)([^)#]+\.md)", _page())
    assert links, "the page should link somewhere"
    missing = sorted(link for link in links if not (PAGE.parent / link).resolve().exists())
    assert not missing, f"links to pages that do not exist: {missing}"
