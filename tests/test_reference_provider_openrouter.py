"""The OpenRouter provider page, pinned against the resolver and its capability profile.

Follows tests/test_reference_provider_openai.py. A provider page makes claims a reader
cannot check — which models exist, which transport they use, which parameters they
accept — so every one of them is asserted here against `capabilities.py`.
"""

import inspect
import re
from pathlib import Path

import pytest

from datafast.llm.capabilities import (
    OPENROUTER_CHAT,
    _SERVED_MODEL_CATALOG,
    resolve_capabilities,
)
from datafast.llm.served_model import openrouter
from datafast.llm.types import BatchMode, EndpointMode, Modality, StructuredOutputMode

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "reference" / "providers" / "openrouter.md"

CATALOGUED = sorted(m for (p, m) in _SERVED_MODEL_CATALOG if p == "openrouter")


def _page() -> str:
    return PAGE.read_text()


def test_the_documented_default_model_is_the_factory_default():
    default = inspect.signature(openrouter).parameters["model_id"].default
    assert default == "openai/gpt-5.4-mini"
    assert f"`{default}`" in _page()


def test_the_page_documents_the_catalog_as_it_really_is():
    """The catalog has no OpenRouter entry, which is itself the page's claim.

    The loop is what keeps this honest if one is ever added: a catalogued model
    that the page never names would be a silent gap.
    """
    assert CATALOGUED == [], "the catalog gained OpenRouter entries; document them"
    assert "No OpenRouter model is catalogued" in _page()
    missing = [m for m in CATALOGUED if f"`{m}`" not in _page()]
    assert not missing, f"catalogued but absent from the model table: {missing}"


def test_every_model_the_page_names_resolves_to_the_profile_it_claims():
    rows = re.findall(
        r"^\|\s*`([\w.\-]+/[\w.\-]+)`\s*\|\s*`(OPENROUTER_\w+)`\s*\|", _page(), re.M
    )
    assert len(rows) >= 3, "the model table should have rows"
    for model_id, profile_name in rows:
        assert profile_name == "OPENROUTER_CHAT", f"unknown profile named: {profile_name}"
        assert resolve_capabilities("openrouter", model_id) is OPENROUTER_CHAT, (
            f"the page puts {model_id} on {profile_name}; the resolver disagrees"
        )


@pytest.mark.parametrize(
    "model_id",
    ["nobody/never-shipped-this", "moonshotai/kimi-k9", "openai/gpt-5.4-mini"],
)
def test_an_uncatalogued_model_id_still_resolves_to_the_provider_default(model_id):
    """The page's claim that *anything else* lands on the same profile."""
    assert "| anything else | `OPENROUTER_CHAT` |" in _page()
    assert resolve_capabilities("openrouter", model_id) is OPENROUTER_CHAT


def test_the_documented_parameters_are_the_ones_the_profile_accepts():
    accepted = OPENROUTER_CHAT.supported_params
    assert accepted, "guard against an empty parameter scan"
    for parameter in accepted:
        assert f"`{parameter}`" in _page(), f"{parameter} is accepted but undocumented"
    # The page says reasoning is not accepted; the profile is where that is decided.
    assert "reasoning_effort" not in accepted
    assert OPENROUTER_CHAT.supports_reasoning is False
    assert OPENROUTER_CHAT.reasoning_off_param is None


def test_the_documented_transport_is_chat_only():
    assert OPENROUTER_CHAT.default_endpoint_mode is EndpointMode.CHAT
    assert OPENROUTER_CHAT.endpoint_modes == frozenset({EndpointMode.CHAT})
    assert OPENROUTER_CHAT.supports_endpoint(EndpointMode.RESPONSES) is False
    assert "Chat Completions only" in _page()


def test_the_documented_batch_mode_is_the_real_one():
    assert OPENROUTER_CHAT.batch_mode is BatchMode.LITELLM_BATCH
    assert "native LiteLLM batch" in _page()


def test_the_documented_structured_output_mode_matches_the_profile():
    assert OPENROUTER_CHAT.structured_output is StructuredOutputMode.JSON_SCHEMA
    assert "`json_schema`" in _page()


def test_the_documented_modalities_match_the_profile():
    assert OPENROUTER_CHAT.modalities == frozenset({Modality.TEXT, Modality.IMAGE})
    assert "text, image" in _page()


def test_a_file_part_raises_rather_than_being_dropped(monkeypatch):
    """The page says a file part stops the call locally. This is that check.

    Only the local message validation runs — nothing is sent anywhere.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-not-a-real-key")
    assert Modality.FILE not in OPENROUTER_CHAT.modalities

    model = openrouter()
    messages = [{"role": "user", "content": [{"type": "file", "file": {"file_data": "x"}}]}]

    with pytest.raises(ValueError) as excinfo:
        model._validate_modalities(messages)

    assert "Modality 'file' is not supported by openrouter/" in str(excinfo.value)
    assert "Modality 'file' is not supported by openrouter/" in _page().replace(
        "<model>", ""
    )


def test_the_page_documents_the_api_key_variable():
    env_key_name = openrouter().config.env_key_name
    assert env_key_name == "OPENROUTER_API_KEY"
    assert f"`{env_key_name}`" in _page()


def test_the_vendor_prefix_survives_into_the_litellm_model_string():
    """The page insists the prefix is part of the id, not something datafast adds."""
    model = openrouter("z-ai/glm-4.6")
    assert model._get_model_string() == "openrouter/z-ai/glm-4.6"
    assert "`z-ai/glm-4.6`" in _page()


@pytest.mark.parametrize("block", re.findall(r"```python\n(.*?)```", PAGE.read_text(), re.DOTALL))
def test_every_example_constructs(block, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-not-a-real-key")
    exec(compile(block, str(PAGE), "exec"), {})


def test_every_page_linked_to_exists():
    links = re.findall(r"\]\((?!https?:)([^)#]+\.md)", _page())
    assert links, "the page should link somewhere"
    missing = sorted(link for link in links if not (PAGE.parent / link).resolve().exists())
    assert not missing, f"links to pages that do not exist: {missing}"
