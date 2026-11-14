"""The Ollama provider page, pinned against its capability profiles and resolver.

Same contract as the OpenAI page test: every claim a reader cannot check — which
names resolve to which profile, which parameters are accepted, how reasoning is
turned off, where the daemon lives — is asserted here against the source.

Nothing in this file contacts the Ollama daemon. Served models are constructed and
inspected only; `probe_capabilities()` is never called.
"""

import inspect
import re
from pathlib import Path

import pytest

from datafast.llm.capabilities import (
    OLLAMA_CHAT,
    OLLAMA_REASONING_CHAT,
    _OLLAMA_REASONING_MODELS,
    _SERVED_MODEL_CATALOG,
    resolve_capabilities,
)
from datafast.llm.served_model import _OllamaServedModel, ollama
from datafast.llm.types import BatchMode, EndpointMode, Modality, StructuredOutputMode

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "reference" / "providers" / "ollama.md"

PROFILES = {
    "OLLAMA_CHAT": OLLAMA_CHAT,
    "OLLAMA_REASONING_CHAT": OLLAMA_REASONING_CHAT,
}
ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`(OLLAMA_\w+)`", re.M)


def _page() -> str:
    return PAGE.read_text()


def test_the_documented_default_model_is_the_factory_default():
    default = inspect.signature(ollama).parameters["model_id"].default
    assert default == "gemma4:12b"
    assert f"`{default}`" in _page()


def test_the_catalog_really_lists_no_ollama_model():
    """The page's opening claim, and the reason it has no catalog table."""
    assert _SERVED_MODEL_CATALOG, "guard against an empty catalog scan"
    catalogued = [model for (provider, model) in _SERVED_MODEL_CATALOG if provider == "ollama"]
    assert not catalogued, f"the page says the catalog lists none; it lists {catalogued}"
    assert "catalog lists no Ollama model" in _page()


def test_the_page_names_every_reasoning_family_the_resolver_matches():
    assert _OLLAMA_REASONING_MODELS, "guard against an empty family scan"
    missing = [name for name in _OLLAMA_REASONING_MODELS if f"`{name}`" not in _page()]
    assert not missing, f"matched by the resolver but absent from the page: {missing}"


@pytest.mark.parametrize("name,profile", ROW.findall(PAGE.read_text()))
def test_every_name_the_page_tables_resolve_as_claimed(name, profile):
    assert resolve_capabilities("ollama", name) is PROFILES[profile], (
        f"the page puts {name} on {profile}; the resolver disagrees"
    )


def test_the_page_tables_are_not_empty():
    rows = ROW.findall(_page())
    assert len(rows) >= len(_OLLAMA_REASONING_MODELS), "the tables lost rows"
    assert {profile for _, profile in rows} == set(PROFILES), "both profiles should appear"


@pytest.mark.parametrize("family", _OLLAMA_REASONING_MODELS)
def test_a_family_name_anywhere_in_the_model_id_matches(family):
    """The page says the match is on what the name *contains*, tag included."""
    assert resolve_capabilities("ollama", f"{family}:8b") is OLLAMA_REASONING_CHAT
    assert resolve_capabilities("ollama", f"my-{family}-v2:latest") is OLLAMA_REASONING_CHAT


@pytest.mark.parametrize("model_id", ["gemma3:4b", "llama3.2", "llama3.2-vision"])
def test_a_name_with_no_family_marker_gets_the_plain_chat_profile(model_id):
    assert f"`{model_id}`" in _page()
    assert resolve_capabilities("ollama", model_id) is OLLAMA_CHAT


def test_the_documented_parameters_are_the_ones_the_profiles_accept():
    for profile in (OLLAMA_CHAT, OLLAMA_REASONING_CHAT):
        for parameter in profile.supported_params:
            assert f"`{parameter}`" in _page(), f"{parameter} is accepted but undocumented"
    assert OLLAMA_REASONING_CHAT.supported_params == (
        OLLAMA_CHAT.supported_params | {"reasoning_effort"}
    )


def test_frequency_penalty_is_genuinely_absent_from_both_profiles():
    """The page's central warning: repeat_penalty is not frequency_penalty."""
    for profile in (OLLAMA_CHAT, OLLAMA_REASONING_CHAT):
        assert "frequency_penalty" not in profile.supported_params
        assert "top_p" in profile.supported_params, "guard: sampling params are not all gone"
    assert "`repeat_penalty`" in _page()


def test_the_documented_reasoning_controls_are_the_real_ones():
    assert OLLAMA_CHAT.supports_reasoning is False
    assert OLLAMA_REASONING_CHAT.supports_reasoning is True
    assert OLLAMA_REASONING_CHAT.reasoning_off_param == ("think", False)
    assert OLLAMA_REASONING_CHAT.reasoning_effort_on == "low"
    assert "`think=False`" in _page()
    assert '`reasoning_effort="low"`' in _page()


def test_the_documented_transport_is_the_only_one_both_profiles_support():
    for profile in (OLLAMA_CHAT, OLLAMA_REASONING_CHAT):
        assert profile.endpoint_modes == frozenset({EndpointMode.CHAT})
        assert profile.default_endpoint_mode is EndpointMode.CHAT
    assert ollama().config.litellm_route == "ollama_chat"
    assert "`ollama_chat`" in _page()


def test_the_documented_batch_mode_is_the_real_one():
    for profile in (OLLAMA_CHAT, OLLAMA_REASONING_CHAT):
        assert profile.batch_mode is BatchMode.FALLBACK_CONCURRENCY
    assert "fallback concurrency" in _page()


def test_the_documented_structured_output_mode_matches_both_profiles():
    for profile in (OLLAMA_CHAT, OLLAMA_REASONING_CHAT):
        assert profile.structured_output is StructuredOutputMode.JSON_SCHEMA
    assert "`json_schema`" in _page()


def test_the_documented_modalities_match_both_profiles():
    """Image is declared for every model, which is exactly the page's warning."""
    expected = frozenset({Modality.TEXT, Modality.IMAGE})
    for profile in (OLLAMA_CHAT, OLLAMA_REASONING_CHAT):
        assert profile.modalities == expected
    assert "text, image" in _page()


def test_ollama_really_needs_no_api_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    for profile in (OLLAMA_CHAT, OLLAMA_REASONING_CHAT):
        assert profile.no_api_key is True
    model = ollama()
    assert model.env_key_name is None
    assert model.api_key is None


def test_the_documented_base_url_default_and_precedence(monkeypatch):
    assert _OllamaServedModel.DEFAULT_API_BASE == "http://localhost:11434"
    assert "`http://localhost:11434`" in _page()
    assert "`OLLAMA_API_BASE`" in _page()

    monkeypatch.delenv("OLLAMA_API_BASE", raising=False)
    assert ollama()._resolved_api_base() == "http://localhost:11434"

    monkeypatch.setenv("OLLAMA_API_BASE", "http://box:11434")
    assert ollama()._resolved_api_base() == "http://box:11434"
    assert ollama(api_base_url="http://other:11434")._resolved_api_base() == "http://other:11434"


def test_the_probe_exists_and_returns_the_names_the_page_lists():
    probe = _OllamaServedModel.probe_capabilities
    assert callable(probe)
    documented = ["completion", "vision", "audio", "thinking", "tools", "embedding", "insert"]
    for name in documented:
        assert f"`{name}`" in _page()
        assert name in probe.__doc__, f"{name} is documented but not a name the probe returns"


@pytest.mark.parametrize("block", re.findall(r"```python\n(.*?)```", PAGE.read_text(), re.DOTALL))
def test_every_example_constructs(block):
    exec(compile(block, str(PAGE), "exec"), {})


def test_every_page_linked_to_exists():
    links = re.findall(r"\]\((?!https?:)([^)#]+\.md)", _page())
    assert links, "the page should link somewhere"
    missing = sorted(link for link in links if not (PAGE.parent / link).resolve().exists())
    assert not missing, f"links to pages that do not exist: {missing}"
