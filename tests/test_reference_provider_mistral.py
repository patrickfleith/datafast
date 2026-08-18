"""The Mistral provider page, pinned against the catalog and its capability profiles.

Same contract as `test_reference_provider_openai.py`: every claim the page makes that a
reader cannot check — which models exist, which profile they resolve to, which
parameters they accept, which reasoning efforts survive — is asserted against
`capabilities.py` and the `mistral()` factory.

Nothing here reaches the network. Examples are executed with the Files API and LiteLLM
stubbed out, and the stubs fail loudly if a call slips through.
"""

import inspect
import re
from pathlib import Path

import httpx
import litellm
import pytest

from datafast.llm.capabilities import (
    MISTRAL_CHAT,
    MISTRAL_REASONING_CHAT,
    _SERVED_MODEL_CATALOG,
    resolve_capabilities,
)
from datafast.llm.served_model import _MistralServedModel, mistral, openai
from datafast.llm.types import BatchMode, EndpointMode, Modality, StructuredOutputMode

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "reference" / "providers" / "mistral.md"

CATALOGUED = sorted(m for (p, m) in _SERVED_MODEL_CATALOG if p == "mistral")
PROFILES = {
    "MISTRAL_REASONING_CHAT": MISTRAL_REASONING_CHAT,
    "MISTRAL_CHAT": MISTRAL_CHAT,
}


def _page() -> str:
    return PAGE.read_text()


def test_the_documented_default_model_is_the_factory_default():
    default = inspect.signature(mistral).parameters["model_id"].default
    assert default == "mistral-small-2603"
    assert f"`{default}`" in _page()


def test_the_documented_api_key_variable_is_the_one_the_factory_reads():
    model = mistral()
    assert model.env_key_name == "MISTRAL_API_KEY"
    assert "`MISTRAL_API_KEY`" in _page()


def test_the_model_table_lists_every_catalogued_mistral_model():
    assert CATALOGUED, "guard against an empty catalog scan"
    missing = [m for m in CATALOGUED if f"`{m}`" not in _page()]
    assert not missing, f"catalogued but absent from the model table: {missing}"


def test_the_page_claims_no_model_that_resolves_elsewhere():
    """A table row for a model that does not resolve as claimed is worse than none."""
    rows = re.findall(r"^\|\s*`([\w.\-]+)`\s*\|\s*`(MISTRAL_\w+)`", _page(), re.M)
    assert len(rows) >= len(CATALOGUED), "the model table should have a row per model"
    for model_id, profile in rows:
        assert resolve_capabilities("mistral", model_id) is PROFILES[profile], (
            f"the page puts {model_id} on {profile}; the resolver disagrees"
        )


@pytest.mark.parametrize(
    "model_id",
    ["magistral-medium-2510", "ministral-8b-2512-reasoning", "mistral-small-reasoning"],
)
def test_uncatalogued_reasoning_names_resolve_to_the_reasoning_profile(model_id):
    """The name-matching rule: `magistral` or `-reasoning` anywhere in the id."""
    assert ("mistral", model_id) not in _SERVED_MODEL_CATALOG, "must be uncatalogued"
    assert resolve_capabilities("mistral", model_id) is MISTRAL_REASONING_CHAT
    assert "`magistral`" in _page() and "`-reasoning`" in _page()


@pytest.mark.parametrize("model_id", ["mistral-tiny", "pixtral-12b"])
def test_uncatalogued_plain_names_resolve_to_the_chat_profile(model_id):
    assert ("mistral", model_id) not in _SERVED_MODEL_CATALOG, "must be uncatalogued"
    assert resolve_capabilities("mistral", model_id) is MISTRAL_CHAT


def test_the_profile_table_matches_the_real_supported_params():
    """The page's central claim: reasoning_effort is the only difference."""
    assert MISTRAL_REASONING_CHAT.supported_params - MISTRAL_CHAT.supported_params == {
        "reasoning_effort"
    }
    assert not MISTRAL_CHAT.supported_params - MISTRAL_REASONING_CHAT.supported_params
    for parameter in MISTRAL_REASONING_CHAT.supported_params | MISTRAL_CHAT.supported_params:
        assert f"`{parameter}`" in _page(), f"{parameter} is accepted but undocumented"


def test_the_documented_transport_is_chat_only_on_both_profiles():
    for profile in (MISTRAL_REASONING_CHAT, MISTRAL_CHAT):
        assert profile.endpoint_modes == frozenset({EndpointMode.CHAT})
        assert profile.default_endpoint_mode is EndpointMode.CHAT
    assert "Chat Completions, always" in _page()


def test_the_documented_batch_mode_is_the_real_one():
    for profile in (MISTRAL_REASONING_CHAT, MISTRAL_CHAT):
        assert profile.batch_mode is BatchMode.LITELLM_BATCH
    assert "native LiteLLM batch" in _page()


def test_the_documented_structured_output_mode_matches_both_profiles():
    for profile in (MISTRAL_REASONING_CHAT, MISTRAL_CHAT):
        assert profile.structured_output is StructuredOutputMode.JSON_SCHEMA
    assert "`json_schema`" in _page()


def test_the_documented_modalities_match_both_profiles():
    expected = frozenset({Modality.TEXT, Modality.IMAGE, Modality.FILE})
    for profile in (MISTRAL_REASONING_CHAT, MISTRAL_CHAT):
        assert profile.modalities == expected
    assert "text, image, file" in _page()


def test_the_documented_reasoning_contract_is_the_profiles_reasoning_contract():
    assert MISTRAL_REASONING_CHAT.supports_reasoning is True
    assert MISTRAL_CHAT.supports_reasoning is False
    assert MISTRAL_REASONING_CHAT.reasoning_effort_on == "high"
    assert MISTRAL_REASONING_CHAT.reasoning_off_param == ("reasoning_effort", "none")
    assert 'reasoning_effort="high"' in _page()
    assert 'reasoning_effort="none"' in _page()


def test_the_documented_effort_allowlist_is_the_real_one():
    """The trap: only two values exist, and the page names both."""
    assert MISTRAL_REASONING_CHAT.reasoning_efforts == frozenset({"high", "none"})
    assert "`low` and `medium` are rejected" in _page()
    assert "HTTP 400" in _page()


@pytest.mark.parametrize("effort", ["low", "medium"])
def test_a_rejected_effort_raises_before_any_request_is_built(effort, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "not-a-real-key")
    model = mistral("mistral-medium-3-5", reasoning_effort=effort)
    with pytest.raises(ValueError, match="is not supported"):
        model._resolve_reasoning_effort()
    # The two allowed values pass the same check.
    for allowed in sorted(MISTRAL_REASONING_CHAT.reasoning_efforts):
        assert mistral("mistral-medium-3-5", reasoning_effort=allowed)._resolve_reasoning_effort()


def test_the_allowlist_escape_hatch_the_page_names_is_really_needed():
    assert MISTRAL_REASONING_CHAT.reasoning_requires_allowlist is True
    assert MISTRAL_CHAT.reasoning_requires_allowlist is False
    assert "allowed_openai_params" in _page()


def test_files_are_documented_as_id_only_because_both_profiles_say_so():
    for profile in (MISTRAL_REASONING_CHAT, MISTRAL_CHAT):
        assert profile.files_require_file_id is True
    assert "uploaded id only" in _page()


def test_the_documented_file_methods_exist_on_mistral_and_nowhere_else(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "not-a-real-key")
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    model = mistral()
    for name in ("upload_file", "delete_file"):
        assert callable(getattr(model, name))
        assert not hasattr(openai(), name), f"{name} is not Mistral-specific after all"
        assert f"`{name}(" in _page()
    signature = inspect.signature(_MistralServedModel.upload_file)
    assert signature.parameters["purpose"].default == "ocr"
    assert signature.parameters["expiry"].default is None
    assert '`upload_file(path, purpose="ocr", expiry=None)`' in _page()


@pytest.fixture
def offline(monkeypatch):
    """Make any network call — Files API or LiteLLM — a loud failure."""

    def boom(*args, **kwargs):
        raise AssertionError("a page example tried to reach the network")

    for name in ("post", "delete", "get", "put"):
        monkeypatch.setattr(httpx, name, boom)
    monkeypatch.setattr(litellm, "completion", boom)
    monkeypatch.setattr(
        _MistralServedModel, "upload_file", lambda self, *a, **k: "file-stub-id"
    )
    monkeypatch.setattr(_MistralServedModel, "delete_file", lambda self, file_id: None)


@pytest.mark.parametrize("block", re.findall(r"```python\n(.*?)```", PAGE.read_text(), re.DOTALL))
def test_every_example_runs(block, offline, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "not-a-real-key")
    exec(compile(block, str(PAGE), "exec"), {})


def test_every_page_linked_to_exists():
    links = re.findall(r"\]\((?!https?:)([^)#]+\.md)", _page())
    assert links, "the page should link somewhere"
    missing = sorted(link for link in links if not (PAGE.parent / link).resolve().exists())
    assert not missing, f"links to pages that do not exist: {missing}"
