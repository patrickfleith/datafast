"""The Gemini provider page, pinned against the catalog and its capability profiles.

Same contract as the OpenAI page test: every claim a reader cannot check — which models
exist, which profile they get, which parameters they accept, what "reasoning off"
actually sends — is asserted here against `capabilities.py` and `served_model.py`.

Nothing here calls a provider. Served models are constructed, and the one behavioural
check builds the request parameters offline.
"""

import inspect
import re
import tomllib
from pathlib import Path

import pytest

from datafast.llm.capabilities import (
    GEMINI_CHAT,
    GEMINI_NO_MINIMAL_CHAT,
    _SERVED_MODEL_CATALOG,
    resolve_capabilities,
)
from datafast.llm.served_model import gemini
from datafast.llm.types import (
    BatchMode,
    EndpointMode,
    Modality,
    NormalizedRequest,
    StructuredOutputMode,
)

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "reference" / "providers" / "gemini.md"

CATALOGUED = sorted(m for (p, m) in _SERVED_MODEL_CATALOG if p == "gemini")
PROFILES = {"GEMINI_CHAT": GEMINI_CHAT, "GEMINI_NO_MINIMAL_CHAT": GEMINI_NO_MINIMAL_CHAT}
REQUEST = NormalizedRequest(messages=[{"role": "user", "content": "hi"}])


def _page() -> str:
    return PAGE.read_text()


def test_the_documented_default_model_is_the_factory_default():
    default = inspect.signature(gemini).parameters["model_id"].default
    assert default == "gemini-3.5-flash-lite"
    assert f"`{default}`" in _page()


def test_the_model_table_lists_every_catalogued_gemini_model():
    assert CATALOGUED, "guard against an empty catalog scan"
    missing = [m for m in CATALOGUED if f"`{m}`" not in _page()]
    assert not missing, f"catalogued but absent from the model table: {missing}"


def test_the_page_claims_no_models_the_catalog_lacks():
    """A table row for a model that does not resolve as claimed is worse than none."""
    rows = re.findall(r"^\|\s*`(gemini-[\w.\-]+)`\s*\|\s*`(GEMINI_\w+)`", _page(), re.M)
    assert len(rows) == len(CATALOGUED), "every catalogued model needs a table row"
    for model_id, profile in rows:
        assert resolve_capabilities("gemini", model_id) is PROFILES[profile], (
            f"the page puts {model_id} on {profile}; the resolver disagrees"
        )


@pytest.mark.parametrize("model_id", ["gemini-4-pro", "some-unreleased-model"])
def test_an_uncatalogued_model_falls_back_to_the_provider_default(model_id):
    assert (model_id, "gemini") not in {(m, p) for (p, m) in _SERVED_MODEL_CATALOG}
    assert resolve_capabilities("gemini", model_id) is GEMINI_CHAT
    assert "provider default" in _page()


def test_the_documented_api_key_variable_is_the_one_the_factory_reads():
    assert gemini().config.env_key_name == "GEMINI_API_KEY"
    assert "`GEMINI_API_KEY`" in _page()


def test_the_documented_transport_is_chat_only_on_both_profiles():
    for profile in PROFILES.values():
        assert profile.default_endpoint_mode is EndpointMode.CHAT
        assert profile.endpoint_modes == frozenset({EndpointMode.CHAT})
    assert "Chat Completions" in _page()


def test_google_generativeai_is_not_required():
    """The page tells readers to install nothing extra, so nothing may import it."""
    assert "`google-generativeai` is not required" in _page()

    dependencies = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    assert dependencies, "guard against an empty dependency scan"
    assert not [d for d in dependencies if "google" in d or "generativeai" in d]

    sources = list((ROOT / "datafast").rglob("*.py"))
    assert sources, "guard against an empty source scan"
    importers = [p.name for p in sources if "import google" in p.read_text()]
    assert not importers, f"these modules import the Google SDK: {importers}"


def test_the_profile_table_documents_every_accepted_parameter():
    accepted = GEMINI_CHAT.supported_params | GEMINI_NO_MINIMAL_CHAT.supported_params
    assert accepted, "guard against an empty parameter set"
    for parameter in accepted:
        assert f"`{parameter}`" in _page(), f"{parameter} is accepted but undocumented"
    # The page says both profiles take the same six parameters.
    assert GEMINI_CHAT.supported_params == GEMINI_NO_MINIMAL_CHAT.supported_params
    assert len(accepted) == 6


def test_the_documented_batch_mode_is_the_real_one():
    for profile in PROFILES.values():
        assert profile.batch_mode is BatchMode.LITELLM_BATCH
    assert "native LiteLLM batch" in _page()


def test_the_documented_structured_output_mode_matches_both_profiles():
    for profile in PROFILES.values():
        assert profile.structured_output is StructuredOutputMode.JSON_SCHEMA
    assert "`json_schema`" in _page()


def test_the_documented_modalities_match_both_profiles():
    expected = frozenset(
        {Modality.TEXT, Modality.IMAGE, Modality.AUDIO, Modality.VIDEO, Modality.FILE}
    )
    for profile in PROFILES.values():
        assert profile.modalities == expected
    assert "text, image, audio, video, file" in _page()


def test_the_documented_reasoning_off_value_is_the_one_that_is_sent():
    """The trap the page is built around: off is explicit, not omission."""
    assert GEMINI_CHAT.supports_reasoning is True
    assert GEMINI_CHAT.reasoning_off_param == ("reasoning_effort", "none")
    assert GEMINI_CHAT.reasoning_always_on is False
    assert 'reasoning_effort="none"' in _page()

    sent = gemini(thinking=False)._build_chat_params(REQUEST, None)
    assert sent["reasoning_effort"] == "none"


def test_the_model_with_a_minimum_effort_refuses_thinking_off():
    assert GEMINI_NO_MINIMAL_CHAT.reasoning_off_param is None
    assert GEMINI_NO_MINIMAL_CHAT.reasoning_always_on is True

    model = gemini("gemini-3.7-flash", thinking=False)
    with pytest.raises(ValueError, match="always reasons"):
        model._build_chat_params(REQUEST, None)


def test_the_documented_accepted_efforts_are_the_real_ones():
    assert GEMINI_CHAT.reasoning_efforts is None  # any value, forwarded unchecked
    assert "forwarded unchecked" in _page()

    efforts = GEMINI_NO_MINIMAL_CHAT.reasoning_efforts
    assert efforts == frozenset({"low", "medium", "high"})
    for effort in efforts:
        assert f"`{effort}`" in _page()
    assert "minimal" not in efforts
    assert "`minimal`" in _page()


def test_the_documented_thinking_on_effort_is_the_real_one():
    for profile in PROFILES.values():
        assert profile.reasoning_effort_on == "low"
    assert 'reasoning_effort="low"' in _page()

    sent = gemini(thinking=True)._build_chat_params(REQUEST, None)
    assert sent["reasoning_effort"] == "low"


def test_the_per_model_minimum_claims_come_from_the_profile_notes():
    """The page states these as fact; the profiles are where the facts are recorded."""
    chat_notes = " ".join(GEMINI_CHAT.notes)
    assert "think by" in chat_notes and "billed" in chat_notes
    minimal_notes = " ".join(GEMINI_NO_MINIMAL_CHAT.notes)
    assert "'medium' for gemini-3.7-flash" in minimal_notes
    assert "which is `medium`" in _page()


@pytest.mark.parametrize("block", re.findall(r"```python\n(.*?)```", PAGE.read_text(), re.DOTALL))
def test_every_example_constructs(block, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    exec(compile(block, str(PAGE), "exec"), {})


def test_every_page_linked_to_exists():
    links = re.findall(r"\]\((?!https?:)([^)#]+\.md)", _page())
    assert links, "the page should link somewhere"
    missing = sorted(link for link in links if not (PAGE.parent / link).resolve().exists())
    assert not missing, f"links to pages that do not exist: {missing}"
