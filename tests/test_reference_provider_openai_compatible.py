"""The OpenAI-compatible page, pinned against the factory and its capability profiles.

This page documents a rule (`provider_id` names the server, never the wire format), a
normalization, and a capability fallback. All three are behaviour a reader cannot see,
so each is asserted here against the real factory and the real resolver. Nothing in this
file calls a served model — every test constructs and inspects.
"""

import inspect
import re
from pathlib import Path

import pytest

from datafast.llm.capabilities import (
    LLAMACPP_CHAT,
    OPENAI_COMPATIBLE_CHAT,
    VLLM_CHAT,
    resolve_capabilities,
)
from datafast.llm.served_model import _WIRE_FORMAT_IDS, openai_compatible
from datafast.llm.types import BatchMode, EndpointMode, Modality, StructuredOutputMode

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "reference" / "providers" / "openai_compatible.md"

BASE_URL = "http://localhost:8000/v1"

# (what the reader passes, what the page says it becomes)
NORMALIZATIONS = [
    (" vllm ", "vllm"),
    ("VLLM", "vllm"),
    ("my-gateway", "my_gateway"),
    ("llama.cpp", "llamacpp"),
    ("llama_cpp", "llamacpp"),
    ("LLAMA-CPP", "llamacpp"),
]


def _page() -> str:
    return PAGE.read_text()


def _code_spans() -> list[str]:
    """Every `backticked` span, so a claim must be written as code, not prose."""
    return re.findall(r"`([^`\n]+)`", _page())


def _build(provider_id: str, **kwargs):
    return openai_compatible("local-model", provider_id=provider_id, **kwargs)


def test_the_page_documents_every_parameter_of_the_factory():
    parameters = inspect.signature(openai_compatible).parameters
    assert set(parameters) == {"model_id", "provider_id", "api_base_url", "kwargs"}
    spans = _code_spans()
    for name, parameter in parameters.items():
        # The catch-all is documented under the name the signature row gives it.
        documented = "config" if parameter.kind is parameter.VAR_KEYWORD else name
        assert any(documented in span for span in spans), f"{name} is undocumented"


def test_provider_id_is_required():
    with pytest.raises(TypeError, match="provider_id"):
        openai_compatible("local-model", api_base_url=BASE_URL)
    assert "`provider_id` is required" in _page()


@pytest.mark.parametrize("wire_format", sorted(_WIRE_FORMAT_IDS))
def test_a_wire_format_provider_id_is_rejected(wire_format):
    """The rule the page is built around: provider_id names the server."""
    assert f"`{wire_format}`" in _page(), f"{wire_format} is rejected but undocumented"
    with pytest.raises(ValueError, match="names a wire format"):
        _build(wire_format, api_base_url=BASE_URL)


def test_the_page_lists_no_rejected_value_the_factory_accepts():
    rejected = re.findall(r"^\|\s*`([\w.]+)`\s*\|\s*a wire format", _page(), re.M)
    assert sorted(rejected) == sorted(_WIRE_FORMAT_IDS)


@pytest.mark.parametrize(("passed", "expected"), NORMALIZATIONS)
def test_the_documented_normalization_is_the_real_one(passed, expected):
    assert f'"{passed}"' in _page(), f"the page does not show {passed!r}"
    assert f"`{expected}`" in _page()
    model = _build(passed, api_base_url=BASE_URL)
    assert model.provider_id == expected


def test_an_unknown_server_name_survives_normalization_unchanged():
    """The page promises free-form ids, so an unheard-of server must stay itself."""
    assert _build("tgi", api_base_url=BASE_URL).provider_id == "tgi"


@pytest.mark.parametrize(
    ("provider_id", "profile"),
    [("vllm", VLLM_CHAT), ("llamacpp", LLAMACPP_CHAT)],
)
def test_a_known_server_gets_its_own_profile(provider_id, profile):
    assert resolve_capabilities(provider_id, "local-model", api_base_url=BASE_URL) is profile
    assert _build(provider_id, api_base_url=BASE_URL).capabilities is profile


def test_an_unknown_server_falls_back_to_the_conservative_profile():
    assert resolve_capabilities("my_gateway", "local-model", api_base_url=BASE_URL) is (
        OPENAI_COMPATIBLE_CHAT
    )
    assert _build("my_gateway", api_base_url=BASE_URL).capabilities is OPENAI_COMPATIBLE_CHAT
    assert "`OPENAI_COMPATIBLE_CHAT`" in _page()


def test_an_unknown_server_without_a_base_url_is_treated_more_cautiously():
    """The page tells the reader to pass api_base_url; this is what skipping it costs."""
    without = resolve_capabilities("my_gateway", "local-model")
    assert without is not OPENAI_COMPATIBLE_CHAT
    assert without.endpoint_modes == frozenset({EndpointMode.CHAT})
    assert OPENAI_COMPATIBLE_CHAT.endpoint_modes == frozenset(
        {EndpointMode.CHAT, EndpointMode.RESPONSES}
    )


def test_the_conservative_profile_accepts_only_the_documented_parameters():
    """The page's central warning: everything but timeout is assumed absent."""
    assert OPENAI_COMPATIBLE_CHAT.supported_params == frozenset({"timeout"})
    for parameter in OPENAI_COMPATIBLE_CHAT.supported_params:
        assert f"`{parameter}`" in _page(), f"{parameter} is accepted but undocumented"
    for parameter in ("temperature", "top_p", "frequency_penalty", "max_completion_tokens"):
        assert parameter not in OPENAI_COMPATIBLE_CHAT.supported_params
        assert f"`{parameter}`" in _page(), f"{parameter} is dropped but undocumented"


def test_the_conservative_profile_transports_batching_and_output_match_the_page():
    assert OPENAI_COMPATIBLE_CHAT.default_endpoint_mode is EndpointMode.CHAT
    assert OPENAI_COMPATIBLE_CHAT.batch_mode is BatchMode.FALLBACK_CONCURRENCY
    assert OPENAI_COMPATIBLE_CHAT.structured_output is StructuredOutputMode.PROMPTED_JSON
    assert "fallback concurrency" in _page()
    assert "`prompted_json`" in _page()


def test_the_conservative_profile_is_text_only():
    assert OPENAI_COMPATIBLE_CHAT.modalities == frozenset({Modality.TEXT})
    assert "Text only means an image part raises." in _page()


def test_the_vllm_and_llamacpp_rows_match_their_profiles():
    for profile in (VLLM_CHAT, LLAMACPP_CHAT):
        assert profile.structured_output is StructuredOutputMode.JSON_SCHEMA
        assert profile.batch_mode is BatchMode.FALLBACK_CONCURRENCY
        for parameter in profile.supported_params:
            assert f"`{parameter}`" in _page(), f"{parameter} is accepted but undocumented"
    assert VLLM_CHAT.supported_params == LLAMACPP_CHAT.supported_params  # the page says "same"
    assert VLLM_CHAT.endpoint_modes == frozenset({EndpointMode.CHAT, EndpointMode.RESPONSES})
    assert LLAMACPP_CHAT.endpoint_modes == frozenset({EndpointMode.CHAT})
    assert VLLM_CHAT.modalities == frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO})
    assert LLAMACPP_CHAT.modalities == frozenset({
        Modality.TEXT,
        Modality.IMAGE,
        Modality.AUDIO,
        Modality.VIDEO,
        Modality.FILE,
    })
    assert "text, image, video" in _page()
    assert "text, image, audio, video, file" in _page()


def test_no_api_key_is_required_or_invented(monkeypatch):
    """The page's API key row: none by default, and no environment variable is read."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
    model = _build("my_gateway", api_base_url=BASE_URL)
    assert model.env_key_name is None
    assert model.api_key is None
    assert model.capabilities.no_api_key is True
    assert _build("my_gateway", api_base_url=BASE_URL, api_key="local").api_key == "local"


def test_the_documented_transport_route_is_the_real_one():
    model = _build("my_gateway", api_base_url=BASE_URL)
    assert model._get_model_string() == "openai/local-model"
    assert model.api_base_url == BASE_URL
    assert "`openai/<model_id>`" in _page()


def test_provider_params_reach_the_request_unchecked():
    """The page's escape hatch for a server that does more than the profile assumes."""
    model = _build("my_gateway", api_base_url=BASE_URL, provider_params={"temperature": 0.7})
    assert model.config.provider_params == {"temperature": 0.7}
    assert "`provider_params`" in _page()


@pytest.mark.parametrize("block", re.findall(r"```python\n(.*?)```", PAGE.read_text(), re.DOTALL))
def test_every_example_constructs(block):
    assert "generate" not in block, "examples construct served models, never call them"
    exec(compile(block, str(PAGE), "exec"), {})


def test_every_page_linked_to_exists():
    links = re.findall(r"\]\((?!https?:)([^)#]+\.md)", _page())
    assert links, "the page should link somewhere"
    missing = sorted(link for link in links if not (PAGE.parent / link).resolve().exists())
    assert not missing, f"links to pages that do not exist: {missing}"
