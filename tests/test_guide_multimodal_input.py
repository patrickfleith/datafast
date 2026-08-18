"""The multimodal input guide, pinned against the code it documents.

1. Every `ContentPart` field and every `Modality` member is named on the page
   (code → docs). A field that exists and is undocumented cannot be used.
2. Every self-contained example executes, against stubbed provider factories.
3. Each normalization and gating claim is proved by a real call — the page tables what
   a part becomes, so the tests build the part and compare.

No test reaches a provider: the factories the examples import are replaced, and the
served models built here are given explicit capabilities and never asked to generate.
"""

import dataclasses
import re
from pathlib import Path

import pytest

import datafast
from datafast.llm.served_model import (
    ServedModel,
    _modality_for_part,
    _normalize_content_part,
    _to_responses_input,
)
from datafast.llm.types import (
    ContentPart,
    EndpointMode,
    Modality,
    ServedModelCapabilities,
)

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "guides" / "multimodal_input.md"

ALL_MODALITIES = frozenset(Modality)


def _page() -> str:
    return PAGE.read_text()


def _code_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", _page(), re.DOTALL)


def _served_model(modalities=ALL_MODALITIES, *, file_id=False, endpoint=EndpointMode.CHAT):
    return ServedModel(
        provider_id="openai",
        model_id="fake",
        api_key="not-a-real-key",
        litellm_route="openai",
        env_key_name="OPENAI_API_KEY",
        capabilities=ServedModelCapabilities(
            endpoint_modes=frozenset({endpoint}),
            default_endpoint_mode=endpoint,
            modalities=frozenset(modalities),
            files_require_file_id=file_id,
        ),
    )


class StubModel:
    """Answers without a network call; stands in for every factory."""

    model_id = "stub"

    def generate(self, prompt=None, messages=None, **kwargs):
        return "stub answer"

    def upload_file(self, path, **kwargs):
        return "file-stub123"

    def delete_file(self, file_id):
        return None


# --- code → docs -----------------------------------------------------------------


def test_every_content_part_field_is_documented():
    names = [f.name for f in dataclasses.fields(ContentPart)]
    assert names, "ContentPart has no fields — check the test, not the page"
    missing = [n for n in names if f"`{n}`" not in _page()]
    assert not missing, f"ContentPart fields undocumented: {missing}"


def test_every_content_part_type_is_documented():
    """The Literal that types the field is the list of accepted values."""
    from typing import get_args

    from datafast.llm.types import ContentPartType

    types = get_args(ContentPartType)
    assert types, "no part types found — check the test, not the page"
    missing = [t for t in types if f'`"{t}"`' not in _page() and f"`{t}`" not in _page()]
    assert not missing, f"content part types undocumented: {missing}"


def test_every_modality_is_documented():
    values = [m.value for m in Modality]
    assert values
    missing = [v for v in values if v not in _page()]
    assert not missing, f"modalities undocumented: {missing}"


def test_the_upload_helpers_belong_to_mistral_alone():
    """The page says so explicitly; a base-class move would make it wrong."""
    from datafast.llm.served_model import _MistralServedModel

    for name in ("upload_file", "delete_file"):
        assert hasattr(_MistralServedModel, name), f"{name} no longer exists"
        assert not hasattr(ServedModel, name), f"{name} is no longer Mistral-only"
        assert f"{name}(" in _page()
    assert isinstance(datafast.mistral(), _MistralServedModel)


# --- examples --------------------------------------------------------------------


@pytest.mark.parametrize("block", _code_blocks(), ids=lambda b: b.split("\n")[0][:40])
def test_every_example_is_valid_python(block):
    compile(block, str(PAGE), "exec")


@pytest.mark.parametrize(
    "block",
    [b for b in _code_blocks() if "import" in b],
    ids=lambda b: b.split("\n")[0][:40],
)
def test_every_self_contained_example_executes(block, monkeypatch):
    """Factories are stubbed on the module, so each example's own import gets them."""
    for factory in ("openai", "anthropic", "gemini", "mistral", "ollama", "openrouter"):
        monkeypatch.setattr(datafast, factory, lambda *a, **k: StubModel())
    exec(compile(block, str(PAGE), "exec"), {})


def test_the_page_has_examples():
    blocks = _code_blocks()
    assert sum("import" in b for b in blocks) >= 5
    assert any("upload_file" in b for b in blocks)


def test_every_relative_link_resolves():
    links = re.findall(r"\]\((\.\./[^)#]+\.md|[a-z_]+\.md)\)", _page())
    assert links, "the page should link somewhere"
    for link in links:
        assert (PAGE.parent / link).resolve().exists(), f"broken link: {link}"


# --- what each part becomes ------------------------------------------------------


@pytest.mark.parametrize(
    "part,expected",
    [
        (
            ContentPart(type="text", text="hi"),
            {"type": "text", "text": "hi"},
        ),
        (
            ContentPart(type="image", url="https://example.com/cat.png"),
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
        ),
        (
            ContentPart(type="audio", data="AAAA", media_type="audio/wav"),
            {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
        ),
        (
            ContentPart(type="video", url="https://example.com/clip.mp4"),
            {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}},
        ),
        (
            ContentPart(type="file", url="file-abc123"),
            {"type": "file", "file": {"file_id": "file-abc123"}},
        ),
    ],
    ids=["text", "image", "audio", "video", "file"],
)
def test_each_part_becomes_the_shape_the_page_tables(part, expected):
    assert _normalize_content_part(part) == expected


def test_a_document_part_normalizes_to_a_file_part():
    """The page calls document an alias; the gate agrees."""
    assert _normalize_content_part(ContentPart(type="document", url="file-1")) == {
        "type": "file",
        "file": {"file_id": "file-1"},
    }
    assert _modality_for_part({"type": "document"}) is Modality.FILE


def test_bytes_are_wrapped_in_a_data_uri_using_media_type():
    part = _normalize_content_part(
        ContentPart(type="image", data="AAAA", media_type="image/png")
    )
    assert part["image_url"]["url"] == "data:image/png;base64,AAAA"


def test_a_complete_data_uri_is_passed_through_untouched():
    part = _normalize_content_part(
        ContentPart(type="image", data="data:image/png;base64,AAAA")
    )
    assert part["image_url"]["url"] == "data:image/png;base64,AAAA"


def test_bytes_without_a_media_type_raise():
    with pytest.raises(ValueError, match="need 'media_type'"):
        _normalize_content_part(ContentPart(type="image", data="AAAA"))


def test_an_image_with_neither_url_nor_data_raises():
    with pytest.raises(ValueError, match="require either 'url' or 'data'"):
        _normalize_content_part(ContentPart(type="image"))


def test_audio_has_no_url_form():
    with pytest.raises(ValueError, match="URL-only audio input"):
        _normalize_content_part(ContentPart(type="audio", url="https://x/y.wav"))


def test_audio_sends_a_bare_format_and_defaults_to_wav():
    from_mime = _normalize_content_part(
        ContentPart(type="audio", data="AAAA", media_type="audio/mp3")
    )
    assert from_mime["input_audio"]["format"] == "mp3"
    default = _normalize_content_part(ContentPart(type="audio", data="AAAA"))
    assert default["input_audio"]["format"] == "wav"


def test_a_file_with_data_is_sent_inline_with_its_filename():
    part = _normalize_content_part(
        ContentPart(
            type="file",
            data="AAAA",
            media_type="application/pdf",
            filename="report.pdf",
        )
    )
    assert part["file"] == {
        "file_data": "data:application/pdf;base64,AAAA",
        "filename": "report.pdf",
    }


def test_a_file_without_a_filename_simply_omits_it():
    """This is why the Responses trap surfaces at the provider, not here."""
    part = _normalize_content_part(
        ContentPart(type="file", data="AAAA", media_type="application/pdf")
    )
    assert "filename" not in part["file"]


def test_provider_options_are_merged_into_the_part_unchecked():
    part = _normalize_content_part(
        ContentPart(type="image", url="u", provider_options={"detail": "high"})
    )
    assert part["image_url"]["detail"] == "high"


def test_media_id_is_forwarded_only_where_the_provider_supports_it():
    part = ContentPart(type="image", url="u", media_id="m1")
    assert _normalize_content_part(part, include_media_uuid=True)["uuid"] == "m1"
    assert "uuid" not in _normalize_content_part(part, include_media_uuid=False)


def test_a_part_that_is_not_a_dict_or_content_part_raises():
    with pytest.raises(ValueError, match="must be dictionaries or ContentPart"):
        _normalize_content_part("just a string")


# --- the modality gate -----------------------------------------------------------


@pytest.mark.parametrize(
    "part_type,modality",
    [
        ("text", Modality.TEXT),
        ("image", Modality.IMAGE),
        ("image_url", Modality.IMAGE),
        ("audio", Modality.AUDIO),
        ("input_audio", Modality.AUDIO),
        ("video", Modality.VIDEO),
        ("video_url", Modality.VIDEO),
        ("file", Modality.FILE),
        ("document", Modality.FILE),
    ],
)
def test_each_part_type_gates_as_the_modality_the_page_says(part_type, modality):
    assert _modality_for_part({"type": part_type}) is modality


def test_an_unknown_part_type_is_treated_as_text():
    """The page warns that a typo sails through; this is the mechanism."""
    assert _modality_for_part({"type": "imgae"}) is Modality.TEXT
    passed_through = _normalize_content_part({"type": "imgae", "url": "u"})
    assert passed_through == {"type": "imgae", "url": "u"}


def test_an_unsupported_modality_raises_and_names_the_served_model():
    model = _served_model({Modality.TEXT})
    with pytest.raises(ValueError, match="Modality 'image' is not supported by"):
        model._prepare_messages(
            [{"role": "user", "content": [ContentPart(type="image", url="u")]}],
            response_format=None,
        )


def test_a_supported_modality_passes_the_gate():
    model = _served_model({Modality.TEXT, Modality.IMAGE})
    messages = model._prepare_messages(
        [{"role": "user", "content": [ContentPart(type="image", url="u")]}],
        response_format=None,
    )
    assert messages[0]["content"][0]["type"] == "image_url"


def test_the_gate_raises_rather_than_going_through_the_unsupported_params_policy():
    """The page contrasts the two; a policy that silenced the gate would break it."""
    from datafast.llm.types import UnsupportedParamsPolicy

    for policy in UnsupportedParamsPolicy:
        model = ServedModel(
            provider_id="openai",
            model_id="fake",
            api_key="k",
            litellm_route="openai",
            env_key_name="OPENAI_API_KEY",
            unsupported_params=policy,
            capabilities=ServedModelCapabilities(
                endpoint_modes=frozenset({EndpointMode.CHAT}),
                default_endpoint_mode=EndpointMode.CHAT,
                modalities=frozenset({Modality.TEXT}),
            ),
        )
        with pytest.raises(ValueError, match="is not supported by"):
            model._prepare_messages(
                [{"role": "user", "content": [ContentPart(type="image", url="u")]}],
                response_format=None,
            )


def test_plain_string_content_still_passes_through():
    model = _served_model({Modality.TEXT})
    assert model._prepare_messages(
        [{"role": "user", "content": "hello"}], response_format=None
    ) == [{"role": "user", "content": "hello"}]


# --- the two traps ---------------------------------------------------------------


def test_a_model_that_requires_a_file_id_refuses_inline_bytes():
    model = _served_model({Modality.TEXT, Modality.FILE}, file_id=True)
    with pytest.raises(ValueError, match="accepts a file only as an id"):
        model._prepare_messages(
            [
                {
                    "role": "user",
                    "content": [
                        ContentPart(
                            type="file",
                            data="AAAA",
                            media_type="application/pdf",
                            filename="a.pdf",
                        )
                    ],
                }
            ],
            response_format=None,
        )


def test_the_same_model_accepts_an_uploaded_id():
    model = _served_model({Modality.TEXT, Modality.FILE}, file_id=True)
    messages = model._prepare_messages(
        [{"role": "user", "content": [ContentPart(type="file", url="file-1")]}],
        response_format=None,
    )
    assert messages[0]["content"][0]["file"] == {"file_id": "file-1"}


def test_mistral_is_the_shipped_provider_that_requires_a_file_id():
    """The page names Mistral specifically; the profiles must still agree."""
    import datafast.llm.capabilities as capabilities

    requiring = sorted(
        name
        for name, value in vars(capabilities).items()
        if isinstance(value, ServedModelCapabilities) and value.files_require_file_id
    )
    assert requiring == ["MISTRAL_CHAT", "MISTRAL_REASONING_CHAT"]
    assert datafast.mistral().capabilities.files_require_file_id is True


# --- what each provider accepts --------------------------------------------------


@pytest.mark.parametrize(
    "factory,expected",
    [
        ("openai", {"text", "image", "file"}),
        ("anthropic", {"text", "image", "file"}),
        ("mistral", {"text", "image", "file"}),
        ("gemini", {"text", "image", "audio", "video", "file"}),
        ("ollama", {"text", "image"}),
        ("openrouter", {"text", "image"}),
    ],
)
def test_the_documented_provider_table_matches_the_profiles(factory, expected):
    model = getattr(datafast, factory)()
    assert {m.value for m in model.capabilities.modalities} == expected
    assert f"`{factory}()`" in _page()


def test_an_unprofiled_openai_compatible_server_is_text_only():
    model = datafast.openai_compatible(
        model_id="x", provider_id="unknown-server", api_base_url="http://localhost:8000/v1"
    )
    assert {m.value for m in model.capabilities.modalities} == {"text"}


def test_no_shipped_profile_declares_the_document_modality():
    """The page says DOCUMENT exists in the enum only."""
    import datafast.llm.capabilities as capabilities

    profiles = [
        value
        for value in vars(capabilities).values()
        if isinstance(value, ServedModelCapabilities)
    ]
    assert profiles, "no profiles found — check the test, not the page"
    assert all(Modality.DOCUMENT not in p.modalities for p in profiles)


def test_vllm_is_the_shipped_profile_that_forwards_a_media_uuid():
    import datafast.llm.capabilities as capabilities

    supporting = sorted(
        name
        for name, value in vars(capabilities).items()
        if isinstance(value, ServedModelCapabilities) and value.supports_media_uuid
    )
    assert supporting == ["VLLM_CHAT"]


# --- the responses endpoint ------------------------------------------------------


def test_parts_are_rewritten_again_for_the_responses_endpoint():
    model = _served_model(endpoint=EndpointMode.RESPONSES)
    normalized = model._prepare_messages(
        [
            {
                "role": "user",
                "content": [
                    ContentPart(type="text", text="hi"),
                    ContentPart(type="image", url="u"),
                    ContentPart(type="file", url="https://example.com/a.pdf"),
                    ContentPart(type="file", url="file-1"),
                ],
            }
        ],
        response_format=None,
    )
    assert [p["type"] for p in _to_responses_input(normalized)[0]["content"]] == [
        "input_text",
        "input_image",
        "input_file",
        "input_file",
    ]
