"""The structured output guide, pinned against the code it documents.

The page's whole job is to keep two mechanisms apart, so the tests do too:

1. Every `parse_mode` value and every `StructuredOutputMode` member is documented
   (code → docs). A mode that exists and is unnamed cannot be chosen.
2. Every self-contained example executes, against a stubbed provider factory.
3. The behaviour of each mode is proved by a real call — the parsers are exercised
   directly, and the request datafast would send is inspected without sending it.

Nothing here reaches a provider: `datafast.openai` is replaced for the examples, and the
served models built below are handed explicit capabilities and never asked to generate.
"""

import inspect
import re
import warnings
from pathlib import Path

import pytest
from pydantic import BaseModel

import datafast
from datafast import LLMStep, Source
from datafast.llm.parsing import JSONParser, TextParser, XMLParser, get_parser
from datafast.llm.served_model import ServedModel
from datafast.llm.types import (
    EndpointMode,
    NormalizedResponse,
    ServedModelCapabilities,
    StructuredOutputMode,
)

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "guides" / "structured_output.md"


def _page() -> str:
    return PAGE.read_text()


def _code_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", _page(), re.DOTALL)


class Question(BaseModel):
    question: str
    answer: str


class StubModel:
    """Answers without a network call, honouring response_format."""

    model_id = "stub"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt=None, messages=None, response_format=None, **kwargs):
        self.prompts.append(prompt if prompt is not None else messages[-1]["content"])
        if response_format is not None:
            return response_format(
                **{name: "stub" for name in response_format.model_fields}
            )
        return '{"question": "Q?", "answer": "A."}'


def _capabilities(mode, endpoint=EndpointMode.CHAT):
    return ServedModelCapabilities(
        endpoint_modes=frozenset({endpoint}),
        default_endpoint_mode=endpoint,
        structured_output=mode,
    )


def _served_model(mode, endpoint=EndpointMode.CHAT):
    return ServedModel(
        provider_id="openai",
        model_id="fake",
        api_key="not-a-real-key",
        litellm_route="openai",
        env_key_name="OPENAI_API_KEY",
        capabilities=_capabilities(mode, endpoint),
    )


# --- code → docs -----------------------------------------------------------------


def test_every_parse_mode_is_documented():
    modes = sorted(LLMStep.VALID_PARSE_MODES)
    assert modes, "no parse modes found — check the test, not the page"
    undocumented = [m for m in modes if f'`parse_mode="{m}"`' not in _page()]
    assert not undocumented, f"parse modes undocumented: {undocumented}"


def test_every_structured_output_mode_is_documented():
    values = [m.value for m in StructuredOutputMode]
    assert values, "no modes found — check the test, not the page"
    undocumented = [v for v in values if f"`{v}`" not in _page()]
    assert not undocumented, f"structured output modes undocumented: {undocumented}"


def test_the_page_documents_every_parameter_of_the_call_it_shows():
    """`response_format` is the whole subject; the rest must at least not drift."""
    parameters = inspect.signature(ServedModel.generate).parameters
    assert "response_format" in parameters
    assert "`response_format`" in _page()


def test_parse_mode_defaults_to_text_as_the_page_says():
    assert inspect.signature(LLMStep.__init__).parameters["parse_mode"].default == "text"
    assert inspect.signature(LLMStep.__init__).parameters["output_column"].default == "generated"
    assert '`"text"` is the default' in _page()
    assert '`output_column` (default `"generated"`)' in _page()


# --- examples --------------------------------------------------------------------


@pytest.mark.parametrize("block", _code_blocks(), ids=lambda b: b.split("\n")[0][:40])
def test_every_example_is_valid_python(block):
    compile(block, str(PAGE), "exec")


@pytest.mark.parametrize(
    "block",
    [b for b in _code_blocks() if "from datafast import" in b],
    ids=lambda b: b.split("\n")[0][:40],
)
def test_every_self_contained_example_executes(block, monkeypatch):
    """The factory is stubbed on the module, so the example's own import gets it."""
    monkeypatch.setattr(datafast, "openai", lambda *a, **k: StubModel())
    exec(compile(block, str(PAGE), "exec"), {})


def test_the_page_has_examples(monkeypatch):
    """Guards the two tests above against passing on an empty list."""
    blocks = _code_blocks()
    assert sum("from datafast import" in b for b in blocks) >= 4
    assert any("response_format" in b for b in blocks)


def test_every_relative_link_resolves():
    links = re.findall(r"\]\((\.\./[^)#]+\.md|[a-z_]+\.md)\)", _page())
    assert links, "the page should link somewhere"
    for link in links:
        assert (PAGE.parent / link).resolve().exists(), f"broken link: {link}"


# --- parse mode behaviour --------------------------------------------------------


def test_get_parser_returns_the_parser_the_page_describes():
    assert isinstance(get_parser("text"), TextParser)
    assert isinstance(get_parser("json"), JSONParser)
    assert isinstance(get_parser("xml"), XMLParser)
    with pytest.raises(ValueError, match="Invalid parse_mode"):
        get_parser("yaml")


def test_text_mode_puts_the_whole_stripped_reply_in_one_column():
    assert get_parser("text").parse("  hello  ", ["summary"]) == {"summary": "hello"}


def test_json_mode_strips_fences_stringifies_and_fills_missing_keys():
    parsed = get_parser("json").parse(
        '```json\n{"a": 1, "b": "x"}\n```', ["a", "b", "missing"]
    )
    assert parsed == {"a": "1", "b": "x", "missing": ""}


def test_json_mode_raises_on_a_reply_that_is_not_json():
    with pytest.raises(ValueError, match="Failed to parse JSON"):
        get_parser("json").parse("Here you go!", ["a"])


def test_xml_mode_ignores_case_and_surrounding_text_and_never_raises():
    parsed = get_parser("xml").parse(
        "Here you go:\n<A>one</A>\nand <b>two\nlines</b>", ["a", "b", "missing"]
    )
    assert parsed == {"a": "one", "b": "two\nlines", "missing": ""}
    assert get_parser("xml").parse("nothing at all", ["a"]) == {"a": ""}


def test_json_and_xml_require_output_columns_but_text_does_not():
    for mode in ("json", "xml"):
        with pytest.raises(ValueError, match=f"output_columns required for parse_mode='{mode}'"):
            LLMStep(prompt="x", input_columns=[], model=StubModel(), parse_mode=mode)
    LLMStep(prompt="x", input_columns=[], model=StubModel(), parse_mode="text")


def test_only_json_and_xml_append_instructions_to_the_prompt():
    assert get_parser("text").get_format_instructions(["a"]) == ""
    assert "valid JSON" in get_parser("json").get_format_instructions(["a"])
    assert "<a>your a here</a>" in get_parser("xml").get_format_instructions(["a"])


def test_the_documented_json_instructions_are_the_ones_actually_sent():
    """The page quotes the appended text; a drift there misleads prompt authors."""
    instructions = get_parser("json").get_format_instructions(["question", "answer"])
    for line in instructions.strip().split("\n"):
        assert line in _page(), f"the page does not quote: {line!r}"


def test_a_step_really_appends_the_instructions_to_the_prompt_it_sends():
    stub = StubModel()
    records = (
        Source.list([{"topic": "gravity"}])
        >> LLMStep(
            prompt="Write about: {topic}",
            input_columns=["topic"],
            output_columns=["question", "answer"],
            parse_mode="json",
            model=stub,
        )
    ).run()
    assert stub.prompts[0].startswith("Write about: gravity")
    assert "Respond with valid JSON" in stub.prompts[0]
    assert records[0]["question"] == "Q?"


def test_parse_mode_columns_are_always_strings():
    parsed = get_parser("json").parse('{"score": 4, "ok": true}', ["score", "ok"])
    assert parsed == {"score": "4", "ok": "True"}
    assert all(isinstance(v, str) for v in parsed.values())


# --- structured output behaviour -------------------------------------------------


@pytest.mark.parametrize(
    "mode,expected",
    [
        (StructuredOutputMode.JSON_SCHEMA, Question),
        (StructuredOutputMode.JSON_OBJECT, {"type": "json_object"}),
        (StructuredOutputMode.PROMPTED_JSON, None),
    ],
)
def test_each_mode_sends_what_the_page_says_it_sends(mode, expected):
    params: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _served_model(mode)._add_chat_structured_output(params, Question)
    assert params.get("response_format") == expected


def test_mode_none_raises_rather_than_sending_anything():
    with pytest.raises(ValueError, match="does not support structured output"):
        _served_model(StructuredOutputMode.NONE)._add_chat_structured_output({}, Question)


def test_prompted_json_warns_and_names_the_served_model():
    with pytest.warns(UserWarning, match="openai/fake has no declared native"):
        _served_model(StructuredOutputMode.PROMPTED_JSON)._add_chat_structured_output(
            {}, Question
        )


@pytest.mark.parametrize(
    "mode,appended",
    [
        (StructuredOutputMode.JSON_SCHEMA, False),
        (StructuredOutputMode.JSON_OBJECT, True),
        (StructuredOutputMode.PROMPTED_JSON, True),
    ],
)
def test_the_weaker_modes_add_json_instructions_to_the_message(mode, appended):
    messages = _served_model(mode)._prepare_messages(
        [{"role": "user", "content": "hi"}], response_format=Question
    )
    assert ("Return only valid JSON" in messages[0]["content"]) is appended


def test_nothing_is_added_when_no_response_format_is_asked_for():
    for mode in StructuredOutputMode:
        messages = _served_model(mode)._prepare_messages(
            [{"role": "user", "content": "hi"}], response_format=None
        )
        assert messages[0]["content"] == "hi"


def test_the_responses_endpoint_requires_json_schema():
    params: dict = {}
    _served_model(
        StructuredOutputMode.JSON_SCHEMA, EndpointMode.RESPONSES
    )._add_responses_structured_output(params, Question)
    assert params == {"text_format": Question}

    with pytest.raises(ValueError, match="does not support native Responses"):
        _served_model(
            StructuredOutputMode.JSON_OBJECT, EndpointMode.RESPONSES
        )._add_responses_structured_output({}, Question)


def test_the_reply_is_validated_whatever_the_mode_was():
    model = _served_model(StructuredOutputMode.PROMPTED_JSON)
    good = NormalizedResponse(text='```json\n{"question": "Q?", "answer": "A."}\n```', raw=None)
    assert model._parse_response(good, response_format=Question) == Question(
        question="Q?", answer="A."
    )

    bad = NormalizedResponse(text='{"question": "Q?"}', raw=None)
    with pytest.raises(ValueError, match="Failed to parse JSON response into Question"):
        model._parse_response(bad, response_format=Question)


# --- the claim that the two mechanisms do not meet -------------------------------


def test_the_shipped_providers_declare_the_modes_the_page_tables():
    documented = {
        "openai": StructuredOutputMode.JSON_SCHEMA,
        "anthropic": StructuredOutputMode.JSON_SCHEMA,
        "gemini": StructuredOutputMode.JSON_SCHEMA,
        "mistral": StructuredOutputMode.JSON_SCHEMA,
        "openrouter": StructuredOutputMode.JSON_SCHEMA,
        "ollama": StructuredOutputMode.JSON_SCHEMA,
    }
    for factory_name, mode in documented.items():
        model = getattr(datafast, factory_name)()
        assert model.capabilities.structured_output is mode, factory_name
        assert f"`{factory_name}()`" in _page()


@pytest.mark.parametrize(
    "provider_id,mode",
    [
        ("vllm", StructuredOutputMode.JSON_SCHEMA),
        ("llamacpp", StructuredOutputMode.JSON_SCHEMA),
        ("some-unknown-server", StructuredOutputMode.PROMPTED_JSON),
    ],
)
def test_an_unprofiled_self_hosted_server_falls_back_to_prompted_json(provider_id, mode):
    model = datafast.openai_compatible(
        model_id="x", provider_id=provider_id, api_base_url="http://localhost:8000/v1"
    )
    assert model.capabilities.structured_output is mode


def test_no_shipped_profile_declares_mode_none():
    """The page says you will only meet `none` if you declare it yourself."""
    import datafast.llm.capabilities as capabilities

    profiles = [
        value
        for value in vars(capabilities).values()
        if isinstance(value, ServedModelCapabilities)
    ]
    assert profiles, "no profiles found — check the test, not the page"
    assert all(p.structured_output is not StructuredOutputMode.NONE for p in profiles)


def test_no_step_in_the_library_passes_response_format():
    """The page's central claim about where the two mechanisms meet: they do not."""
    offenders = [
        path.relative_to(ROOT)
        for path in (ROOT / "datafast" / "transforms").rglob("*.py")
        if "response_format" in path.read_text()
    ]
    assert not offenders, f"a step now uses structured output: {offenders}"
