"""The direct-call guide, pinned against the code it documents.

1. Every parameter of all four methods and every `NormalizedResponse` field is named on
   the page (code → docs).
2. Every self-contained example executes, against a stubbed provider factory.
3. Each behavioural claim — the input/output shape table, the endpoint asymmetry, the
   error contract, the batching fallback — is proved by a real call with LiteLLM's
   transport replaced, so nothing leaves the machine.
"""

import dataclasses
import inspect
import re
import types
import warnings
from pathlib import Path

import pytest
from pydantic import BaseModel

import datafast
import datafast.llm.served_model as served_model_module
from datafast.llm.served_model import ServedModel
from datafast.llm.types import (
    BatchMode,
    EndpointMode,
    NormalizedResponse,
    ServedModelCapabilities,
    StructuredOutputMode,
)

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "guides" / "calling_a_served_model.md"

METHODS = [
    ServedModel.generate,
    ServedModel.generate_batch,
    ServedModel.generate_response,
    ServedModel.generate_batch_response,
]


def _page() -> str:
    return PAGE.read_text()


def _code_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", _page(), re.DOTALL)


def _fake_chat_response(content='{"a": "1"}'):
    message = types.SimpleNamespace(
        content=content,
        reasoning_content="because",
        thinking_blocks=[{"type": "thinking", "thinking": "hmm"}],
        images=None,
        audio=None,
    )
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=message)], model="fake", usage=None
    )


@pytest.fixture
def transport(monkeypatch):
    """Replace LiteLLM's call with a recorder. Nothing reaches a network."""
    calls: list[dict] = []

    def fake_completion(**params):
        calls.append(params)
        return _fake_chat_response()

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)
    return calls


def _model(*, batch_mode=BatchMode.FALLBACK_CONCURRENCY, endpoint=EndpointMode.CHAT, **kwargs):
    return ServedModel(
        provider_id="openai",
        model_id="fake",
        api_key="not-a-real-key",
        litellm_route="openai",
        env_key_name="OPENAI_API_KEY",
        capabilities=ServedModelCapabilities(
            endpoint_modes=frozenset({endpoint}),
            default_endpoint_mode=endpoint,
            batch_mode=batch_mode,
            structured_output=StructuredOutputMode.JSON_SCHEMA,
        ),
        **kwargs,
    )


class StubModel:
    """Stands in for a provider factory inside the page's examples."""

    model_id = "stub"

    def generate(self, prompt=None, messages=None, **kwargs):
        if isinstance(prompt, list) or (
            isinstance(messages, list) and messages and isinstance(messages[0], list)
        ):
            return ["stub answer"] * len(prompt if prompt is not None else messages)
        return "stub answer"

    def generate_response(self, prompt=None, messages=None, **kwargs):
        return NormalizedResponse(text="stub answer", raw=None, reasoning_content="because")


# --- code → docs -----------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS, ids=lambda m: m.__name__)
def test_every_parameter_of_every_method_is_documented(method):
    parameters = [
        p.name
        for p in inspect.signature(method).parameters.values()
        if p.kind is not p.VAR_KEYWORD and p.name != "self"
    ]
    assert parameters, f"{method.__name__} takes nothing — check the test, not the page"
    missing = [p for p in parameters if f"`{p}`" not in _page()]
    assert not missing, f"{method.__name__} takes {missing}, undocumented on the page"


@pytest.mark.parametrize("method", METHODS, ids=lambda m: m.__name__)
def test_every_method_is_named_on_the_page(method):
    assert f"`{method.__name__}()`" in _page()


def test_every_normalized_response_field_is_documented():
    names = [f.name for f in dataclasses.fields(NormalizedResponse)]
    assert names, "NormalizedResponse has no fields — check the test, not the page"
    missing = [n for n in names if f"`{n}`" not in _page()]
    assert not missing, f"NormalizedResponse fields undocumented: {missing}"


def test_generate_response_really_has_no_response_format():
    """The page tells readers they cannot have both; that must stay true."""
    for method in (ServedModel.generate_response, ServedModel.generate_batch_response):
        assert "response_format" not in inspect.signature(method).parameters
    for method in (ServedModel.generate, ServedModel.generate_batch):
        assert "response_format" in inspect.signature(method).parameters


# --- examples --------------------------------------------------------------------


@pytest.mark.parametrize("block", _code_blocks(), ids=lambda b: b.split("\n")[0][:40])
def test_every_example_is_valid_python(block):
    compile(block, str(PAGE), "exec")


@pytest.mark.parametrize(
    "block",
    [b for b in _code_blocks() if "from datafast import" in b],
    ids=lambda b: b.split("\n")[0][:40],
)
def test_every_self_contained_example_executes(block, monkeypatch, capsys):
    for factory in ("openai", "anthropic", "gemini", "mistral", "ollama", "openrouter"):
        monkeypatch.setattr(datafast, factory, lambda *a, **k: StubModel())
    exec(compile(block, str(PAGE), "exec"), {})


def test_the_page_has_examples():
    blocks = _code_blocks()
    assert sum("from datafast import" in b for b in blocks) >= 3
    assert any("generate_response" in b for b in blocks)


def test_every_relative_link_resolves():
    links = re.findall(r"\]\((\.\./[^)#]+\.md|[a-z_]+\.md)\)", _page())
    assert links, "the page should link somewhere"
    for link in links:
        assert (PAGE.parent / link).resolve().exists(), f"broken link: {link}"


# --- input and output shapes -----------------------------------------------------


def test_a_single_prompt_returns_a_string(transport):
    assert _model().generate("hi") == '{"a": "1"}'


def test_a_list_of_prompts_returns_a_list(transport):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _model().generate(["a", "b"])
    assert result == ['{"a": "1"}', '{"a": "1"}']


def test_a_single_conversation_returns_a_string(transport):
    assert _model().generate(messages=[{"role": "user", "content": "hi"}]) == '{"a": "1"}'


def test_a_list_of_conversations_returns_a_list(transport):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _model().generate(
            messages=[
                [{"role": "user", "content": "a"}],
                [{"role": "user", "content": "b"}],
            ]
        )
    assert result == ['{"a": "1"}', '{"a": "1"}']


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({}, "Either prompt or messages must be provided"),
        (
            {"prompt": "x", "messages": [{"role": "user", "content": "y"}]},
            "Provide either prompt or messages, not both",
        ),
        ({"prompt": []}, "prompt list cannot be empty"),
        ({"prompt": 123}, "prompt must be a string or list of strings"),
        ({"messages": "hi"}, "Invalid messages format"),
    ],
    ids=["neither", "both", "empty", "wrong type", "bad messages"],
)
def test_bad_input_raises_before_anything_is_sent(transport, kwargs, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        _model().generate(**kwargs)
    assert not transport, "a request went out despite invalid input"


def test_generate_batch_preserves_order_and_accepts_an_empty_list(transport):
    model = _model()
    assert model.generate_batch([]) == []
    assert not transport

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = model.generate_batch(
            [
                [{"role": "user", "content": "first"}],
                [{"role": "user", "content": "second"}],
            ]
        )
    assert len(results) == 2
    sent = [call["messages"][0]["content"] for call in transport]
    assert sent == ["first", "second"]


def test_a_length_mismatch_on_a_batch_raises(transport):
    with pytest.raises(ValueError, match="previous_response_ids length must match"):
        _model().generate_batch(
            [[{"role": "user", "content": "a"}]], previous_response_ids=[None, None]
        )


def test_response_format_returns_a_validated_object(transport):
    class Answer(BaseModel):
        a: str

    assert _model().generate("hi", response_format=Answer) == Answer(a="1")


# --- NormalizedResponse ----------------------------------------------------------


def test_generate_response_carries_the_text_and_the_trace(transport):
    response = _model().generate_response("hi")
    assert isinstance(response, NormalizedResponse)
    assert response.text == '{"a": "1"}'
    assert response.reasoning_content == "because"
    assert response.thinking_blocks == [{"type": "thinking", "thinking": "hmm"}]


def test_raw_is_the_providers_own_object(transport):
    response = _model().generate_response("hi")
    assert response.raw.choices[0].message.content == '{"a": "1"}'


def test_generate_response_mirrors_its_input_shape(transport):
    model = _model()
    assert isinstance(model.generate_response("hi"), NormalizedResponse)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        many = model.generate_response(["a", "b"])
    assert isinstance(many, list) and len(many) == 2


def test_generate_batch_response_returns_one_response_per_item(transport):
    model = _model()
    assert model.generate_batch_response([]) == []
    responses = model.generate_batch_response([[{"role": "user", "content": "a"}]])
    assert len(responses) == 1 and isinstance(responses[0], NormalizedResponse)


def test_the_chat_endpoint_never_fills_output_items(transport):
    """Half of the endpoint asymmetry the page tables."""
    response = _model().generate_response("hi")
    assert response.output_items == []
    assert response.thinking_blocks, "chat should fill thinking_blocks"


def test_the_responses_endpoint_never_fills_thinking_blocks(monkeypatch):
    """The other half, read off the code that builds each response."""
    source = inspect.getsource(ServedModel._execute_single)
    responses_branch, chat_branch = source.split("params = self._build_chat_params")
    assert "thinking_blocks" not in responses_branch
    assert "output_items" not in chat_branch
    assert "output_items" in responses_branch
    assert "thinking_blocks" in chat_branch


# --- batching --------------------------------------------------------------------


def test_a_model_without_native_batching_warns_and_still_answers(transport):
    model = _model(batch_mode=BatchMode.FALLBACK_CONCURRENCY)
    with pytest.warns(UserWarning, match="does not expose native batching"):
        results = model.generate(["a", "b"])
    assert len(results) == 2


def test_a_single_input_never_warns(transport):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _model(batch_mode=BatchMode.FALLBACK_CONCURRENCY).generate("hi")


def test_the_documented_providers_declare_native_batching():
    for factory in ("anthropic", "gemini", "mistral", "openrouter"):
        model = getattr(datafast, factory)()
        assert model.capabilities.batch_mode is BatchMode.LITELLM_BATCH, factory
        assert f"`{factory}()`" in _page()


def test_openais_default_model_is_the_documented_exception():
    """The page says the warning fires on OpenAI's default; that is worth pinning."""
    model = datafast.openai()
    assert model.endpoint_mode is EndpointMode.RESPONSES
    assert model.capabilities.batch_mode is not BatchMode.LITELLM_BATCH


# --- the error contract ----------------------------------------------------------


def test_a_provider_failure_is_wrapped_in_runtime_error(monkeypatch):
    def boom(**params):
        raise ConnectionError("the network is down")

    monkeypatch.setattr(served_model_module.litellm, "completion", boom)
    model = _model(retry_limit=0)
    with pytest.raises(RuntimeError, match="Error generating response with openai"):
        model.generate("hi")


def test_a_value_error_passes_through_unwrapped(monkeypatch):
    def boom(**params):
        raise ValueError("a validation problem")

    monkeypatch.setattr(served_model_module.litellm, "completion", boom)
    model = _model(retry_limit=0)
    with pytest.raises(ValueError, match="a validation problem"):
        model.generate("hi")


def _raising(exception, attempts):
    def boom(**params):
        attempts.append(1)
        raise exception

    return boom


def test_a_retryable_failure_is_retried_before_it_raises(monkeypatch):
    """Only the four retryable kinds get a second chance; this is one of them."""
    from litellm import exceptions as litellm_exceptions

    attempts: list[int] = []
    monkeypatch.setattr(
        served_model_module.litellm,
        "completion",
        _raising(
            litellm_exceptions.APIConnectionError(
                message="down", llm_provider="openai", model="fake"
            ),
            attempts,
        ),
    )
    model = _model(retry_limit=2)
    model._sleep = lambda seconds: None  # instance attribute, set in __init__
    with pytest.raises(RuntimeError):
        model.generate("hi")
    assert len(attempts) == 3, "retry_limit counts retries after the first attempt"


def test_a_failure_that_would_fail_again_is_not_retried(monkeypatch):
    """The page says a bad key or malformed request raises on the first failure."""
    attempts: list[int] = []
    monkeypatch.setattr(
        served_model_module.litellm,
        "completion",
        _raising(ConnectionError("not a litellm error"), attempts),
    )
    model = _model(retry_limit=5)
    model._sleep = lambda seconds: None
    with pytest.raises(RuntimeError):
        model.generate("hi")
    assert len(attempts) == 1


# --- the runnable examples the page points at ------------------------------------


def test_the_example_directory_the_page_names_exists():
    directory = ROOT / "examples" / "providers"
    assert directory.is_dir(), "the page points readers at examples/providers/"
    assert "`examples/providers/`" in _page()
    assert any(directory.glob("*/0*.py")), "no runnable provider examples found"
