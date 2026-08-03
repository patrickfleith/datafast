import pytest
from litellm import exceptions as litellm_exceptions
from pydantic import BaseModel

import datafast.llm.served_model as served_model_module
from datafast import LLMStep, ListSink, Source
from datafast.llm import (
    ContentPart,
    EndpointMode,
    Modality,
    RetryPolicy,
    gemini,
    mistral,
    ollama,
    openai,
    openai_compatible,
    openrouter,
)
from datafast.llm.capabilities import resolve_capabilities


class SimpleSchema(BaseModel):
    answer: str


class _DummyMessage:
    def __init__(
        self,
        content,
        reasoning_content=None,
        thinking_blocks=None,
        images=None,
        audio=None,
    ):
        self.content = content
        self.reasoning_content = reasoning_content
        self.thinking_blocks = thinking_blocks
        self.images = images
        self.audio = audio


class _DummyChoice:
    def __init__(
        self,
        content,
        reasoning_content=None,
        thinking_blocks=None,
        images=None,
        audio=None,
    ):
        self.message = _DummyMessage(
            content,
            reasoning_content,
            thinking_blocks,
            images,
            audio,
        )


class _DummyChatResponse:
    def __init__(
        self,
        content,
        reasoning_content=None,
        thinking_blocks=None,
        images=None,
        audio=None,
    ):
        self.choices = [
            _DummyChoice(
                content,
                reasoning_content,
                thinking_blocks,
                images,
                audio,
            )
        ]


class _DummyResponsesResponse:
    def __init__(self, output_text=None, output=None, reasoning_content=None):
        self.output_text = output_text
        self.output = output
        self.reasoning_content = reasoning_content


@pytest.fixture(autouse=True)
def _disable_served_model_side_effects(monkeypatch):
    monkeypatch.setattr(served_model_module, "load_env_once", lambda: None)
    monkeypatch.setattr(
        served_model_module,
        "maybe_configure_langfuse_tracing",
        lambda load_env=False: False,
    )


def test_factories_resolve_expected_served_models():
    hosted = openai(api_key="test-key")
    local = openai_compatible(
        "ministral-8b-2512",
        api_base_url="http://localhost:8000/v1",
    )

    assert hosted.provider_id == "openai"
    assert hosted.endpoint_mode == EndpointMode.RESPONSES
    assert hosted._get_model_string() == "openai/gpt-5.5"

    assert local.provider_id == "openai_compatible"
    assert local.endpoint_mode == EndpointMode.CHAT
    assert local.api_base_url == "http://localhost:8000/v1"


def test_served_model_suppresses_litellm_debug_info_by_default(monkeypatch):
    monkeypatch.delenv(served_model_module.LITELLM_SUPPRESS_DEBUG_ENV, raising=False)
    monkeypatch.setattr(served_model_module.litellm, "suppress_debug_info", False)

    served_model_module.openrouter(model_id="demo-model", api_key="test-key")

    assert served_model_module.litellm.suppress_debug_info is True


def test_served_model_allows_litellm_debug_opt_out(monkeypatch):
    monkeypatch.setenv(served_model_module.LITELLM_SUPPRESS_DEBUG_ENV, "0")
    monkeypatch.setattr(served_model_module.litellm, "suppress_debug_info", False)

    served_model_module.openrouter(model_id="demo-model", api_key="test-key")

    assert served_model_module.litellm.suppress_debug_info is False


def test_openai_compatible_backend_profiles_are_distinct():
    generic = openai_compatible(
        "local-model",
        api_base_url="http://localhost:8000/v1",
    )
    vllm = openai_compatible(
        "local-model",
        api_base_url="http://localhost:8000/v1",
        backend="vllm",
    )
    llamacpp = openai_compatible(
        "local-model",
        api_base_url="http://localhost:8080/v1",
        backend="llamacpp",
    )

    assert generic.provider_id == "openai_compatible"
    assert generic.capabilities.modalities == frozenset({Modality.TEXT})

    assert vllm.provider_id == "vllm"
    assert vllm.capabilities.supports_endpoint(EndpointMode.RESPONSES)
    assert Modality.IMAGE in vllm.capabilities.modalities
    assert Modality.VIDEO in vllm.capabilities.modalities

    assert llamacpp.provider_id == "llamacpp"
    assert Modality.AUDIO in llamacpp.capabilities.modalities
    assert Modality.FILE in llamacpp.capabilities.modalities


def test_input_validation_rejects_missing_or_ambiguous_inputs():
    model = openrouter(model_id="demo-model", api_key="test-key")

    with pytest.raises(ValueError, match="Either prompt or messages"):
        model.generate()

    with pytest.raises(ValueError, match="either prompt or messages"):
        model.generate(prompt="hello", messages=[{"role": "user", "content": "hi"}])


def test_unsupported_params_warn_and_omit(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _DummyChatResponse("ok")

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    model = openai_compatible(
        "local-model",
        api_base_url="http://localhost:8000/v1",
        temperature=0.7,
    )

    with pytest.warns(UserWarning, match="temperature"):
        assert model.generate(prompt="ping") == "ok"

    assert "temperature" not in captured
    assert captured["api_base"] == "http://localhost:8000/v1"


def test_unsupported_params_fail_before_dispatch(monkeypatch):
    def fake_completion(**kwargs):
        raise AssertionError("request should not be dispatched")

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    model = openai_compatible(
        "local-model",
        api_base_url="http://localhost:8000/v1",
        temperature=0.7,
        unsupported_params="fail",
    )

    with pytest.raises(ValueError, match="temperature"):
        model.generate(prompt="ping")


def test_chat_endpoint_warns_and_omits_previous_response_id(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _DummyChatResponse("ok")

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    model = openrouter(model_id="demo-model", api_key="test-key")

    with pytest.warns(UserWarning, match="previous_response_id"):
        assert model.generate(prompt="ping", previous_response_id="resp_old") == "ok"

    assert "previous_response_id" not in captured


def test_openrouter_thinking_warns_and_omits_reasoning_param(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _DummyChatResponse("ok")

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    model = openrouter(
        model_id="nvidia/nemotron-3-super-120b-a12b:nitro",
        api_key="test-key",
        thinking=True,
    )

    with pytest.warns(UserWarning, match="reasoning_effort"):
        assert model.generate(prompt="ping") == "ok"

    assert "reasoning_effort" not in captured
    assert "reasoning" not in captured


def test_mistral_reasoning_capability_resolution():
    # Reasoning-capable Mistral served models: magistral family plus the documented
    # mistral-medium/small snapshots.
    for model_id in (
        "mistral-medium-3-5",
        "mistral-small-2603",
        "magistral-medium-2509",
        "magistral-small-latest",
    ):
        caps = resolve_capabilities("mistral", model_id)
        assert caps.supports_reasoning is True
        assert "reasoning_effort" in caps.supported_params
        assert caps.reasoning_requires_allowlist is True

    # Non-reasoning Mistral served models keep the plain hosted-chat profile.
    for model_id in ("mistral-large-2512", "mistral-tiny"):
        caps = resolve_capabilities("mistral", model_id)
        assert caps.supports_reasoning is False
        assert "reasoning_effort" not in caps.supported_params


def test_mistral_reasoning_effort_is_forwarded_with_allowlist(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _DummyChatResponse("ok", reasoning_content="chain of thought")

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    model = mistral(
        model_id="mistral-medium-3-5",
        api_key="test-key",
        reasoning_effort="high",
    )

    response = model.generate_response(prompt="think it through")

    assert captured["reasoning_effort"] == "high"
    # LiteLLM only forwards reasoning_effort for a subset of Mistral models; the
    # allowlist forces it through for the rest.
    assert "reasoning_effort" in captured["allowed_openai_params"]
    assert response.reasoning_content == "chain of thought"


def test_mistral_thinking_true_uses_high_effort(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _DummyChatResponse("ok")

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    model = mistral(model_id="mistral-medium-3-5", api_key="test-key", thinking=True)

    assert model.generate(prompt="ping") == "ok"
    # The Mistral API accepts only 'high' and 'none'; a generic 'low' 400s.
    assert captured["reasoning_effort"] == "high"


def test_mistral_thinking_false_disables_reasoning(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _DummyChatResponse("ok")

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    model = mistral(model_id="mistral-small-2603", api_key="test-key", thinking=False)

    assert model.generate(prompt="ping") == "ok"
    assert captured["reasoning_effort"] == "none"


def test_mistral_rejects_unsupported_reasoning_effort(monkeypatch):
    monkeypatch.setattr(
        served_model_module.litellm,
        "completion",
        lambda **kwargs: _DummyChatResponse("ok"),
    )

    model = mistral(
        model_id="mistral-medium-3-5",
        api_key="test-key",
        reasoning_effort="low",
    )

    with pytest.raises(ValueError, match="high, none"):
        model.generate(prompt="ping")


def test_mistral_without_reasoning_effort_stays_plain(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _DummyChatResponse("ok")

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    model = mistral(model_id="mistral-medium-3-5", api_key="test-key")

    assert model.generate(prompt="ping") == "ok"
    assert "reasoning_effort" not in captured
    assert "allowed_openai_params" not in captured


def test_mistral_non_reasoning_model_warns_and_omits_reasoning_effort(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _DummyChatResponse("ok")

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    model = mistral(
        model_id="mistral-large-2512",
        api_key="test-key",
        reasoning_effort="high",
    )

    with pytest.warns(UserWarning, match="reasoning_effort"):
        assert model.generate(prompt="ping") == "ok"

    assert "reasoning_effort" not in captured
    assert "allowed_openai_params" not in captured


def test_ollama_reasoning_capability_resolution():
    # Thinking-capable Ollama families gain a mapped reasoning control; models
    # tagged "-thinking" (e.g. lfm2.5-thinking) match too.
    for model_id in (
        "deepseek-r1:8b",
        "qwen3:8b",
        "gpt-oss:20b",
        "magistral:latest",
        "gemma4:12b",
        "lfm2.5-thinking:1.2b",
    ):
        caps = resolve_capabilities("ollama", model_id)
        assert caps.supports_reasoning is True
        assert "reasoning_effort" in caps.supported_params

    # Other Ollama models keep the plain chat profile.
    for model_id in ("gemma3:4b", "llama3.2"):
        caps = resolve_capabilities("ollama", model_id)
        assert caps.supports_reasoning is False
        assert "reasoning_effort" not in caps.supported_params


def test_ollama_reasoning_effort_is_forwarded(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _DummyChatResponse("ok", reasoning_content="chain of thought")

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    model = ollama(model_id="deepseek-r1:8b", reasoning_effort="high")

    response = model.generate_response(prompt="think it through")

    assert captured["reasoning_effort"] == "high"
    # LiteLLM maps reasoning_effort onto Ollama's think param natively, so no
    # allowlist escape hatch is needed.
    assert "allowed_openai_params" not in captured
    assert response.reasoning_content == "chain of thought"


def test_ollama_thinking_true_defaults_to_low_effort(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _DummyChatResponse("ok")

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    model = ollama(model_id="qwen3:8b", thinking=True)

    assert model.generate(prompt="ping") == "ok"
    assert captured["reasoning_effort"] == "low"


def test_ollama_thinking_false_sends_think_false(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _DummyChatResponse("ok")

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    model = ollama(model_id="qwen3:8b", thinking=False)

    assert model.generate(prompt="ping") == "ok"
    # Omitting the parameter would leave the model default, which is on for
    # qwen3. think=false is passed directly: LiteLLM's reasoning_effort mapping
    # sends the literal string for gpt-oss, which Ollama rejects.
    assert captured["think"] is False
    assert "reasoning_effort" not in captured


def test_ollama_non_reasoning_model_thinking_false_stays_plain(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _DummyChatResponse("ok")

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    model = ollama(model_id="gemma3:4b", thinking=False)

    assert model.generate(prompt="ping") == "ok"
    assert "think" not in captured
    assert "reasoning_effort" not in captured


def test_ollama_non_reasoning_model_warns_and_omits_reasoning_effort(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _DummyChatResponse("ok")

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    model = ollama(model_id="gemma3:4b", reasoning_effort="high")

    with pytest.warns(UserWarning, match="reasoning_effort"):
        assert model.generate(prompt="ping") == "ok"

    assert "reasoning_effort" not in captured


def test_gemini_reasoning_capability_resolution():
    # All catalogued Gemini models support reasoning natively.
    for model_id in ("gemini-3.5-flash", "gemini-3.1-flash-lite"):
        caps = resolve_capabilities("gemini", model_id)
        assert caps.supports_reasoning is True
        assert "reasoning_effort" in caps.supported_params


def test_gemini_reasoning_effort_is_forwarded(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _DummyChatResponse("ok", reasoning_content="chain of thought")

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    model = gemini(
        model_id="gemini-3.5-flash", api_key="test-key", reasoning_effort="high"
    )

    response = model.generate_response(prompt="think it through")

    assert captured["reasoning_effort"] == "high"
    # LiteLLM forwards reasoning_effort to gemini/* natively, so no allowlist.
    assert "allowed_openai_params" not in captured
    assert response.reasoning_content == "chain of thought"


def test_gemini_thinking_true_defaults_to_low_effort(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _DummyChatResponse("ok")

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    model = gemini(
        model_id="gemini-3.5-flash", api_key="test-key", thinking=True
    )

    assert model.generate(prompt="ping") == "ok"
    assert captured["reasoning_effort"] == "low"


def test_gemini_thinking_false_disables_reasoning(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _DummyChatResponse("ok")

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    model = gemini(model_id="gemini-3.5-flash", api_key="test-key", thinking=False)

    assert model.generate(prompt="ping") == "ok"
    # Gemini 3 models think by default, so omitting the parameter would still
    # bill reasoning tokens.
    assert captured["reasoning_effort"] == "none"


def test_provider_params_escape_hatch_is_forwarded(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _DummyChatResponse("ok")

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    model = openai_compatible(
        "local-model",
        api_base_url="http://localhost:8000/v1",
        provider_params={"extra_body": {"backend_hint": "vllm"}},
    )

    assert model.generate(prompt="ping") == "ok"
    assert captured["extra_body"] == {"backend_hint": "vllm"}


def test_content_parts_normalize_multimodal_and_document_shapes():
    vllm = openai_compatible(
        "local-model",
        api_base_url="http://localhost:8000/v1",
        backend="vllm",
    )
    prepared = vllm._prepare_messages(
        [
            {
                "role": "user",
                "content": [
                    ContentPart(type="text", text="What is in this image?"),
                    ContentPart(
                        type="image",
                        url="https://example.com/image.png",
                        media_id="img-123",
                    ),
                    ContentPart(
                        type="video",
                        url="https://example.com/video.mp4",
                        media_id="vid-123",
                    ),
                ],
            }
        ],
        response_format=None,
    )

    assert prepared[0]["content"] == [
        {"type": "text", "text": "What is in this image?"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.png"},
            "uuid": "img-123",
        },
        {
            "type": "video_url",
            "video_url": {"url": "https://example.com/video.mp4"},
            "uuid": "vid-123",
        },
    ]

    llamacpp = openai_compatible(
        "local-model",
        api_base_url="http://localhost:8080/v1",
        backend="llamacpp",
    )
    prepared = llamacpp._prepare_messages(
        [
            {
                "role": "user",
                "content": [
                    ContentPart(
                        type="document",
                        data="data:application/pdf;base64,abc",
                        media_type="application/pdf",
                    ),
                ],
            }
        ],
        response_format=None,
    )

    assert prepared[0]["content"] == [
        {
            "type": "file",
            "file": {"file_data": "data:application/pdf;base64,abc"},
        }
    ]


def test_litellm_unsupported_params_can_retry_with_drop_params(monkeypatch):
    unsupported_error = type("UnsupportedParamsError", (Exception,), {})
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise unsupported_error("bad param")
        return _DummyChatResponse("ok")

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    model = openrouter(model_id="demo-model", api_key="test-key")

    with pytest.warns(UserWarning, match="drop_params=True"):
        assert model.generate(prompt="ping") == "ok"

    assert calls[0].get("drop_params") is None
    assert calls[1]["drop_params"] is True


def _retryable(cls=litellm_exceptions.RateLimitError):
    """A litellm exception the model treats as retryable."""
    return cls(message="boom", llm_provider="openrouter", model="demo-model")


def test_retryable_error_is_retried_until_bounded_limit(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise _retryable()
        return _DummyChatResponse("ok")

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    model = openrouter(
        model_id="demo-model", api_key="test-key", retry_limit=2
    )
    model._sleep = lambda delay: None

    assert model.generate(prompt="ping") == "ok"
    assert len(calls) == 3  # initial attempt + two bounded retries


def test_non_retryable_error_fails_without_retry(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        raise _retryable(litellm_exceptions.AuthenticationError)

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    model = openrouter(model_id="demo-model", api_key="test-key")
    model._sleep = lambda delay: None

    with pytest.raises(RuntimeError):
        model.generate(prompt="ping")
    assert len(calls) == 1


def test_backoff_grows_across_retries(monkeypatch):
    def fake_completion(**kwargs):
        raise _retryable()

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    delays = []
    model = openrouter(
        model_id="demo-model",
        api_key="test-key",
        retry_policy=RetryPolicy(max_retries=3, jitter=0.0),
    )
    model._sleep = delays.append

    with pytest.raises(RuntimeError):
        model.generate(prompt="ping")
    assert delays == [1.0, 2.0, 4.0]


def test_jitter_stays_within_expected_range(monkeypatch):
    def fake_completion(**kwargs):
        raise _retryable()

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    delays = []
    model = openrouter(
        model_id="demo-model",
        api_key="test-key",
        retry_policy=RetryPolicy(max_retries=3, jitter=0.25),
    )
    model._sleep = delays.append

    with pytest.raises(RuntimeError):
        model.generate(prompt="ping")
    for attempt, delay in enumerate(delays):
        base = 1.0 * (2 ** attempt)
        assert base <= delay <= base * 1.25


def test_timeout_is_forwarded_and_failure_surfaces(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _DummyChatResponse("ok")

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    model = openrouter(
        model_id="demo-model", api_key="test-key", timeout=30
    )
    assert model.generate(prompt="ping") == "ok"
    assert captured["timeout"] == 30

    def raise_timeout(**kwargs):
        raise _retryable(litellm_exceptions.Timeout)

    monkeypatch.setattr(served_model_module.litellm, "completion", raise_timeout)
    model = openrouter(
        model_id="demo-model", api_key="test-key", retry_limit=0
    )
    with pytest.raises(RuntimeError, match="openrouter"):
        model.generate(prompt="ping")


def test_rpm_limit_throttles_before_dispatch(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return _DummyChatResponse("ok")

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    clock = [0.0]
    delays = []
    monkeypatch.setattr(served_model_module.time, "monotonic", lambda: clock[0])

    model = openrouter(
        model_id="demo-model", api_key="test-key", rpm_limit=2
    )

    def fake_sleep(delay):
        delays.append(delay)
        clock[0] += delay

    model._sleep = fake_sleep

    for _ in range(3):
        model.generate(prompt="ping")

    assert len(calls) == 3
    assert delays == [61.0]  # third request waits for the window to clear


def test_batch_retry_preserves_output_order(monkeypatch):
    def fake_batch_completion(**kwargs):
        return [_DummyChatResponse("a"), _retryable(), _DummyChatResponse("c")]

    monkeypatch.setattr(
        served_model_module.litellm, "batch_completion", fake_batch_completion
    )
    monkeypatch.setattr(
        served_model_module.litellm,
        "completion",
        lambda **kwargs: _DummyChatResponse("b"),
    )

    model = openrouter(model_id="demo-model", api_key="test-key")

    assert model.generate(prompt=["a", "b", "c"]) == ["a", "b", "c"]


def test_generate_response_preserves_litellm_reasoning_metadata(monkeypatch):
    monkeypatch.setattr(
        served_model_module.litellm,
        "completion",
        lambda **kwargs: _DummyChatResponse(
            "final answer",
            reasoning_content="internal summary",
            thinking_blocks=[
                {
                    "type": "thinking",
                    "thinking": "visible thinking block",
                    "signature": "sig",
                }
            ],
            images=[{"type": "image", "url": "https://example.com/out.png"}],
            audio={"id": "audio-1", "expires_at": 123},
        ),
    )

    model = openrouter(model_id="demo-model", api_key="test-key")
    response = model.generate_response(prompt="ping")

    assert response.text == "final answer"
    assert response.reasoning_content == "internal summary"
    assert response.thinking_blocks == [
        {
            "type": "thinking",
            "thinking": "visible thinking block",
            "signature": "sig",
        }
    ]
    assert response.images == [
        {"type": "image", "url": "https://example.com/out.png"}
    ]
    assert response.audio == {"id": "audio-1", "expires_at": 123}


def test_responses_full_response_preserves_output_items_and_media(monkeypatch):
    output = [
        {"type": "reasoning", "summary": [{"text": "short rationale"}]},
        {"type": "image_generation_call", "result": "base64-image"},
        {
            "type": "message",
            "content": [{"type": "output_text", "text": "Here is the image."}],
        },
    ]

    monkeypatch.setattr(
        served_model_module.litellm,
        "responses",
        lambda **kwargs: _DummyResponsesResponse(output=output),
    )

    model = openai(model_id="gpt-5.5", api_key="test-key")
    response = model.generate_response(prompt="make an image")

    assert response.text == "Here is the image."
    assert response.reasoning_content == "short rationale"
    assert response.images == [
        {"type": "image_generation_call", "result": "base64-image"}
    ]
    assert response.output_items == output


def test_responses_endpoint_maps_reasoning_state_and_structured_output(monkeypatch):
    captured = {}

    def fake_responses(**kwargs):
        captured.update(kwargs)
        return _DummyResponsesResponse('{"answer": "Paris"}')

    monkeypatch.setattr(served_model_module.litellm, "responses", fake_responses)

    model = openai(
        model_id="gpt-5.5",
        api_key="test-key",
        thinking=True,
        max_completion_tokens=64,
    )

    result = model.generate(
        messages=[{"role": "user", "content": "capital?"}],
        response_format=SimpleSchema,
        previous_response_id="resp_previous",
        metadata={"purpose": "test"},
    )

    assert result == SimpleSchema(answer="Paris")
    assert captured["model"] == "openai/gpt-5.5"
    assert captured["previous_response_id"] == "resp_previous"
    assert captured["reasoning"] == {"effort": "low"}
    assert captured["max_output_tokens"] == 64
    assert captured["text_format"] is SimpleSchema
    # Trace metadata must stay off the Responses wire (OpenAI requires
    # string-only metadata); it rides LiteLLM's logging-only kwarg instead.
    assert "metadata" not in captured
    assert captured["litellm_metadata"]["purpose"] == "test"
    assert captured["input"] == [{"role": "user", "content": "capital?"}]


def test_fallback_batching_preserves_order(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs["messages"][0]["content"])
        return _DummyChatResponse(f"reply:{kwargs['messages'][0]['content']}")

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    model = openai_compatible(
        "local-model",
        api_base_url="http://localhost:8000/v1",
        max_concurrent=1,
    )

    with pytest.warns(UserWarning, match="Falling back"):
        result = model.generate(prompt=["one", "two", "three"])

    assert result == ["reply:one", "reply:two", "reply:three"]
    assert calls == ["one", "two", "three"]


def test_structured_output_validation_error_is_clear(monkeypatch):
    monkeypatch.setattr(
        served_model_module.litellm,
        "completion",
        lambda **kwargs: _DummyChatResponse("not json"),
    )

    model = openrouter(model_id="demo-model", api_key="test-key")

    with pytest.raises(ValueError, match="Failed to parse JSON response"):
        model.generate(prompt="answer in json", response_format=SimpleSchema)


def test_runner_dispatches_same_model_batches_through_generate_batch():
    class FakeBatchModel:
        provider_id = "fake"
        model_id = "fake-model"

        def __init__(self):
            self.batches = []

        def generate_batch(self, messages, metadata=None, response_format=None):
            self.batches.append({"messages": messages, "metadata": metadata})
            return ["first", "second"]

    model = FakeBatchModel()
    sink = ListSink()
    pipeline = (
        Source.list([{"topic": "alpha"}, {"topic": "beta"}])
        >> LLMStep(
            prompt="Write about {topic}.",
            input_columns=["topic"],
            output_column="result",
            model=model,
        )
        >> sink
    )

    output = pipeline.run(batch_size=2)

    assert output == [
        {"topic": "alpha", "result": "first", "_model": "fake-model"},
        {"topic": "beta", "result": "second", "_model": "fake-model"},
    ]
    assert len(model.batches) == 1
    assert [batch[0]["content"] for batch in model.batches[0]["messages"]] == [
        "Write about alpha.",
        "Write about beta.",
    ]
    assert len(model.batches[0]["metadata"]) == 2
