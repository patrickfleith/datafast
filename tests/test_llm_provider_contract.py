import pytest
from pydantic import BaseModel

import datafast.llm.provider as provider_module
from datafast import LLMStep, ListSink, Source
from datafast.llm import (
    ContentPart,
    EndpointMode,
    Modality,
    OpenAIProvider,
    OpenRouterProvider,
    openai,
    openai_compatible,
)


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
def _disable_provider_side_effects(monkeypatch):
    monkeypatch.setattr(provider_module, "load_env_once", lambda: None)
    monkeypatch.setattr(
        provider_module,
        "maybe_configure_langfuse_tracing",
        lambda load_env=False: False,
    )


def test_factories_resolve_expected_targets():
    hosted = openai(api_key="test-key")
    local = openai_compatible(
        "ministral-8b-2512",
        api_base_url="http://localhost:8000/v1",
    )

    assert hosted.provider_name == "openai"
    assert hosted.endpoint_mode == EndpointMode.RESPONSES
    assert hosted._get_model_string() == "openai/gpt-5.5"

    assert local.provider_name == "openai_compatible"
    assert local.endpoint_mode == EndpointMode.CHAT
    assert local.api_base_url == "http://localhost:8000/v1"


def test_provider_suppresses_litellm_debug_info_by_default(monkeypatch):
    monkeypatch.delenv(provider_module.LITELLM_SUPPRESS_DEBUG_ENV, raising=False)
    monkeypatch.setattr(provider_module.litellm, "suppress_debug_info", False)

    provider_module.OpenRouterProvider(model_id="demo-model", api_key="test-key")

    assert provider_module.litellm.suppress_debug_info is True


def test_provider_allows_litellm_debug_opt_out(monkeypatch):
    monkeypatch.setenv(provider_module.LITELLM_SUPPRESS_DEBUG_ENV, "0")
    monkeypatch.setattr(provider_module.litellm, "suppress_debug_info", False)

    provider_module.OpenRouterProvider(model_id="demo-model", api_key="test-key")

    assert provider_module.litellm.suppress_debug_info is False


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

    assert generic.provider_name == "openai_compatible"
    assert generic.capabilities.modalities == frozenset({Modality.TEXT})

    assert vllm.provider_name == "vllm"
    assert vllm.capabilities.supports_endpoint(EndpointMode.RESPONSES)
    assert Modality.IMAGE in vllm.capabilities.modalities
    assert Modality.VIDEO in vllm.capabilities.modalities

    assert llamacpp.provider_name == "llamacpp"
    assert Modality.AUDIO in llamacpp.capabilities.modalities
    assert Modality.FILE in llamacpp.capabilities.modalities


def test_input_validation_rejects_missing_or_ambiguous_inputs():
    provider = OpenRouterProvider(model_id="demo-model", api_key="test-key")

    with pytest.raises(ValueError, match="Either prompt or messages"):
        provider.generate()

    with pytest.raises(ValueError, match="either prompt or messages"):
        provider.generate(prompt="hello", messages=[{"role": "user", "content": "hi"}])


def test_unsupported_params_warn_and_omit(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _DummyChatResponse("ok")

    monkeypatch.setattr(provider_module.litellm, "completion", fake_completion)

    provider = openai_compatible(
        "local-model",
        api_base_url="http://localhost:8000/v1",
        temperature=0.7,
    )

    with pytest.warns(UserWarning, match="temperature"):
        assert provider.generate(prompt="ping") == "ok"

    assert "temperature" not in captured
    assert captured["api_base"] == "http://localhost:8000/v1"


def test_unsupported_params_fail_before_dispatch(monkeypatch):
    def fake_completion(**kwargs):
        raise AssertionError("request should not be dispatched")

    monkeypatch.setattr(provider_module.litellm, "completion", fake_completion)

    provider = openai_compatible(
        "local-model",
        api_base_url="http://localhost:8000/v1",
        temperature=0.7,
        unsupported_params="fail",
    )

    with pytest.raises(ValueError, match="temperature"):
        provider.generate(prompt="ping")


def test_chat_endpoint_warns_and_omits_previous_response_id(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _DummyChatResponse("ok")

    monkeypatch.setattr(provider_module.litellm, "completion", fake_completion)

    provider = OpenRouterProvider(model_id="demo-model", api_key="test-key")

    with pytest.warns(UserWarning, match="previous_response_id"):
        assert provider.generate(prompt="ping", previous_response_id="resp_old") == "ok"

    assert "previous_response_id" not in captured


def test_openrouter_thinking_warns_and_omits_reasoning_param(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _DummyChatResponse("ok")

    monkeypatch.setattr(provider_module.litellm, "completion", fake_completion)

    provider = OpenRouterProvider(
        model_id="nvidia/nemotron-3-super-120b-a12b:nitro",
        api_key="test-key",
        thinking=True,
    )

    with pytest.warns(UserWarning, match="reasoning_effort"):
        assert provider.generate(prompt="ping") == "ok"

    assert "reasoning_effort" not in captured
    assert "reasoning" not in captured


def test_provider_params_escape_hatch_is_forwarded(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _DummyChatResponse("ok")

    monkeypatch.setattr(provider_module.litellm, "completion", fake_completion)

    provider = openai_compatible(
        "local-model",
        api_base_url="http://localhost:8000/v1",
        provider_params={"extra_body": {"backend_hint": "vllm"}},
    )

    assert provider.generate(prompt="ping") == "ok"
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

    monkeypatch.setattr(provider_module.litellm, "completion", fake_completion)

    provider = OpenRouterProvider(model_id="demo-model", api_key="test-key")

    with pytest.warns(UserWarning, match="drop_params=True"):
        assert provider.generate(prompt="ping") == "ok"

    assert calls[0].get("drop_params") is None
    assert calls[1]["drop_params"] is True


def test_generate_response_preserves_litellm_reasoning_metadata(monkeypatch):
    monkeypatch.setattr(
        provider_module.litellm,
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

    provider = OpenRouterProvider(model_id="demo-model", api_key="test-key")
    response = provider.generate_response(prompt="ping")

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
        provider_module.litellm,
        "responses",
        lambda **kwargs: _DummyResponsesResponse(output=output),
    )

    provider = OpenAIProvider(model_id="gpt-5.5", api_key="test-key")
    response = provider.generate_response(prompt="make an image")

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

    monkeypatch.setattr(provider_module.litellm, "responses", fake_responses)

    provider = OpenAIProvider(
        model_id="gpt-5.5",
        api_key="test-key",
        thinking=True,
        max_completion_tokens=64,
    )

    result = provider.generate(
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
    assert captured["metadata"]["purpose"] == "test"


def test_fallback_batching_preserves_order(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs["messages"][0]["content"])
        return _DummyChatResponse(f"reply:{kwargs['messages'][0]['content']}")

    monkeypatch.setattr(provider_module.litellm, "completion", fake_completion)

    provider = openai_compatible(
        "local-model",
        api_base_url="http://localhost:8000/v1",
        max_concurrent=1,
    )

    with pytest.warns(UserWarning, match="Falling back"):
        result = provider.generate(prompt=["one", "two", "three"])

    assert result == ["reply:one", "reply:two", "reply:three"]
    assert calls == ["one", "two", "three"]


def test_structured_output_validation_error_is_clear(monkeypatch):
    monkeypatch.setattr(
        provider_module.litellm,
        "completion",
        lambda **kwargs: _DummyChatResponse("not json"),
    )

    provider = OpenRouterProvider(model_id="demo-model", api_key="test-key")

    with pytest.raises(ValueError, match="Failed to parse JSON response"):
        provider.generate(prompt="answer in json", response_format=SimpleSchema)


def test_runner_dispatches_same_model_batches_through_generate_batch():
    class FakeBatchModel:
        provider_name = "fake"
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
