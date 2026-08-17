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
    anthropic,
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
        provider_id="tgi",
        api_base_url="http://localhost:8000/v1",
    )

    assert hosted.provider_id == "openai"
    assert hosted.endpoint_mode == EndpointMode.RESPONSES
    assert hosted._get_model_string() == "openai/gpt-5.5"

    assert local.provider_id == "tgi"
    assert local.endpoint_mode == EndpointMode.CHAT
    assert local.api_base_url == "http://localhost:8000/v1"


def test_ollama_default_model_is_the_multimodal_reasoning_one():
    """The default is gemma4:12b, which resolves to OLLAMA_REASONING_CHAT — so
    calling ollama() with no id gets reasoning and vision rather than the plain
    profile gemma3 resolved to. It is also what the example scripts use."""
    model = ollama()

    assert model._get_model_string() == "ollama_chat/gemma4:12b"
    assert model.capabilities.supports_reasoning is True
    assert Modality.IMAGE in model.capabilities.modalities


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


def test_openai_compatible_provider_profiles_are_distinct():
    generic = openai_compatible(
        "local-model",
        provider_id="tgi",
        api_base_url="http://localhost:8000/v1",
    )
    vllm = openai_compatible(
        "local-model",
        api_base_url="http://localhost:8000/v1",
        provider_id="vllm",
    )
    llamacpp = openai_compatible(
        "local-model",
        api_base_url="http://localhost:8080/v1",
        provider_id="llamacpp",
    )

    assert generic.provider_id == "tgi"
    assert generic.capabilities.modalities == frozenset({Modality.TEXT})

    assert vllm.provider_id == "vllm"
    assert vllm.capabilities.supports_endpoint(EndpointMode.RESPONSES)
    assert Modality.IMAGE in vllm.capabilities.modalities
    assert Modality.VIDEO in vllm.capabilities.modalities

    assert llamacpp.provider_id == "llamacpp"
    assert Modality.AUDIO in llamacpp.capabilities.modalities
    assert Modality.FILE in llamacpp.capabilities.modalities


def test_openai_compatible_rejects_a_wire_format_as_provider_id():
    # provider_id names the server, never the wire format used to reach it.
    for value in ("openai_compatible", "openai-compatible", "OpenAI_Compatible"):
        with pytest.raises(ValueError, match="names a wire format"):
            openai_compatible(
                "local-model",
                provider_id=value,
                api_base_url="http://localhost:8000/v1",
            )


def test_openai_compatible_requires_a_provider_id():
    with pytest.raises(TypeError, match="provider_id"):
        openai_compatible("local-model", api_base_url="http://localhost:8000/v1")


def test_openai_compatible_normalizes_provider_id():
    for value in ("llama.cpp", "LLAMA-CPP", " llamacpp "):
        model = openai_compatible(
            "local-model",
            provider_id=value,
            api_base_url="http://localhost:8080/v1",
        )
        assert model.provider_id == "llamacpp"


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
        provider_id="tgi",
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
        provider_id="tgi",
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
        # Reasoning is mainline now and marked in the id rather than by family.
        "ministral-3-8b-reasoning-2512",
    ):
        caps = resolve_capabilities("mistral", model_id)
        assert caps.supports_reasoning is True
        assert "reasoning_effort" in caps.supported_params
        assert caps.reasoning_requires_allowlist is True
        assert caps.files_require_file_id is True

    # Non-reasoning Mistral served models keep the plain hosted-chat profile, but
    # the file carrier is a property of the API, so it applies to them too.
    for model_id in ("mistral-large-2512", "mistral-tiny"):
        caps = resolve_capabilities("mistral", model_id)
        assert caps.supports_reasoning is False
        assert "reasoning_effort" not in caps.supported_params
        assert caps.files_require_file_id is True


def test_mistral_rejects_inline_file_data(monkeypatch):
    """Mistral's chat API takes a file only as an uploaded id, so inline data has to
    fail here rather than as a 422 from the provider."""
    monkeypatch.setattr(
        served_model_module.litellm,
        "completion",
        lambda **kwargs: pytest.fail("no request should be sent"),
    )

    model = mistral(model_id="mistral-small-2603", api_key="test-key")
    messages = [
        {
            "role": "user",
            "content": [
                ContentPart(type="file", data="abc", media_type="application/pdf"),
            ],
        }
    ]

    with pytest.raises(ValueError, match="upload API"):
        model.generate(messages=messages)


def test_mistral_forwards_an_uploaded_file_id(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _DummyChatResponse("ok")

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    model = mistral(model_id="mistral-small-2603", api_key="test-key")
    messages = [
        {
            "role": "user",
            "content": [ContentPart(type="file", url="file-abc123")],
        }
    ]

    model.generate(messages=messages)

    part = captured["messages"][0]["content"][0]
    assert part == {"type": "file", "file": {"file_id": "file-abc123"}}


def test_mistral_upload_file_returns_the_id(monkeypatch, tmp_path):
    """LiteLLM has no Files support for Mistral, so upload_file posts to the API
    itself; this pins the request it builds without sending one."""
    captured = {}

    class _DummyUploadResponse:
        def json(self):
            return {"id": "file-abc123", "purpose": "ocr"}

        def raise_for_status(self):
            return None

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _DummyUploadResponse()

    monkeypatch.setattr(served_model_module.httpx, "post", fake_post)

    document = tmp_path / "report.pdf"
    document.write_bytes(b"%PDF-1.4 ...")
    model = mistral(model_id="mistral-small-2603", api_key="test-key")

    file_id = model.upload_file(document)

    assert file_id == "file-abc123"
    assert captured["url"] == "https://api.mistral.ai/v1/files"
    # No expiry unless asked for, so the account's own retention stays in force.
    assert captured["data"] == {"purpose": "ocr"}
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["files"]["file"][0] == "report.pdf"

    model.upload_file(document, expiry=1)

    assert captured["data"] == {"purpose": "ocr", "expiry": 1}


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


def test_anthropic_thinking_warns_and_omits_temperature(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _DummyChatResponse("ok")

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    model = anthropic(
        model_id="claude-haiku-4-5",
        api_key="test-key",
        thinking=True,
        temperature=0.0,
    )

    # Anthropic 400s on any temperature but 1 once thinking is on.
    with pytest.warns(UserWarning, match="temperature"):
        assert model.generate(prompt="ping") == "ok"

    assert "temperature" not in captured
    assert captured["reasoning_effort"] == "low"


def test_anthropic_without_thinking_keeps_temperature(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _DummyChatResponse("ok")

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    model = anthropic(
        model_id="claude-haiku-4-5",
        api_key="test-key",
        temperature=0.0,
    )

    assert model.generate(prompt="ping") == "ok"
    assert captured["temperature"] == 0.0


def test_sonnet_5_thinking_false_sends_the_native_disable(monkeypatch):
    """claude-sonnet-5 thinks by default, so omitting the parameter is not off.

    LiteLLM maps reasoning_effort='none' to dropping the parameter, which lands
    back on that default — the off switch has to be Anthropic's own thinking
    block.
    """
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _DummyChatResponse("ok")

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    model = anthropic(model_id="claude-sonnet-5", api_key="test-key", thinking=False)

    assert model.generate(prompt="ping") == "ok"
    assert captured["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in captured


def test_sonnet_5_refuses_a_caller_temperature(monkeypatch):
    """Anthropic rejects any temperature but 1 on this line, reasoning or not,
    so it is unsupported rather than locked while thinking like on haiku."""
    monkeypatch.setattr(
        served_model_module.litellm,
        "completion",
        lambda **kwargs: _DummyChatResponse("ok"),
    )

    model = anthropic(
        model_id="claude-sonnet-5",
        api_key="test-key",
        thinking=False,
        temperature=0.0,
    )

    with pytest.warns(UserWarning, match="temperature"):
        assert model.generate(prompt="ping") == "ok"


def test_sonnet_5_rejects_an_effort_that_reads_as_off():
    """'none' would read as off while leaving the model's default in force, and
    'minimal' is silently mapped to 'low'. Both are refused with the real list."""
    caps = resolve_capabilities("anthropic", "claude-sonnet-5")
    assert caps.reasoning_efforts == {"low", "medium", "high", "xhigh", "max"}

    for effort in ("none", "minimal"):
        model = anthropic(
            model_id="claude-sonnet-5", api_key="test-key", reasoning_effort=effort
        )
        with pytest.raises(ValueError, match="not supported"):
            model.generate(prompt="ping")


def test_sonnet_4_6_keeps_the_older_profile():
    """The new profile is per-model, not per-provider: 4.6 still takes a
    temperature while thinking is off, and still has no explicit disable."""
    caps = resolve_capabilities("anthropic", "claude-sonnet-4-6")

    assert "temperature" in caps.supported_params
    assert caps.reasoning_off_param is None


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


def test_sampling_params_are_gated_by_the_profile(monkeypatch):
    """top_p and frequency_penalty are config fields, so a profile that omits them
    drops the value rather than forwarding it. OPENAI_RESPONSES is the case that
    matters: its reasoning models 400 on a sampling control."""
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return _DummyResponsesResponse("ok")

    monkeypatch.setattr(served_model_module.litellm, "responses", capture)
    monkeypatch.setattr(
        served_model_module.litellm,
        "completion",
        lambda **kwargs: captured.update(kwargs) or _DummyChatResponse("ok"),
    )

    chat_model = openai(
        model_id="gpt-4o-mini", api_key="test-key", top_p=0.85, frequency_penalty=0.2
    )
    chat_model.generate(prompt="ping")
    assert captured["top_p"] == 0.85
    assert captured["frequency_penalty"] == 0.2

    captured.clear()
    reasoning_model = openai(model_id="gpt-5.4-mini", api_key="test-key", top_p=0.85)
    with pytest.warns(UserWarning, match="top_p"):
        reasoning_model.generate(prompt="ping")
    assert "top_p" not in captured


def test_ollama_takes_repeat_penalty_and_refuses_frequency_penalty(monkeypatch):
    """LiteLLM renames frequency_penalty onto Ollama's repeat_penalty without
    rescaling, and their neutral points differ (0 vs 1.0), so datafast does not
    offer it. repeat_penalty itself rides provider_params."""
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _DummyChatResponse("ok")

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    model = ollama(model_id="gemma3:4b", top_p=0.85, frequency_penalty=0.15)

    with pytest.warns(UserWarning, match="frequency_penalty"):
        assert model.generate(prompt="ping") == "ok"

    assert captured["top_p"] == 0.85
    assert "frequency_penalty" not in captured

    captured.clear()
    ollama(model_id="gemma3:4b", repeat_penalty=1.2).generate(prompt="ping")
    assert captured["repeat_penalty"] == 1.2


def test_ollama_probe_capabilities_reads_the_daemon(monkeypatch):
    """Name heuristics cannot know which model is pulled, so the probe asks
    /api/show. This pins the request without a daemon running."""
    captured = {}

    class _DummyShowResponse:
        def json(self):
            return {"capabilities": ["completion", "vision", "tools", "thinking"]}

        def raise_for_status(self):
            return None

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _DummyShowResponse()

    monkeypatch.delenv("OLLAMA_API_BASE", raising=False)
    monkeypatch.setattr(served_model_module.httpx, "post", fake_post)

    capabilities = ollama(model_id="gemma4:12b").probe_capabilities()

    assert capabilities == frozenset({"completion", "vision", "tools", "thinking"})
    assert captured["url"] == "http://localhost:11434/api/show"
    assert captured["json"] == {"model": "gemma4:12b"}
    # No auth: Ollama is keyless, which is what no_api_key on the profile records.
    assert "headers" not in captured


def test_ollama_probe_capabilities_follows_the_generate_daemon(monkeypatch):
    """The probe must land on the same host the generate calls reach, so it resolves
    the base URL the way LiteLLM does: explicit api_base_url, then OLLAMA_API_BASE."""
    urls = []

    class _DummyShowResponse:
        def json(self):
            return {}

        def raise_for_status(self):
            return None

    def fake_post(url, **kwargs):
        urls.append(url)
        return _DummyShowResponse()

    monkeypatch.setattr(served_model_module.httpx, "post", fake_post)
    monkeypatch.setenv("OLLAMA_API_BASE", "http://gpu-box:11434")

    # A model with no capabilities key answers with an empty set rather than None.
    assert ollama(model_id="qwen3:8b").probe_capabilities() == frozenset()
    ollama(model_id="qwen3:8b", api_base_url="http://other-box:11434/").probe_capabilities()

    assert urls == [
        "http://gpu-box:11434/api/show",
        "http://other-box:11434/api/show",
    ]


def test_gemini_reasoning_capability_resolution():
    # All catalogued Gemini models support reasoning natively.
    for model_id in (
        "gemini-3.7-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
    ):
        caps = resolve_capabilities("gemini", model_id)
        assert caps.supports_reasoning is True
        assert "reasoning_effort" in caps.supported_params


def test_gemini_flash_lite_can_request_reasoning_off():
    """The lite line accepts 'minimal', which is what 'none' resolves to."""
    caps = resolve_capabilities("gemini", "gemini-3.5-flash-lite")

    assert caps.reasoning_always_on is False
    assert caps.reasoning_off_param == ("reasoning_effort", "none")


def test_gemini_flash_rejects_thinking_false():
    """gemini-3.7-flash has no level below 'low', so thinking=False has nothing
    to send. Refusing beats sending nothing and reasoning at 'medium'."""
    model = gemini(model_id="gemini-3.7-flash", api_key="test-key", thinking=False)

    with pytest.raises(ValueError, match="always reasons"):
        model.generate(prompt="ping")


def test_gemini_flash_rejects_an_effort_below_its_floor():
    model = gemini(
        model_id="gemini-3.7-flash", api_key="test-key", reasoning_effort="minimal"
    )

    with pytest.raises(ValueError, match="not supported"):
        model.generate(prompt="ping")


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
        provider_id="tgi",
        api_base_url="http://localhost:8000/v1",
        provider_params={"extra_body": {"backend_hint": "vllm"}},
    )

    assert model.generate(prompt="ping") == "ok"
    assert captured["extra_body"] == {"backend_hint": "vllm"}


def test_content_parts_normalize_multimodal_and_document_shapes():
    vllm = openai_compatible(
        "local-model",
        api_base_url="http://localhost:8000/v1",
        provider_id="vllm",
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
        provider_id="llamacpp",
    )
    prepared = llamacpp._prepare_messages(
        [
            {
                "role": "user",
                "content": [
                    ContentPart(
                        type="document",
                        data="abc",
                        media_type="application/pdf",
                    ),
                    ContentPart(
                        type="file",
                        data="data:application/pdf;base64,def",
                        filename="report.pdf",
                    ),
                ],
            }
        ],
        response_format=None,
    )

    # Raw base64 gets wrapped into a data URI, just like image parts; an
    # already-built URI passes through untouched.
    assert prepared[0]["content"] == [
        {
            "type": "file",
            "file": {"file_data": "data:application/pdf;base64,abc"},
        },
        {
            "type": "file",
            "file": {
                "file_data": "data:application/pdf;base64,def",
                "filename": "report.pdf",
            },
        },
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


def test_responses_endpoint_converts_file_parts_to_input_file(monkeypatch):
    """The Responses API rejects chat `file` parts, so they become `input_file`:
    inline data stays as `file_data`, an http(s) reference becomes `file_url`."""
    captured = {}

    def fake_responses(**kwargs):
        captured.update(kwargs)
        return _DummyResponsesResponse("ok")

    monkeypatch.setattr(served_model_module.litellm, "responses", fake_responses)

    model = openai(model_id="gpt-5.5", api_key="test-key")
    model.generate(
        messages=[
            {
                "role": "user",
                "content": [
                    ContentPart(
                        type="file",
                        data="abc",
                        media_type="application/pdf",
                        filename="report.pdf",
                    ),
                    ContentPart(type="file", url="https://example.com/report.pdf"),
                ],
            }
        ]
    )

    assert captured["input"][0]["content"] == [
        {
            "type": "input_file",
            "file_data": "data:application/pdf;base64,abc",
            "filename": "report.pdf",
        },
        {"type": "input_file", "file_url": "https://example.com/report.pdf"},
    ]


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


def test_openai_model_id_picks_the_endpoint_and_temperature_support(monkeypatch):
    """No openai catalog id maps to OPENAI_CHAT — a gpt-4-class id gets there through
    the reasoning-model prefix fallback, and only there is temperature supported."""
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return _DummyResponsesResponse("ok")

    monkeypatch.setattr(served_model_module.litellm, "responses", capture)
    monkeypatch.setattr(
        served_model_module.litellm,
        "completion",
        lambda **kwargs: captured.update(kwargs) or _DummyChatResponse("ok"),
    )

    reasoning_model = openai(
        model_id="gpt-5.4-mini", api_key="test-key", temperature=0.0
    )
    assert reasoning_model.endpoint_mode == EndpointMode.RESPONSES
    with pytest.warns(UserWarning, match="temperature"):
        reasoning_model.generate(prompt="ping")
    assert "temperature" not in captured

    captured.clear()
    chat_model = openai(model_id="gpt-4o-mini", api_key="test-key", temperature=0.0)
    assert chat_model.endpoint_mode == EndpointMode.CHAT
    chat_model.generate(prompt="ping")
    assert captured["temperature"] == 0.0


def test_openai_thinking_false_sends_the_off_effort(monkeypatch):
    """gpt-5.5 defaults to effort 'medium', so omitting the parameter would
    reason despite thinking=False. The off value is an effort like any other
    and must carry the Responses wrapper, not ride as a bare kwarg."""
    captured = {}

    def fake_responses(**kwargs):
        captured.update(kwargs)
        return _DummyResponsesResponse("ok")

    monkeypatch.setattr(served_model_module.litellm, "responses", fake_responses)

    model = openai(model_id="gpt-5.5", api_key="test-key", thinking=False)

    assert model.generate(prompt="ping") == "ok"
    assert captured["reasoning"] == {"effort": "none"}
    assert "reasoning_effort" not in captured


def test_openai_reasoning_summary_rides_with_the_effort(monkeypatch):
    """OpenAI returns a reasoning summary only when asked, and the ask shares the
    `reasoning` object with the effort — so it must be merged into it rather than
    replace it, which is all provider_params could ever do."""
    captured = {}

    def fake_responses(**kwargs):
        captured.update(kwargs)
        return _DummyResponsesResponse("ok", reasoning_content="because")

    monkeypatch.setattr(served_model_module.litellm, "responses", fake_responses)

    model = openai(
        model_id="gpt-5.5",
        api_key="test-key",
        thinking=True,
        reasoning_summary="auto",
    )

    response = model.generate_response(prompt="ping")
    assert captured["reasoning"] == {"effort": "low", "summary": "auto"}
    assert response.reasoning_content == "because"


def test_openai_reasoning_summary_alone_still_sends_the_reasoning_object(monkeypatch):
    """A summary with no effort is a valid ask: it leaves the served model's own
    default effort in force, so the reasoning object must still go out."""
    captured = {}

    def fake_responses(**kwargs):
        captured.update(kwargs)
        return _DummyResponsesResponse("ok")

    monkeypatch.setattr(served_model_module.litellm, "responses", fake_responses)

    model = openai(model_id="gpt-5.5", api_key="test-key", reasoning_summary="detailed")

    assert model.generate(prompt="ping") == "ok"
    assert captured["reasoning"] == {"summary": "detailed"}


def test_reasoning_summary_warns_where_it_cannot_be_carried(monkeypatch):
    """A summary rides inside the Responses `reasoning` object, so a chat endpoint
    has nowhere to put it — and thinking=False leaves nothing to summarise."""
    captured = {}

    monkeypatch.setattr(
        served_model_module.litellm,
        "completion",
        lambda **kwargs: captured.update(kwargs) or _DummyChatResponse("ok"),
    )
    monkeypatch.setattr(
        served_model_module.litellm,
        "responses",
        lambda **kwargs: captured.update(kwargs) or _DummyResponsesResponse("ok"),
    )

    chat_model = anthropic(api_key="test-key", thinking=True, reasoning_summary="auto")
    with pytest.warns(UserWarning, match="reasoning_summary"):
        chat_model.generate(prompt="ping")
    assert "reasoning" not in captured

    captured.clear()
    off_model = openai(
        model_id="gpt-5.5", api_key="test-key", thinking=False, reasoning_summary="auto"
    )
    with pytest.warns(UserWarning, match="reasoning_summary"):
        off_model.generate(prompt="ping")
    assert captured["reasoning"] == {"effort": "none"}


def test_fallback_batching_preserves_order(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs["messages"][0]["content"])
        return _DummyChatResponse(f"reply:{kwargs['messages'][0]['content']}")

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    model = openai_compatible(
        "local-model",
        provider_id="tgi",
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
