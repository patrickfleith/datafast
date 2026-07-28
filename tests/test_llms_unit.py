import pytest

import datafast.llm.provider as provider_module
import datafast.llms as llms_module
from datafast.llms import OpenRouterProvider


@pytest.fixture(autouse=True)
def _disable_provider_side_effects(monkeypatch):
    # Patch the bindings LLMProvider.__init__ actually calls (imported into
    # datafast.llm.provider); patching datafast.llms would be a no-op.
    monkeypatch.setattr(provider_module, "load_env_once", lambda: None)
    monkeypatch.setattr(
        provider_module,
        "maybe_configure_langfuse_tracing",
        lambda load_env=False: False,
    )


class _DummyMessage:
    def __init__(self, content: str, **extra: object) -> None:
        self.content = content
        for key, value in extra.items():
            setattr(self, key, value)


class _DummyChoice:
    def __init__(self, content: str, **extra: object) -> None:
        self.message = _DummyMessage(content, **extra)


class _DummyResponse:
    def __init__(self, content: str, **extra: object) -> None:
        self.choices = [_DummyChoice(content, **extra)]


def test_openrouter_single_messages_use_completion(monkeypatch):
    calls = {"completion": 0, "batch_completion": 0}

    def fake_completion(**kwargs):
        calls["completion"] += 1
        assert kwargs["messages"] == [{"role": "user", "content": "ping"}]
        return _DummyResponse("ok")

    def fake_batch_completion(**kwargs):
        calls["batch_completion"] += 1
        raise AssertionError("single-message requests should not use batch_completion")

    monkeypatch.setattr(llms_module.litellm, "completion", fake_completion)
    monkeypatch.setattr(llms_module.litellm, "batch_completion", fake_batch_completion)

    provider = OpenRouterProvider(model_id="demo-model", api_key="test-key")

    response = provider.generate(messages=[{"role": "user", "content": "ping"}])

    assert response == "ok"
    assert calls == {"completion": 1, "batch_completion": 0}


def test_openrouter_batch_messages_use_batch_completion(monkeypatch):
    calls = {"completion": 0, "batch_completion": 0}

    def fake_completion(**kwargs):
        calls["completion"] += 1
        raise AssertionError("batched requests should not use completion")

    def fake_batch_completion(**kwargs):
        calls["batch_completion"] += 1
        assert len(kwargs["messages"]) == 2
        return [_DummyResponse("first"), _DummyResponse("second")]

    monkeypatch.setattr(llms_module.litellm, "completion", fake_completion)
    monkeypatch.setattr(llms_module.litellm, "batch_completion", fake_batch_completion)

    provider = OpenRouterProvider(model_id="demo-model", api_key="test-key")

    response = provider.generate(messages=[
        [{"role": "user", "content": "one"}],
        [{"role": "user", "content": "two"}],
    ])

    assert response == ["first", "second"]
    assert calls == {"completion": 0, "batch_completion": 1}


def test_openrouter_generate_response_reads_reasoning_field(monkeypatch):
    monkeypatch.setattr(
        llms_module.litellm,
        "completion",
        lambda **kwargs: _DummyResponse(
            "final answer",
            reasoning="hidden chain of thought summary",
        ),
    )

    provider = OpenRouterProvider(model_id="demo-model", api_key="test-key")

    response = provider.generate_response(prompt="solve this")

    assert response.text == "final answer"
    assert response.reasoning_content == "hidden chain of thought summary"
