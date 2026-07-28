import datafast.llms as llms_module
from datafast.llms import OpenRouterProvider


class _DummyMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _DummyChoice:
    def __init__(self, content: str) -> None:
        self.message = _DummyMessage(content)


class _DummyResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_DummyChoice(content)]


def test_openrouter_single_messages_use_completion(monkeypatch):
    monkeypatch.setattr(llms_module, "load_env_once", lambda: None)
    monkeypatch.setattr(
        llms_module,
        "maybe_configure_langfuse_tracing",
        lambda load_env=False: False,
    )

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
    monkeypatch.setattr(llms_module, "load_env_once", lambda: None)
    monkeypatch.setattr(
        llms_module,
        "maybe_configure_langfuse_tracing",
        lambda load_env=False: False,
    )

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
