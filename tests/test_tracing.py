"""Tests for optional Langfuse tracing integration."""

from __future__ import annotations

import litellm
import pytest

from datafast import LLMStep, ListSink, OllamaProvider, Source
from datafast.tracing import configure_langfuse_tracing
import datafast.tracing as tracing_module


@pytest.fixture(autouse=True)
def reset_tracing_state(monkeypatch):
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "failure_callback", [])
    monkeypatch.setattr(tracing_module, "_ENV_LOADED", True)
    monkeypatch.setattr(tracing_module, "_LANGFUSE_AUTO_DISABLED", False)
    monkeypatch.setattr(tracing_module, "_MISSING_LANGFUSE_WARNING_EMITTED", False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)


def test_configure_langfuse_tracing_registers_callbacks_once(monkeypatch):
    monkeypatch.setattr(
        tracing_module.importlib.util,
        "find_spec",
        lambda name: object() if name == "langfuse" else None,
    )

    enabled = configure_langfuse_tracing(
        public_key="pk-test",
        secret_key="sk-test",
        host="https://langfuse.example",
        load_env=False,
    )

    assert enabled is True
    assert litellm.success_callback == ["langfuse"]
    assert litellm.failure_callback == ["langfuse"]

    enabled_again = configure_langfuse_tracing(
        public_key="pk-test",
        secret_key="sk-test",
        load_env=False,
    )

    assert enabled_again is True
    assert litellm.success_callback == ["langfuse"]
    assert litellm.failure_callback == ["langfuse"]


def test_configure_langfuse_tracing_warns_when_dependency_missing(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setattr(tracing_module.importlib.util, "find_spec", lambda name: None)

    with pytest.warns(RuntimeWarning, match="pip install datafast\\[langfuse\\]"):
        enabled = configure_langfuse_tracing(load_env=False, strict=False)

    assert enabled is False
    assert litellm.success_callback == []
    assert litellm.failure_callback == []


def test_ollama_provider_auto_enables_langfuse_from_environment(monkeypatch):
    monkeypatch.setattr(tracing_module, "_ENV_LOADED", False)
    monkeypatch.setattr(tracing_module, "load_dotenv", lambda override=False: None)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setattr(
        tracing_module.importlib.util,
        "find_spec",
        lambda name: object() if name == "langfuse" else None,
    )

    provider = OllamaProvider(model_id="gemma3:4b")

    assert provider.model_id == "gemma3:4b"
    assert litellm.success_callback == ["langfuse"]
    assert litellm.failure_callback == ["langfuse"]


def test_explicit_disable_prevents_auto_reenable(monkeypatch):
    monkeypatch.setattr(tracing_module, "_ENV_LOADED", False)
    monkeypatch.setattr(tracing_module, "load_dotenv", lambda override=False: None)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setattr(
        tracing_module.importlib.util,
        "find_spec",
        lambda name: object() if name == "langfuse" else None,
    )

    configure_langfuse_tracing(enabled=False, load_env=False)
    provider = OllamaProvider(model_id="gemma3:4b")

    assert provider.model_id == "gemma3:4b"
    assert litellm.success_callback == []
    assert litellm.failure_callback == []


def test_runner_attaches_datafast_trace_metadata():
    class FakeModel:
        provider_name = "fake"
        model_id = "fake-model"

        def __init__(self) -> None:
            self.calls: list[dict] = []

        def generate(self, messages, metadata=None, response_format=None):
            self.calls.append({"messages": messages, "metadata": metadata})
            return "done"

    model = FakeModel()
    sink = ListSink()

    pipeline = (
        Source.list([{"topic": "robotics"}])
        >> LLMStep(
            prompt="Write one short line about {topic}.",
            input_columns=["topic"],
            output_column="result",
            model=model,
        ).as_step("generate_copy")
        >> sink
    )

    output = pipeline.run()

    assert output == [{"topic": "robotics", "result": "done", "_model": "fake-model"}]
    assert len(model.calls) == 1

    metadata = model.calls[0]["metadata"]
    assert metadata["trace_name"] == "datafast.generate_copy"
    assert metadata["datafast_step"] == "generate_copy"
    assert metadata["datafast_step_type"] == "LLMStep"
    assert metadata["datafast_provider"] == "fake"
    assert metadata["datafast_model_id"] == "fake-model"
    assert metadata["datafast_record_index"] == 0
    assert metadata["datafast_prompt_index"] == 0
    assert metadata["datafast_output_index"] == 0
    assert metadata["datafast_call_id"] == "0_0_fake-model__0"
    assert metadata["session_id"].startswith("datafast-run-")
