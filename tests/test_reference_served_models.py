"""The shared served-model reference, pinned against the config and the enums.

Template for the seven provider pages. The load-bearing check is the first one: a
configuration field that exists and is undocumented is a field nobody can use.
"""

import re
from dataclasses import fields
from pathlib import Path

import pytest

from datafast.llm.types import (
    BatchMode,
    EndpointMode,
    RetryPolicy,
    ServedModelConfig,
    StructuredOutputMode,
    UnsupportedParamsPolicy,
)

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "reference" / "served_models.md"

FACTORY_DEFAULTS = {
    "openai": ("gpt-5.5", "OPENAI_API_KEY"),
    "anthropic": ("claude-haiku-4-5", "ANTHROPIC_API_KEY"),
    "gemini": ("gemini-3.5-flash-lite", "GEMINI_API_KEY"),
    "mistral": ("mistral-small-2603", "MISTRAL_API_KEY"),
    "openrouter": ("openai/gpt-5.4-mini", "OPENROUTER_API_KEY"),
}


def _page() -> str:
    return PAGE.read_text()


def test_every_config_field_is_documented():
    declared = [f.name for f in fields(ServedModelConfig)]
    assert len(declared) > 15, "guard against an empty introspection"
    missing = [f for f in declared if f"`{f}`" not in _page()]
    assert not missing, f"ServedModelConfig fields absent from the page: {missing}"


def test_every_retry_policy_field_and_default_is_documented():
    for field in fields(RetryPolicy):
        assert f"`{field.name}`" in _page(), f"{field.name} undocumented"
        assert f"({field.default})" in _page(), f"{field.name}'s default is not on the page"


def test_the_documented_max_concurrent_default_is_the_real_one():
    default = next(f for f in fields(ServedModelConfig) if f.name == "max_concurrent").default
    assert default == 4 and "`4`" in _page()


@pytest.mark.parametrize("factory,expected", FACTORY_DEFAULTS.items(), ids=lambda x: str(x))
def test_the_factory_table_names_the_real_default_and_key(factory, expected):
    """Each row pairs a factory with the model and key its own code declares."""
    import inspect

    from datafast import llm

    model_id, env_key = expected
    signature = inspect.signature(getattr(llm, factory))
    assert signature.parameters["model_id"].default == model_id, "default model changed"
    assert re.search(rf"\|\s*`{re.escape(factory)}\(\)`\s*\|\s*`{re.escape(model_id)}`\s*\|\s*`{env_key}`", _page()), (
        f"the factory table should pair {factory}() with {model_id} and {env_key}"
    )


@pytest.mark.parametrize("policy", [p.value for p in UnsupportedParamsPolicy])
def test_every_unsupported_params_policy_is_documented(policy):
    assert f"`{policy}`" in _page()


@pytest.mark.parametrize("mode", [m.value for m in StructuredOutputMode])
def test_every_structured_output_mode_is_documented(mode):
    assert f"`{mode}`" in _page()


@pytest.mark.parametrize("mode", [m.value for m in BatchMode])
def test_every_batch_mode_is_documented(mode):
    assert f"`{mode}`" in _page()


@pytest.mark.parametrize("mode", [m.value for m in EndpointMode])
def test_every_endpoint_mode_is_documented(mode):
    assert f'"{mode}"' in _page() or f"`{mode}`" in _page()


def test_max_tokens_really_is_an_accepted_alias():
    """The page tells readers to use it, so it had better work."""
    from datafast import openai

    assert openai(max_tokens=128).config.max_completion_tokens == 128


def test_construction_does_not_validate_the_key_as_the_page_says(monkeypatch):
    from datafast import openai

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # .env is loaded once at first construction and would supply the key otherwise.
    monkeypatch.setattr("datafast.tracing.load_env_once", lambda: None)
    assert openai().config.api_key is None


def test_every_page_linked_to_exists():
    links = re.findall(r"\]\((?!https?:)([^)#]+\.md)", _page())
    missing = sorted(link for link in links if not (PAGE.parent / link).resolve().exists())
    assert not missing, f"links to pages that do not exist: {missing}"
