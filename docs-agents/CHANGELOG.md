# Changelog

## [Unreleased]

### Added

### Changed

- **Breaking: renamed the provider layer to the served-model vocabulary.** A *provider*
  is now strictly the server, a *model* is the LLM it serves, and a **served model** is
  the two together plus its configuration — the object you construct and call.
  - `LLMProvider` → `ServedModel`, `TargetConfig` → `ServedModelConfig`,
    `TargetCapabilities` → `ServedModelCapabilities`.
  - The seven per-provider subclasses (`OpenAIProvider`, `AnthropicProvider`, …) are now
    private. The lowercase factories — `openai`, `anthropic`, `gemini`, `mistral`,
    `openrouter`, `ollama`, `openai_compatible` — are the only public entry points and
    are unchanged. `ServedModel` is exported for type annotations.
  - Fields: `provider` → `provider_id`, `litellm_provider` → `litellm_route`,
    `provider_name` → `provider_id`.
  - `datafast/llm/provider.py` → `datafast/llm/served_model.py`; the `datafast.llms`
    compatibility module is removed — import from `datafast` or `datafast.llm`.
  - Langfuse trace metadata: the `datafast_provider` key is now `datafast_provider_id`,
    matching its sibling `datafast_model_id`. The tracing component for served-model
    calls is now `served_model.generate`.
  - `LLMStep(model=...)` is unchanged.

### Fixed

## [0.1.0] — YYYY-MM-DD

- Initial release.
