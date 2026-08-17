# Changelog

## [Unreleased]

### Added

- **`top_p` and `frequency_penalty` as real config fields.** Both were declared by
  the capability profiles but had no field, so a caller's value slipped through
  `provider_params` unchecked — including to `OPENAI_RESPONSES`, whose reasoning
  models reject sampling controls. They are now gated like `temperature`: sent
  where the profile declares them, dropped under the `unsupported_params` policy
  where it does not.
- **`repeat_penalty` documented as Ollama's repetition control.** Ollama speaks its
  own API, where the knob is a multiplier neutral at `1.0` and values below it
  *reward* repetition; LiteLLM renames `frequency_penalty` onto it without
  rescaling, so an OpenAI-style `0.15` degenerated the output. The Ollama profiles
  no longer declare `frequency_penalty`; pass `ollama(repeat_penalty=1.2)` on its
  own scale instead. vLLM and llama.cpp are unaffected — they are reached over the
  OpenAI wire format, where `frequency_penalty` keeps its usual meaning.
- **`reasoning_summary` on served models.** OpenAI returns a reasoning summary only
  when asked, and the ask shares the Responses `reasoning` object with the effort.
  `openai(reasoning_summary="auto")` merges into that object, so `reasoning_content`
  is now reachable without hand-writing the provider's request shape through
  `provider_params` — which replaced the whole `reasoning` key rather than adding
  to it. Declared by `OPENAI_RESPONSES` only; elsewhere it goes through the
  `unsupported_params` policy like any other unsupported parameter.

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
