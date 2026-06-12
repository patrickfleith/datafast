# LLM Provider Requirements (Draft)

## Goal

Design a clean model-provider layer for `datafast/llms.py` with one stable Datafast API, while resolving actual support per target model or deployment.

The key design rule is:

- The public API should provide a uniform core model.
- The public API should also provide ergonomic provider-specific entry points.
- Capabilities should be resolved per target: provider + endpoint + model + optional self-hosted server behavior.

## Core Design Principles

- Keep a small common config surface for normal usage.
- Do not assume all models under one provider support the same parameters.
- Do not silently pass unsupported parameters unless that behavior is explicitly enabled.
- Preserve provider or server defaults when the user does not override them.
- Separate Datafast-level config from provider-specific request mapping.

## Common Datafast Config

Every target should support these common fields when applicable:

- `model_id`
- `temperature`
- `rpm_limit`
- `timeout`

Optional fields, only sent when supported:

- `max_completion_tokens`
- `thinking`
- `reasoning_effort`
- `api_key`
- `api_base_url`
- retry limit
- `unsupported_params`

`unsupported_params` should control how Datafast handles user-specified parameters that are known to be unsupported by the resolved target.

- `fail`: raise a clear error before sending the request
- `warn`: omit the unsupported parameter and emit a warning
- `quiet`: omit the unsupported parameter silently

Default:

- `unsupported_params="warn"`

## Public API Ergonomics

The public API should expose provider-specific entry points such as:

- `openai(...)`
- `anthropic(...)`
- `openrouter(...)`
- `mistral(...)`
- `ollama(...)`

Requirements:

- Provider-specific entry points should be the primary ergonomic API for users.
- They should make provider choice explicit and easy to read in pipelines.
- They should expose sensible provider-specific defaults and validation.
- They should share the same common config surface where possible.
- They may expose provider-specific options when needed, without forcing those options into every provider API.
- They should remain thin wrappers over a shared internal target/config system.
- Core execution behavior such as retries, batching, capability resolution, caching, and parsing should not live separately in each provider wrapper.

## Capability Resolution

Requirements should be defined around resolved target capabilities, not provider classes alone.

That means:

- OpenAI-compatible transport does not imply OpenAI-equivalent features.
- OpenRouter support is model-specific, not just provider-specific.
- Local servers such as Ollama, vLLM, and `llama.cpp` may expose different controls even when they look OpenAI-compatible.
- Local servers may emulate an endpoint shape without matching the full upstream semantics.
- When support is unknown, the safe default is to omit optional params rather than optimistically send them.

The design should allow:

- capability mapping per model or deployment
- endpoint-mode resolution per target, especially chat completions vs Responses API
- provider-specific parameter aliases
- explicit escape hatches for provider-specific params
- controlled dropping of unsupported params when intentionally enabled

Unsupported-parameter handling should be explicit and user-configurable through `unsupported_params`.

- The policy should apply to Datafast-known unsupported parameters for the resolved target.
- The default behavior should be `warn`.
- `quiet` should be allowed for users who intentionally want best-effort portability.
- `fail` should be available for users who want strict validation.

Some targets may work best through `completion()` and others through `responses()`. The public Datafast API should not force users to care about that distinction, but the internal adapter layer should.

Requirements should also allow target-level compatibility notes such as:

- chat endpoint requires a compatible chat template
- a parameter is accepted but ignored
- an endpoint is available but implemented as an internal translation layer

## Request / Response Model

Datafast should expose one request model that supports:

- single request
- concurrent batch requests
- prompt input
- message input
- structured output via Pydantic

The execution layer should support both:

- native same-target batching for many inputs to one resolved model/deployment when available
- fallback concurrency when native batching is unavailable

If native batching is unavailable and Datafast falls back to parallel single requests, the user should be warned that a fallback execution path is being used.

The message model should support both:

- simple text messages
- typed multimodal content parts

Supported content parts should include a common shape for:

- text
- image
- audio
- video
- file
- document

This keeps the public API compatible with multimodal-capable chat models without forcing separate provider APIs for each modality.

Content parts should also be able to carry optional stable media IDs / UUIDs for targets that can reuse multimodal processing across requests.

## Multimodal Requirements

- Multimodal input support must be capability-aware per target.
- A model that supports text-only should still work with the same public call shape.
- A model that supports image, audio, video, document, or file inputs should accept typed content parts in `messages`.
- The design should also allow non-text outputs when supported, especially image-generation-capable chat models.
- Structured output and multimodal input should coexist when the target supports both.
- The design should support targets that expose multimodal and reasoning features primarily through the Responses API.
- The design should not assume all local backends support the same modalities. For example, support for image, audio, video, and document inputs may differ substantially between vLLM and `llama.cpp`.
- The design should allow target-specific media options when needed, without polluting the common API surface.

## Reliability and Execution

Every LLM call should have a standard execution policy:

- bounded retries
- exponential backoff
- jitter
- retryable vs non-retryable error handling
- consistent timeout handling
- client-side RPM throttling

Batch execution should:

- preserve input order
- apply the same retry and timeout rules as single requests
- use native same-target batching when available
- fall back to controlled concurrency when native batching is unavailable
- warn the user when fallback concurrency is used instead of native batching

## Endpoint Mode Requirements

The design should explicitly allow multiple endpoint modes behind one public API.

- Some targets should be called through chat completions.
- Some targets should be called through the Responses API.
- Endpoint choice should be resolved per target capability, not hardcoded per provider class.
- Responses API support matters for targets that expose reasoning, multimodal I/O, image generation, or session continuity through that endpoint.
- When the Responses API is used, the design should allow carrying forward response-session state such as `previous_response_id` when needed.
- The requirements should not assume that every Responses API implementation is native. A local backend may expose `/v1/responses` by translating it into another internal request shape.

## Caching Requirements

Caching should be part of the design, but not assumed to behave the same across targets.

The requirements should distinguish:

- provider-native prompt caching
- gateway or routing-layer caching
- local server prefix / KV caching
- optional client-side result caching

Key requirements:

- caching must be explicit and correctness-preserving
- cache behavior must be capability-aware per target
- cache keys or cache hints must account for model, endpoint, relevant generation params, and multimodal inputs
- provider-specific caching controls should be supported through the mapping layer or escape hatch
- the public API should not promise identical cache semantics across OpenAI, Anthropic, Mistral, OpenRouter, Ollama, vLLM, and `llama.cpp`

The requirements should also distinguish between:

- provider-side prompt caching semantics
- prefix / KV-cache reuse for repeated prompt prefixes
- multimodal preprocessing cache reuse keyed by stable media identity

In particular, local backends may expose caching mainly as performance-oriented KV reuse rather than provider-managed prompt caching. That should be modeled explicitly.

## What To Keep From The Current Design

The current `llms.py` points to a few good design directions that should remain in the requirements:

- one stable API for single and batch calls
- first-class structured output
- proactive client-side rate limiting
- standard retry behavior
- graceful fallback when a target lacks native batching
- support for local backends without requiring an API key
- tracing / metadata hooks on every request

## Recommended Direction

The optimal design is:

- provider-specific public factories as thin entry points
- one common Datafast request/config model
- one target capability layer
- one shared execution layer for retries, throttling, batching, caching, and parsing
- thin internal provider adapters that only map Datafast requests into target-specific LiteLLM calls

The capability layer should be able to describe at least:

- supported endpoint modes
- supported modalities
- structured-output mechanism
- cache mechanism type
- chat-template or prompt-format requirements
- parameter caveats such as unsupported, ignored, translated, or model-dependent

This keeps the user-facing API simple while allowing model-specific behavior where it actually belongs.

## References

- LiteLLM provider-specific params: <https://docs.litellm.ai/docs/completion/provider_specific_params>
- LiteLLM drop unsupported params: <https://docs.litellm.ai/docs/completion/drop_params>
- LiteLLM retries / fallbacks: <https://docs.litellm.ai/docs/completion/reliable_completions>
- LiteLLM batching: <https://docs.litellm.ai/docs/completion/batching>
- LiteLLM Responses API: <https://docs.litellm.ai/docs/response_api>
- LiteLLM structured output / JSON mode: <https://docs.litellm.ai/docs/completion/json_mode>
- LiteLLM reasoning content: <https://docs.litellm.ai/docs/reasoning_content>
- LiteLLM vision: <https://docs.litellm.ai/docs/completion/vision>
- LiteLLM audio: <https://docs.litellm.ai/docs/completion/audio>
- LiteLLM document understanding: <https://docs.litellm.ai/docs/completion/document_understanding>
- LiteLLM image generation in chat: <https://docs.litellm.ai/docs/completion/image_generation_chat>
- vLLM online serving: <https://docs.vllm.ai/en/latest/serving/online_serving/>
- vLLM structured outputs: <https://docs.vllm.ai/en/latest/features/structured_outputs/>
- vLLM automatic prefix caching: <https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/>
- vLLM multimodal inputs: <https://docs.vllm.ai/en/latest/features/multimodal_inputs/>
- llama.cpp server: <https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md>
- llama.cpp multimodal: <https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/multimodal.md>
