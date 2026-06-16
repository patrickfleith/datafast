"""Capability resolution for Datafast LLM targets."""

from __future__ import annotations

from datafast.llm.types import (
    BatchMode,
    CacheMode,
    EndpointMode,
    Modality,
    StructuredOutputMode,
    TargetCapabilities,
)


COMMON_CHAT_PARAMS = frozenset({
    "temperature",
    "max_completion_tokens",
    "timeout",
})

SAMPLING_CHAT_PARAMS = frozenset({
    "top_p",
    "frequency_penalty",
})

REASONING_PARAMS = frozenset({
    "thinking",
    "reasoning_effort",
})

RESPONSES_PARAMS = frozenset({
    "temperature",
    "max_completion_tokens",
    "timeout",
    "thinking",
    "reasoning_effort",
    "previous_response_id",
})


HOSTED_CHAT = TargetCapabilities(
    endpoint_modes=frozenset({EndpointMode.CHAT}),
    default_endpoint_mode=EndpointMode.CHAT,
    supported_params=COMMON_CHAT_PARAMS | SAMPLING_CHAT_PARAMS,
    structured_output=StructuredOutputMode.JSON_SCHEMA,
    batch_mode=BatchMode.LITELLM_BATCH,
    cache_mode=CacheMode.PROVIDER_PROMPT,
)


OPENAI_RESPONSES = TargetCapabilities(
    endpoint_modes=frozenset({EndpointMode.CHAT, EndpointMode.RESPONSES}),
    default_endpoint_mode=EndpointMode.RESPONSES,
    supported_params=RESPONSES_PARAMS | SAMPLING_CHAT_PARAMS,
    structured_output=StructuredOutputMode.JSON_SCHEMA,
    batch_mode=BatchMode.FALLBACK_CONCURRENCY,
    cache_mode=CacheMode.PROVIDER_PROMPT,
    supports_reasoning=True,
)


OPENAI_CHAT = TargetCapabilities(
    endpoint_modes=frozenset({EndpointMode.CHAT, EndpointMode.RESPONSES}),
    default_endpoint_mode=EndpointMode.CHAT,
    supported_params=COMMON_CHAT_PARAMS | SAMPLING_CHAT_PARAMS,
    structured_output=StructuredOutputMode.JSON_SCHEMA,
    batch_mode=BatchMode.LITELLM_BATCH,
    cache_mode=CacheMode.PROVIDER_PROMPT,
)


ANTHROPIC_CHAT = TargetCapabilities(
    endpoint_modes=frozenset({EndpointMode.CHAT}),
    default_endpoint_mode=EndpointMode.CHAT,
    supported_params=COMMON_CHAT_PARAMS | REASONING_PARAMS,
    structured_output=StructuredOutputMode.JSON_SCHEMA,
    batch_mode=BatchMode.LITELLM_BATCH,
    cache_mode=CacheMode.PROVIDER_PROMPT,
    supports_reasoning=True,
    supports_thinking=True,
)


OPENROUTER_CHAT = TargetCapabilities(
    endpoint_modes=frozenset({EndpointMode.CHAT}),
    default_endpoint_mode=EndpointMode.CHAT,
    supported_params=COMMON_CHAT_PARAMS | SAMPLING_CHAT_PARAMS,
    modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
    structured_output=StructuredOutputMode.JSON_SCHEMA,
    batch_mode=BatchMode.LITELLM_BATCH,
    cache_mode=CacheMode.ROUTER,
    notes=(
        "OpenRouter capabilities remain model and routed-provider dependent.",
        "Reasoning controls are omitted by default; pass provider_params for "
        "model-specific OpenRouter/LiteLLM escape hatches.",
    ),
)


OLLAMA_CHAT = TargetCapabilities(
    endpoint_modes=frozenset({EndpointMode.CHAT}),
    default_endpoint_mode=EndpointMode.CHAT,
    supported_params=COMMON_CHAT_PARAMS | SAMPLING_CHAT_PARAMS,
    structured_output=StructuredOutputMode.JSON_OBJECT,
    batch_mode=BatchMode.FALLBACK_CONCURRENCY,
    cache_mode=CacheMode.LOCAL_KV,
    no_api_key=True,
    notes=("Structured output uses Ollama JSON mode plus Datafast validation.",),
)


VLLM_CHAT = TargetCapabilities(
    endpoint_modes=frozenset({EndpointMode.CHAT, EndpointMode.RESPONSES}),
    default_endpoint_mode=EndpointMode.CHAT,
    supported_params=COMMON_CHAT_PARAMS | SAMPLING_CHAT_PARAMS,
    modalities=frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO}),
    structured_output=StructuredOutputMode.JSON_SCHEMA,
    batch_mode=BatchMode.FALLBACK_CONCURRENCY,
    cache_mode=CacheMode.LOCAL_KV,
    no_api_key=True,
    requires_chat_template=True,
    notes=(
        "vLLM exposes OpenAI-compatible chat and Responses endpoints, but "
        "feature coverage remains model and server-version dependent.",
        "Multimodal support depends on the served model; stable media UUIDs "
        "can be passed with ContentPart.media_id.",
    ),
)


LLAMACPP_CHAT = TargetCapabilities(
    endpoint_modes=frozenset({EndpointMode.CHAT}),
    default_endpoint_mode=EndpointMode.CHAT,
    supported_params=COMMON_CHAT_PARAMS | SAMPLING_CHAT_PARAMS,
    modalities=frozenset({
        Modality.TEXT,
        Modality.IMAGE,
        Modality.AUDIO,
        Modality.VIDEO,
        Modality.FILE,
    }),
    structured_output=StructuredOutputMode.JSON_SCHEMA,
    batch_mode=BatchMode.FALLBACK_CONCURRENCY,
    cache_mode=CacheMode.LOCAL_KV,
    no_api_key=True,
    requires_chat_template=True,
    notes=(
        "llama.cpp server is OpenAI-compatible for chat, with JSON schema "
        "support through response_format.",
        "Multimodal inputs and reasoning controls are model and build dependent; "
        "use provider_params for llama.cpp-specific extra_body fields.",
    ),
)


OPENAI_COMPATIBLE_CHAT = TargetCapabilities(
    endpoint_modes=frozenset({EndpointMode.CHAT, EndpointMode.RESPONSES}),
    default_endpoint_mode=EndpointMode.CHAT,
    supported_params=frozenset({"timeout"}),
    structured_output=StructuredOutputMode.PROMPTED_JSON,
    batch_mode=BatchMode.FALLBACK_CONCURRENCY,
    cache_mode=CacheMode.LOCAL_KV,
    no_api_key=True,
    requires_chat_template=True,
    notes=("OpenAI-compatible transport does not imply OpenAI feature support.",),
)


_CATALOG: dict[tuple[str, str], TargetCapabilities] = {
    ("openai", "gpt-5.5"): OPENAI_RESPONSES,
    ("openai", "gpt-5.4"): OPENAI_RESPONSES,
    ("openai", "gpt-5.4-mini"): OPENAI_RESPONSES,
    ("openai", "gpt-5.4-nano"): OPENAI_RESPONSES,
    ("anthropic", "claude-sonnet-4-6"): ANTHROPIC_CHAT,
    ("anthropic", "claude-haiku-4-5"): ANTHROPIC_CHAT,
    ("gemini", "gemini-2.5-pro"): HOSTED_CHAT,
    ("gemini", "gemini-3.5-flash"): HOSTED_CHAT,
    ("gemini", "gemini-3.1-flash-lite"): HOSTED_CHAT,
    ("mistral", "mistral-medium-3-5"): HOSTED_CHAT,
    ("mistral", "mistral-large-2512"): HOSTED_CHAT,
    ("mistral", "mistral-small-2603"): HOSTED_CHAT,
    ("mistral", "ministral-14b-2512"): OPENAI_COMPATIBLE_CHAT,
    ("mistral", "ministral-8b-2512"): OPENAI_COMPATIBLE_CHAT,
    ("mistral", "ministral-3b-2512"): OPENAI_COMPATIBLE_CHAT,
}

_PROVIDER_DEFAULTS: dict[str, TargetCapabilities] = {
    "anthropic": ANTHROPIC_CHAT,
    "gemini": HOSTED_CHAT,
    "llamacpp": LLAMACPP_CHAT,
    "mistral": HOSTED_CHAT,
    "ollama": OLLAMA_CHAT,
    "openrouter": OPENROUTER_CHAT,
    "vllm": VLLM_CHAT,
}

_OPENAI_COMPATIBLE_PROVIDERS = frozenset({
    "openai_compatible",
})


def resolve_capabilities(
    provider: str,
    model_id: str,
    *,
    api_base_url: str | None = None,
    explicit: TargetCapabilities | None = None,
) -> TargetCapabilities:
    """Resolve target capabilities with conservative defaults."""
    if explicit is not None:
        return explicit

    normalized_provider = provider.lower()
    normalized_model = model_id.lower()

    catalog_match = _CATALOG.get((normalized_provider, normalized_model))
    if catalog_match is not None:
        return catalog_match

    if normalized_provider == "openai":
        return _resolve_openai_capabilities(normalized_model)

    provider_default = _PROVIDER_DEFAULTS.get(normalized_provider)
    if provider_default is not None:
        return provider_default

    if normalized_provider in _OPENAI_COMPATIBLE_PROVIDERS:
        return OPENAI_COMPATIBLE_CHAT

    if api_base_url:
        return OPENAI_COMPATIBLE_CHAT

    return _unknown_capabilities()


def _resolve_openai_capabilities(model_id: str) -> TargetCapabilities:
    if _looks_like_openai_reasoning_model(model_id):
        return OPENAI_RESPONSES
    return OPENAI_CHAT


def _looks_like_openai_reasoning_model(model_id: str) -> bool:
    return (
        model_id.startswith("gpt-5")
        or model_id.startswith("o1")
        or model_id.startswith("o3")
        or model_id.startswith("o4")
    )


def _unknown_capabilities() -> TargetCapabilities:
    return TargetCapabilities(
        endpoint_modes=frozenset({EndpointMode.CHAT}),
        default_endpoint_mode=EndpointMode.CHAT,
        supported_params=frozenset({"timeout"}),
        structured_output=StructuredOutputMode.PROMPTED_JSON,
        batch_mode=BatchMode.FALLBACK_CONCURRENCY,
        notes=("Unknown target; optional Datafast parameters are omitted by default.",),
    )


__all__ = [
    "ANTHROPIC_CHAT",
    "HOSTED_CHAT",
    "LLAMACPP_CHAT",
    "OLLAMA_CHAT",
    "OPENAI_CHAT",
    "OPENAI_COMPATIBLE_CHAT",
    "OPENAI_RESPONSES",
    "OPENROUTER_CHAT",
    "VLLM_CHAT",
    "resolve_capabilities",
]
