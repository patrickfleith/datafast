"""Capability resolution for Datafast served models."""

from __future__ import annotations

from datafast.llm.types import (
    BatchMode,
    CacheMode,
    EndpointMode,
    Modality,
    StructuredOutputMode,
    ServedModelCapabilities,
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

# previous_response_id is a Responses-API concept; reasoning models reject
# sampling controls such as temperature/top_p, so they are excluded here and
# surface through the unsupported_params policy instead.
RESPONSES_PARAMS = frozenset({
    "max_completion_tokens",
    "timeout",
    "thinking",
    "reasoning_effort",
    "previous_response_id",
})


HOSTED_CHAT = ServedModelCapabilities(
    endpoint_modes=frozenset({EndpointMode.CHAT}),
    default_endpoint_mode=EndpointMode.CHAT,
    supported_params=COMMON_CHAT_PARAMS | SAMPLING_CHAT_PARAMS,
    modalities=frozenset({Modality.TEXT, Modality.IMAGE, Modality.FILE}),
    structured_output=StructuredOutputMode.JSON_SCHEMA,
    batch_mode=BatchMode.LITELLM_BATCH,
    cache_mode=CacheMode.PROVIDER_PROMPT,
)


MISTRAL_REASONING_CHAT = ServedModelCapabilities(
    endpoint_modes=frozenset({EndpointMode.CHAT}),
    default_endpoint_mode=EndpointMode.CHAT,
    supported_params=(
        COMMON_CHAT_PARAMS | SAMPLING_CHAT_PARAMS | frozenset({"reasoning_effort"})
    ),
    modalities=frozenset({Modality.TEXT, Modality.IMAGE, Modality.FILE}),
    structured_output=StructuredOutputMode.JSON_SCHEMA,
    batch_mode=BatchMode.LITELLM_BATCH,
    cache_mode=CacheMode.PROVIDER_PROMPT,
    supports_reasoning=True,
    reasoning_requires_allowlist=True,
    notes=(
        "Reasoning is opt-in via reasoning_effort. Magistral models enable it "
        "natively; mistral-medium/small accept it server-side but LiteLLM only "
        "forwards it through the allowed_openai_params escape hatch.",
    ),
)


GEMINI_CHAT = ServedModelCapabilities(
    endpoint_modes=frozenset({EndpointMode.CHAT}),
    default_endpoint_mode=EndpointMode.CHAT,
    supported_params=(
        COMMON_CHAT_PARAMS | SAMPLING_CHAT_PARAMS | frozenset({"reasoning_effort"})
    ),
    modalities=frozenset({
        Modality.TEXT,
        Modality.IMAGE,
        Modality.AUDIO,
        Modality.VIDEO,
        Modality.FILE,
    }),
    structured_output=StructuredOutputMode.JSON_SCHEMA,
    batch_mode=BatchMode.LITELLM_BATCH,
    cache_mode=CacheMode.PROVIDER_PROMPT,
    supports_reasoning=True,
    notes=(
        "Reasoning is forwarded natively via reasoning_effort (thinking=True "
        "maps to effort 'low'); LiteLLM handles gemini/* without an allowlist.",
    ),
)


OPENAI_RESPONSES = ServedModelCapabilities(
    endpoint_modes=frozenset({EndpointMode.CHAT, EndpointMode.RESPONSES}),
    default_endpoint_mode=EndpointMode.RESPONSES,
    supported_params=RESPONSES_PARAMS,
    modalities=frozenset({Modality.TEXT, Modality.IMAGE, Modality.FILE}),
    structured_output=StructuredOutputMode.JSON_SCHEMA,
    batch_mode=BatchMode.FALLBACK_CONCURRENCY,
    cache_mode=CacheMode.PROVIDER_PROMPT,
    supports_reasoning=True,
)


OPENAI_CHAT = ServedModelCapabilities(
    endpoint_modes=frozenset({EndpointMode.CHAT, EndpointMode.RESPONSES}),
    default_endpoint_mode=EndpointMode.CHAT,
    supported_params=COMMON_CHAT_PARAMS | SAMPLING_CHAT_PARAMS,
    modalities=frozenset({Modality.TEXT, Modality.IMAGE, Modality.FILE}),
    structured_output=StructuredOutputMode.JSON_SCHEMA,
    batch_mode=BatchMode.LITELLM_BATCH,
    cache_mode=CacheMode.PROVIDER_PROMPT,
)


ANTHROPIC_CHAT = ServedModelCapabilities(
    endpoint_modes=frozenset({EndpointMode.CHAT}),
    default_endpoint_mode=EndpointMode.CHAT,
    supported_params=COMMON_CHAT_PARAMS | REASONING_PARAMS,
    modalities=frozenset({Modality.TEXT, Modality.IMAGE, Modality.FILE}),
    structured_output=StructuredOutputMode.JSON_SCHEMA,
    batch_mode=BatchMode.LITELLM_BATCH,
    cache_mode=CacheMode.PROVIDER_PROMPT,
    supports_reasoning=True,
    supports_thinking=True,
)


OPENROUTER_CHAT = ServedModelCapabilities(
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


OLLAMA_CHAT = ServedModelCapabilities(
    endpoint_modes=frozenset({EndpointMode.CHAT}),
    default_endpoint_mode=EndpointMode.CHAT,
    supported_params=COMMON_CHAT_PARAMS | SAMPLING_CHAT_PARAMS,
    modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
    structured_output=StructuredOutputMode.JSON_SCHEMA,
    batch_mode=BatchMode.FALLBACK_CONCURRENCY,
    cache_mode=CacheMode.LOCAL_KV,
    no_api_key=True,
    notes=(
        "Structured output uses Ollama schema-constrained decoding via "
        "LiteLLM's response_format translation, plus Datafast validation.",
        "Image input requires a vision-capable Ollama model (e.g. gemma3, "
        "gemma4, llama3.2-vision); text-only models will reject it server-side.",
    ),
)


OLLAMA_REASONING_CHAT = ServedModelCapabilities(
    endpoint_modes=frozenset({EndpointMode.CHAT}),
    default_endpoint_mode=EndpointMode.CHAT,
    supported_params=(
        COMMON_CHAT_PARAMS | SAMPLING_CHAT_PARAMS | frozenset({"reasoning_effort"})
    ),
    modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
    structured_output=StructuredOutputMode.JSON_SCHEMA,
    batch_mode=BatchMode.FALLBACK_CONCURRENCY,
    cache_mode=CacheMode.LOCAL_KV,
    no_api_key=True,
    supports_reasoning=True,
    notes=(
        "Thinking-capable models (deepseek-r1, qwen3, gpt-oss, magistral) accept "
        "reasoning via thinking/reasoning_effort; LiteLLM maps it onto Ollama's "
        "think parameter and normalizes the trace into reasoning_content.",
        "gpt-oss honors the effort level (low/medium/high); other thinking models "
        "treat any level as on/off.",
    ),
)


VLLM_CHAT = ServedModelCapabilities(
    endpoint_modes=frozenset({EndpointMode.CHAT, EndpointMode.RESPONSES}),
    default_endpoint_mode=EndpointMode.CHAT,
    supported_params=COMMON_CHAT_PARAMS | SAMPLING_CHAT_PARAMS,
    modalities=frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO}),
    structured_output=StructuredOutputMode.JSON_SCHEMA,
    batch_mode=BatchMode.FALLBACK_CONCURRENCY,
    cache_mode=CacheMode.LOCAL_KV,
    supports_media_uuid=True,
    no_api_key=True,
    requires_chat_template=True,
    notes=(
        "vLLM exposes OpenAI-compatible chat and Responses endpoints, but "
        "feature coverage remains model and server-version dependent.",
        "Multimodal support depends on the served model; stable media UUIDs "
        "can be passed with ContentPart.media_id.",
    ),
)


LLAMACPP_CHAT = ServedModelCapabilities(
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


OPENAI_COMPATIBLE_CHAT = ServedModelCapabilities(
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


_SERVED_MODEL_CATALOG: dict[tuple[str, str], ServedModelCapabilities] = {
    ("openai", "gpt-5.5"): OPENAI_RESPONSES,
    ("openai", "gpt-5.4"): OPENAI_RESPONSES,
    ("openai", "gpt-5.4-mini"): OPENAI_RESPONSES,
    ("openai", "gpt-5.4-nano"): OPENAI_RESPONSES,
    ("anthropic", "claude-sonnet-4-6"): ANTHROPIC_CHAT,
    ("anthropic", "claude-haiku-4-5"): ANTHROPIC_CHAT,
    ("gemini", "gemini-3.5-flash"): GEMINI_CHAT,
    ("gemini", "gemini-3.1-flash-lite"): GEMINI_CHAT,
    ("mistral", "mistral-medium-3-5"): MISTRAL_REASONING_CHAT,
    ("mistral", "mistral-large-2512"): HOSTED_CHAT,
    ("mistral", "mistral-small-2603"): MISTRAL_REASONING_CHAT,
    ("mistral", "ministral-14b-2512"): OPENAI_COMPATIBLE_CHAT,
    ("mistral", "ministral-8b-2512"): OPENAI_COMPATIBLE_CHAT,
    ("mistral", "ministral-3b-2512"): OPENAI_COMPATIBLE_CHAT,
}

_PROVIDER_DEFAULTS: dict[str, ServedModelCapabilities] = {
    "anthropic": ANTHROPIC_CHAT,
    "gemini": GEMINI_CHAT,
    "llamacpp": LLAMACPP_CHAT,
    "openrouter": OPENROUTER_CHAT,
    "vllm": VLLM_CHAT,
}

_OPENAI_COMPATIBLE_PROVIDERS = frozenset({
    "openai_compatible",
})


def resolve_capabilities(
    provider_id: str,
    model_id: str,
    *,
    api_base_url: str | None = None,
    explicit: ServedModelCapabilities | None = None,
) -> ServedModelCapabilities:
    """Resolve served-model capabilities with conservative defaults."""
    if explicit is not None:
        return explicit

    normalized_provider = provider_id.lower()
    normalized_model = model_id.lower()

    catalog_match = _SERVED_MODEL_CATALOG.get((normalized_provider, normalized_model))
    if catalog_match is not None:
        return catalog_match

    if normalized_provider == "openai":
        return _resolve_openai_capabilities(normalized_model)

    if normalized_provider == "mistral":
        return _resolve_mistral_capabilities(normalized_model)

    if normalized_provider == "ollama":
        return _resolve_ollama_capabilities(normalized_model)

    provider_default = _PROVIDER_DEFAULTS.get(normalized_provider)
    if provider_default is not None:
        return provider_default

    if normalized_provider in _OPENAI_COMPATIBLE_PROVIDERS:
        return OPENAI_COMPATIBLE_CHAT

    if api_base_url:
        return OPENAI_COMPATIBLE_CHAT

    return _unknown_capabilities()


def _resolve_openai_capabilities(model_id: str) -> ServedModelCapabilities:
    if _looks_like_openai_reasoning_model(model_id):
        return OPENAI_RESPONSES
    return OPENAI_CHAT


def _resolve_mistral_capabilities(model_id: str) -> ServedModelCapabilities:
    # Magistral is Mistral's reasoning family; LiteLLM enables reasoning_effort
    # for any model whose id contains "magistral". Everything else falls back to
    # the standard hosted-chat profile.
    if "magistral" in model_id:
        return MISTRAL_REASONING_CHAT
    return HOSTED_CHAT


# Ollama model families that accept the `think` parameter (LiteLLM maps
# reasoning_effort onto it). "-thinking" also matches models an author has
# explicitly tagged as a thinking variant, e.g. lfm2.5-thinking. Models outside
# this set reject reasoning.
_OLLAMA_REASONING_MODELS = (
    "deepseek-r1",
    "deepseek-v3.1",
    "qwen3",
    "qwq",
    "gpt-oss",
    "magistral",
    "gemma4",
    "-thinking",
)


def _resolve_ollama_capabilities(model_id: str) -> ServedModelCapabilities:
    # Only thinking-capable families accept a reasoning control; every other
    # Ollama model keeps the plain chat profile.
    if any(family in model_id for family in _OLLAMA_REASONING_MODELS):
        return OLLAMA_REASONING_CHAT
    return OLLAMA_CHAT


def _looks_like_openai_reasoning_model(model_id: str) -> bool:
    return (
        model_id.startswith("gpt-5")
        or model_id.startswith("o1")
        or model_id.startswith("o3")
        or model_id.startswith("o4")
    )


def _unknown_capabilities() -> ServedModelCapabilities:
    return ServedModelCapabilities(
        endpoint_modes=frozenset({EndpointMode.CHAT}),
        default_endpoint_mode=EndpointMode.CHAT,
        supported_params=frozenset({"timeout"}),
        structured_output=StructuredOutputMode.PROMPTED_JSON,
        batch_mode=BatchMode.FALLBACK_CONCURRENCY,
        notes=(
            "Unknown served model; optional Datafast parameters are omitted by default.",
        ),
    )


__all__ = [
    "ANTHROPIC_CHAT",
    "GEMINI_CHAT",
    "HOSTED_CHAT",
    "LLAMACPP_CHAT",
    "MISTRAL_REASONING_CHAT",
    "OLLAMA_CHAT",
    "OLLAMA_REASONING_CHAT",
    "OPENAI_CHAT",
    "OPENAI_COMPATIBLE_CHAT",
    "OPENAI_RESPONSES",
    "OPENROUTER_CHAT",
    "VLLM_CHAT",
    "resolve_capabilities",
]
