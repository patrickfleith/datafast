"""Capability resolution for Datafast served models."""

from __future__ import annotations

from dataclasses import replace

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

# Ollama speaks its own API rather than the OpenAI wire format, and its
# repetition control is repeat_penalty: a multiplier neutral at 1.0, where
# values below 1.0 *reward* repetition. LiteLLM renames frequency_penalty onto
# it without rescaling, so an OpenAI-style 0.15 arrives as strong repetition
# encouragement. Datafast therefore does not offer frequency_penalty here;
# repeat_penalty goes through provider_params, e.g. ollama(repeat_penalty=1.2).
OLLAMA_SAMPLING_CHAT_PARAMS = frozenset({"top_p"})

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
    files_require_file_id=True,
    supports_reasoning=True,
    reasoning_requires_allowlist=True,
    reasoning_effort_on="high",
    reasoning_off_param=("reasoning_effort", "none"),
    reasoning_efforts=frozenset({"high", "none"}),
    notes=(
        "Reasoning is opt-in via reasoning_effort. Magistral models enable it "
        "natively; mistral-medium/small accept it server-side but LiteLLM only "
        "forwards it through the allowed_openai_params escape hatch.",
        "The Mistral API accepts only 'high' and 'none'; 'low'/'medium' are "
        "rejected with a 400.",
        "File input means an uploaded file_id only: upload_file() returns one, "
        "and it goes in a file part's url. Mistral's chat API rejects inline "
        "base64 file data with a 422, so Datafast refuses it client-side.",
    ),
)


# Mistral models with no reasoning control. Identical to HOSTED_CHAT except for the
# file carrier, which is a property of the Mistral chat API rather than of the model.
MISTRAL_CHAT = replace(HOSTED_CHAT, files_require_file_id=True)


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
    reasoning_off_param=("reasoning_effort", "none"),
    notes=(
        "Reasoning is forwarded natively via reasoning_effort (thinking=True "
        "maps to effort 'low'); LiteLLM handles gemini/* without an allowlist.",
        "Reasoning must be requested off explicitly: Gemini 3 models think by "
        "default, so omitting the parameter still bills reasoning tokens.",
        "thinking=False does not mean no reasoning. Gemini 3 has no off "
        "switch — LiteLLM turns 'none' into the model's lowest thinking level "
        "with the trace hidden, so those tokens are still billed.",
        "temperature, top_p and top_k are deprecated for Gemini 3+ and slated "
        "for removal, and any temperature below 1.0 is warned against. They "
        "stay supported here because they still function and datafast never "
        "sends one unless a caller asks; drop them once Google removes them.",
    ),
)


# Gemini 3 models whose lowest thinking level is 'low' rather than 'minimal'.
# They reject the value 'none' resolves to, so there is nothing thinking=False
# could send: the only honest answer is to refuse it.
GEMINI_NO_MINIMAL_CHAT = replace(
    GEMINI_CHAT,
    reasoning_off_param=None,
    reasoning_always_on=True,
    reasoning_efforts=frozenset({"low", "medium", "high"}),
    notes=GEMINI_CHAT.notes[:1]
    + (
        "'minimal' is rejected outright, which is what 'none' maps to, so "
        "thinking=False raises rather than silently reasoning at the model's "
        "own default ('medium' for gemini-3.7-flash).",
        "At effort 'low' the reasoning is real but invisible: no "
        "reasoning_content and no thinking_blocks come back, only an opaque "
        "thought_signatures entry. Callers who need a readable trace should "
        "ask for a higher effort.",
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
    reasoning_locks_temperature=True,
    notes=(
        "Anthropic accepts only temperature=1 while thinking is enabled, so "
        "Datafast omits temperature on reasoning requests and lets the "
        "provider default apply.",
    ),
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
    supported_params=COMMON_CHAT_PARAMS | OLLAMA_SAMPLING_CHAT_PARAMS,
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
        "frequency_penalty is not supported: Ollama's repetition control is "
        "repeat_penalty, a multiplier neutral at 1.0 where lower values reward "
        "repetition, and LiteLLM renames frequency_penalty onto it without "
        "rescaling. Pass repeat_penalty through provider_params instead.",
    ),
)


OLLAMA_REASONING_CHAT = ServedModelCapabilities(
    endpoint_modes=frozenset({EndpointMode.CHAT}),
    default_endpoint_mode=EndpointMode.CHAT,
    supported_params=(
        COMMON_CHAT_PARAMS
        | OLLAMA_SAMPLING_CHAT_PARAMS
        | frozenset({"reasoning_effort"})
    ),
    modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
    structured_output=StructuredOutputMode.JSON_SCHEMA,
    batch_mode=BatchMode.FALLBACK_CONCURRENCY,
    cache_mode=CacheMode.LOCAL_KV,
    no_api_key=True,
    supports_reasoning=True,
    reasoning_off_param=("think", False),
    notes=(
        "Thinking-capable models (deepseek-r1, qwen3, gpt-oss, magistral) accept "
        "reasoning via thinking/reasoning_effort; LiteLLM maps it onto Ollama's "
        "think parameter and normalizes the trace into reasoning_content.",
        "gpt-oss honors the effort level (low/medium/high); other thinking models "
        "treat any level as on/off.",
        "Reasoning must be turned off explicitly: omitting the parameter leaves "
        "the model default, which is on for qwen3. think=false is passed "
        "directly because LiteLLM's reasoning_effort mapping sends the literal "
        "string for gpt-oss, which Ollama rejects.",
        "frequency_penalty is not supported: Ollama's repetition control is "
        "repeat_penalty, a multiplier neutral at 1.0 where lower values reward "
        "repetition, and LiteLLM renames frequency_penalty onto it without "
        "rescaling. Pass repeat_penalty through provider_params instead.",
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
    ("gemini", "gemini-3.7-flash"): GEMINI_NO_MINIMAL_CHAT,
    ("gemini", "gemini-3.5-flash"): GEMINI_CHAT,
    ("gemini", "gemini-3.5-flash-lite"): GEMINI_CHAT,
    ("gemini", "gemini-3.1-flash-lite"): GEMINI_CHAT,
    ("mistral", "mistral-medium-3-5"): MISTRAL_REASONING_CHAT,
    ("mistral", "mistral-large-2512"): MISTRAL_CHAT,
    ("mistral", "mistral-small-2603"): MISTRAL_REASONING_CHAT,
    # Ministral 3 is served from Mistral's own API — hosted chat with vision and
    # native schema support, not a self-hosted OpenAI-compatible endpoint.
    ("mistral", "ministral-14b-2512"): MISTRAL_CHAT,
    ("mistral", "ministral-8b-2512"): MISTRAL_CHAT,
    ("mistral", "ministral-3b-2512"): MISTRAL_CHAT,
}

_PROVIDER_DEFAULTS: dict[str, ServedModelCapabilities] = {
    "anthropic": ANTHROPIC_CHAT,
    "gemini": GEMINI_CHAT,
    "llamacpp": LLAMACPP_CHAT,
    "openrouter": OPENROUTER_CHAT,
    "vllm": VLLM_CHAT,
}


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

    # A self-hosted server we have no profile for: assume only what the
    # OpenAI-compatible wire format itself guarantees.
    if api_base_url:
        return OPENAI_COMPATIBLE_CHAT

    return _unknown_capabilities()


def _resolve_openai_capabilities(model_id: str) -> ServedModelCapabilities:
    if _looks_like_openai_reasoning_model(model_id):
        return OPENAI_RESPONSES
    return OPENAI_CHAT


def _resolve_mistral_capabilities(model_id: str) -> ServedModelCapabilities:
    # Magistral was Mistral's reasoning family, and LiteLLM still keys its
    # reasoning_effort support off that name. Reasoning has since moved into the
    # mainline models, which mark it in the id instead — Ministral 3 ships
    # "-reasoning" post-trained variants beside the instruct ones. Matching both
    # keeps an uncatalogued reasoning model from silently resolving to a profile
    # with reasoning switched off.
    if "magistral" in model_id or "-reasoning" in model_id:
        return MISTRAL_REASONING_CHAT
    return MISTRAL_CHAT


# Ollama model families that accept the `think` parameter (LiteLLM maps
# reasoning_effort onto it). "-thinking" also matches models an author has
# explicitly tagged as a thinking variant, e.g. lfm2.5-thinking. Models outside
# this set reject reasoning.
#
# Entries name families that are wholly thinking-capable, never a family with a
# mix: matching "nemotron" would catch nemotron-3-super, which does think, along
# with nemotron-mini and the Nemotron-70B-Instruct models, which do not. A false
# positive sends `think` to a model that rejects it, so the bar for adding a name
# is the whole family, and single models that think without saying so in their id
# are left to `probe_capabilities()`.
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
    "MISTRAL_CHAT",
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
