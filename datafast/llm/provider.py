"""Provider exports for the pipeline-first datafast API."""

from datafast.llms import (
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    MistralProvider,
    OpenRouterProvider,
    OllamaProvider,
)


def openai(model_id: str = "gpt-5-mini-2025-08-07", **kwargs) -> OpenAIProvider:
    """Create an OpenAI provider instance."""
    return OpenAIProvider(model_id=model_id, **kwargs)


def anthropic(
    model_id: str = "claude-haiku-4-5-20251001",
    **kwargs,
) -> AnthropicProvider:
    """Create an Anthropic provider instance."""
    return AnthropicProvider(model_id=model_id, **kwargs)


def gemini(model_id: str = "gemini-2.0-flash", **kwargs) -> GeminiProvider:
    """Create a Gemini provider instance."""
    return GeminiProvider(model_id=model_id, **kwargs)


def mistral(model_id: str = "mistral-small-latest", **kwargs) -> MistralProvider:
    """Create a Mistral provider instance."""
    return MistralProvider(model_id=model_id, **kwargs)


def openrouter(
    model_id: str = "openai/gpt-5-mini",
    **kwargs,
) -> OpenRouterProvider:
    """Create an OpenRouter provider instance."""
    return OpenRouterProvider(model_id=model_id, **kwargs)


def ollama(model_id: str = "gemma3:4b", **kwargs) -> OllamaProvider:
    """Create an Ollama provider instance."""
    return OllamaProvider(model_id=model_id, **kwargs)


__all__ = [
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "MistralProvider",
    "OpenRouterProvider",
    "OllamaProvider",
    "openai",
    "anthropic",
    "gemini",
    "mistral",
    "openrouter",
    "ollama",
]
