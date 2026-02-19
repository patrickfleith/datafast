"""LLM providers using LiteLLM for unified API access."""

import os
from abc import ABC, abstractmethod
from typing import Any

import litellm
from loguru import logger


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        timeout: int | None = None,
    ) -> None:
        """
        Initialize the LLM provider.

        Args:
            model_id: The model identifier.
            api_key: API key (if None, will get from environment).
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling parameter (0.0 to 1.0).
            frequency_penalty: Penalty for token frequency (-2.0 to 2.0).
            timeout: Request timeout in seconds.
        """
        self.model_id = model_id
        self.api_key = api_key or self._get_api_key()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.timeout = timeout

        self._configure_env()
        logger.info(f"Initialized {self.provider_name} | Model: {self.model_id}")

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name used by LiteLLM."""
        pass

    @property
    @abstractmethod
    def env_key_name(self) -> str:
        """Return the environment variable name for API key."""
        pass

    def _get_api_key(self) -> str:
        """Get API key from environment variables."""
        api_key = os.getenv(self.env_key_name)
        if not api_key:
            logger.error(f"Missing API key | Set {self.env_key_name} environment variable")
            raise ValueError(
                f"{self.env_key_name} environment variable not set. "
                f"Please set it or provide an API key when initializing the provider."
            )
        return api_key

    def _configure_env(self) -> None:
        """Configure environment variables for API key."""
        if self.api_key:
            os.environ[self.env_key_name] = self.api_key

    def _get_model_string(self) -> str:
        """Get the full model string for LiteLLM."""
        return f"{self.provider_name}/{self.model_id}"

    def _build_completion_params(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Build parameters for LiteLLM completion call."""
        params: dict[str, Any] = {
            "model": self._get_model_string(),
            "messages": messages,
        }

        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        if self.timeout is not None:
            params["timeout"] = self.timeout

        return params

    def generate(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Generate a single completion from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            The generated text content.

        Raises:
            RuntimeError: If generation fails.
        """
        try:
            params = self._build_completion_params(messages)
            response = litellm.completion(**params)
            content = response.choices[0].message.content
            return content.strip() if content else ""

        except Exception as e:
            logger.error(
                f"Generation failed | Provider: {self.provider_name} | "
                f"Model: {self.model_id} | Error: {e}"
            )
            raise RuntimeError(f"Error generating response: {e}") from e

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_id={self.model_id!r})"


class OpenRouterProvider(LLMProvider):
    """OpenRouter provider using LiteLLM."""

    @property
    def provider_name(self) -> str:
        return "openrouter"

    @property
    def env_key_name(self) -> str:
        return "OPENROUTER_API_KEY"

    def __init__(
        self,
        model_id: str = "openai/gpt-4o-mini",
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        timeout: int | None = None,
    ) -> None:
        """
        Initialize the OpenRouter provider.

        Args:
            model_id: The model ID (e.g., "openai/gpt-4o-mini", "anthropic/claude-3-haiku").
            api_key: API key (if None, will get from OPENROUTER_API_KEY env var).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling parameter.
            frequency_penalty: Penalty for token frequency.
            timeout: Request timeout in seconds.
        """
        super().__init__(
            model_id=model_id,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            timeout=timeout,
        )


class OllamaProvider(LLMProvider):
    """Ollama provider using LiteLLM. Typically runs locally without API key."""

    @property
    def provider_name(self) -> str:
        return "ollama_chat"

    @property
    def env_key_name(self) -> str:
        return "OLLAMA_API_BASE"

    def _get_api_key(self) -> str:
        """Ollama doesn't require an API key."""
        return ""

    def __init__(
        self,
        model_id: str = "llama3",
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        timeout: int | None = None,
    ) -> None:
        """
        Initialize the Ollama provider.

        Args:
            model_id: The model ID (e.g., "llama3", "mistral", "codellama").
            api_base: Base URL for Ollama API (default: "http://localhost:11434").
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling parameter.
            frequency_penalty: Penalty for token frequency.
            timeout: Request timeout in seconds.
        """
        if api_base:
            os.environ["OLLAMA_API_BASE"] = api_base

        super().__init__(
            model_id=model_id,
            api_key="",
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            timeout=timeout,
        )


def openrouter(
    model_id: str = "openai/gpt-4o-mini",
    **kwargs,
) -> OpenRouterProvider:
    """
    Create an OpenRouter provider.

    Args:
        model_id: The model ID (e.g., "openai/gpt-4o-mini").
        **kwargs: Additional arguments passed to OpenRouterProvider.

    Returns:
        Configured OpenRouterProvider instance.

    Example:
        >>> model = openrouter("anthropic/claude-3-haiku", temperature=0.7)
    """
    return OpenRouterProvider(model_id=model_id, **kwargs)


def ollama(
    model_id: str = "llama3",
    **kwargs,
) -> OllamaProvider:
    """
    Create an Ollama provider.

    Args:
        model_id: The model ID (e.g., "llama3", "mistral").
        **kwargs: Additional arguments passed to OllamaProvider.

    Returns:
        Configured OllamaProvider instance.

    Example:
        >>> model = ollama("codellama", temperature=0.2)
    """
    return OllamaProvider(model_id=model_id, **kwargs)
