"""LLM providers for datafast using LiteLLM.

This module provides classes for different LLM providers (OpenAI, Anthropic, Gemini)
with a unified interface using LiteLLM under the hood.
"""

from typing import Any, Type, TypeVar
from abc import ABC, abstractmethod
import os
import time
import traceback

# Pydantic
from pydantic import BaseModel

# LiteLLM
import litellm
from litellm.utils import ModelResponse

# Internal imports
from .llm_utils import get_messages

# Type aliases for Python 3.10+
Message = dict[str, str]
Messages = list[Message]
T = TypeVar('T', bound=BaseModel)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        rpm_limit: int | None = None,
    ):
        """Initialize the LLM provider with common parameters.

        Args:
            model_id: The model identifier
            api_key: API key (if None, will get from environment)
            temperature: The sampling temperature to be used, between 0 and 2. Higher values like 0.8 produce more random outputs, while lower values like 0.2 make outputs more focused and deterministic
            max_completion_tokens: An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens.
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            frequency_penalty: Penalty for token frequency (-2.0 to 2.0)
        """
        self.model_id = model_id
        self.api_key = api_key or self._get_api_key()

        # Set generation parameters
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty

        # Rate limiting
        self.rpm_limit = rpm_limit
        self._request_timestamps: list[float] = []

        # Configure environment with API key if needed
        self._configure_env()

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

    def _respect_rate_limit(self) -> None:
        """Block execution to ensure we do not exceed the rpm_limit."""
        if self.rpm_limit is None:
            return
        current = time.monotonic()
        # Keep only timestamps within the last minute
        self._request_timestamps = [
            ts for ts in self._request_timestamps if current - ts < 60]
        if len(self._request_timestamps) < self.rpm_limit:
            return
        # Need to wait until the earliest request is outside the 60-second window
        earliest = self._request_timestamps[0]
        # Add a 1s margin to avoid accidental rate limit exceedance
        sleep_time = 61 - (current - earliest)
        if sleep_time > 0:
            print("Waiting for rate limit...")
            time.sleep(sleep_time)

    @staticmethod
    def _strip_code_fences(content: str) -> str:
        """Strip markdown code fences from content if present.
        
        Args:
            content: The content string that may contain code fences
            
        Returns:
            Content with code fences removed
        """
        if not content:
            return content
        
        content = content.strip()
        
        # Check for code fences with optional language identifier
        if content.startswith('```'):
            # Find the end of the first line (language identifier)
            first_newline = content.find('\n')
            if first_newline != -1:
                content = content[first_newline + 1:]
            else:
                # No newline after opening fence, remove just the fence
                content = content[3:]
        
        # Remove closing fence
        if content.endswith('```'):
            content = content[:-3]
        
        return content.strip()

    def generate(
        self,
        prompt: str | list[str] | None = None,
        messages: list[Messages] | Messages | None = None,
        response_format: Type[T] | None = None,
    ) -> str | list[str] | T | list[T]:
        """
        Generate responses from the LLM using single or batch inference.

        Args:
            prompt: Single text prompt (str) or list of text prompts for batch processing
            messages: Single message list or list of message lists for batch processing
            response_format: Optional Pydantic model class for structured output

        Returns:
            Single string/model or list of strings/models depending on input type.

        Raises:
            ValueError: If neither prompt nor messages is provided, or if both are provided.
            RuntimeError: If there's an error during generation.
        """
        # Validate inputs
        if prompt is None and messages is None:
            raise ValueError("Either prompts or messages must be provided")
        if prompt is not None and messages is not None:
            raise ValueError("Provide either prompts or messages, not both")

        # Determine if this is a single input or batch input
        single_input = False
        batch_prompts = None
        batch_messages = None

        if prompt is not None:
            if isinstance(prompt, str):
                # Single prompt - convert to batch
                batch_prompts = [prompt]
                single_input = True
            elif isinstance(prompt, list):
                # Already a list of prompts
                batch_prompts = prompt
                single_input = False
            else:
                raise ValueError("prompt must be a string or list of strings")

        if messages is not None:
            if isinstance(messages, list) and len(messages) > 0:
                # Check if it's a single message list or batch
                if isinstance(messages[0], dict):
                    # Single message list - convert to batch
                    batch_messages = [messages]
                    single_input = True
                elif isinstance(messages[0], list):
                    # Already a batch of message lists
                    batch_messages = messages
                    single_input = False
                else:
                    raise ValueError("Invalid messages format")
            else:
                raise ValueError("messages cannot be empty")

        try:
            # Append JSON formatting instructions if response_format is provided
            json_instructions = (
                "\nReturn only valid JSON. To do so, don't include ```json ``` markdown "
                "or code fences around the JSON. Use double quotes for all keys and values. "
                "Escape internal quotes and newlines (use \\n). Do not include trailing commas."
            )
            
            # Convert batch prompts to messages if needed
            batch_to_send = []
            if batch_prompts is not None:
                for one_prompt in batch_prompts:
                    # Append JSON instructions to prompt if response_format is provided
                    modified_prompt = one_prompt + json_instructions if response_format is not None else one_prompt
                    batch_to_send.append(get_messages(modified_prompt))
            else:
                batch_to_send = batch_messages
                # Append JSON instructions to the last user message if response_format is provided
                if response_format is not None:
                    for message_list in batch_to_send:
                        for msg in reversed(message_list):
                            if msg.get("role") == "user":
                                msg["content"] += json_instructions
                                break

            # Enforce rate limit per batch
            self._respect_rate_limit()

            # Prepare completion parameters for batch
            completion_params = {
                "model": self._get_model_string(),
                "messages": batch_to_send,
                "temperature": self.temperature,
                "max_tokens": self.max_completion_tokens,
                "top_p": self.top_p,
                "frequency_penalty": self.frequency_penalty,
            }
            if response_format is not None:
                completion_params["response_format"] = response_format

            # Call LiteLLM completion with batch messages
            response: list[ModelResponse] = litellm.batch_completion(
                **completion_params)

            # Record timestamp for rate limiting
            if self.rpm_limit is not None:
                self._request_timestamps.append(time.monotonic())

            # Extract content from each response
            results = []
            for one_response in response:
                content = one_response.choices[0].message.content
                
                if response_format is not None:
                    # Strip code fences before validation
                    content = self._strip_code_fences(content)
                    results.append(
                        response_format.model_validate_json(content))
                else:
                    # Strip leading/trailing whitespace for text responses
                    results.append(content.strip() if content else content)

            # Return single result for backward compatibility
            if single_input and len(results) == 1:
                return results[0]
            return results

        except Exception as e:
            error_trace = traceback.format_exc()
            raise RuntimeError(
                f"Error generating batch response with {self.provider_name}:\n{error_trace}"
            )


class OpenAIProvider(LLMProvider):
    """OpenAI provider using litellm."""

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def env_key_name(self) -> str:
        return "OPENAI_API_KEY"

    def __init__(
        self,
        model_id: str = "gpt-5-mini-2025-08-07",
        api_key: str | None = None,
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
    ):
        """Initialize the OpenAI provider.

        Args:
            model_id: The model ID (defaults to gpt-5-mini-2025-08-07)
            api_key: API key (if None, will get from environment)
            temperature: The sampling temperature to be used, between 0 and 2. Higher values like 0.8 produce more random outputs, while lower values like 0.2 make outputs more focused and deterministic
            max_completion_tokens: An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens.
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            frequency_penalty: Penalty for token frequency (-2.0 to 2.0)
        """
        super().__init__(
            model_id=model_id,
            api_key=api_key,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
        )


class AnthropicProvider(LLMProvider):
    """Anthropic provider using litellm."""

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def env_key_name(self) -> str:
        return "ANTHROPIC_API_KEY"

    def __init__(
        self,
        model_id: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        # top_p: float | None = None, # Not properly supported by anthropic models 4.5
        # frequency_penalty: float | None = None,  # Not supported by anthropic models 4.5
    ):
        """Initialize the Anthropic provider.

        Args:
            model_id: The model ID (defaults to claude-haiku-4-5-20251001)
            api_key: API key (if None, will get from environment)
            temperature: Temperature for generation (0.0 to 1.0)
            max_completion_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter (0.0 to 1.0)
        """
        super().__init__(
            model_id=model_id,
            api_key=api_key,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )


class GeminiProvider(LLMProvider):
    """Google Gemini provider using litellm."""

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def env_key_name(self) -> str:
        return "GEMINI_API_KEY"

    def __init__(
        self,
        model_id: str = "gemini-2.0-flash",
        api_key: str | None = None,
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        rpm_limit: int | None = None,
    ):
        """Initialize the Gemini provider.

        Args:
            model_id: The model ID (defaults to gemini-2.0-flash)
            api_key: API key (if None, will get from environment)
            temperature: Temperature for generation (0.0 to 1.0)
            max_completion_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            frequency_penalty: Penalty for token frequency (-2.0 to 2.0)
        """
        super().__init__(
            model_id=model_id,
            api_key=api_key,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            rpm_limit=rpm_limit,
        )


class OllamaProvider(LLMProvider):
    """Ollama provider using litellm.

    Note: Ollama typically doesn't require an API key as it's usually run locally.
    """

    @property
    def provider_name(self) -> str:
        return "ollama_chat"

    @property
    def env_key_name(self) -> str:
        return "OLLAMA_API_BASE"

    def _get_api_key(self) -> str:
        """Override to handle Ollama not requiring an API key.

        Returns an empty string since Ollama typically doesn't need an API key.
        OLLAMA_API_BASE can be used to set a custom base URL.
        """
        return ""

    def __init__(
        self,
        model_id: str = "gemma3:4b",
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        api_base: str | None = None,
        rpm_limit: int | None = None,
    ):
        """Initialize the Ollama provider.

        Args:
            model_id: The model ID (defaults to llama3)
            temperature: Temperature for generation (0.0 to 1.0)
            max_completion_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            frequency_penalty: Penalty for token frequency (-2.0 to 2.0)
            api_base: Base URL for Ollama API (e.g., "http://localhost:11434")
        """
        # Set API base URL if provided
        if api_base:
            os.environ["OLLAMA_API_BASE"] = api_base

        super().__init__(
            model_id=model_id,
            api_key="",  # Pass empty string since parent class requires this parameter
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            rpm_limit=rpm_limit,
        )


class OpenRouterProvider(LLMProvider):
    """OpenRouter provider using litellm"""

    @property
    def provider_name(self) -> str:
        return "openrouter"
    
    @property
    def env_key_name(self) -> str:
        return "OPENROUTER_API_KEY"
    
    def __init__(
            self,
            model_id: str = "openai/gpt-5-mini",  # for default model
            api_key: str | None = None,
            temperature: float | None = None,
            max_completion_tokens: int | None = None,
            top_p: float | None = None,
            frequency_penalty: float | None = None,
    ):
        """Initialize the OpenRouter provider.

        Args:
            model_id: The model ID (defaults to openai/gpt-5-mini)
            api_key: API key (if None, will get from environment)
            temperature: Temperature for generation (0.0 to 1.0)
            max_completion_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            frequency_penalty: Penalty for token frequency (-2.0 to 2.0)
        """
        super().__init__(
            model_id = model_id,
            api_key = api_key,
            temperature = temperature,
            max_completion_tokens = max_completion_tokens,
            top_p = top_p,
            frequency_penalty = frequency_penalty,
        )