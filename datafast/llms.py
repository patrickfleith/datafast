"""LLM providers for datafast using LiteLLM.

This module provides classes for different LLM providers (OpenAI, Anthropic, Gemini)
with a unified interface using LiteLLM under the hood.
"""

from typing import Any, Type, TypeVar
from abc import ABC, abstractmethod
import os
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
    
    def generate(
        self, 
        prompt: str | None = None, 
        messages: Messages | None = None,
        response_format: Type[T] | None = None
    ) -> str | T:
        """Generate a response from the LLM.

        Args:
            prompt: Text prompt (use either prompt or messages, not both)
            messages: List of message dictionaries with role and content (use either prompt or messages, not both)
            response_format: Optional Pydantic model class for structured output

        Returns:
            Either a string response or a Pydantic model instance if response_format is provided
        
        Raises:
            ValueError: If neither prompt nor messages is provided, or if both are provided
            RuntimeError: If there's an error during generation
        """
        # Validate inputs
        if prompt is None and messages is None:
            raise ValueError("Either prompt or messages must be provided")
        if prompt is not None and messages is not None:
            raise ValueError("Provide either prompt or messages, not both")
        
        try:
            # Convert string prompt to messages if needed
            if prompt is not None:
                messages_to_send = get_messages(prompt)
            else:
                messages_to_send = messages
            
            # Prepare completion parameters
            completion_params = {
                "model": self._get_model_string(),
                "messages": messages_to_send,
                "temperature": self.temperature,
                "max_tokens": self.max_completion_tokens,
                "top_p": self.top_p,
                "frequency_penalty": self.frequency_penalty,
            }
            
            # Add response format if provided
            if response_format is not None:
                completion_params["response_format"] = response_format
            
            # Call LiteLLM completion
            response: ModelResponse = litellm.completion(**completion_params)
            
            # Extract content from response
            content = response.choices[0].message.content
            
            # Parse and validate if response_format is provided
            if response_format is not None:
                return response_format.model_validate_json(content)
            else:
                return content
                
        except Exception as e:
            error_trace = traceback.format_exc()
            raise RuntimeError(f"Error generating response with {self.provider_name}:\n{error_trace}")


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
        model_id: str = "gpt-4.1-mini-2025-04-14",
        api_key: str | None = None,
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
    ):
        """Initialize the OpenAI provider.
        
        Args:
            model_id: The model ID (defaults to gpt-4.1-mini-2025-04-14)
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
        model_id: str = "claude-3-5-haiku-latest",
        api_key: str | None = None,
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        top_p: float | None = None,
        # frequency_penalty: float | None = None,  # Not supported by anthropic
    ):
        """Initialize the Anthropic provider.
        
        Args:
            model_id: The model ID (defaults to claude-3-5-haiku-latest)
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
            top_p=top_p,
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
        )
