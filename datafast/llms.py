from .llm_utils import get_messages
from pydantic import BaseModel, Field
import os
from typing import Any, Dict, List, Optional, Type, Union
import litellm
from litellm.utils import ModelResponse


class LLMProvider:
    """Unified LLM provider using LiteLLM for all model providers."""

    # Default models for each provider
    DEFAULT_MODELS = {
        "anthropic": "claude-3-5-haiku-latest",
        "gemini": "gemini-1.5-flash",
        "openai": "gpt-4o-mini",
        "huggingface": "meta-llama/Llama-3.3-70B-Instruct",
        "ollama": "gemma3:12b",
    }

    # Environment variable names for API keys
    ENV_KEYS = {
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "huggingface": "HF_TOKEN",
        "ollama": "not_needed",  # Ollama doesn't need an API key
    }

    def __init__(
        self, 
        provider: str,
        model_id: str | None = None, 
        api_key: str | None = None,
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
    ):
        """Initialize the LLM provider.
        
        Args:
            provider: Provider name ('anthropic', 'gemini', 'openai', 'huggingface', 'ollama')
            model_id: Optional model identifier. If not provided, uses provider's default
            api_key: Optional API key. If not provided, gets from environment
            temperature: Temperature for generation (0.0 to 1.0)
            max_completion_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            frequency_penalty: Penalty for token frequency (-2.0 to 2.0)
            Note: We align with LiteLLM Input Params: https://docs.litellm.ai/docs/completion/input
        """
        self.provider = provider.lower()
        
        # Validate provider
        if self.provider not in self.DEFAULT_MODELS:
            raise ValueError(f"Unknown provider: {provider}")
            
        # Set model ID (use default if not provided)
        self.model_id = model_id or self.DEFAULT_MODELS.get(self.provider, "")
        if self.model_id == "":
            raise ValueError(f"No model ID provided for provider: {self.provider}")
        
        # Set API key (get from environment if not provided)
        self.api_key = api_key or self._get_api_key()
        
        # Set generation parameters
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        
        # Configure LiteLLM with the API key if needed
        if self.provider in self.ENV_KEYS and self.api_key != "not_needed":
            env_key = self.ENV_KEYS[self.provider]
            os.environ[env_key] = self.api_key

    def _get_api_key(self) -> str:
        """Get API key from environment"""
        env_key = self.ENV_KEYS.get(self.provider)
        if not env_key:
            return ""  # No API key needed or unknown provider
            
        api_key = os.getenv(env_key)
        if not api_key:
            raise ValueError(f"{env_key} environment variable is not set or unknown for this provider.")
            
        return api_key

    def _get_model_string(self) -> str:
        """Get the full model string for LiteLLM"""
        return f"{self.provider}/{self.model_id}"

    def generate(self, prompt: str | list[dict[str, str]], response_format: type[BaseModel] | None = None) -> str | BaseModel:
        """Generate a response from the LLM.

        Args:
            prompt: The input prompt to send to the model (string or message list)
            response_format: Optional Pydantic model class defining the expected response structure.
                            If provided, returns a structured response as a Pydantic model instance.
                            If None, returns the raw text response as a string.

        Returns:
            Either a string response or a Pydantic model instance if response_format is provided
        """
        try:
            # Convert string prompt to messages if needed
            if isinstance(prompt, str):
                messages = get_messages(prompt)
            else:
                messages = prompt
            
            # Prepare completion parameters
            completion_params = {
                "model": self._get_model_string(),
                "messages": messages,
                "temperature": self.temperature,
                "max_completion_tokens": self.max_completion_tokens,
                "top_p": self.top_p,
                "frequency_penalty": self.frequency_penalty,
            }
            
            # Add response format if provided
            if response_format:
                # For OpenAI, we need to specify that we want JSON output
                completion_params["response_format"] = response_format
            
            # Call LiteLLM completion with the appropriate parameters
            response = litellm.completion(**completion_params)
            
            # Extract the content from the response
            content = response.choices[0].message.content
            
            # Parse and validate if response_format is provided, otherwise return raw content
            if response_format:
                return response_format.model_validate_json(content)
            else:
                return content
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            raise RuntimeError(f"Error generating response with {self.provider}:\n{error_trace}")


def create_provider(
    provider: str, 
    model_id: str | None = None, 
    api_key: str | None = None,
    temperature: float | None = None,
    max_completion_tokens: int | None = None,
    top_p: float | None = None,
    frequency_penalty: float | None = None,
) -> LLMProvider:
    """Create an LLM provider for structured text generation.

    Args:
        provider: Provider name ('anthropic', 'gemini', 'openai', 'huggingface', 'ollama')
        model_id: Optional model identifier. If not provided, uses provider's default
        api_key: Optional API key. If not provided, gets from environment
        temperature: Temperature for generation (0.0 to 1.0)
        max_completion_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter (0.0 to 1.0)
        frequency_penalty: Penalty for token frequency (-2.0 to 2.0)

    Returns:
        An initialized LLM provider
    """
    return LLMProvider(
        provider=provider, 
        model_id=model_id, 
        api_key=api_key,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty
    )
