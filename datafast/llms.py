from .llm_utils import get_messages
import anthropic
from pydantic import BaseModel
import os
import instructor
import google.generativeai as genai
from openai import OpenAI
from huggingface_hub import InferenceClient
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Base class for LLM providers."""

    ENV_KEY_NAME: str = ""  # Override in subclasses
    DEFAULT_MODEL: str = ""  # Override in subclasses

    def __init__(self, model_id: str | None = None, api_key: str | None = None):
        self.model_id = model_id or self.DEFAULT_MODEL
        self.api_key = api_key or self._get_api_key()
        self.client = self._initialize_client()

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name"""
        pass

    def _get_api_key(self) -> str:
        """Get API key from environment"""
        api_key = os.getenv(self.ENV_KEY_NAME)
        if not api_key:
            raise ValueError(f"{self.ENV_KEY_NAME} environment variable is not set")
        return api_key

    @abstractmethod
    def _initialize_client(self):
        """Initialize the LLM client"""
        pass

    def generate(self, prompt: str | list[dict[str, str]], response_format: type[BaseModel]) -> BaseModel:
        """Generate a structured response from the LLM.

        Args:
            prompt (str, list[dict[str, str]]): The input prompt to send to the model
            response_format: A Pydantic model class defining the expected response
            structure

        Returns:
            An instance of the response_format model containing the structured
            response

        Example:
            class MovieReview(BaseModel):
                rating: int
                text: str

            provider = create_provider('anthropic')  # Uses default model
            review = provider.generate("Review Inception", MovieReview)
            print(f"Rating: {review.rating}")
        """
        try:
            return self._generate_impl(prompt, response_format)
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            raise RuntimeError(f"Error generating response with {self.name}:\n{error_trace}")

    @abstractmethod
    def _generate_impl(
        self, prompt: str | list[dict[str, str]], response_format: type[BaseModel]
    ) -> BaseModel:
        """Implementation of generate() to be provided by subclasses"""
        pass


class AnthropicProvider(LLMProvider):
    """Claude provider for structured text generation."""

    ENV_KEY_NAME = "ANTHROPIC_API_KEY"
    DEFAULT_MODEL = "claude-3-5-haiku-latest"

    def __init__(
        self,
        model_id: str | None = None, # TODO: are these arguments needed?
        api_key: str | None = None,
        max_tokens: int = 2056,
        temperature: float = 0.3,
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        super().__init__(model_id, api_key)

    @property
    def name(self) -> str:
        return "anthropic"

    def _initialize_client(self):
        try:
            anthropic_model = anthropic.Anthropic(api_key=self.api_key)
            return instructor.from_anthropic(
                anthropic_model, mode=instructor.Mode.ANTHROPIC_TOOLS
            )
        except Exception as e:
            raise ValueError(f"Error initializing Anthropic client: {str(e)}")

    def _generate_impl(
        self, prompt: str | list[dict[str, str]], response_format: type[BaseModel]
    ) -> BaseModel:
        return self.client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            messages=get_messages(prompt) if isinstance(prompt, str) else prompt,
            temperature=self.temperature,
            response_model=response_format,
        )


class GoogleProvider(LLMProvider):
    """Google Gemini provider for structured text generation."""

    ENV_KEY_NAME = "GOOGLE_API_KEY"
    DEFAULT_MODEL = "gemini-1.5-flash"

    @property
    def name(self) -> str:
        return "google"

    def _initialize_client(self):
        try:
            genai.configure(api_key=self.api_key)
            google_model = genai.GenerativeModel(model_name=self.model_id)
            return instructor.from_gemini(
                client=google_model, mode=instructor.Mode.GEMINI_JSON
            )
        except Exception as e:
            raise ValueError(
                f"Invalid model ID or model initialization error: {str(e)}"
            )

    def _generate_impl(
        self, prompt: str | list[dict[str, str]], response_format: type[BaseModel]
    ) -> BaseModel:
        return self.client.messages.create(
            messages=get_messages(prompt) if isinstance(prompt, str) else prompt,
            response_model=response_format,
        )


class OpenAIProvider(LLMProvider):
    """OpenAI provider for structured text generation."""

    ENV_KEY_NAME = "OPENAI_API_KEY"
    DEFAULT_MODEL = "gpt-4o-mini"

    @property
    def name(self) -> str:
        return "openai"

    def _initialize_client(self):
        try:
            openai_model = OpenAI(api_key=self.api_key)
            return instructor.from_openai(
                client=openai_model, mode=instructor.Mode.JSON
            )
        except Exception as e:
            raise ValueError(f"Error initializing OpenAI client: {str(e)}")

    def _generate_impl(
        self, prompt: str | list[dict[str, str]], response_format: type[BaseModel]
    ) -> BaseModel:
        return self.client.chat.completions.create(
            model=self.model_id,
            messages=get_messages(prompt) if isinstance(prompt, str) else prompt,
            response_model=response_format,
        )


class HuggingFaceProvider(LLMProvider):
    """Hugging Face provider for structured text generation."""

    ENV_KEY_NAME = "HF_TOKEN"
    DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

    @property
    def name(self) -> str:
        return "huggingface"

    def _initialize_client(self):
        try:
            return InferenceClient(model=self.model_id, api_key=self.api_key)
        except ImportError as e:
            raise ImportError(f"huggingface_hub package not installed. Install it with 'pip install huggingface_hub': {str(e)}")
        except Exception as e:
            raise ValueError(f"Error initializing Hugging Face client: {str(e)}")
    
    def _generate_impl(self, prompt: str | list[dict[str, str]], response_format: type[BaseModel]) -> BaseModel:
        # Check prompt type
        if isinstance(prompt, list):
            # For now, we don't support message-based prompts for HuggingFace
            raise NotImplementedError("Message-based prompts are not yet supported for the HuggingFace provider")
        else:
            prompt_text = prompt
        
        # Get schema for the response format
        schema = response_format.model_json_schema()
        
        # Generate response with structure enforcement
        response = self.client.text_generation(
            prompt=prompt_text,
            grammar={"type": "json", "value": schema},
        )
        
        # Parse and validate the response against the Pydantic model
        return response_format.model_validate_json(response)


class OllamaProvider(LLMProvider):
    """Ollama provider for structured text generation."""

    # No API key needed for local Ollama
    DEFAULT_MODEL = "gemma3:12b"

    @property
    def name(self) -> str:
        return "ollama"
        
    def _get_api_key(self) -> str:
        """Override _get_api_key since Ollama doesn't need an API key"""
        return "not_needed"  # Return a dummy value

    def _initialize_client(self):
        try:
            import ollama
            return ollama
        except ImportError as e:
            raise ImportError(f"Ollama package not installed. Install it with 'pip install ollama': {str(e)}")
        except Exception as e:
            raise ValueError(f"Error initializing Ollama client: {str(e)}")

    def _generate_impl(
        self, prompt: str | list[dict[str, str]], response_format: type[BaseModel]
    ) -> BaseModel:
        # Convert prompt to messages format if it's a string
        messages = get_messages(prompt) if isinstance(prompt, str) else prompt
        
        # Get schema for the response format
        schema = response_format.model_json_schema()
        
        # Call the Ollama chat API
        response = self.client.chat(
            messages=messages,
            model=self.model_id,
            format=schema,
        )
        
        # Parse the response content and validate against the Pydantic model
        # Unlike other providers that use instructor and return the parsed model directly,
        # we need to manually parse the JSON response here
        return response_format.model_validate_json(response.message.content)


def create_provider(
    provider: str, model_id: str | None = None, **kwargs
) -> LLMProvider:
    """Create an LLM provider for structured text generation.

    Args:
        provider: Provider name ('anthropic', 'google', 'openai', 'ollama')
        model_id: Optional model identifier. If not provided, uses provider's default
        **kwargs: Additional provider-specific arguments

    Returns:
        An initialized LLM provider
    """
    provider_map = {
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "openai": OpenAIProvider,
        "huggingface": HuggingFaceProvider,
        "ollama": OllamaProvider,
    }

    provider_class = provider_map.get(provider.lower())
    if not provider_class:
        raise ValueError(f"Unknown provider: {provider}")

    return provider_class(model_id=model_id, **kwargs)
