from .llm_utils import get_messages
import anthropic
from pydantic import BaseModel
import os
import instructor
import google.generativeai as genai
from openai import OpenAI
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

    def generate(self, prompt: str, response_format: type[BaseModel]) -> BaseModel:
        """Generate a structured response from the LLM.

        Args:
            prompt: The input prompt to send to the model
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
            raise RuntimeError(f"Error generating response with {self.name}: {str(e)}")

    @abstractmethod
    def _generate_impl(
        self, prompt: str, response_format: type[BaseModel]
    ) -> BaseModel:
        """Implementation of generate() to be provided by subclasses"""
        pass


class AnthropicProvider(LLMProvider):
    """Claude provider for structured text generation."""

    ENV_KEY_NAME = "ANTHROPIC_API_KEY"
    DEFAULT_MODEL = "claude-3-5-haiku-latest"

    def __init__(
        self,
        model_id: str | None = None,
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
        self, prompt: str, response_format: type[BaseModel]
    ) -> BaseModel:
        return self.client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            messages=get_messages(prompt),
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
        self, prompt: str, response_format: type[BaseModel]
    ) -> BaseModel:
        return self.client.messages.create(
            messages=get_messages(prompt),
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
        self, prompt: str, response_format: type[BaseModel]
    ) -> BaseModel:
        return self.client.chat.completions.create(
            model=self.model_id,
            messages=get_messages(prompt),
            response_model=response_format,
        )


def create_provider(
    provider: str, model_id: str | None = None, **kwargs
) -> LLMProvider:
    """Create an LLM provider for structured text generation.

    Args:
        provider: Provider name ('anthropic', 'google', or 'openai')
        model_id: Optional model identifier. If not provided, uses provider's default
        **kwargs: Additional provider-specific arguments

    Returns:
        An initialized LLM provider
    """
    provider_map = {
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "openai": OpenAIProvider,
    }

    provider_class = provider_map.get(provider.lower())
    if not provider_class:
        raise ValueError(f"Unknown provider: {provider}")

    return provider_class(model_id=model_id, **kwargs)
