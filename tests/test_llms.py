from datafast.llms import OpenAIProvider, AnthropicProvider, GeminiProvider, OllamaProvider
from dotenv import load_dotenv
import pytest
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

load_dotenv('secrets.env')


class SimpleResponse(BaseModel):
    """Simple response model for testing structured output."""
    answer: str = Field(description="The answer to the question")
    reasoning: str = Field(description="The reasoning behind the answer")


class Attribute(BaseModel):
    """Attribute of a landmark with value and importance."""
    name: str = Field(description="Name of the attribute")
    value: str = Field(description="Value of the attribute")
    importance: float = Field(description="Importance score between 0 and 1")
    
    @field_validator('importance')
    @classmethod
    def check_importance(cls, v: float) -> float:
        """Validate importance is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Importance must be between 0 and 1")
        return v


class LandmarkInfo(BaseModel):
    """Information about a landmark with attributes."""
    name: str = Field(description="The name of the landmark")
    location: str = Field(description="Where the landmark is located")
    description: str = Field(description="A brief description of the landmark")
    year_built: Optional[int] = Field(None, description="Year when the landmark was built")
    attributes: List[Attribute] = Field(description="List of attributes about the landmark")
    visitor_rating: float = Field(description="Average visitor rating from 0 to 5")
    
    @field_validator('visitor_rating')
    @classmethod
    def check_rating(cls, v: float) -> float:
        """Validate rating is between 0 and 5."""
        if not 0 <= v <= 5:
            raise ValueError("Rating must be between 0 and 5")
        return v

@pytest.mark.integration
def test_openai_provider():
    """Test the OpenAI provider with text response."""
    provider = OpenAIProvider()
    response = provider.generate(prompt="What is the capital of France? Answer in one word.")
    assert "Paris" in response

@pytest.mark.integration
def test_anthropic_provider():
    """Test the Anthropic provider with text response."""
    provider = AnthropicProvider()
    response = provider.generate(prompt="What is the capital of France? Answer in one word.")
    assert "Paris" in response

@pytest.mark.integration
def test_gemini_provider():
    """Test the Gemini provider with text response."""
    provider = GeminiProvider()
    response = provider.generate(prompt="What is the capital of France? Answer in one word.")
    assert "Paris" in response


@pytest.mark.integration
def test_openai_structured_output():
    """Test the OpenAI provider with structured output."""
    provider = OpenAIProvider()
    prompt = """What is the capital of France? 
    Provide a short answer and a brief explanation of why Paris is the capital.
    Format your response as JSON with 'answer' and 'reasoning' fields."""
    
    response = provider.generate(
        prompt=prompt,
        response_format=SimpleResponse
    )
    
    assert isinstance(response, SimpleResponse)
    assert "Paris" in response.answer
    assert len(response.reasoning) > 10  # Make sure we have some reasoning text


@pytest.mark.integration
def test_anthropic_structured_output():
    """Test the Anthropic provider with structured output."""
    provider = AnthropicProvider()
    prompt = """What is the capital of France? 
    Provide a short answer and a brief explanation of why Paris is the capital.
    Format your response as JSON with 'answer' and 'reasoning' fields."""
    
    response = provider.generate(
        prompt=prompt,
        response_format=SimpleResponse
    )
    
    assert isinstance(response, SimpleResponse)
    assert "Paris" in response.answer
    assert len(response.reasoning) > 10


@pytest.mark.integration
def test_gemini_structured_output():
    """Test the Gemini provider with structured output."""
    provider = GeminiProvider()
    prompt = """What is the capital of France? 
    Provide a short answer and a brief explanation of why Paris is the capital.
    Format your response as JSON with 'answer' and 'reasoning' fields."""
    
    response = provider.generate(
        prompt=prompt,
        response_format=SimpleResponse
    )
    
    assert isinstance(response, SimpleResponse)
    assert "Paris" in response.answer
    assert len(response.reasoning) > 10


@pytest.mark.integration
def test_openai_with_messages():
    """Test OpenAI provider with messages input instead of prompt."""
    provider = OpenAIProvider()
    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides brief, accurate answers."},
        {"role": "user", "content": "What is the capital of France? Answer in one word."}
    ]
    
    response = provider.generate(messages=messages)
    assert "Paris" in response


@pytest.mark.integration
def test_anthropic_with_messages():
    """Test Anthropic provider with messages input instead of prompt."""
    provider = AnthropicProvider()
    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides brief, accurate answers."},
        {"role": "user", "content": "What is the capital of France? Answer in one word."}
    ]
    
    response = provider.generate(messages=messages)
    assert "Paris" in response


@pytest.mark.integration
def test_gemini_with_messages():
    """Test Gemini provider with messages input instead of prompt."""
    provider = GeminiProvider()
    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides brief, accurate answers."},
        {"role": "user", "content": "What is the capital of France? Answer in one word."}
    ]
    
    response = provider.generate(messages=messages)
    assert "Paris" in response


@pytest.mark.integration
def test_openai_messages_with_structured_output():
    """Test OpenAI provider with messages input and structured output."""
    provider = OpenAIProvider()
    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides answers in JSON format."},
        {"role": "user", "content": """What is the capital of France? 
        Provide a short answer and a brief explanation of why Paris is the capital.
        Format your response as JSON with 'answer' and 'reasoning' fields."""}
    ]
    
    response = provider.generate(
        messages=messages,
        response_format=SimpleResponse
    )
    
    assert isinstance(response, SimpleResponse)
    assert "Paris" in response.answer
    assert len(response.reasoning) > 10


@pytest.mark.integration
def test_openai_with_all_parameters():
    """Test OpenAI provider with all optional parameters specified."""
    provider = OpenAIProvider(
        model_id="gpt-4.1-mini-2025-04-14",
        temperature=0.2,
        max_completion_tokens=100,
        top_p=0.9,
        frequency_penalty=0.1
    )
    
    prompt = "What is the capital of France? Answer in one word."
    response = provider.generate(prompt=prompt)
    
    assert "Paris" in response


@pytest.mark.integration
def test_anthropic_messages_with_structured_output():
    """Test the Anthropic provider with messages input and structured output."""
    provider = AnthropicProvider()
    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides answers in JSON format."},
        {"role": "user", "content": """What is the capital of France? 
        Provide a short answer and a brief explanation of why Paris is the capital.
        Format your response as JSON with 'answer' and 'reasoning' fields."""}
    ]
    
    response = provider.generate(
        messages=messages,
        response_format=SimpleResponse
    )
    
    assert isinstance(response, SimpleResponse)
    assert "Paris" in response.answer
    assert len(response.reasoning) > 10


@pytest.mark.integration
def test_gemini_messages_with_structured_output():
    """Test the Gemini provider with messages input and structured output."""
    provider = GeminiProvider()
    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides answers in JSON format."},
        {"role": "user", "content": """What is the capital of France? 
        Provide a short answer and a brief explanation of why Paris is the capital.
        Format your response as JSON with 'answer' and 'reasoning' fields."""}
    ]
    
    response = provider.generate(
        messages=messages,
        response_format=SimpleResponse
    )
    
    assert isinstance(response, SimpleResponse)
    assert "Paris" in response.answer
    assert len(response.reasoning) > 10


@pytest.mark.integration
def test_anthropic_with_all_parameters():
    """Test Anthropic provider with all optional parameters specified."""
    provider = AnthropicProvider(
        model_id="claude-3-5-haiku-latest",
        temperature=0.3,
        max_completion_tokens=200,
        top_p=0.95,
    )
    
    prompt = "What is the capital of France? Answer in one word."
    response = provider.generate(prompt=prompt)
    
    assert "Paris" in response


@pytest.mark.integration
def test_gemini_with_all_parameters():
    """Test Gemini provider with all optional parameters specified."""
    provider = GeminiProvider(
        model_id="gemini-2.0-flash",
        temperature=0.4,
        max_completion_tokens=150,
        top_p=0.85,
        frequency_penalty=0.15
    )
    
    prompt = "What is the capital of France? Answer in one word."
    response = provider.generate(prompt=prompt)
    
    assert "Paris" in response


@pytest.mark.integration
def test_openai_structured_landmark_info():
    """Test OpenAI with a structured landmark info response."""
    provider = OpenAIProvider(temperature=0.1, max_completion_tokens=800)
    
    prompt = """
    Provide detailed information about the Eiffel Tower in Paris.
    
    Return your response as a structured JSON object with the following elements:
    - name: The name of the landmark (Eiffel Tower)
    - location: Where it's located (Paris, France)
    - description: A brief description of the landmark (2-3 sentences)
    - year_built: The year when it was built (as a number)
    - attributes: A list of at least 3 attribute objects, each containing:
      - name: The name of the attribute (e.g., "height", "material", "architect")
      - value: The value of the attribute (e.g., "330 meters", "wrought iron", "Gustave Eiffel")
      - importance: An importance score between 0 and 1
    - visitor_rating: Average visitor rating from 0 to 5 (e.g., 4.5)
    
    Make sure your response is properly structured and can be parsed as valid JSON.
    """
    
    response = provider.generate(prompt=prompt, response_format=LandmarkInfo)
    
    # Verify the structure was correctly generated and parsed
    assert isinstance(response, LandmarkInfo)
    assert "Eiffel Tower" in response.name
    assert "Paris" in response.location
    assert len(response.description) > 20
    assert response.year_built is not None and response.year_built > 1800
    assert len(response.attributes) >= 3
    
    # Verify nested objects
    for attr in response.attributes:
        assert 0 <= attr.importance <= 1
        assert len(attr.name) > 0
        assert len(attr.value) > 0
    
    # Verify rating field
    assert 0 <= response.visitor_rating <= 5


@pytest.mark.integration
def test_anthropic_structured_landmark_info():
    """Test Anthropic with a structured landmark info response."""
    provider = AnthropicProvider(temperature=0.1, max_completion_tokens=800)
    
    prompt = """
    Provide detailed information about the Golden Gate Bridge in San Francisco.
    
    Return your response as a structured JSON object with the following elements:
    - name: The name of the landmark (Golden Gate Bridge)
    - location: Where it's located (San Francisco, USA)
    - description: A brief description of the landmark (2-3 sentences)
    - year_built: The year when it was built (as a number)
    - attributes: A list of at least 3 attribute objects, each containing:
      - name: The name of the attribute (e.g., "length", "color", "architect")
      - value: The value of the attribute (e.g., "1.7 miles", "International Orange", "Joseph Strauss")
      - importance: An importance score between 0 and 1
    - visitor_rating: Average visitor rating from 0 to 5 (e.g., 4.8)
    
    Make sure your response is properly structured and can be parsed as valid JSON.
    """
    
    response = provider.generate(prompt=prompt, response_format=LandmarkInfo)
    
    # Verify the structure was correctly generated and parsed
    assert isinstance(response, LandmarkInfo)
    assert "Golden Gate Bridge" in response.name
    assert "Francisco" in response.location
    assert len(response.description) > 20
    assert response.year_built is not None and response.year_built > 1900
    assert len(response.attributes) >= 3
    
    # Verify nested objects
    for attr in response.attributes:
        assert 0 <= attr.importance <= 1
        assert len(attr.name) > 0
        assert len(attr.value) > 0
    
    # Verify rating field
    assert 0 <= response.visitor_rating <= 5


@pytest.mark.integration
def test_gemini_structured_landmark_info():
    """Test Gemini with a structured landmark info response."""
    provider = GeminiProvider(temperature=0.1, max_completion_tokens=800)
    
    prompt = """
    Provide detailed information about the Great Wall of China.
    
    Return your response as a structured JSON object with the following elements:
    - name: The name of the landmark (Great Wall of China)
    - location: Where it's located (Northern China)
    - description: A brief description of the landmark (2-3 sentences)
    - year_built: The year when construction began (as a number)
    - attributes: A list of at least 3 attribute objects, each containing:
      - name: The name of the attribute (e.g., "length", "material", "dynasties")
      - value: The value of the attribute (e.g., "13,171 miles", "stone, brick, wood, etc.", "multiple including Qin, Han, Ming")
      - importance: An importance score between 0 and 1
    - visitor_rating: Average visitor rating from 0 to 5 (e.g., 4.7)
    
    Make sure your response is properly structured and can be parsed as valid JSON.
    """
    
    response = provider.generate(prompt=prompt, response_format=LandmarkInfo)
    
    # Verify the structure was correctly generated and parsed
    assert isinstance(response, LandmarkInfo)
    assert "Great Wall" in response.name
    assert "China" in response.location
    assert len(response.description) > 20
    assert response.year_built is not None
    assert len(response.attributes) >= 3
    
    # Verify nested objects
    for attr in response.attributes:
        assert 0 <= attr.importance <= 1
        assert len(attr.name) > 0
        assert len(attr.value) > 0
    
    # Verify rating field
    assert 0 <= response.visitor_rating <= 5

# import litellm
# litellm._turn_on_debug() # turn on debug to see the request

@pytest.mark.integration
def test_ollama_provider():
    """Test the Ollama provider with text response."""
    provider = OllamaProvider(model_id="gemma3:4b")
    response = provider.generate(prompt="What is the capital of France? Answer in one word.")
    assert "Paris" in response


@pytest.mark.integration
def test_ollama_structured_output():
    """Test the Ollama provider with structured output."""
    provider = OllamaProvider(model_id="gemma3:4b")
    prompt = """What is the capital of France? 
    Provide a short answer and a brief explanation of why Paris is the capital.
    Format your response as JSON with 'answer' and 'reasoning' fields."""
    
    response = provider.generate(
        prompt=prompt,
        response_format=SimpleResponse
    )
    
    assert isinstance(response, SimpleResponse)
    assert "Paris" in response.answer
    assert len(response.reasoning) > 10


@pytest.mark.integration
def test_ollama_with_messages():
    """Test Ollama provider with messages input instead of prompt."""
    provider = OllamaProvider()
    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides brief, accurate answers."},
        {"role": "user", "content": "What is the capital of France? Answer in one word."}
    ]
    
    response = provider.generate(messages=messages)
    assert "Paris" in response


@pytest.mark.integration
def test_ollama_messages_with_structured_output():
    """Test the Ollama provider with messages input and structured output."""
    provider = OllamaProvider()
    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides answers in JSON format."},
        {"role": "user", "content": """What is the capital of France? 
        Provide a short answer and a brief explanation of why Paris is the capital.
        Format your response as JSON with 'answer' and 'reasoning' fields."""}
    ]
    
    response = provider.generate(
        messages=messages,
        response_format=SimpleResponse
    )
    
    assert isinstance(response, SimpleResponse)
    assert "Paris" in response.answer
    assert len(response.reasoning) > 10


@pytest.mark.integration
def test_ollama_with_all_parameters():
    """Test Ollama provider with all optional parameters specified."""
    provider = OllamaProvider(
        model_id="gemma3:4b",
        temperature=0.4,
        max_completion_tokens=150,
        top_p=0.85,
        frequency_penalty=0.15,
        api_base="http://localhost:11434"
    )
    
    prompt = "What is the capital of France? Answer in one word."
    response = provider.generate(prompt=prompt)
    
    assert "Paris" in response


@pytest.mark.integration
def test_ollama_structured_landmark_info():
    """Test Ollama with a structured landmark info response."""
    provider = OllamaProvider(temperature=0.1, max_completion_tokens=800)
    
    prompt = """
    Provide detailed information about the Sydney Opera House.
    
    Return your response as a structured JSON object with the following elements:
    - name: The name of the landmark (Sydney Opera House)
    - location: Where it's located (Sydney, Australia)
    - description: A brief description of the landmark (2-3 sentences)
    - year_built: The year when it was completed (as a number)
    - attributes: A list of at least 3 attribute objects, each containing:
      - name: The name of the attribute (e.g., "architect", "style", "height")
      - value: The value of the attribute (e.g., "Jørn Utzon", "Expressionist", "65 meters")
      - importance: An importance score between 0 and 1
    - visitor_rating: Average visitor rating from 0 to 5 (e.g., 4.9)
    
    Make sure your response is properly structured and can be parsed as valid JSON.
    """
    
    response = provider.generate(prompt=prompt, response_format=LandmarkInfo)
    
    # Verify the structure was correctly generated and parsed
    assert isinstance(response, LandmarkInfo)
    assert "Opera House" in response.name
    assert "Sydney" in response.location
    assert len(response.description) > 20
    assert response.year_built is not None and response.year_built > 1900
    assert len(response.attributes) >= 3
    
    # Verify nested objects
    for attr in response.attributes:
        assert 0 <= attr.importance <= 1
        assert len(attr.name) > 0
        assert len(attr.value) > 0
    
    # Verify rating field
    assert 0 <= response.visitor_rating <= 5
