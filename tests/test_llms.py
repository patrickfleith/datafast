from datafast.llms import OpenAIProvider, GeminiProvider
from dotenv import load_dotenv
import pytest
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

load_dotenv()


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
    year_built: Optional[int] = Field(
        None, description="Year when the landmark was built")
    attributes: List[Attribute] = Field(
        description="List of attributes about the landmark")
    visitor_rating: float = Field(
        description="Average visitor rating from 0 to 5")

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
    response = provider.generate(
        prompt="What is the capital of France? Answer in one word.")
    assert "Paris" in response


@pytest.mark.integration
def test_gemini_provider():
    """Test the Gemini provider with text response."""
    provider = GeminiProvider()
    response = provider.generate(
        prompt="What is the capital of France? Answer in one word.")
    assert "Paris" in response


@pytest.mark.slow
@pytest.mark.integration
def test_gemini_rpm_limit_real():
    """Test GeminiProvider RPM limit (15 requests/minute) is enforced with real waiting."""
    import time
    prompts_count = 17
    rpm = 15
    provider = GeminiProvider(
        model_id="gemini-2.5-flash-lite-preview-06-17", rpm_limit=rpm)
    prompt = [f"Test request {i}" for i in range(prompts_count)]
    start = time.monotonic()
    for prompt in prompt:
        provider.generate(prompt=prompt)
    elapsed = time.monotonic() - start
    # 17 requests, rpm=15, donc on doit attendre au moins ~60s pour les 2 requêtes au-delà de la limite
    assert elapsed >= 59, f"Elapsed time too short for RPM limit: {elapsed:.2f}s for {prompts_count} requests with rpm={rpm}"


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
    # Make sure we have some reasoning text
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
        model_id="gpt-5-mini-2025-08-07",
        temperature=0.2,
        max_completion_tokens=100,
        top_p=0.9,
        frequency_penalty=0.1
    )

    prompt = "What is the capital of France? Answer in one word."
    response = provider.generate(prompt=prompt)

    assert "Paris" in response


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


"******* Batch Inference Tests *******"
"Similar to previous tests but for batch inputs"

# Batch tests to add to your existing test file


@pytest.mark.integration
def test_openai_batch_prompts():
    """Test the OpenAI provider with batch prompts."""
    provider = OpenAIProvider()
    prompt = [
        "What is the capital of France? Answer in one word.",
        "What is the capital of Germany? Answer in one word.",
        "What is the capital of Italy? Answer in one word."
    ]

    responses = provider.generate(prompt=prompt)

    assert len(responses) == 3
    assert isinstance(responses, list)
    assert all(isinstance(r, str) for r in responses)
    assert "Paris" in responses[0]
    assert "Berlin" in responses[1]
    assert "Rome" in responses[2]


@pytest.mark.integration
def test_gemini_batch_prompts():
    """Test the Gemini provider with batch prompts."""
    provider = GeminiProvider()
    prompt = [
        "What is 2+2? Answer with just the number.",
        "What is 3+3? Answer with just the number.",
        "What is 4+4? Answer with just the number."
    ]

    responses = provider.generate(prompt=prompt)

    assert len(responses) == 3
    assert isinstance(responses, list)
    assert all(isinstance(r, str) for r in responses)
    assert "4" in responses[0]
    assert "6" in responses[1]
    assert "8" in responses[2]


@pytest.mark.integration
def test_openai_batch_messages():
    """Test OpenAI provider with batch messages."""
    provider = OpenAIProvider()
    messages = [
        [
            {"role": "system", "content": "You are a helpful assistant that provides brief, accurate answers."},
            {"role": "user", "content": "What is the capital of France? One word."}
        ],
        [
            {"role": "system", "content": "You are a helpful assistant that provides brief, accurate answers."},
            {"role": "user", "content": "What is the capital of Japan? One word."}
        ]
    ]

    responses = provider.generate(messages=messages)

    assert len(responses) == 2
    assert isinstance(responses, list)
    assert all(isinstance(r, str) for r in responses)
    assert "Paris" in responses[0]
    assert "Tokyo" in responses[1]


@pytest.mark.integration
def test_gemini_batch_messages():
    """Test Gemini provider with batch messages."""
    provider = GeminiProvider()
    messages = [
        [
            {"role": "system", "content": "You are a helpful assistant that provides brief, accurate answers."},
            {"role": "user", "content": "What is 5+5? Just the number."}
        ],
        [
            {"role": "system", "content": "You are a helpful assistant that provides brief, accurate answers."},
            {"role": "user", "content": "What is 7+3? Just the number."}
        ]
    ]

    responses = provider.generate(messages=messages)

    assert len(responses) == 2
    assert isinstance(responses, list)
    assert all(isinstance(r, str) for r in responses)
    assert "10" in responses[0]
    assert "10" in responses[1]


@pytest.mark.integration
def test_openai_batch_structured_output():
    """Test OpenAI provider with batch structured output."""
    provider = OpenAIProvider()
    prompt = [
        """What is the capital of France? 
        Provide a short answer and brief reasoning.
        Format as JSON with 'answer' and 'reasoning' fields.""",
        """What is the capital of Japan?
        Provide a short answer and brief reasoning.
        Format as JSON with 'answer' and 'reasoning' fields."""
    ]

    responses = provider.generate(
        prompt=prompt,
        response_format=SimpleResponse
    )

    assert len(responses) == 2
    assert all(isinstance(r, SimpleResponse) for r in responses)
    assert "Paris" in responses[0].answer
    assert "Tokyo" in responses[1].answer
    assert len(responses[0].reasoning) > 5
    assert len(responses[1].reasoning) > 5


@pytest.mark.integration
def test_gemini_batch_structured_output():
    """Test Gemini provider with batch structured output."""
    provider = GeminiProvider()
    prompt = [
        """What is 8*3? Provide the answer and show your work.
        Format as JSON with 'answer' and 'reasoning' fields.""",
        """What is 9*4? Provide the answer and show your work.
        Format as JSON with 'answer' and 'reasoning' fields."""
    ]

    responses = provider.generate(
        prompt=prompt,
        response_format=SimpleResponse
    )

    assert len(responses) == 2
    assert all(isinstance(r, SimpleResponse) for r in responses)
    assert "24" in responses[0].answer
    assert "36" in responses[1].answer
    assert len(responses[0].reasoning) > 5
    assert len(responses[1].reasoning) > 5


@pytest.mark.integration
def test_openai_batch_messages_with_structured_output():
    """Test OpenAI provider with batch messages and structured output."""
    provider = OpenAIProvider()
    messages = [
        [
            {"role": "system", "content": "You are a helpful assistant that provides answers in JSON format."},
            {"role": "user", "content": """What is the capital of Brazil? 
            Provide a short answer and brief reasoning.
            Format as JSON with 'answer' and 'reasoning' fields."""}
        ],
        [
            {"role": "system", "content": "You are a helpful assistant that provides answers in JSON format."},
            {"role": "user", "content": """What is the capital of Argentina?
            Provide a short answer and brief reasoning.
            Format as JSON with 'answer' and 'reasoning' fields."""}
        ]
    ]

    responses = provider.generate(
        messages=messages,
        response_format=SimpleResponse
    )

    assert len(responses) == 2
    assert all(isinstance(r, SimpleResponse) for r in responses)
    assert "Brasília" in responses[0].answer or "Brasilia" in responses[0].answer
    assert "Buenos Aires" in responses[1].answer
    assert len(responses[0].reasoning) > 5
    assert len(responses[1].reasoning) > 5


@pytest.mark.integration
def test_gemini_batch_messages_with_structured_output():
    """Test Gemini provider with batch messages and structured output."""
    provider = GeminiProvider()
    messages = [
        [
            {"role": "system", "content": "You are a helpful assistant that provides answers in JSON format."},
            {"role": "user", "content": """What is 12/3? Provide the answer and show your work.
            Format as JSON with 'answer' and 'reasoning' fields."""}
        ],
        [
            {"role": "system", "content": "You are a helpful assistant that provides answers in JSON format."},
            {"role": "user", "content": """What is 15/5? Provide the answer and show your work.
            Format as JSON with 'answer' and 'reasoning' fields."""}
        ]
    ]

    responses = provider.generate(
        messages=messages,
        response_format=SimpleResponse
    )

    assert len(responses) == 2
    assert all(isinstance(r, SimpleResponse) for r in responses)
    assert "4" in responses[0].answer
    assert "3" in responses[1].answer
    assert len(responses[0].reasoning) > 5
    assert len(responses[1].reasoning) > 5


@pytest.mark.integration
def test_openai_batch_with_all_parameters():
    """Test OpenAI provider with batch processing and all optional parameters."""
    provider = OpenAIProvider(
        model_id="gpt-5-mini-2025-08-07",
        temperature=0.1,
        max_completion_tokens=50,
        top_p=0.9,
        frequency_penalty=0.1
    )

    prompt = [
        "What is the capital of Sweden? Answer in one word.",
        "What is the capital of Norway? Answer in one word."
    ]

    responses = provider.generate(prompt=prompt)

    assert len(responses) == 2
    assert "Stockholm" in responses[0]
    assert "Oslo" in responses[1]





@pytest.mark.integration
def test_gemini_batch_with_all_parameters():
    """Test Gemini provider with batch processing and all optional parameters."""
    provider = GeminiProvider(
        model_id="gemini-2.0-flash",
        temperature=0.1,
        max_completion_tokens=50,
        top_p=0.9,
        frequency_penalty=0.1
    )

    prompt = [
        "What is the capital of Belgium? Answer in one word.",
        "What is the capital of Netherlands? Answer in one word."
    ]

    responses = provider.generate(prompt=prompt)

    assert len(responses) == 2
    assert "Brussels" in responses[0]
    assert "Amsterdam" in responses[1]


@pytest.mark.integration
def test_openai_batch_landmark_info():
    """Test OpenAI with batch structured landmark info responses."""
    provider = OpenAIProvider(temperature=0.1, max_completion_tokens=800)

    prompt = [
        """
        Provide detailed information about the Statue of Liberty.
        
        Return your response as a structured JSON object with the following elements:
        - name: The name of the landmark (Statue of Liberty)
        - location: Where it's located (New York, USA)
        - description: A brief description of the landmark (2-3 sentences)
        - year_built: The year when it was completed (as a number)
        - attributes: A list of at least 3 attribute objects, each containing:
          - name: The name of the attribute (e.g., "height", "material", "sculptor")
          - value: The value of the attribute (e.g., "93 meters", "copper", "Frédéric Auguste Bartholdi")
          - importance: An importance score between 0 and 1
        - visitor_rating: Average visitor rating from 0 to 5 (e.g., 4.6)
        
        Make sure your response is properly structured and can be parsed as valid JSON.
        """,
        """
        Provide detailed information about Big Ben in London.
        
        Return your response as a structured JSON object with the following elements:
        - name: The name of the landmark (Big Ben)
        - location: Where it's located (London, UK)
        - description: A brief description of the landmark (2-3 sentences)
        - year_built: The year when it was completed (as a number)
        - attributes: A list of at least 3 attribute objects, each containing:
          - name: The name of the attribute (e.g., "height", "clock", "architect")
          - value: The value of the attribute (e.g., "96 meters", "Great Clock", "Augustus Pugin")
          - importance: An importance score between 0 and 1
        - visitor_rating: Average visitor rating from 0 to 5 (e.g., 4.4)
        
        Make sure your response is properly structured and can be parsed as valid JSON.
        """
    ]

    responses = provider.generate(
        prompt=prompt, response_format=LandmarkInfo)

    # Verify we got 2 responses
    assert len(responses) == 2
    assert all(isinstance(r, LandmarkInfo) for r in responses)

    # Verify first response (Statue of Liberty)
    assert "Statue of Liberty" in responses[0].name
    assert "New York" in responses[0].location
    assert len(responses[0].description) > 20
    assert responses[0].year_built is not None and responses[0].year_built > 1800
    assert len(responses[0].attributes) >= 3

    # Verify second response (Big Ben)
    assert "Big Ben" in responses[1].name
    assert "London" in responses[1].location
    assert len(responses[1].description) > 20
    assert responses[1].year_built is not None and responses[1].year_built > 1800
    assert len(responses[1].attributes) >= 3

    # Verify nested objects for both responses
    for response in responses:
        for attr in response.attributes:
            assert 0 <= attr.importance <= 1
            assert len(attr.name) > 0
            assert len(attr.value) > 0
        assert 0 <= response.visitor_rating <= 5


