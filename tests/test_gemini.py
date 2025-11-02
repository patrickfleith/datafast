from datafast.llms import GeminiProvider
from dotenv import load_dotenv
import pytest
from tests.test_schemas import (
    SimpleResponse,
    LandmarkInfo,
    PersonaContent,
    QASet,
    MCQSet,
)
import time

load_dotenv()

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
def test_gemini_persona_content_generation():
    """Test generating tweets and bio for a persona using Gemini."""
    provider = GeminiProvider(
        temperature=0.7,
        max_completion_tokens=1000
    )
    
    prompt = """
    Generate social media content for the following persona:
    
    Persona: A passionate environmental scientist who loves hiking and photography, 
    advocates for climate action, and enjoys sharing nature facts with humor.
    
    Create exactly 5 tweets and 1 bio for this persona.
    """
    time.sleep(60)
    response = provider.generate(prompt=prompt, response_format=PersonaContent)
    
    assert isinstance(response, PersonaContent)
    assert len(response.tweets) == 5
    assert all(len(tweet) > 0 for tweet in response.tweets)
    assert len(response.bio) > 20


@pytest.mark.integration
def test_gemini_qa_generation():
    """Test generating Q&A pairs on machine learning using Gemini."""
    provider = GeminiProvider(
        temperature=0.5,
        max_completion_tokens=1500
    )
    
    prompt = """
    Generate exactly 5 questions and their correct answers about machine learning topics.
    
    Topics to cover: supervised learning, neural networks, overfitting, gradient descent, and cross-validation.
    
    Each question should be clear and the answer should be concise but complete.
    """
    
    response = provider.generate(prompt=prompt, response_format=QASet)
    
    assert isinstance(response, QASet)
    assert len(response.questions) == 5
    for qa in response.questions:
        assert len(qa.question) > 10
        assert len(qa.answer) > 10


@pytest.mark.integration
def test_gemini_mcq_generation():
    """Test generating multiple choice questions using Gemini."""
    provider = GeminiProvider(
        temperature=0.5,
        max_completion_tokens=1500
    )
    
    prompt = """
    Generate exactly 3 multiple choice questions about machine learning.
    
    For each question, provide:
    - The question itself
    - One correct answer
    - Three plausible but incorrect answers
    
    Topics: neural networks, decision trees, and ensemble methods.
    """
    
    response = provider.generate(prompt=prompt, response_format=MCQSet)
    
    assert isinstance(response, MCQSet)
    assert len(response.questions) == 3
    for mcq in response.questions:
        assert len(mcq.question) > 10
        assert len(mcq.correct_answer) > 0
        assert len(mcq.incorrect_answers) == 3
        assert all(len(ans) > 0 for ans in mcq.incorrect_answers)

