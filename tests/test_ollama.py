from datafast.llms import OllamaProvider
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


class PersonaContent(BaseModel):
    """Generated content for a persona including tweets and bio."""
    tweets: List[str] = Field(description="List of 5 tweets for the persona")
    bio: str = Field(description="Biography for the persona")

    @field_validator('tweets')
    @classmethod
    def check_tweets_count(cls, v: List[str]) -> List[str]:
        """Validate that exactly 5 tweets are provided."""
        if len(v) != 5:
            raise ValueError("Must provide exactly 5 tweets")
        return v


class QAItem(BaseModel):
    """Question and answer pair."""
    question: str = Field(description="The question")
    answer: str = Field(description="The correct answer")


class QASet(BaseModel):
    """Set of questions and answers."""
    questions: List[QAItem] = Field(description="List of question-answer pairs")

    @field_validator('questions')
    @classmethod
    def check_qa_count(cls, v: List[QAItem]) -> List[QAItem]:
        """Validate that exactly 5 Q&A pairs are provided."""
        if len(v) != 5:
            raise ValueError("Must provide exactly 5 question-answer pairs")
        return v


class MCQQuestion(BaseModel):
    """Multiple choice question with one correct and three incorrect answers."""
    question: str = Field(description="The question")
    correct_answer: str = Field(description="The correct answer")
    incorrect_answers: List[str] = Field(description="List of 3 incorrect answers")

    @field_validator('incorrect_answers')
    @classmethod
    def check_incorrect_count(cls, v: List[str]) -> List[str]:
        """Validate that exactly 3 incorrect answers are provided."""
        if len(v) != 3:
            raise ValueError("Must provide exactly 3 incorrect answers")
        return v


class MCQSet(BaseModel):
    """Set of multiple choice questions."""
    questions: List[MCQQuestion] = Field(description="List of MCQ questions")

    @field_validator('questions')
    @classmethod
    def check_questions_count(cls, v: List[MCQQuestion]) -> List[MCQQuestion]:
        """Validate that exactly 3 questions are provided."""
        if len(v) != 3:
            raise ValueError("Must provide exactly 3 questions")
        return v


@pytest.mark.integration
def test_ollama_provider():
    """Test the Ollama provider with text response."""
    provider = OllamaProvider(model_id="gemma3:4b")
    response = provider.generate(
        prompt="What is the capital of France? Answer in one word.")
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


@pytest.mark.integration
def test_ollama_batch_prompts():
    """Test Ollama provider with batch prompts."""
    provider = OllamaProvider(model_id="gemma3:4b")
    prompt = [
        "What is the capital of France? Answer in one word.",
        "What is the capital of Germany? Answer in one word."
    ]

    responses = provider.generate(prompt=prompt)

    assert len(responses) == 2
    assert isinstance(responses, list)
    assert all(isinstance(r, str) for r in responses)
    assert "Paris" in responses[0]
    assert "Berlin" in responses[1]


@pytest.mark.integration
def test_ollama_batch_messages():
    """Test Ollama provider with batch messages."""
    provider = OllamaProvider()
    messages = [
        [
            {"role": "system", "content": "You are a helpful assistant that provides brief, accurate answers."},
            {"role": "user", "content": "What is 6+4? Just the number."}
        ],
        [
            {"role": "system", "content": "You are a helpful assistant that provides brief, accurate answers."},
            {"role": "user", "content": "What is 8+2? Just the number."}
        ]
    ]

    responses = provider.generate(messages=messages)

    assert len(responses) == 2
    assert isinstance(responses, list)
    assert all(isinstance(r, str) for r in responses)
    assert "10" in responses[0]
    assert "10" in responses[1]


@pytest.mark.integration
def test_ollama_batch_structured_output():
    """Test Ollama provider with batch structured output."""
    provider = OllamaProvider()
    prompt = [
        """What is the capital of Spain? 
        Provide a short answer and brief reasoning.
        Format as JSON with 'answer' and 'reasoning' fields.""",
        """What is the capital of Portugal?
        Provide a short answer and brief reasoning.
        Format as JSON with 'answer' and 'reasoning' fields."""
    ]

    responses = provider.generate(
        prompt=prompt,
        response_format=SimpleResponse
    )

    assert len(responses) == 2
    assert all(isinstance(r, SimpleResponse) for r in responses)
    assert "Madrid" in responses[0].answer
    assert "Lisbon" in responses[1].answer
    assert len(responses[0].reasoning) > 5
    assert len(responses[1].reasoning) > 5


@pytest.mark.integration
class TestOllama:
    """Tests mirroring OpenRouter structure using Ollama with structured outputs."""

    def test_persona_content_generation(self):
        """Generate tweets and a bio for a persona using Ollama."""
        provider = OllamaProvider(model_id="gemma3:4b", temperature=0.5, max_completion_tokens=2000)
        prompt = """
        Generate social media content for the following persona:
        
        Persona: A tech entrepreneur who is passionate about AI ethics, loves reading sci-fi novels,
        practices meditation, and frequently shares insights about startup culture.
        
        Create exactly 5 tweets and 1 bio for this persona.
        """
        response = provider.generate(prompt=prompt, response_format=PersonaContent)
        assert isinstance(response, PersonaContent)
        assert len(response.tweets) == 5
        assert all(len(tweet) > 0 for tweet in response.tweets)
        assert len(response.bio) > 20

    def test_qa_generation(self):
        """Generate 5 Q&A pairs on machine learning topics using Ollama."""
        provider = OllamaProvider(model_id="gemma3:4b", temperature=0.5, max_completion_tokens=1500)
        prompt = """
        Generate exactly 5 questions and their correct answers about machine learning topics.
        
        Topics to cover: reinforcement learning, convolutional neural networks, regularization, 
        backpropagation, and feature engineering.
        
        Each question should be clear and the answer should be concise but complete.
        """
        response = provider.generate(prompt=prompt, response_format=QASet)
        assert isinstance(response, QASet)
        assert len(response.questions) == 5
        for qa in response.questions:
            assert len(qa.question) > 10
            assert len(qa.answer) > 10

    def test_mcq_generation(self):
        """Generate 3 MCQs on ML topics using Ollama."""
        provider = OllamaProvider(model_id="gemma3:4b", temperature=0.5, max_completion_tokens=1500)
        prompt = """
        Generate exactly 3 multiple choice questions about machine learning.
        
        For each question, provide:
        - The question itself
        - One correct answer
        - Three plausible but incorrect answers
        
        Topics: recurrent neural networks, k-means clustering, and support vector machines.
        """
        response = provider.generate(prompt=prompt, response_format=MCQSet)
        assert isinstance(response, MCQSet)
        assert len(response.questions) == 3
        for mcq in response.questions:
            assert len(mcq.question) > 10
            assert len(mcq.correct_answer) > 0
            assert len(mcq.incorrect_answers) == 3
            assert all(len(ans) > 0 for ans in mcq.incorrect_answers)
