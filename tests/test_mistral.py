from datafast.llms import MistralProvider
from dotenv import load_dotenv
import pytest
from tests.test_schemas import (
    SimpleResponse,
    LandmarkInfo,
    PersonaContent,
    QASet,
    MCQSet,
)

load_dotenv()


@pytest.mark.integration
class TestMistralProvider:
    """Test suite for Mistral provider with various input types and configurations."""

    def test_basic_text_response(self):
        """Test the Mistral provider with a basic text response."""
        provider = MistralProvider()
        response = provider.generate(prompt="What is the capital of France? Answer in one word.")
        assert "Paris" in response

    def test_structured_output(self):
        """Test the Mistral provider with structured output."""
        provider = MistralProvider()
        prompt = """What is the capital of France?
        Provide a short answer and a brief explanation of why Paris is the capital."""

        response = provider.generate(
            prompt=prompt,
            response_format=SimpleResponse,
        )

        assert isinstance(response, SimpleResponse)
        assert "Paris" in response.answer
        assert len(response.reasoning) > 10

    def test_with_messages(self):
        """Test Mistral provider with messages input instead of prompt."""
        provider = MistralProvider()
        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides brief, accurate answers."},
            {"role": "user", "content": "What is the capital of France? Answer in one word."},
        ]

        response = provider.generate(messages=messages)
        assert "Paris" in response

    def test_messages_with_structured_output(self):
        """Test Mistral provider with messages input and structured output."""
        provider = MistralProvider()
        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides brief, accurate answers."},
            {"role": "user", "content": """What is the capital of France?
            Provide a short answer and a brief explanation of why Paris is the capital."""},
        ]

        response = provider.generate(
            messages=messages,
            response_format=SimpleResponse,
        )

        assert isinstance(response, SimpleResponse)
        assert "Paris" in response.answer
        assert len(response.reasoning) > 10

    def test_with_all_parameters(self):
        """Test Mistral provider with all optional parameters specified."""
        provider = MistralProvider(
            model_id="mistral-small-latest",
            temperature=0.3,
            max_completion_tokens=300,
            top_p=0.85,
        )

        response = provider.generate(prompt="What is the capital of France? Answer in one word.")
        assert "Paris" in response

    def test_structured_landmark_info(self):
        """Test Mistral provider with a structured landmark info response."""
        provider = MistralProvider(temperature=0.6, max_completion_tokens=2000)

        prompt = """
        Extract structured landmark details about the Great Wall of China from the passage below.

        Passage:
        "The Great Wall of China stands across northern China, originally begun in 220 BCE to guard imperial borders.
        Spanning roughly 13,171 miles, it threads over mountains and deserts, symbolising centuries of engineering prowess and cultural unity.
        Construction and major reinforcement during the Ming dynasty in the 14th century gave the wall its iconic form, using stone and brick to fortify older earthen ramparts.
        Key attributes include: overall length of about 13,171 miles (importance 0.9), primary materials of stone and brick with tamped earth cores (importance 0.7), and critical Ming dynasty stewardship that restored and expanded the fortifications (importance 0.8).
        Today's visitors typically rate the experience around 4.6 out of 5, citing sweeping views and the wall's historical resonance."
        """

        response = provider.generate(prompt=prompt, response_format=LandmarkInfo)

        assert isinstance(response, LandmarkInfo)
        assert "Great Wall" in response.name
        assert "China" in response.location
        assert len(response.description) > 20
        assert response.year_built is not None
        assert len(response.attributes) >= 3

        for attr in response.attributes:
            assert 0 <= attr.importance <= 1
            assert len(attr.name) > 0
            assert len(attr.value) > 0

        assert 0 <= response.visitor_rating <= 5


@pytest.mark.integration
class TestMistralMedium:
    """Test suite for mistral-medium-latest model."""

    def test_persona_content_generation(self):
        """Test generating tweets and bio for a persona using Mistral Medium."""
        provider = MistralProvider(
            model_id="mistral-medium-latest",
            temperature=0.7,
            max_completion_tokens=1000,
        )

        prompt = """
        Generate social media content for the following persona:

        Persona: A passionate environmental scientist who loves hiking and photography,
        advocates for climate action, and enjoys sharing nature facts with humor.

        Create exactly 5 tweets and 1 bio for this persona.
        """

        response = provider.generate(prompt=prompt, response_format=PersonaContent)

        assert isinstance(response, PersonaContent)
        assert len(response.tweets) == 5
        assert all(len(tweet) > 0 for tweet in response.tweets)
        assert len(response.bio) > 20

    def test_qa_generation(self):
        """Test generating Q&A pairs on machine learning using Mistral Medium."""
        provider = MistralProvider(
            model_id="mistral-medium-latest",
            temperature=0.5,
            max_completion_tokens=1500,
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

    def test_mcq_generation(self):
        """Test generating multiple choice questions using Mistral Medium."""
        provider = MistralProvider(
            model_id="mistral-medium-latest",
            temperature=0.5,
            max_completion_tokens=1500,
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
