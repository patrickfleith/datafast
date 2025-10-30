from datafast.llms import OpenRouterProvider
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
    def check_incorrect_count(cls, v: List[str]) -> List[str]:
        """Validate that exactly 3 incorrect answers are provided."""
        if len(v) != 3:
            raise ValueError("Must provide exactly 3 incorrect answers")
        return v


class MCQSet(BaseModel):
    """Set of multiple choice questions."""
    questions: List[MCQQuestion] = Field(description="List of MCQ questions")

    @field_validator('questions')
    def check_questions_count(cls, v: List[MCQQuestion]) -> List[MCQQuestion]:
        """Validate that exactly 3 questions are provided."""
        if len(v) != 3:
            raise ValueError("Must provide exactly 3 questions")
        return v


@pytest.mark.integration
class TestOpenRouterProvider:
    """Test suite for OpenRouter provider with various input types and configurations."""

    def test_basic_text_response(self):
        """Test the OpenRouter provider with text response."""
        provider = OpenRouterProvider()
        response = provider.generate(prompt="What is the capital of France? Answer in one word.")
        assert "Paris" in response

    def test_structured_output(self):
        """Test the OpenRouter provider with structured output."""
        provider = OpenRouterProvider()
        prompt = """What is the capital of France? 
        Provide a short answer and a brief explanation of why Paris is the capital."""
        
        response = provider.generate(
            prompt=prompt,
            response_format=SimpleResponse
        )
        
        assert isinstance(response, SimpleResponse)
        assert "Paris" in response.answer
        assert len(response.reasoning) > 10

    def test_with_messages(self):
        """Test OpenRouter provider with messages input instead of prompt."""
        provider = OpenRouterProvider()
        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides brief, accurate answers."},
            {"role": "user", "content": "What is the capital of France? Answer in one word."}
        ]
        
        response = provider.generate(messages=messages)
        assert "Paris" in response

    def test_messages_with_structured_output(self):
        """Test OpenRouter provider with messages input and structured output."""
        provider = OpenRouterProvider()
        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides brief, accurate answers."},
            {"role": "user", "content": """What is the capital of France? 
            Provide a short answer and a brief explanation of why Paris is the capital."""}
        ]
        
        response = provider.generate(
            messages=messages,
            response_format=SimpleResponse
        )
        
        assert isinstance(response, SimpleResponse)
        assert "Paris" in response.answer
        assert len(response.reasoning) > 10

    def test_with_all_parameters(self):
        """Test OpenRouter provider with all optional parameters specified."""
        provider = OpenRouterProvider(
            model_id="meta-llama/llama-3.3-70b-instruct",
            max_completion_tokens=300,
            top_p=0.85,
        )
        
        response = provider.generate(prompt="What is the capital of France? Answer in one word.")
        
        assert "Paris" in response

    def test_structured_landmark_info(self):
        """Test OpenRouter with a structured landmark info response."""
        provider = OpenRouterProvider(temperature=0.6, max_completion_tokens=2000)
        
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
class TestOpenRouterGLM46:
    """Test suite for z-ai/glm-4.6 model via OpenRouter."""

    def test_persona_content_generation(self):
        """Test generating tweets and bio for a persona using GLM-4.6."""
        provider = OpenRouterProvider(
            model_id="z-ai/glm-4.6",
            temperature=0.5,
            max_completion_tokens=2000
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
        """Test generating Q&A pairs on machine learning using GLM-4.6."""
        provider = OpenRouterProvider(
            model_id="z-ai/glm-4.6",
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

    def test_mcq_generation(self):
        """Test generating multiple choice questions using GLM-4.6."""
        provider = OpenRouterProvider(
            model_id="z-ai/glm-4.6",
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


@pytest.mark.integration
class TestOpenRouterQwen3:
    """Test suite for qwen/qwen3-next-80b-a3b-instruct model via OpenRouter."""

    def test_persona_content_generation(self):
        """Test generating tweets and bio for a persona using Qwen3."""
        provider = OpenRouterProvider(
            model_id="qwen/qwen3-next-80b-a3b-instruct",
            temperature=0.5,
            max_completion_tokens=2000
        )
        
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
        """Test generating Q&A pairs on machine learning using Qwen3."""
        provider = OpenRouterProvider(
            model_id="qwen/qwen3-next-80b-a3b-instruct",
            temperature=0.5,
            max_completion_tokens=1500
        )
        
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
        """Test generating multiple choice questions using Qwen3."""
        provider = OpenRouterProvider(
            model_id="qwen/qwen3-next-80b-a3b-instruct",
            temperature=0.5,
            max_completion_tokens=1500
        )
        
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


@pytest.mark.integration
class TestOpenRouterLlama33:
    """Test suite for meta-llama/llama-3.3-70b-instruct model via OpenRouter."""

    def test_persona_content_generation(self):
        """Test generating tweets and bio for a persona using Llama 3.3."""
        provider = OpenRouterProvider(
            model_id="meta-llama/llama-3.3-70b-instruct",
            temperature=0.7,
            max_completion_tokens=1000
        )
        
        prompt = """
        Generate social media content for the following persona:
        
        Persona: A professional chef who specializes in fusion cuisine, loves traveling to discover 
        new ingredients, teaches cooking classes, and shares culinary tips with enthusiasm.
        
        Create exactly 5 tweets and 1 bio for this persona.
        """
        
        response = provider.generate(prompt=prompt, response_format=PersonaContent)
        
        assert isinstance(response, PersonaContent)
        assert len(response.tweets) == 5
        assert all(len(tweet) > 0 for tweet in response.tweets)
        assert len(response.bio) > 20

    def test_qa_generation(self):
        """Test generating Q&A pairs on machine learning using Llama 3.3."""
        provider = OpenRouterProvider(
            model_id="meta-llama/llama-3.3-70b-instruct",
            temperature=0.5,
            max_completion_tokens=1500
        )
        
        prompt = """
        Generate exactly 5 questions and their correct answers about machine learning topics.
        
        Topics to cover: transfer learning, attention mechanisms, batch normalization,
        dropout, and hyperparameter tuning.
        
        Each question should be clear and the answer should be concise but complete.
        """
        
        response = provider.generate(prompt=prompt, response_format=QASet)
        
        assert isinstance(response, QASet)
        assert len(response.questions) == 5
        for qa in response.questions:
            assert len(qa.question) > 10
            assert len(qa.answer) > 10

    def test_mcq_generation(self):
        """Test generating multiple choice questions using Llama 3.3."""
        provider = OpenRouterProvider(
            model_id="meta-llama/llama-3.3-70b-instruct",
            temperature=0.5,
            max_completion_tokens=1500
        )
        
        prompt = """
        Generate exactly 3 multiple choice questions about machine learning.
        
        For each question, provide:
        - The question itself
        - One correct answer
        - Three plausible but incorrect answers
        
        Topics: transformers, random forests, and principal component analysis.
        """
        
        response = provider.generate(prompt=prompt, response_format=MCQSet)
        
        assert isinstance(response, MCQSet)
        assert len(response.questions) == 3
        for mcq in response.questions:
            assert len(mcq.question) > 10
            assert len(mcq.correct_answer) > 0
            assert len(mcq.incorrect_answers) == 3
            assert all(len(ans) > 0 for ans in mcq.incorrect_answers)


@pytest.mark.integration
class TestOpenRouterGemini25Flash:
    """Test suite for google/gemini-2.5-flash model via OpenRouter."""

    def test_persona_content_generation(self):
        """Test generating tweets and bio for a persona using Gemini 2.5 Flash."""
        provider = OpenRouterProvider(
            model_id="google/gemini-2.5-flash",
            temperature=0.7,
            max_completion_tokens=1000
        )
        
        prompt = """
        Generate social media content for the following persona:
        
        Persona: A data scientist who is passionate about open source, enjoys playing chess,
        contributes to educational content, and advocates for diversity in tech.
        
        Create exactly 5 tweets and 1 bio for this persona.
        """
        
        response = provider.generate(prompt=prompt, response_format=PersonaContent)
        
        assert isinstance(response, PersonaContent)
        assert len(response.tweets) == 5
        assert all(len(tweet) > 0 for tweet in response.tweets)
        assert len(response.bio) > 20

    def test_qa_generation(self):
        """Test generating Q&A pairs on machine learning using Gemini 2.5 Flash."""
        provider = OpenRouterProvider(
            model_id="google/gemini-2.5-flash",
            temperature=0.5,
            max_completion_tokens=1500
        )
        
        prompt = """
        Generate exactly 5 questions and their correct answers about machine learning topics.
        
        Topics to cover: generative adversarial networks, autoencoders, dimensionality reduction,
        bias-variance tradeoff, and model evaluation metrics.
        
        Each question should be clear and the answer should be concise but complete.
        """
        
        response = provider.generate(prompt=prompt, response_format=QASet)
        
        assert isinstance(response, QASet)
        assert len(response.questions) == 5
        for qa in response.questions:
            assert len(qa.question) > 10
            assert len(qa.answer) > 10

    def test_mcq_generation(self):
        """Test generating multiple choice questions using Gemini 2.5 Flash."""
        provider = OpenRouterProvider(
            model_id="google/gemini-2.5-flash",
            temperature=0.5,
            max_completion_tokens=1500
        )
        
        prompt = """
        Generate exactly 3 multiple choice questions about machine learning.
        
        For each question, provide:
        - The question itself
        - One correct answer
        - Three plausible but incorrect answers
        
        Topics: LSTM networks, gradient boosting, and model interpretability.
        """
        
        response = provider.generate(prompt=prompt, response_format=MCQSet)
        
        assert isinstance(response, MCQSet)
        assert len(response.questions) == 3
        for mcq in response.questions:
            assert len(mcq.question) > 10
            assert len(mcq.correct_answer) > 0
            assert len(mcq.incorrect_answers) == 3
            assert all(len(ans) > 0 for ans in mcq.incorrect_answers)

