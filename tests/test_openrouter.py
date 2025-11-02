from datafast.llms import OpenRouterProvider
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

