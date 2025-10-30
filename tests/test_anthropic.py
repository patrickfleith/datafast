from datafast.llms import AnthropicProvider
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
class TestAnthropicSonnet45:
    """Anthropic tests for claude-sonnet-4-5-20250929."""

    def test_persona_content_generation(self):
        provider = AnthropicProvider(
            model_id="claude-sonnet-4-5-20250929",
            temperature=0.5,
            max_completion_tokens=2000,
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
        provider = AnthropicProvider(
            model_id="claude-sonnet-4-5-20250929",
            temperature=0.5,
            max_completion_tokens=1500,
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
        provider = AnthropicProvider(
            model_id="claude-sonnet-4-5-20250929",
            temperature=0.5,
            max_completion_tokens=1500,
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
class TestAnthropicHaiku45:
    """Anthropic tests for claude-haiku-4-5-20251001."""

    def test_persona_content_generation(self):
        provider = AnthropicProvider(
            model_id="claude-haiku-4-5-20251001",
            temperature=0.5,
            max_completion_tokens=2000,
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
        provider = AnthropicProvider(
            model_id="claude-haiku-4-5-20251001",
            temperature=0.5,
            max_completion_tokens=1500,
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
        provider = AnthropicProvider(
            model_id="claude-haiku-4-5-20251001",
            temperature=0.5,
            max_completion_tokens=1500,
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
class TestAnthropicProvider:
    """General Anthropic provider tests mirroring OpenRouter structure."""

    def test_basic_text_response(self):
        provider = AnthropicProvider()
        response = provider.generate(
            prompt="What is the capital of France? Answer in one word.")
        assert "Paris" in response

    def test_structured_output(self):
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

    def test_with_messages(self):
        provider = AnthropicProvider()
        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides brief, accurate answers."},
            {"role": "user", "content": "What is the capital of France? Answer in one word."}
        ]

        response = provider.generate(messages=messages)
        assert "Paris" in response

    def test_messages_with_structured_output(self):
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

    def test_with_all_parameters(self):
        provider = AnthropicProvider(
            model_id="claude-haiku-4-5-20251001",
            temperature=0.3,
            max_completion_tokens=100,
        )

        prompt = "What is the capital of France? Answer in one word"
        response = provider.generate(prompt=prompt)

        assert "Paris" in response

    def test_structured_landmark_info(self):
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

    def test_batch_prompts(self):
        provider = AnthropicProvider()
        prompt = [
            "What is the capital of France? Answer in one word.",
            "What is the capital of Spain? Answer in one word.",
            "What is the capital of Portugal? Answer in one word."
        ]

        responses = provider.generate(prompt=prompt)

        assert len(responses) == 3
        assert isinstance(responses, list)
        assert all(isinstance(r, str) for r in responses)
        assert "Paris" in responses[0]
        assert "Madrid" in responses[1]
        assert "Lisbon" in responses[2]

    def test_batch_messages(self):
        provider = AnthropicProvider()
        messages = [
            [
                {"role": "system", "content": "You are a helpful assistant that provides brief, accurate answers."},
                {"role": "user", "content": "What is the capital of Canada? One word."}
            ],
            [
                {"role": "system", "content": "You are a helpful assistant that provides brief, accurate answers."},
                {"role": "user", "content": "What is the capital of Australia? One word."}
            ]
        ]

        responses = provider.generate(messages=messages)

        assert len(responses) == 2
        assert isinstance(responses, list)
        assert all(isinstance(r, str) for r in responses)
        assert "Ottawa" in responses[0]
        assert "Canberra" in responses[1]

    def test_batch_structured_output(self):
        provider = AnthropicProvider()
        prompt = [
            """What is the capital of Germany? 
            Provide a short answer and brief reasoning.
            Format as JSON with 'answer' and 'reasoning' fields.""",
            """What is the capital of Italy?
            Provide a short answer and brief reasoning.
            Format as JSON with 'answer' and 'reasoning' fields."""
        ]

        responses = provider.generate(
            prompt=prompt,
            response_format=SimpleResponse
        )

        assert len(responses) == 2
        assert all(isinstance(r, SimpleResponse) for r in responses)
        assert "Berlin" in responses[0].answer
        assert "Rome" in responses[1].answer
        assert len(responses[0].reasoning) > 5
        assert len(responses[1].reasoning) > 5

    def test_batch_messages_with_structured_output(self):
        provider = AnthropicProvider()
        messages = [
            [
                {"role": "system", "content": "You are a helpful assistant that provides answers in JSON format."},
                {"role": "user", "content": """What is the capital of Egypt? 
                Provide a short answer and brief reasoning.
                Format as JSON with 'answer' and 'reasoning' fields."""}
            ],
            [
                {"role": "system", "content": "You are a helpful assistant that provides answers in JSON format."},
                {"role": "user", "content": """What is the capital of Morocco?
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
        assert "Cairo" in responses[0].answer
        assert "Rabat" in responses[1].answer
        assert len(responses[0].reasoning) > 5
        assert len(responses[1].reasoning) > 5

    def test_batch_with_all_parameters(self):
        provider = AnthropicProvider(
            model_id="claude-haiku-4-5-20251001",
            temperature=0.1,
            max_completion_tokens=50
        )

        prompt = [
            "What is the capital of Denmark? Answer in one word.",
            "What is the capital of Finland? Answer in one word."
        ]

        responses = provider.generate(prompt=prompt)

        assert len(responses) == 2
        assert "Copenhagen" in responses[0]
        assert "Helsinki" in responses[1]

    def test_batch_validation_errors(self):
        provider = AnthropicProvider()

        # Test no inputs provided
        with pytest.raises(ValueError, match="Either prompts or messages must be provided"):
            provider.generate()

        # Test both inputs provided
        with pytest.raises(ValueError, match="Provide either prompts or messages, not both"):
            provider.generate(
                prompt=["test"],
                messages=[[{"role": "user", "content": "test"}]]
            )