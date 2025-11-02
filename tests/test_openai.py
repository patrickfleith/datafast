from datafast.llms import OpenAIProvider
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
class TestOpenAIProvider:
    """OpenAI provider tests using the default model gpt-5-mini-2025-08-07."""

    def test_basic_text_response(self):
        provider = OpenAIProvider()
        response = provider.generate(
            prompt="What is the capital of France? Answer in one word.")
        assert "Paris" in response

    def test_structured_output(self):
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
        assert len(response.reasoning) > 10

    def test_with_messages(self):
        provider = OpenAIProvider()
        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides brief, accurate answers."},
            {"role": "user", "content": "What is the capital of France? Answer in one word."}
        ]

        response = provider.generate(messages=messages)
        assert "Paris" in response

    def test_messages_with_structured_output(self):
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

    def test_with_all_parameters(self):
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

    def test_structured_landmark_info(self):
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

        assert isinstance(response, LandmarkInfo)
        assert "Eiffel Tower" in response.name
        assert "Paris" in response.location
        assert len(response.description) > 20
        assert response.year_built is not None and response.year_built > 1800
        assert len(response.attributes) >= 3

        for attr in response.attributes:
            assert 0 <= attr.importance <= 1
            assert len(attr.name) > 0
            assert len(attr.value) > 0

        assert 0 <= response.visitor_rating <= 5

    def test_batch_prompts(self):
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

    def test_batch_messages(self):
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

    def test_batch_structured_output(self):
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

    def test_batch_messages_with_structured_output(self):
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

    def test_batch_with_all_parameters(self):
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

    def test_batch_landmark_info(self):
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
            prompt=prompt,
            response_format=LandmarkInfo
        )

        assert len(responses) == 2
        assert all(isinstance(r, LandmarkInfo) for r in responses)

        assert "Statue of Liberty" in responses[0].name
        assert "New York" in responses[0].location
        assert len(responses[0].description) > 20
        assert responses[0].year_built is not None and responses[0].year_built > 1800
        assert len(responses[0].attributes) >= 3

        assert "Big Ben" in responses[1].name
        assert "London" in responses[1].location
        assert len(responses[1].description) > 20
        assert responses[1].year_built is not None and responses[1].year_built > 1800
        assert len(responses[1].attributes) >= 3

        for response in responses:
            for attr in response.attributes:
                assert 0 <= attr.importance <= 1
                assert len(attr.name) > 0
                assert len(attr.value) > 0
            assert 0 <= response.visitor_rating <= 5

    def test_batch_validation_errors(self):
        provider = OpenAIProvider()

        with pytest.raises(ValueError, match="Either prompts or messages must be provided"):
            provider.generate()

        with pytest.raises(ValueError, match="Provide either prompts or messages, not both"):
            provider.generate(
                prompt=["test"],
                messages=[[{"role": "user", "content": "test"}]]
            )
