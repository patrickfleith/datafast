from datafast.llms import OpenAIProvider
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
            max_completion_tokens=1000,
            reasoning_effort="low"
        )

        prompt = "What is the capital of France? Answer in one word."
        response = provider.generate(prompt=prompt)

        assert "Paris" in response

    def test_structured_landmark_info(self):
        provider = OpenAIProvider(max_completion_tokens=1000)

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
            max_completion_tokens=1000,
            reasoning_effort="low"
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
        provider = OpenAIProvider(max_completion_tokens=1000)

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

    def test_persona_content_generation(self):
        """Test generating tweets and bio for a persona using OpenAI."""
        provider = OpenAIProvider(max_completion_tokens=1000)
        
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
        """Test generating Q&A pairs on machine learning using OpenAI."""
        provider = OpenAIProvider(max_completion_tokens=1500)
        
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
        """Test generating multiple choice questions using OpenAI."""
        provider = OpenAIProvider(max_completion_tokens=1500)
        
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
