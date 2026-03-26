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
class TestMistralLarge:
    """Test suite for mistral-large-2512 model."""

    def test_persona_content_generation(self):
        provider = MistralProvider(
            model_id="mistral-large-2512",
            temperature=0.5,
            max_completion_tokens=2000,
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
        provider = MistralProvider(
            model_id="mistral-large-2512",
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
        provider = MistralProvider(
            model_id="mistral-large-2512",
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

    def test_structured_landmark_info(self):
        provider = MistralProvider(
            model_id="mistral-large-2512",
            temperature=0.1,
            max_completion_tokens=800
        )
        prompt = """
        Provide detailed information about the Great Wall of China.

        Return your response as a structured JSON object with the following elements:
        - name: The name of the landmark (Great Wall of China)
        - location: Where it's located (China)
        - description: A brief description of the landmark (2-3 sentences)
        - year_built: The year when construction began (as a number)
        - attributes: A list of at least 3 attribute objects, each containing:
          - name: The name of the attribute (e.g., "length", "material", "purpose")
          - value: The value of the attribute (e.g., "13,171 miles", "stone and brick", "defense")
          - importance: An importance score between 0 and 1
        - visitor_rating: Average visitor rating from 0 to 5 (e.g., 4.6)

        Make sure your response is properly structured and can be parsed as valid JSON.
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

    def test_simple_response_structured(self):
        provider = MistralProvider(
            model_id="mistral-large-2512",
            temperature=0.3,
            max_completion_tokens=500
        )
        prompt = """What is machine learning?
        Provide a short answer and a brief explanation.
        Format your response as JSON with 'answer' and 'reasoning' fields."""
        response = provider.generate(prompt=prompt, response_format=SimpleResponse)
        assert isinstance(response, SimpleResponse)
        assert len(response.answer) > 10
        assert len(response.reasoning) > 20

    def test_batch_structured_output(self):
        provider = MistralProvider(
            model_id="mistral-large-2512",
            temperature=0.5,
            max_completion_tokens=1000
        )
        prompt = [
            """What is deep learning?
            Provide a short answer and brief reasoning.
            Format as JSON with 'answer' and 'reasoning' fields.""",
            """What is natural language processing?
            Provide a short answer and brief reasoning.
            Format as JSON with 'answer' and 'reasoning' fields."""
        ]
        responses = provider.generate(prompt=prompt, response_format=SimpleResponse)
        assert len(responses) == 2
        assert all(isinstance(r, SimpleResponse) for r in responses)
        assert "deep learning" in responses[0].answer.lower() or "neural" in responses[0].answer.lower()
        assert "nlp" in responses[1].answer.lower() or "language" in responses[1].answer.lower()
        assert all(len(r.reasoning) > 10 for r in responses)


@pytest.mark.integration
class TestMagistralMedium:
    """Test suite for magistral-medium-2509 model."""

    def test_persona_content_generation(self):
        provider = MistralProvider(
            model_id="magistral-medium-2509",
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
        provider = MistralProvider(
            model_id="magistral-medium-2509",
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
        provider = MistralProvider(
            model_id="magistral-medium-2509",
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

    def test_structured_landmark_info(self):
        provider = MistralProvider(
            model_id="magistral-medium-2509",
            temperature=0.1,
            max_completion_tokens=800
        )
        prompt = """
        Provide detailed information about the Colosseum in Rome.

        Return your response as a structured JSON object with the following elements:
        - name: The name of the landmark (Colosseum)
        - location: Where it's located (Rome, Italy)
        - description: A brief description of the landmark (2-3 sentences)
        - year_built: The year when it was built (as a number)
        - attributes: A list of at least 3 attribute objects, each containing:
          - name: The name of the attribute (e.g., "capacity", "material", "architect")
          - value: The value of the attribute (e.g., "50,000 spectators", "concrete and stone", "Emperor Vespasian")
          - importance: An importance score between 0 and 1
        - visitor_rating: Average visitor rating from 0 to 5 (e.g., 4.7)

        Make sure your response is properly structured and can be parsed as valid JSON.
        """
        response = provider.generate(prompt=prompt, response_format=LandmarkInfo)
        assert isinstance(response, LandmarkInfo)
        assert "Colosseum" in response.name
        assert "Rome" in response.location
        assert len(response.description) > 20
        assert response.year_built is not None and response.year_built > 0
        assert len(response.attributes) >= 3
        for attr in response.attributes:
            assert 0 <= attr.importance <= 1
            assert len(attr.name) > 0
            assert len(attr.value) > 0
        assert 0 <= response.visitor_rating <= 5

    def test_simple_response_structured(self):
        provider = MistralProvider(
            model_id="magistral-medium-2509",
            temperature=0.3,
            max_completion_tokens=500
        )
        prompt = """What is quantum computing?
        Provide a short answer and a brief explanation.
        Format your response as JSON with 'answer' and 'reasoning' fields."""
        response = provider.generate(prompt=prompt, response_format=SimpleResponse)
        assert isinstance(response, SimpleResponse)
        assert len(response.answer) > 10
        assert len(response.reasoning) > 20

    def test_batch_structured_output(self):
        provider = MistralProvider(
            model_id="magistral-medium-2509",
            temperature=0.5,
            max_completion_tokens=1000
        )
        prompt = [
            """What is artificial intelligence?
            Provide a short answer and brief reasoning.
            Format as JSON with 'answer' and 'reasoning' fields.""",
            """What is blockchain technology?
            Provide a short answer and brief reasoning.
            Format as JSON with 'answer' and 'reasoning' fields."""
        ]
        responses = provider.generate(prompt=prompt, response_format=SimpleResponse)
        assert len(responses) == 2
        assert all(isinstance(r, SimpleResponse) for r in responses)
        assert "intelligence" in responses[0].answer.lower() or "ai" in responses[0].answer.lower()
        assert "blockchain" in responses[1].answer.lower() or "distributed" in responses[1].answer.lower()
        assert all(len(r.reasoning) > 10 for r in responses)

    def test_batch_qa_generation(self):
        provider = MistralProvider(
            model_id="magistral-medium-2509",
            temperature=0.5,
            max_completion_tokens=2000
        )
        prompt = [
            """Generate exactly 3 questions and their correct answers about Python programming.
            Topics: data structures, functions, object-oriented programming.
            Each question should be clear and the answer should be concise but complete.""",
            """Generate exactly 3 questions and their correct answers about web development.
            Topics: HTML, CSS, JavaScript.
            Each question should be clear and the answer should be concise but complete."""
        ]
        responses = provider.generate(prompt=prompt, response_format=QASet)
        assert len(responses) == 2
        assert all(isinstance(r, QASet) for r in responses)
        assert all(len(r.questions) == 3 for r in responses)
        for qa_set in responses:
            for qa in qa_set.questions:
                assert len(qa.question) > 10
                assert len(qa.answer) > 10

    def test_batch_mcq_generation(self):
        provider = MistralProvider(
            model_id="magistral-medium-2509",
            temperature=0.5,
            max_completion_tokens=2000
        )
        prompt = [
            """Generate exactly 2 multiple choice questions about databases.
            For each question, provide: the question, one correct answer, and three incorrect answers.
            Topics: SQL, NoSQL.""",
            """Generate exactly 2 multiple choice questions about cloud computing.
            For each question, provide: the question, one correct answer, and three incorrect answers.
            Topics: AWS, Azure."""
        ]
        responses = provider.generate(prompt=prompt, response_format=MCQSet)
        assert len(responses) == 2
        assert all(isinstance(r, MCQSet) for r in responses)
        assert all(len(r.questions) == 2 for r in responses)
        for mcq_set in responses:
            for mcq in mcq_set.questions:
                assert len(mcq.question) > 10
                assert len(mcq.correct_answer) > 0
                assert len(mcq.incorrect_answers) == 3
                assert all(len(ans) > 0 for ans in mcq.incorrect_answers)


@pytest.mark.integration
class TestMistralProvider:
    """General Mistral provider tests mirroring OpenAI/Anthropic structure."""

    def test_basic_text_response(self):
        provider = MistralProvider()
        response = provider.generate(
            prompt="What is the capital of France? Answer in one word.")
        assert "Paris" in response

    def test_structured_output(self):
        provider = MistralProvider()
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
        provider = MistralProvider()
        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides brief, accurate answers."},
            {"role": "user", "content": "What is the capital of France? Answer in one word."}
        ]

        response = provider.generate(messages=messages)
        assert "Paris" in response

    def test_messages_with_structured_output(self):
        provider = MistralProvider()
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
        provider = MistralProvider(
            model_id="mistral-large-2512",
            temperature=0.3,
            max_completion_tokens=100,
        )

        prompt = "What is the capital of France? Answer in one word"
        response = provider.generate(prompt=prompt)

        assert "Paris" in response

    def test_structured_landmark_info(self):
        provider = MistralProvider(temperature=0.1, max_completion_tokens=800)

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
        provider = MistralProvider()
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
        provider = MistralProvider()
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
        provider = MistralProvider()
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
        provider = MistralProvider()
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
        provider = MistralProvider(
            model_id="mistral-large-2512",
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
        provider = MistralProvider()

        with pytest.raises(ValueError, match="Either prompts or messages must be provided"):
            provider.generate()

        with pytest.raises(ValueError, match="Provide either prompts or messages, not both"):
            provider.generate(
                prompt=["test"],
                messages=[[{"role": "user", "content": "test"}]]
            )

    def test_persona_content_generation(self):
        provider = MistralProvider(max_completion_tokens=1000)
        
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
        provider = MistralProvider(max_completion_tokens=1500)
        
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
        provider = MistralProvider(max_completion_tokens=1500)
        
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
