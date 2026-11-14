"""Live structured output against Ollama.

`OLLAMA_CHAT` claims `JSON_SCHEMA`, which is a real claim: LiteLLM turns
`response_format` into Ollama's `format` field carrying the raw JSON schema
(`llms/ollama/chat/transformation.py:177-179`), and the daemon then constrains
decoding to it. That is genuine schema-constrained generation, not the prompted
JSON a 0.6B model would otherwise struggle to produce — which is also what makes
these assertions safe on a model this small.
"""

import pytest
from pydantic import BaseModel, Field

pytestmark = [pytest.mark.live, pytest.mark.ollama]


class Attribute(BaseModel):
    name: str = Field(description="What the attribute is called")
    value: str = Field(description="Its value")


class Landmark(BaseModel):
    name: str = Field(description="The landmark's name")
    # min_length becomes minItems in the JSON schema, which is what makes the
    # nested-object branch of the grammar unavoidable rather than optional.
    attributes: list[Attribute] = Field(
        min_length=2, description="Two notable attributes"
    )


class CapitalFact(BaseModel):
    city: str = Field(description="The capital city")
    # Copied from the question rather than recalled, which is what lets the
    # concurrency test below key on it.
    country: str = Field(description="The country named in the question, verbatim")


def test_structured_output_from_a_prompt(served_model):
    response = served_model().generate(
        prompt="What is the capital of France?",
        response_format=CapitalFact,
    )

    assert isinstance(response, CapitalFact)
    assert "Paris" in response.city
    assert "France" in response.country


def test_nested_schema_is_honoured(served_model):
    """A flat two-string schema says little about constrained decoding — a nested
    list of objects is where a grammar either holds or falls apart.

    The list carries `minItems: 2`, so the two items are the grammar's obligation
    rather than the model's goodwill. Without it a 0.6B model happily returns
    `attributes: []`, which satisfies the schema while never once entering the
    nested-object branch — the test would pass having proved nothing.
    """
    response = served_model(max_completion_tokens=600).generate(
        prompt="Describe the Eiffel Tower and two of its notable attributes.",
        response_format=Landmark,
    )

    assert isinstance(response, Landmark)
    assert "Eiffel" in response.name
    assert len(response.attributes) >= 2
    for attribute in response.attributes:
        assert isinstance(attribute, Attribute)
        assert attribute.name and attribute.value


def test_concurrent_structured_output_keeps_input_order(served_model):
    """What is under test is that the thread pool returns responses in submission
    order, so the assertion keys on the country named in the prompt rather than on
    the capital the model has to know. A 0.6B model under constrained decoding will
    sometimes fill the wrong field — it answered "Portugal" as the city once — and
    that is a fact about the model, not about ordering. Field placement is
    `test_structured_output_from_a_prompt`'s job; here any field will do.
    """
    prompts = [
        "What is the capital of Germany?",
        "What is the capital of Portugal?",
        "What is the capital of Japan?",
    ]

    with pytest.warns(UserWarning, match="does not expose native batching"):
        responses = served_model(max_concurrent=2).generate(
            prompt=prompts,
            response_format=CapitalFact,
        )

    assert len(responses) == 3
    assert all(isinstance(response, CapitalFact) for response in responses)
    for expected, response in zip(("Germany", "Portugal", "Japan"), responses):
        assert expected in response.model_dump_json()
