"""Live structured output against OpenRouter (native JSON schema).

These are the tests that make the provider pin in `conftest.py` load-bearing:
structured output is exactly the capability the routed endpoints disagree about,
so they run on `served_model` — the suite's default endpoint rejects
json_schema outright, despite the catalog claiming otherwise.
"""

import pytest
from pydantic import BaseModel, Field

pytestmark = [pytest.mark.live, pytest.mark.openrouter]


class CapitalFact(BaseModel):
    city: str = Field(description="The capital city")
    country: str = Field(description="The country it is the capital of")


class Attribute(BaseModel):
    name: str = Field(description="The attribute's name")
    value: str = Field(description="The attribute's value")


class Landmark(BaseModel):
    name: str = Field(description="The landmark's name")
    # min_length becomes minItems in the JSON schema, which is what makes the
    # nested-object branch unavoidable rather than optional.
    attributes: list[Attribute] = Field(
        min_length=2, description="Two notable attributes"
    )


def test_structured_output_from_a_prompt(served_model):
    response = served_model().generate(
        prompt="What is the capital of France?",
        response_format=CapitalFact,
    )

    assert isinstance(response, CapitalFact)
    assert "Paris" in response.city
    assert "France" in response.country


def test_structured_output_from_messages(served_model):
    """The schema has to survive the messages path as well as the prompt one —
    they build the request differently."""
    messages = [
        {"role": "system", "content": "You answer factual questions briefly."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    response = served_model().generate(
        messages=messages,
        response_format=CapitalFact,
    )

    assert isinstance(response, CapitalFact)
    assert "Paris" in response.city
    assert "France" in response.country


def test_nested_schema_is_honoured(served_model):
    """A flat two-string schema says little about the Pydantic-to-provider
    translation; a nested list of objects is where it can actually break."""
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
    prompts = [
        "What is the capital of Germany?",
        "What is the capital of Portugal?",
        "What is the capital of Japan?",
    ]

    responses = served_model(max_concurrent=3).generate(
        prompt=prompts,
        response_format=CapitalFact,
    )

    assert len(responses) == 3
    assert all(isinstance(response, CapitalFact) for response in responses)
    for expected, response in zip(("Berlin", "Lisbon", "Tokyo"), responses):
        assert expected in response.city
