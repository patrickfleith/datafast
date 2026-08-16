"""Live structured output against Gemini (native JSON schema)."""

import pytest
from pydantic import BaseModel, Field

pytestmark = [pytest.mark.live, pytest.mark.gemini]


class Attribute(BaseModel):
    name: str = Field(description="The attribute's name")
    value: str = Field(description="The attribute's value")


class Landmark(BaseModel):
    name: str = Field(description="The landmark's name")
    # min_length becomes minItems in the JSON schema, which is what makes the
    # nested-object branch of the schema unavoidable rather than optional.
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
    """Gemini's structured output is a native schema rather than a prompt
    instruction, so a nested list of objects is where that claim is worth
    testing — a flat two-string schema would say little."""
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
    for expected, response in zip(("Germany", "Portugal", "Japan"), responses):
        assert isinstance(response, CapitalFact)
        assert expected in response.country
