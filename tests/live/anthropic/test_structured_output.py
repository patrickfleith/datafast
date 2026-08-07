"""Live structured output against Anthropic (native JSON schema)."""

import pytest
from pydantic import BaseModel, Field

pytestmark = [pytest.mark.live, pytest.mark.anthropic]


class CapitalFact(BaseModel):
    city: str = Field(description="The capital city")
    country: str = Field(description="The country it is the capital of")


def test_structured_output_from_a_prompt(served_model):
    response = served_model().generate(
        prompt="What is the capital of France?",
        response_format=CapitalFact,
    )

    assert isinstance(response, CapitalFact)
    assert "Paris" in response.city
    assert "France" in response.country


def test_structured_output_honours_the_system_prompt(served_model):
    """Steering applies to schema fields too, not only to free text."""
    messages = [
        {
            "role": "system",
            "content": "Always name countries by their ISO 3166 alpha-2 code.",
        },
        {"role": "user", "content": "What is the capital of France?"},
    ]

    response = served_model().generate(
        messages=messages,
        response_format=CapitalFact,
    )

    assert isinstance(response, CapitalFact)
    assert "Paris" in response.city
    assert response.country.strip().upper() == "FR"


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
