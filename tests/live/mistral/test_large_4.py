"""Live checks for Mistral Large 4, which has its own reasoning levels.

The rest of this suite runs on mistral-small-2603. Large 4 takes 'low', 'medium' and
'high' and has no 'none', so it needs its own calls to prove those values reach the
model. The refusal of `thinking=False` happens before any request, so it is covered
in `tests/test_served_model_contract.py`.

Large 4 may reason even when no effort is set, and reasoning tokens count against
max_completion_tokens, so every test here uses a higher ceiling.
"""

import pytest
from pydantic import BaseModel, Field

from datafast import mistral
from datafast.llm import ContentPart

pytestmark = [pytest.mark.live, pytest.mark.mistral]

MODEL_ID = "mistral-large-4"
REASONING_TOKENS = 2048
PROMPT = "A shop sells pens at 3 for $2. What do 12 pens cost?"


class CapitalFact(BaseModel):
    city: str = Field(description="The capital city")
    country: str = Field(description="The country it is the capital of")


@pytest.fixture
def large_4(require_api_key):
    """Factory for a live Large 4 served model; overrides merge over the defaults."""
    require_api_key("MISTRAL_API_KEY")

    def _make(**overrides):
        params = {"temperature": 0.0, "max_completion_tokens": REASONING_TOKENS}
        params.update(overrides)
        return mistral(model_id=MODEL_ID, **params)

    return _make


def test_simple_prompt(large_4):
    response = large_4().generate(
        prompt="What is the capital of France? Answer in one word."
    )

    assert "Paris" in response


def test_thinking_true_returns_a_reasoning_trace(large_4):
    response = large_4(thinking=True).generate_response(prompt=PROMPT)

    assert response.reasoning_content or response.thinking_blocks
    assert "8" in response.text


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_each_documented_effort_is_accepted(large_4, effort):
    response = large_4(reasoning_effort=effort).generate_response(prompt=PROMPT)

    assert "8" in response.text


def test_structured_output(large_4):
    response = large_4().generate(
        prompt="What is the capital of France?",
        response_format=CapitalFact,
    )

    assert isinstance(response, CapitalFact)
    assert "Paris" in response.city
    assert "France" in response.country


@pytest.mark.multimodal
def test_image_input_reaches_the_model(large_4, image_asset):
    messages = [
        {
            "role": "user",
            "content": [
                ContentPart(
                    type="text",
                    text="What colour is this square? Answer with one word.",
                ),
                ContentPart(type="image", data=image_asset),
            ],
        }
    ]

    response = large_4().generate(messages=messages)

    assert "red" in response.lower()


@pytest.mark.multimodal
def test_uploaded_document_reaches_the_model(large_4, document_path):
    # upload_file has no retries, so a transient 502 from Mistral fails this test.
    model = large_4()
    # `expiry` backs up the delete below if this process dies mid-test.
    file_id = model.upload_file(document_path, expiry=1)

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    ContentPart(type="file", url=file_id),
                    ContentPart(
                        type="text",
                        text="What is the passphrase in this document? Answer with the passphrase only.",
                    ),
                ],
            }
        ]

        response = model.generate(messages=messages)
    finally:
        model.delete_file(file_id)

    assert "ZEBRAFISH-42" in response.upper()
