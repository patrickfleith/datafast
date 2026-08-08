"""Live text generation against Mistral."""

import pytest

pytestmark = [pytest.mark.live, pytest.mark.mistral]


def test_simple_prompt(served_model):
    response = served_model().generate(
        prompt="What is the capital of France? Answer in one word."
    )

    assert "Paris" in response


def test_user_and_system_messages(served_model):
    messages = [
        {"role": "system", "content": "You answer factual questions briefly."},
        {
            "role": "user",
            "content": "What is the capital of France? Answer in one word.",
        },
    ]

    response = served_model().generate(messages=messages)

    assert "Paris" in response


def test_concurrent_prompts_keep_input_order(served_model):
    """Mistral batches natively (litellm.batch_completion), so no fallback warning
    is expected here — unlike the OpenAI suite."""
    prompts = [
        "What is the capital of France? Answer in one word.",
        "What is the capital of Spain? Answer in one word.",
        "What is the capital of Italy? Answer in one word.",
        "What is the capital of Japan? Answer in one word.",
    ]

    responses = served_model(max_concurrent=4).generate(prompt=prompts)

    assert len(responses) == 4
    for expected, response in zip(("Paris", "Madrid", "Rome", "Tokyo"), responses):
        assert expected in response
