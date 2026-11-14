"""Live text generation against Anthropic."""

import pytest

pytestmark = [pytest.mark.live, pytest.mark.anthropic]


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


def test_system_prompt_steers_the_answer_format(served_model):
    """The system message must steer the model, not just ride along unused."""
    messages = [
        {
            "role": "system",
            "content": (
                "Reply with a single lowercase word. "
                "No punctuation, no explanation, no preamble."
            ),
        },
        {"role": "user", "content": "What is the capital of France?"},
    ]

    response = served_model().generate(messages=messages)

    assert response.strip() == "paris"


def test_concurrent_prompts_keep_input_order(served_model):
    """Anthropic batches natively (litellm.batch_completion), bounded by max_concurrent."""
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


def test_concurrent_message_lists_keep_input_order(served_model):
    """A batch of message lists is a different input shape from a batch of
    prompts — each element is itself a list, so a flattening bug would only show
    up here."""
    messages = [
        [
            {"role": "system", "content": "You answer factual questions briefly."},
            {"role": "user", "content": "What is the capital of Canada? One word."},
        ],
        [
            {"role": "system", "content": "You answer factual questions briefly."},
            {"role": "user", "content": "What is the capital of Australia? One word."},
        ],
    ]

    responses = served_model(max_concurrent=2).generate(messages=messages)

    assert len(responses) == 2
    for expected, response in zip(("Ottawa", "Canberra"), responses):
        assert expected in response
