"""Live text generation against OpenAI (Responses endpoint).

Every openai catalog id resolves to OPENAI_RESPONSES, so these calls go through
`litellm.responses` rather than `litellm.completion` — a transport the Anthropic
suite never touches. They are the only proof that datafast's Responses request
shape is accepted and that `_extract_responses_text` reads a real payload.
"""

import pytest

pytestmark = [pytest.mark.live, pytest.mark.openai]


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
    """OpenAI has no native batching on this endpoint, so datafast warns and falls
    back to a bounded thread pool. Order must survive the pool."""
    prompts = [
        "What is the capital of France? Answer in one word.",
        "What is the capital of Spain? Answer in one word.",
        "What is the capital of Italy? Answer in one word.",
        "What is the capital of Japan? Answer in one word.",
    ]

    with pytest.warns(UserWarning, match="does not expose native batching"):
        responses = served_model(max_concurrent=4).generate(prompt=prompts)

    assert len(responses) == 4
    for expected, response in zip(("Paris", "Madrid", "Rome", "Tokyo"), responses):
        assert expected in response


def test_concurrent_message_lists_keep_input_order(served_model):
    """A batch of message lists is a different input shape from a batch of
    prompts — each element is itself a list, so a flattening bug would only show
    up here. It goes through the same fallback pool."""
    messages = [
        [
            {"role": "system", "content": "You answer factual questions briefly."},
            {"role": "user", "content": "What is the capital of France? One word."},
        ],
        [
            {"role": "system", "content": "You answer factual questions briefly."},
            {"role": "user", "content": "What is the capital of Japan? One word."},
        ],
    ]

    with pytest.warns(UserWarning, match="does not expose native batching"):
        responses = served_model(max_concurrent=2).generate(messages=messages)

    assert len(responses) == 2
    for expected, response in zip(("Paris", "Tokyo"), responses):
        assert expected in response


def test_temperature_is_dropped_for_responses(served_model):
    """OPENAI_RESPONSES omits temperature because the API rejects sampling
    controls. Dropping it client-side is what keeps this call from 400ing."""
    with pytest.warns(UserWarning, match="temperature"):
        response = served_model(temperature=0.0).generate(
            prompt="What is the capital of France? Answer in one word."
        )

    assert "Paris" in response
