"""Live text generation against OpenRouter."""

import pytest

pytestmark = [pytest.mark.live, pytest.mark.openrouter]


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


def test_sampling_params_are_accepted(served_model):
    """OPENROUTER_CHAT claims SAMPLING_CHAT_PARAMS on top of the common ones, and
    only a real call proves LiteLLM forwards them to openrouter/* rather than
    dropping them."""
    response = served_model(top_p=0.85, frequency_penalty=0.1).generate(
        prompt="What is the capital of France? Answer in one word."
    )

    assert "Paris" in response


def test_concurrent_prompts_keep_input_order(served_model):
    """OpenRouter batches natively (litellm.batch_completion), so no fallback
    warning is expected here — unlike the OpenAI suite."""
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


def test_reasoning_controls_are_refused(served_model):
    """OPENROUTER_CHAT deliberately declares no reasoning controls, because what
    a reasoning request means here depends on the routed model rather than on
    the provider. So `thinking=True` goes through the unsupported_params policy
    instead of being sent, and the call still returns an answer.

    This is the profile's note ("pass provider_params for model-specific
    OpenRouter/LiteLLM escape hatches") stated as a test.
    """
    with pytest.warns(UserWarning, match="reasoning_effort"):
        response = served_model(thinking=True).generate(
            prompt="What is the capital of France? Answer in one word."
        )

    assert "Paris" in response
