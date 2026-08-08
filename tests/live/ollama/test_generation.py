"""Live text generation against a local Ollama daemon.

The transport is `litellm.completion` on the `ollama_chat` route, which rewrites
the request into Ollama's own shape rather than an OpenAI-compatible one:
`max_completion_tokens` becomes `num_predict` inside an `options` dict, and
`frequency_penalty` becomes `repeat_penalty`
(`llms/ollama/chat/transformation.py:161-172`). Nothing mocked can show that the
daemon accepts the result.

Ollama has no batch endpoint, so `OLLAMA_CHAT` is `FALLBACK_CONCURRENCY` and the
concurrency test expects datafast's warning — the same path openai takes, but
here every concurrent request competes for one machine's GPU.
"""

import pytest

pytestmark = [pytest.mark.live, pytest.mark.ollama]


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


def test_top_p_is_accepted(served_model):
    """`top_p` means the same thing to Ollama as to OpenAI and passes through
    unchanged, so a caller-set value must not disturb the answer.

    `frequency_penalty` is deliberately not tested alongside it. LiteLLM maps it to
    Ollama's `repeat_penalty` without rescaling (`transformation.py:171`), but their
    neutral points differ: 0 is neutral for `frequency_penalty`, while
    `repeat_penalty` is a multiplier where 1.0 is neutral and lower values *reward*
    repetition. A mild OpenAI-style 0.15 therefore arrives as strong repetition
    encouragement and degenerates the output. Asserting a good answer under it would
    only encode the confusion.
    """
    response = served_model(top_p=0.85).generate(
        prompt="What is the capital of France? Answer in one word."
    )

    assert "Paris" in response


def test_batch_of_message_lists(served_model):
    """`generate` takes a list of message lists as well as a list of prompts, and
    that shape has its own normalization path. Distinct arithmetic per item so a
    swapped result cannot pass."""
    conversations = [
        [
            {"role": "system", "content": "Answer with the number only."},
            {"role": "user", "content": "What is 6 + 4?"},
        ],
        [
            {"role": "system", "content": "Answer with the number only."},
            {"role": "user", "content": "What is 20 + 3?"},
        ],
    ]

    with pytest.warns(UserWarning, match="does not expose native batching"):
        responses = served_model(max_concurrent=2).generate(messages=conversations)

    assert len(responses) == 2
    assert "10" in responses[0]
    assert "23" in responses[1]


def test_concurrent_prompts_keep_input_order(served_model):
    """`max_concurrent` stays at 2: these requests share one local GPU, so a wider
    pool would slow the suite down rather than speed it up. Order must still
    survive the pool.

    The prompts are echoes rather than questions. Asking for capital cities made the
    assertion depend on the model's knowledge — a 0.6B answered "Veneto" for Italy —
    and a wrong answer in the right slot is indistinguishable from a right answer in
    the wrong slot. Echoing four maximally distinct tokens isolates what is actually
    under test: that `executor.map` returns results in submission order.
    """
    words = ("ALPHA", "BRAVO", "CHARLIE", "DELTA")
    prompts = [f"Reply with exactly this one word: {word}" for word in words]

    with pytest.warns(UserWarning, match="does not expose native batching"):
        responses = served_model(max_concurrent=2).generate(prompt=prompts)

    assert len(responses) == 4
    for expected, response in zip(words, responses):
        assert expected in response.upper()
