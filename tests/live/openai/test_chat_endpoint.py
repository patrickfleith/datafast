"""Live coverage of the OPENAI_CHAT profile.

No openai id in the catalog reaches this profile — every `gpt-5.x` id resolves to
OPENAI_RESPONSES. A `gpt-4o-mini` falls through to OPENAI_CHAT, which is the only
place `litellm.completion` and native batching run under `provider_id="openai"`,
and the only openai profile where a temperature is genuinely supported.
"""

import pytest

pytestmark = [pytest.mark.live, pytest.mark.openai]


def test_chat_profile_accepts_temperature_and_batches_natively(
    chat_served_model, recwarn
):
    prompts = [
        "What is the capital of France? Answer in one word.",
        "What is the capital of Spain? Answer in one word.",
    ]

    responses = chat_served_model(max_concurrent=2).generate(prompt=prompts)

    for expected, response in zip(("Paris", "Madrid"), responses):
        assert expected in response

    messages = [str(warning.message) for warning in recwarn]
    assert not [m for m in messages if "temperature" in m]
    assert not [m for m in messages if "native batching" in m]
