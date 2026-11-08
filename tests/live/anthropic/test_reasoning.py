"""Live reasoning controls against Anthropic, on claude-haiku-5-5.

The prompt is hard because thinking is adaptive: the model declines to reason on
anything easy. Tests that need a trace assert only the trace, since the reasoning
can use the whole budget and leave no answer.
"""

import pytest

pytestmark = [pytest.mark.live, pytest.mark.anthropic]

THINKING_TOKENS = 8192
PROMPT = (
    "Three people A, B, C each hold a distinct integer from 1 to 9, with "
    "A+B=C+4 and B*C=A+30. Find every solution."
)


def test_thinking_true_returns_a_reasoning_trace(served_model):
    response = served_model(
        thinking=True, max_completion_tokens=THINKING_TOKENS
    ).generate_response(prompt=PROMPT)

    assert response.reasoning_content or response.thinking_blocks


def test_thinking_false_actually_stops_the_reasoning(served_model):
    """Haiku 5.5 reasons by default, so only the native disable is honoured."""
    response = served_model(thinking=False).generate_response(prompt=PROMPT)

    assert not response.thinking_blocks
    assert response.text


def test_explicit_reasoning_effort_is_accepted(served_model):
    """'low' differs from the 'high' this model gets by default, so this proves
    a caller's level reaches the API."""
    response = served_model(
        thinking=None,
        reasoning_effort="low",
        max_completion_tokens=THINKING_TOKENS,
    ).generate_response(prompt="A shop sells pens at 3 for $2. What do 12 pens cost?")

    assert "8" in response.text


def test_thinking_drops_a_caller_temperature_on_haiku_4_5(served_model):
    """`reasoning_locks_temperature` applies only to ANTHROPIC_CHAT, so this
    needs a model on that profile. Anthropic 400s on any temperature but 1 while
    thinking is on, so datafast has to drop it."""
    with pytest.warns(UserWarning, match="temperature"):
        response = served_model(
            model_id="claude-haiku-4-5",
            thinking=True,
            temperature=0.0,
            max_completion_tokens=2048,
        ).generate_response(prompt=PROMPT)

    assert response.reasoning_content or response.thinking_blocks
