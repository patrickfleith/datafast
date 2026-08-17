"""Live reasoning controls against Anthropic.

`thinking=True` resolves to the served model's own "on" level, which for
ANTHROPIC_CHAT is reasoning_effort='low'. LiteLLM translates that into
Anthropic's native thinking block, and the shape depends on the model:
claude-haiku-4-5 gets {"type": "enabled", "budget_tokens": 1024}, while
claude-sonnet-4-6 gets {"type": "adaptive"}.

The budget must fit inside max_tokens (Anthropic's minimum is 1024), so these
tests raise the shared fixture's 300-token ceiling. They also clear the
fixture's temperature: Anthropic accepts only temperature=1 while thinking is
on, so datafast drops a caller-set temperature and warns about it.
"""

import pytest

pytestmark = [pytest.mark.live, pytest.mark.anthropic]

THINKING_TOKENS = 2048
# reasoning_effort='high' asks for a larger budget than THINKING_TOKENS fits.
HIGH_EFFORT_TOKENS = 8192
PROMPT = "A shop sells pens at 3 for $2. What do 12 pens cost?"


def test_thinking_true_returns_a_reasoning_trace(served_model):
    """The trace must come back alongside the answer, not instead of it."""
    response = served_model(
        thinking=True,
        temperature=None,
        max_completion_tokens=THINKING_TOKENS,
    ).generate_response(prompt=PROMPT)

    assert response.reasoning_content or response.thinking_blocks
    assert "8" in response.text


def test_thinking_drops_a_caller_temperature(served_model):
    """`reasoning_locks_temperature` exists because Anthropic 400s on any
    temperature but 1 while thinking is on. Keeping the fixture's temperature=0
    proves datafast drops it rather than letting the call fail."""
    with pytest.warns(UserWarning, match="temperature"):
        response = served_model(
            thinking=True,
            max_completion_tokens=THINKING_TOKENS,
        ).generate_response(prompt=PROMPT)

    assert response.reasoning_content or response.thinking_blocks


def test_explicit_reasoning_effort(served_model):
    """ANTHROPIC_CHAT sets no `reasoning_efforts` allowlist, so the level is
    forwarded unchecked and only the real API confirms it is accepted. 'high'
    is the level worth spending a call on: `thinking=True` already covers
    'low', and 'high' asks for the largest thinking budget."""
    response = served_model(
        reasoning_effort="high",
        temperature=None,
        max_completion_tokens=HIGH_EFFORT_TOKENS,
    ).generate_response(prompt=PROMPT)

    assert response.reasoning_content or response.thinking_blocks
    assert "8" in response.text
