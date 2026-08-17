"""Live reasoning controls for the adaptive-thinking Anthropic line.

The rest of this suite runs on claude-haiku-4-5, which reasons only when asked.
claude-sonnet-5 is the opposite — it reasons unless told not to — and the two
tests here are the pair that proves it, since either one alone would pass on a
model that never reasons at all.

The prompt is deliberately hard: thinking is adaptive, so the model declines to
reason on anything easy and a simple prompt would prove nothing.
"""

import pytest

from datafast import anthropic

pytestmark = [pytest.mark.live, pytest.mark.anthropic]

MODEL_ID = "claude-sonnet-5"
# Thinking is billed out of the same budget as the answer, and how much of it
# gets spent varies per call — this prompt sometimes reasons until the budget is
# gone and returns no answer at all (finish_reason='length'), even at 10000.
# That is why the reasoning-on test below asserts the trace and not the answer:
# requiring both would make it flaky. It is also why the profile carries a note
# telling callers to raise max_completion_tokens when reasoning is on.
THINKING_TOKENS = 4000
PROMPT = (
    "Three people A, B, C each hold a distinct integer from 1 to 9, with "
    "A+B=C+4 and B*C=A+30. Find every solution."
)


@pytest.fixture
def sonnet(require_api_key):
    require_api_key("ANTHROPIC_API_KEY")

    def _make(**overrides):
        return anthropic(
            model_id=MODEL_ID, max_completion_tokens=THINKING_TOKENS, **overrides
        )

    return _make


def test_thinking_true_returns_a_reasoning_trace(sonnet):
    """The trace arrives as thinking blocks and nothing else: their text is
    empty and so is reasoning_content, because Anthropic omits the written
    summary by default and Datafast exposes no control to ask for one. Only the
    blocks' presence is worth asserting — the answer may not fit the budget."""
    response = sonnet(thinking=True).generate_response(prompt=PROMPT)

    assert response.thinking_blocks


def test_thinking_false_actually_stops_the_reasoning(sonnet):
    """The point of ANTHROPIC_ADAPTIVE_CHAT. Omitting the parameter — or
    sending reasoning_effort='none', which LiteLLM turns into omitting it —
    leaves this model reasoning, so only the native disable is honoured. With
    reasoning off the whole budget goes to the answer, so this is also where
    the suite checks one comes back."""
    response = sonnet(thinking=False).generate_response(prompt=PROMPT)

    assert not response.thinking_blocks
    assert response.text


def test_caller_temperature_is_refused(sonnet):
    """Anthropic accepts only temperature=1 on this line, whether or not
    reasoning is on, so Datafast drops it instead of letting the call 400."""
    with pytest.warns(UserWarning, match="temperature"):
        response = sonnet(thinking=False, temperature=0.0).generate_response(
            prompt=PROMPT
        )

    assert response.text
