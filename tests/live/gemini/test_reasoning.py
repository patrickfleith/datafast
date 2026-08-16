"""Live reasoning controls against Gemini.

Gemini 3 has no off switch: 'minimal' is the floor everywhere, and whether a
model even accepts that floor is per-model. That split is why the suite runs
two models — `gemini-3.5-flash-lite` takes 'minimal', `gemini-3.7-flash` does
not — and it is the whole reason `GEMINI_NO_MINIMAL_CHAT` exists alongside
`GEMINI_CHAT`.

These tests only raise the fixture's token ceiling, to leave room for a trace
alongside the answer. Gemini does not lock temperature while thinking the way
Anthropic does, and the fixture pins no temperature to begin with.
"""

import pytest

pytestmark = [pytest.mark.live, pytest.mark.gemini]

THINKING_TOKENS = 2048
PROMPT = "A shop sells pens at 3 for $2. What do 12 pens cost?"


def test_thinking_true_returns_a_reasoning_trace(served_model):
    """The trace must come back alongside the answer, not instead of it."""
    response = served_model(
        thinking=True,
        max_completion_tokens=THINKING_TOKENS,
    ).generate_response(prompt=PROMPT)

    assert response.reasoning_content or response.thinking_blocks
    assert "8" in response.text


def test_thinking_false_leaves_no_trace(served_model):
    """`thinking=False` sends reasoning_effort='none', which LiteLLM turns into
    the lowest thinking level with the trace hidden. So the assertion is that no
    trace comes back — not that no reasoning tokens were spent, which is not
    something Gemini 3 lets any caller ask for."""
    response = served_model(thinking=False).generate_response(prompt=PROMPT)

    assert not response.reasoning_content
    assert not response.thinking_blocks
    assert "8" in response.text


def test_explicit_reasoning_effort(served_model):
    """GEMINI_CHAT sets no `reasoning_efforts` allowlist, so the level is
    forwarded unchecked and only the real API confirms it is accepted."""
    response = served_model(
        reasoning_effort="high",
        max_completion_tokens=THINKING_TOKENS,
    ).generate_response(prompt=PROMPT)

    assert response.reasoning_content or response.thinking_blocks
    assert "8" in response.text


def test_flash_accepts_its_floor_effort(flash_served_model):
    """'low' is the lowest level gemini-3.7-flash accepts, and pinning that
    live is what makes GEMINI_NO_MINIMAL_CHAT's allowlist a measurement rather
    than a transcription of the docs.

    No trace is asserted, unlike the flash-lite tests above. At 'low' this
    model returns neither reasoning_content nor thinking_blocks — only an
    opaque `thought_signatures` entry in provider_specific_fields, which shows
    it reasoned without exposing a summary. Asserting a visible trace here
    would be asserting a second, weaker claim that does not hold.
    """
    response = flash_served_model(
        reasoning_effort="low",
        max_completion_tokens=THINKING_TOKENS,
    ).generate_response(prompt=PROMPT)

    assert "8" in response.text


def test_flash_rejects_the_minimal_level(flash_served_model):
    """The premise the whole profile rests on: gemini-3.7-flash really does
    refuse 'minimal'.

    Datafast blocks that value client-side, so the only way to ask the API
    itself is to go around the allowlist with provider_params, which is applied
    last and overwrites the resolved effort. If this test ever passes a value
    through, Google has widened the model and the profile should follow.
    """
    model = flash_served_model(
        reasoning_effort="low",
        provider_params={"reasoning_effort": "minimal"},
    )

    # Google's message shouts the level back: "Thinking level MINIMAL is not
    # supported for this model."
    with pytest.raises(RuntimeError, match="(?i)thinking level minimal"):
        model.generate(prompt=PROMPT)
