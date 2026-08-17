"""Live reasoning controls against OpenAI (Responses endpoint).

The assertions differ from the Anthropic suite for two reasons:

- On Responses the effort is sent as `reasoning={"effort": ...}` and the trace comes
  back as an `output` item, so `thinking_blocks` is always empty. `output_items` is
  the Responses-only field to assert on.
- OpenAI only returns a reasoning *summary* when asked for one, so `reasoning_content`
  stays empty under plain `thinking=True` and needs `reasoning_summary`.

Reasoning tokens count against `max_output_tokens`, so these tests raise the
fixture's 300-token ceiling.
"""

import pytest

pytestmark = [pytest.mark.live, pytest.mark.openai]

REASONING_TOKENS = 2048
SUMMARY_TOKENS = 4096
PROMPT = "A shop sells pens at 3 for $2. What do 12 pens cost?"
# A trivial prompt at effort 'low' returns a reasoning item with an empty summary,
# so the summary test needs something worth summarising.
HARD_PROMPT = (
    "A freight train leaves at 09:00 travelling 60 km/h. A second leaves the same "
    "station at 11:30 travelling 95 km/h on the same track. At what clock time does "
    "the second catch the first? Give the final time only."
)


def test_thinking_true_emits_a_reasoning_item(served_model):
    """`thinking=True` resolves to effort 'low' and must produce a reasoning item
    alongside the answer, not instead of it."""
    response = served_model(
        thinking=True,
        max_completion_tokens=REASONING_TOKENS,
    ).generate_response(prompt=PROMPT)

    assert "8" in response.text
    assert any(item.get("type") == "reasoning" for item in response.output_items)


def test_reasoning_summary_populates_reasoning_content(served_model):
    """Proves `reasoning_summary` reaches the provider and that the response
    extractor parses the summary payload it comes back with.

    Effort must be 'high' here: OpenAI returns an empty summary for cheap reasoning,
    leaving only `encrypted_content`, and then there is genuinely nothing to extract.
    """
    response = served_model(
        max_completion_tokens=SUMMARY_TOKENS,
        reasoning_effort="high",
        reasoning_summary="auto",
    ).generate_response(prompt=HARD_PROMPT)

    assert "15:47" in response.text
    assert response.reasoning_content
