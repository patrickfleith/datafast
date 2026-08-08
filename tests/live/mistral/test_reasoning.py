"""Live reasoning controls against Mistral.

This is the module the whole Mistral suite exists for. `MISTRAL_REASONING_CHAT` sets
`reasoning_requires_allowlist=True` because the installed LiteLLM only recognises
`reasoning_effort` for magistral ids and would silently drop it for
mistral-medium/small; datafast forces it through with
`allowed_openai_params=["reasoning_effort"]` (`served_model.py:616`). Mocked tests
prove datafast *sets* that key — only a real call proves the effort arrives.

Mistral also accepts only 'high' and 'none'; 'low'/'medium' 400. The client-side
rejection is covered in `tests/test_served_model_contract.py`, so what is left to
verify here is that the two allowed values are in fact accepted server-side.

Reasoning tokens count against max_completion_tokens, so these tests raise the
fixture's 300-token ceiling.
"""

import pytest

pytestmark = [pytest.mark.live, pytest.mark.mistral]

REASONING_TOKENS = 2048
PROMPT = "A shop sells pens at 3 for $2. What do 12 pens cost?"


def test_thinking_true_returns_a_reasoning_trace(served_model):
    """`thinking=True` resolves to this profile's own "on" level, 'high'. The trace
    must come back alongside the answer, not instead of it — and it only comes back
    at all if the allowlist got the param past LiteLLM."""
    response = served_model(
        thinking=True,
        max_completion_tokens=REASONING_TOKENS,
    ).generate_response(prompt=PROMPT)

    assert response.reasoning_content or response.thinking_blocks
    assert "8" in response.text


def test_thinking_false_sends_the_off_value(served_model):
    """`reasoning_off_param` is `("reasoning_effort", "none")`, so turning reasoning
    off means sending a value rather than omitting the param. 'none' is only a legal
    value because Mistral says so — this is the call that confirms it."""
    response = served_model(
        thinking=False,
        max_completion_tokens=REASONING_TOKENS,
    ).generate_response(prompt=PROMPT)

    assert "8" in response.text
    assert not response.reasoning_content
