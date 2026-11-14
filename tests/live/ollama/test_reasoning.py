"""Live reasoning controls against Ollama.

Ollama does not take `reasoning_effort`. LiteLLM translates it into the daemon's
own boolean `think` field, and the translation is lossy: for gpt-oss the effort
string is forwarded verbatim, for every other model it collapses to
`think = effort in {"low", "medium", "high"}`
(`llms/ollama/chat/transformation.py:180-184`). That asymmetry is why
`OLLAMA_REASONING_CHAT` turns reasoning off with `("think", False)` rather than
with `reasoning_effort="none"`: the "none" string would reach a gpt-oss daemon
literally and be rejected.

Turning it off explicitly is not optional here — qwen3 thinks by default, so
omitting the field leaves reasoning on. These are the tests that show the trace
appears and disappears on command, and that LiteLLM's remapping of Ollama's
`thinking` field onto `reasoning_content` (`:352-355`) reaches datafast.

The ceiling is raised because thinking tokens count against `num_predict`, and a
0.6B model is verbose in proportion to its size.
"""

import pytest

pytestmark = [pytest.mark.live, pytest.mark.ollama]

REASONING_TOKENS = 2048
PROMPT = "A shop sells pens at 3 for $2. What do 12 pens cost?"


def test_thinking_true_returns_a_reasoning_trace(served_model):
    """`thinking=True` resolves to this profile's "on" level, 'low', which LiteLLM
    collapses to `think=true` for a non-gpt-oss model."""
    response = served_model(
        thinking=True,
        max_completion_tokens=REASONING_TOKENS,
    ).generate_response(prompt=PROMPT)

    assert response.reasoning_content
    assert "8" in response.text


def test_thinking_false_leaves_no_trace(served_model):
    """qwen3 thinks unless told not to, so an empty trace here is the whole point:
    it proves `think=false` arrived rather than that the model chose not to think."""
    response = served_model(
        thinking=False,
        max_completion_tokens=REASONING_TOKENS,
    ).generate_response(prompt=PROMPT)

    assert "8" in response.text
    assert not response.reasoning_content
