"""Live checks for behaviour specific to individual Claude 5.5 models.

Everything general runs on claude-haiku-5-5 in the rest of this suite.
"""

import pytest
from pydantic import BaseModel

pytestmark = [pytest.mark.live, pytest.mark.anthropic]

PROMPT = "What is the capital of France? Answer in one word."


class Capital(BaseModel):
    city: str


@pytest.mark.parametrize("model_id", ["claude-opus-5-5", "claude-sonnet-5-5"])
def test_structured_output_works_on_models_that_refuse_forced_tool_use(
    served_model, model_id
):
    """Haiku 5.5 accepts forced tool use, so passing there proves nothing here."""
    result = served_model(
        model_id=model_id,
        thinking=None,
        reasoning_effort="low",
        max_completion_tokens=4000,
    ).generate(prompt=PROMPT, response_format=Capital)

    assert isinstance(result, Capital)


def test_sonnet_5_5_accepts_its_own_off_switch(served_model):
    """Sonnet 5.5 is the only model that needs thinking={'type': 'between_tools'}."""
    response = served_model(model_id="claude-sonnet-5-5").generate_response(prompt=PROMPT)

    assert response.text
    assert not response.thinking_blocks
