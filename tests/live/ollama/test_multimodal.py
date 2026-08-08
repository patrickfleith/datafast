"""Live image input against Ollama.

Ollama takes images as bare base64 in an `images` array beside the text, not as a
data URI inside the content list. LiteLLM does that conversion — it pulls the
image parts out of the message and strips the `data:image/png;base64,` prefix
(`litellm_core_utils/prompt_templates/common_utils.py:1530`). So datafast's
`ContentPart(type="image")` is correct here for a reason that has nothing to do
with the OpenAI shape it produces, and only a real call shows the conversion
survives end to end.

This runs on gemma4:12b rather than the suite's default model: both Ollama
profiles declare `Modality.IMAGE`, but qwen3:0.6b has no vision at all.
"""

import pytest

from datafast.llm import ContentPart

pytestmark = [pytest.mark.live, pytest.mark.ollama, pytest.mark.multimodal]


def test_image_input_reaches_the_model(vision_served_model, image_asset):
    messages = [
        {
            "role": "user",
            "content": [
                ContentPart(
                    type="text",
                    text="What colour is this square? Answer with one word.",
                ),
                ContentPart(type="image", data=image_asset),
            ],
        }
    ]

    response = vision_served_model().generate(messages=messages)

    assert "red" in response.lower()
