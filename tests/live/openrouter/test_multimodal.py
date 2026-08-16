"""Live multimodal input against OpenRouter.

`OPENROUTER_CHAT` declares TEXT and IMAGE only, so there is no document test
here — unlike the gemini and mistral suites. The routed model accepts video
too, but datafast does not declare it for this provider, and a live test should
pin what datafast claims rather than what the model happens to allow.

Image support is another thing the routed endpoints disagree about — the
DeepInfra variants of this model answer an image content part with a 405 — so
this module leans on the same provider pin as the rest of the suite.
"""

import pytest

from datafast.llm import ContentPart

pytestmark = [pytest.mark.live, pytest.mark.openrouter, pytest.mark.multimodal]


def test_image_input_reaches_the_model(served_model, image_asset):
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

    response = served_model().generate(messages=messages)

    assert "red" in response.lower()
