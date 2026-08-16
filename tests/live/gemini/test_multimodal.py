"""Live multimodal input against Gemini.

`GEMINI_CHAT` declares IMAGE, AUDIO, VIDEO and FILE. Only IMAGE and FILE are
covered here, matching the other suites and the shared `tests/live/assets`
fixtures; AUDIO and VIDEO stay declared-but-unproven until assets exist for
them.
"""

import pytest

from datafast.llm import ContentPart

pytestmark = [pytest.mark.live, pytest.mark.gemini, pytest.mark.multimodal]


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


def test_document_input_reaches_the_model(served_model, document_asset):
    """Gemini takes a PDF inline as a data URI — no upload step, unlike
    Mistral, whose chat API accepts only a file id."""
    messages = [
        {
            "role": "user",
            "content": [
                ContentPart(type="file", data=document_asset),
                ContentPart(
                    type="text",
                    text="What is the passphrase in this document? Answer with the passphrase only.",
                ),
            ],
        }
    ]

    response = served_model().generate(messages=messages)

    assert "ZEBRAFISH-42" in response.upper()
