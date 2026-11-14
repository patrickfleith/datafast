"""Live multimodal input against OpenAI (Responses endpoint).

`OPENAI_RESPONSES` declares `Modality.IMAGE` and `Modality.FILE`. The Responses API
rejects chat-style parts, so datafast rewrites them into `input_image` / `input_file`
— a conversion only the real API can accept or reject. The shared `image_asset` /
`document_asset` fixtures carry a deterministic answer, so this asserts the same
thing the Anthropic module does.
"""

import pytest

from datafast.llm import ContentPart

pytestmark = [pytest.mark.live, pytest.mark.openai, pytest.mark.multimodal]


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
    """The declared `Modality.FILE` must survive the round trip, not just pass
    client-side validation. OpenAI rejects inline file data without a filename."""
    messages = [
        {
            "role": "user",
            "content": [
                ContentPart(
                    type="file", data=document_asset, filename="passphrase.pdf"
                ),
                ContentPart(
                    type="text",
                    text="What is the passphrase in this document? Answer with the passphrase only.",
                ),
            ],
        }
    ]

    response = served_model().generate(messages=messages)

    assert "ZEBRAFISH-42" in response.upper()
