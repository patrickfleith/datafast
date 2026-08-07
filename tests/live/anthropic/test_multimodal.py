"""Live multimodal input against Anthropic.

`ANTHROPIC_CHAT` declares `Modality.IMAGE` and `Modality.FILE`. Mocked tests
only prove datafast builds a content part; the shape it builds — a base64
`data:` URI for images, `file.file_data` for documents — is something only the
real API can accept or reject. The `image_asset` / `document_asset` fixtures
come from `tests/live/assets` and have a deterministic answer, so every
provider's multimodal module can assert the same thing.
"""

import pytest

from datafast.llm import ContentPart

pytestmark = [pytest.mark.live, pytest.mark.anthropic, pytest.mark.multimodal]


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
    client-side validation."""
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
