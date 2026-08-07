"""Live multimodal input against Anthropic.

`ANTHROPIC_CHAT` declares `Modality.IMAGE` and `Modality.FILE`. Mocked tests
only prove datafast builds a content part; the shape it builds — a base64
`data:` URI for images, `file.file_data` for documents — is something only the
real API can accept or reject. The assets next to this module are generated
fixtures with a deterministic answer: a solid red square, and a PDF holding one
passphrase.
"""

import base64
from pathlib import Path

import pytest

from datafast.llm import ContentPart

pytestmark = [pytest.mark.live, pytest.mark.anthropic, pytest.mark.multimodal]

IMAGE_PATH = Path(__file__).parent / "sample_square.png"
DOCUMENT_PATH = Path(__file__).parent / "sample_passphrase.pdf"


def _as_base64(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("ascii")


def test_image_input_reaches_the_model(served_model):
    messages = [
        {
            "role": "user",
            "content": [
                ContentPart(
                    type="text",
                    text="What colour is this square? Answer with one word.",
                ),
                ContentPart(
                    type="image",
                    data=_as_base64(IMAGE_PATH),
                    media_type="image/png",
                ),
            ],
        }
    ]

    response = served_model().generate(messages=messages)

    assert "red" in response.lower()


def test_document_input_reaches_the_model(served_model):
    """The declared `Modality.FILE` must survive the round trip, not just pass
    client-side validation."""
    messages = [
        {
            "role": "user",
            "content": [
                ContentPart(
                    type="file",
                    data=f"data:application/pdf;base64,{_as_base64(DOCUMENT_PATH)}",
                    media_type="application/pdf",
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
