"""Live multimodal input against Mistral.

Images arrive inline, documents do not. Mistral's chat API accepts a document only
as an uploaded `file_id` — LiteLLM says so outright in
`llms/mistral/chat/transformation.py:296` and flattens `file.file_id` to a top-level
`file_id`. Inline `file_data` gets a 422, which is why the profile sets
`files_require_file_id` and datafast refuses it client-side.

The document test therefore drives the whole supported path: `upload_file` posts to
the Files API, the returned id goes in a file part's url, and `delete_file` cleans up.
Nothing else in the suite touches that API.
"""

import pytest

from datafast.llm import ContentPart

pytestmark = [pytest.mark.live, pytest.mark.mistral, pytest.mark.multimodal]


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


def test_uploaded_document_reaches_the_model(served_model, document_path):
    """Upload, ask, delete. The upload is the only way a document reaches Mistral,
    so this is the single test covering `upload_file` / `delete_file` against the
    real Files API."""
    model = served_model()
    # `expiry` backs up the delete below: if this process dies mid-test, Mistral
    # drops the file on its own rather than leaving it in the account for good.
    file_id = model.upload_file(document_path, expiry=1)

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    ContentPart(type="file", url=file_id),
                    ContentPart(
                        type="text",
                        text="What is the passphrase in this document? Answer with the passphrase only.",
                    ),
                ],
            }
        ]

        response = model.generate(messages=messages)
    finally:
        model.delete_file(file_id)

    assert "ZEBRAFISH-42" in response.upper()
