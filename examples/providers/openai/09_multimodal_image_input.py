"""OpenAI example with text plus image input.

The image ships with this example and is sent as base64 bytes via
``ContentPart(data=...)``, so the request does not depend on OpenAI being able
to fetch a remote URL.
"""

import base64
from pathlib import Path

from dotenv import load_dotenv

from datafast import openai
from datafast.llm import ContentPart


# Swap this for any OpenAI model that supports image input.
MODEL_ID = "gpt-5.4-mini"
IMAGE_PATH = Path(__file__).parent / "sample_lion.jpg"


def main() -> None:
    load_dotenv()

    image_b64 = base64.standard_b64encode(IMAGE_PATH.read_bytes()).decode("ascii")
    messages = [
        {
            "role": "user",
            "content": [
                ContentPart(
                    type="text",
                    text="Describe this image in two short bullet points.",
                ),
                ContentPart(
                    type="image",
                    data=image_b64,
                    media_type="image/jpeg",
                ),
            ],
        }
    ]

    model = openai(MODEL_ID)
    response = model.generate(messages=messages)
    print(response.strip())


if __name__ == "__main__":
    main()
