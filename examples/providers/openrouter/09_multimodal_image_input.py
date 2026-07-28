"""OpenRouter example with text plus image input.

The image ships with this example and is sent as base64 bytes via
``ContentPart(data=...)``, so the request does not depend on a routed provider
being able to fetch a remote URL.
"""

import base64
from pathlib import Path

from dotenv import load_dotenv

from datafast import openrouter
from datafast.llm import ContentPart


# Swap this for any OpenRouter model on your account that supports image input.
MODEL_ID = "openai/gpt-5-mini"
IMAGE_PATH = Path(__file__).parent / "sample_hot_air_balloon.jpg"


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

    model = openrouter(MODEL_ID, temperature=0)
    response = model.generate(messages=messages)
    print(response.strip())


if __name__ == "__main__":
    main()
