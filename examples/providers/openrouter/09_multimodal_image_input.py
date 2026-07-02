"""OpenRouter example with text plus image input."""

from dotenv import load_dotenv

from datafast import openrouter
from datafast.llm import ContentPart


# Swap this for any OpenRouter model on your account that supports image input.
MODEL_ID = "openai/gpt-5-mini"
IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/4/40/Portrait_of_a_father.jpg"
)
MESSAGES = [
    {
        "role": "user",
        "content": [
            ContentPart(
                type="text",
                text="Describe this image in two short bullet points.",
            ),
            ContentPart(
                type="image",
                url=IMAGE_URL,
                media_id="boardwalk-demo-image",
            ),
        ],
    }
]


def main() -> None:
    load_dotenv()

    model = openrouter(MODEL_ID, temperature=0)
    response = model.generate(messages=MESSAGES)
    print(response.strip())


if __name__ == "__main__":
    main()
