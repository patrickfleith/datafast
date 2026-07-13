"""OpenAI example using explicit chat messages."""

from dotenv import load_dotenv

from datafast import openai
from datafast.llm_utils import format_generated_responses


MODEL_ID = "gpt-5.4-mini"
MESSAGES = [
    {
        "role": "system",
        "content": "You are a concise technical assistant. Answer in exactly two bullets.",
    },
    {
        "role": "user",
        "content": "Explain why teams use OpenAI models for structured data generation.",
    },
]


def main() -> None:
    load_dotenv()

    model = openai(MODEL_ID)
    response = model.generate(messages=MESSAGES)
    print(format_generated_responses(MESSAGES[-1]["content"], response))


if __name__ == "__main__":
    main()
