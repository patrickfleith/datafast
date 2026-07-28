"""OpenRouter example using explicit chat messages."""

from dotenv import load_dotenv

from datafast import openrouter
from datafast.llm_utils import format_generated_responses


MODEL_ID = "openai/gpt-5.4-mini"
MESSAGES = [
    {
        "role": "system",
        "content": "You are a concise technical assistant. Answer in exactly two bullets.",
    },
    {
        "role": "user",
        "content": "Explain why teams use an LLM router.",
    },
]


def main() -> None:
    load_dotenv()

    model = openrouter(MODEL_ID, temperature=0)
    response = model.generate(messages=MESSAGES)
    print(format_generated_responses(MESSAGES[-1]["content"], response))


if __name__ == "__main__":
    main()
