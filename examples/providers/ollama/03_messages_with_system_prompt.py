"""Ollama example using explicit chat messages."""

from dotenv import load_dotenv

from datafast import ollama
from datafast.llm_utils import format_generated_responses


MODEL_ID = "gemma4:12b"
MESSAGES = [
    {
        "role": "system",
        "content": "You are a concise technical assistant. Answer in exactly two bullets.",
    },
    {
        "role": "user",
        "content": "Explain why teams run local models with Ollama for data generation.",
    },
]


def main() -> None:
    load_dotenv()

    model = ollama(MODEL_ID, temperature=0)
    response = model.generate(messages=MESSAGES)
    print(format_generated_responses(MESSAGES[-1]["content"], response))


if __name__ == "__main__":
    main()
