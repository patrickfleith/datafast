"""Minimal OpenRouter example with a batch of prompts."""

from dotenv import load_dotenv

from datafast import openrouter
from datafast.llm_utils import format_generated_responses


MODEL_ID = "openai/gpt-5.4-mini"
PROMPTS = [
    "Give a one-sentence definition of synthetic data.",
    "Give a one-sentence definition of retrieval-augmented generation.",
    "Give a one-sentence definition of tool calling.",
]


def main() -> None:
    load_dotenv()

    model = openrouter(MODEL_ID, temperature=0)
    responses = model.generate(prompt=PROMPTS)
    print(format_generated_responses(PROMPTS, responses))


if __name__ == "__main__":
    main()
