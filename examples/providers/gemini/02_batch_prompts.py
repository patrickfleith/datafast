"""Minimal Gemini example with a batch of prompts."""

from dotenv import load_dotenv

from datafast import gemini
from datafast.llm_utils import format_generated_responses


MODEL_ID = "gemini-3.1-flash-lite"
PROMPTS = [
    "Give a one-sentence definition of synthetic data.",
    "Give a one-sentence definition of retrieval-augmented generation.",
    "Give a one-sentence definition of tool calling.",
]


def main() -> None:
    load_dotenv()

    model = gemini(MODEL_ID, temperature=0)
    responses = model.generate(prompt=PROMPTS)
    print(format_generated_responses(PROMPTS, responses))


if __name__ == "__main__":
    main()
