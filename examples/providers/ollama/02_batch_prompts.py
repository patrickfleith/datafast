"""Ollama example with a batch of prompts.

Ollama has no native batch endpoint, so Datafast falls back to bounded parallel
single requests and emits a UserWarning. On a small machine keep concurrency low
(`max_concurrent=1`) so you don't run several 12B generations at once.
"""

from dotenv import load_dotenv

from datafast import ollama
from datafast.llm_utils import format_generated_responses


MODEL_ID = "gemma4:12b"
PROMPTS = [
    "Give a one-sentence definition of synthetic data.",
    "Give a one-sentence definition of retrieval-augmented generation.",
    "Give a one-sentence definition of tool calling.",
]


def main() -> None:
    load_dotenv()

    model = ollama(MODEL_ID, temperature=0, max_concurrent=1)
    responses = model.generate(prompt=PROMPTS)
    print(format_generated_responses(PROMPTS, responses))


if __name__ == "__main__":
    main()
