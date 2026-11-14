"""Ollama example with batched structured responses.

Combines schema-constrained decoding with the bounded-concurrency batch fallback.
Kept at `max_concurrent=1` so a small machine runs the generations one at a time.
"""

from dotenv import load_dotenv
from pydantic import BaseModel

from datafast import ollama


MODEL_ID = "gemma4:12b"
PROMPTS = [
    "Return JSON for Python with fields language, category, and one_sentence_use_case.",
    "Return JSON for Rust with fields language, category, and one_sentence_use_case.",
    "Return JSON for SQL with fields language, category, and one_sentence_use_case.",
]


class LanguageCard(BaseModel):
    language: str
    category: str
    one_sentence_use_case: str


def main() -> None:
    load_dotenv()

    model = ollama(MODEL_ID, temperature=0, max_concurrent=1)
    responses = model.generate(prompt=PROMPTS, response_format=LanguageCard)

    for response in responses:
        print(response.model_dump_json(indent=2))
        print()


if __name__ == "__main__":
    main()
