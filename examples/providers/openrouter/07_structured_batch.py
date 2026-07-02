"""OpenRouter example with batched structured responses."""

from dotenv import load_dotenv
from pydantic import BaseModel

from datafast import openrouter


# MODEL_ID = "openai/gpt-5.4-mini"
MODEL_ID = "google/gemma-4-26b-a4b-it:nitro"
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

    model = openrouter(MODEL_ID, temperature=0)
    responses = model.generate(prompt=PROMPTS, response_format=LanguageCard)

    for response in responses:
        print(response.model_dump_json(indent=2))
        print()


if __name__ == "__main__":
    main()
