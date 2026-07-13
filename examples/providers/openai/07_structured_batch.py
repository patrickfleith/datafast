"""OpenAI example with batched structured responses."""

from dotenv import load_dotenv
from pydantic import BaseModel

from datafast import openai


MODEL_ID = "gpt-5.4-mini"
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

    model = openai(MODEL_ID)
    responses = model.generate(prompt=PROMPTS, response_format=LanguageCard)

    for response in responses:
        print(response.model_dump_json(indent=2))
        print()


if __name__ == "__main__":
    main()
