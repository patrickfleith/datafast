"""OpenAI example with structured output validation."""

from dotenv import load_dotenv
from pydantic import BaseModel

from datafast import openai


MODEL_ID = "gpt-5.4-mini"
PROMPT = "Return a JSON object describing OpenAI in two short sentences."


class ProviderSummary(BaseModel):
    name: str
    summary: str
    best_for: str


def main() -> None:
    load_dotenv()

    model = openai(MODEL_ID)
    response = model.generate(prompt=PROMPT, response_format=ProviderSummary)
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
