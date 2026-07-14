"""Gemini example with structured output validation."""

from dotenv import load_dotenv
from pydantic import BaseModel

from datafast import gemini


MODEL_ID = "gemini-3.1-flash-lite"
PROMPT = "Return a JSON object describing Google Gemini in two short sentences."


class ProviderSummary(BaseModel):
    name: str
    summary: str
    best_for: str


def main() -> None:
    load_dotenv()

    model = gemini(MODEL_ID, temperature=0)
    response = model.generate(prompt=PROMPT, response_format=ProviderSummary)
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
