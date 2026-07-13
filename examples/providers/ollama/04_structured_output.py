"""Ollama example with structured output validation.

Ollama supports schema-constrained decoding, so Datafast passes the Pydantic
schema through as the response format and validates the result. Constrained
decoding is what makes even a local model reliably return valid JSON.
"""

from dotenv import load_dotenv
from pydantic import BaseModel

from datafast import ollama


MODEL_ID = "gemma4:12b"
PROMPT = "Return a JSON object describing the Ollama runtime in two short sentences."


class ProviderSummary(BaseModel):
    name: str
    summary: str
    best_for: str


def main() -> None:
    load_dotenv()

    model = ollama(MODEL_ID, temperature=0)
    response = model.generate(prompt=PROMPT, response_format=ProviderSummary)
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
