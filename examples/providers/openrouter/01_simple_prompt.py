"""Minimal OpenRouter example with a single prompt."""

from dotenv import load_dotenv

from datafast import openrouter
from datafast.llm_utils import format_generated_responses


MODEL_ID = "openai/gpt-5.4-mini"
PROMPT = "Write one sentence explaining what OpenRouter is."


def main() -> None:
    load_dotenv()

    model = openrouter(MODEL_ID, temperature=0)
    response = model.generate(prompt=PROMPT)
    print(format_generated_responses(PROMPT, response))


if __name__ == "__main__":
    main()
