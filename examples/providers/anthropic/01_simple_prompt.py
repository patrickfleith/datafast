"""Minimal Anthropic example with a single prompt."""

from dotenv import load_dotenv

from datafast import anthropic
from datafast.llm_utils import format_generated_responses


MODEL_ID = "claude-haiku-4-5"
PROMPT = "Write one sentence explaining what Anthropic is."


def main() -> None:
    load_dotenv()

    model = anthropic(MODEL_ID, temperature=0)
    response = model.generate(prompt=PROMPT)
    print(format_generated_responses(PROMPT, response))


if __name__ == "__main__":
    main()
