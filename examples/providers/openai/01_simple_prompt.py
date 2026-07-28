"""Minimal OpenAI example with a single prompt."""

from dotenv import load_dotenv

from datafast import openai
from datafast.llm_utils import format_generated_responses


MODEL_ID = "gpt-5.4-mini"
PROMPT = "Write one sentence explaining what OpenAI is."


def main() -> None:
    load_dotenv()

    model = openai(MODEL_ID)
    response = model.generate(prompt=PROMPT)
    print(format_generated_responses(PROMPT, response))


if __name__ == "__main__":
    main()
