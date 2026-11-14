"""Minimal Mistral example with a single prompt."""

from dotenv import load_dotenv

from datafast import mistral
from datafast.llm_utils import format_generated_responses


MODEL_ID = "mistral-small-2603"
PROMPT = "Write one sentence explaining what Mistral AI is."


def main() -> None:
    load_dotenv()

    model = mistral(MODEL_ID, temperature=0)
    response = model.generate(prompt=PROMPT)
    print(format_generated_responses(PROMPT, response))


if __name__ == "__main__":
    main()
