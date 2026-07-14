"""Minimal Gemini example with a single prompt."""

from dotenv import load_dotenv

from datafast import gemini
from datafast.llm_utils import format_generated_responses


MODEL_ID = "gemini-3.1-flash-lite"
PROMPT = "Write one sentence explaining what Google Gemini is."


def main() -> None:
    load_dotenv()

    model = gemini(MODEL_ID, temperature=0)
    response = model.generate(prompt=PROMPT)
    print(format_generated_responses(PROMPT, response))


if __name__ == "__main__":
    main()
