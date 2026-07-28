"""Minimal Ollama example with a single prompt.

Ollama runs locally and needs no API key. Make sure the Ollama server is
running and the model is pulled: `ollama pull gemma4:12b`.
"""

from dotenv import load_dotenv

from datafast import ollama
from datafast.llm_utils import format_generated_responses


MODEL_ID = "gemma4:12b"
PROMPT = "Write one sentence explaining what a local LLM is."


def main() -> None:
    load_dotenv()

    model = ollama(MODEL_ID, temperature=0)
    response = model.generate(prompt=PROMPT)
    print(format_generated_responses(PROMPT, response))


if __name__ == "__main__":
    main()
