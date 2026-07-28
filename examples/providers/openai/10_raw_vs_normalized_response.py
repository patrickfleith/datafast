"""OpenAI example comparing normalized fields with raw payload fields.

gpt-5 models run on the Responses API, so the raw payload exposes ``output``
items and ``output_text`` rather than the chat-completions ``choices`` array.
"""

from __future__ import annotations

from dotenv import load_dotenv

from datafast import openai


MODEL_ID = "gpt-5.4"
PROMPT = (
    "A train travels 60 miles per hour for 2.5 hours. "
    "Work it out carefully, then give the final answer in one short sentence."
)


def main() -> None:
    load_dotenv()

    model = openai(
        MODEL_ID,
        provider_params={"reasoning": {"effort": "high", "summary": "auto"}},
    )
    response = model.generate_response(prompt=PROMPT)

    usage = getattr(response.raw, "usage", None)
    output = getattr(response.raw, "output", None)
    raw_text = getattr(response.raw, "output_text", None)

    print("Comparison")
    print("----------")
    print(f"normalized.text: {response.text.strip()!r}")
    print(f"raw output_text: {raw_text!r}")
    print(f"reasoning_content: {bool(response.reasoning_content)}")
    print(f"output_items: {len(response.output_items)}")
    if response.reasoning_content:
        print()
        print("Normalized reasoning")
        print("--------------------")
        print(response.reasoning_content)
    print()
    print("Raw")
    print("---")
    print(f"raw_type: {type(response.raw).__name__}")
    print(f"has_usage: {usage is not None}")
    print(f"has_output: {output is not None}")
    if usage is not None:
        print(f"input_tokens: {getattr(usage, 'input_tokens', None)}")
        print(f"output_tokens: {getattr(usage, 'output_tokens', None)}")


if __name__ == "__main__":
    main()
