"""Ollama example comparing normalized fields with raw payload fields.

The raw payload from a local Ollama call carries timing/token counters that make
it easy to estimate tokens per second, which matters a lot when running a 12B
model on modest hardware.
"""

from __future__ import annotations

from dotenv import load_dotenv

from datafast import ollama


MODEL_ID = "gemma4:12b"
PROMPT = (
    "A train travels 60 miles per hour for 2.5 hours. "
    "Work it out carefully, then give the final answer in one short sentence."
)


def _get_attr_or_key(value, name: str):
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _first_choice_message(raw_response):
    choices = _get_attr_or_key(raw_response, "choices") or []
    if not choices:
        return None
    return _get_attr_or_key(choices[0], "message")


def main() -> None:
    load_dotenv()

    model = ollama(MODEL_ID, temperature=0)
    response = model.generate_response(prompt=PROMPT)

    usage = getattr(response.raw, "usage", None)
    message = _first_choice_message(response.raw)
    choices = getattr(response.raw, "choices", None)
    raw_text = _get_attr_or_key(message, "content")

    print("Comparison")
    print("----------")
    print(f"normalized.text: {response.text.strip()!r}")
    print(f"raw choices[0].message.content: {raw_text!r}")
    print(f"reasoning_content: {bool(response.reasoning_content)}")
    print(f"thinking_blocks: {len(response.thinking_blocks)}")
    print()
    print("Raw")
    print("---")
    print(f"raw_type: {type(response.raw).__name__}")
    print(f"has_usage: {usage is not None}")
    print(f"has_choices: {choices is not None}")
    if usage is not None:
        print(f"prompt_tokens: {getattr(usage, 'prompt_tokens', None)}")
        print(f"completion_tokens: {getattr(usage, 'completion_tokens', None)}")

    # Ollama-specific timings surface on the raw response or its hidden params
    # (in nanoseconds). Print them when the installed LiteLLM version exposes them.
    hidden = getattr(response.raw, "_hidden_params", None) or {}
    for field in ("eval_count", "eval_duration", "total_duration"):
        value = _get_attr_or_key(response.raw, field) or _get_attr_or_key(hidden, field)
        if value is not None:
            print(f"{field}: {value}")


if __name__ == "__main__":
    main()
