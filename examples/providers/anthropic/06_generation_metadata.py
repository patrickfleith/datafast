"""Anthropic example returning normalized response metadata."""

from dotenv import load_dotenv

from datafast import anthropic


MODEL_ID = "claude-sonnet-4-6"
PROMPT = (
    "A train travels 60 miles per hour for 2.5 hours. "
    "Work it out carefully, then give the final answer in one short sentence."
)


def main() -> None:
    load_dotenv()

    # Extended thinking is enabled natively; temperature is left unset because
    # Anthropic requires the default temperature while thinking is on.
    model = anthropic(MODEL_ID, thinking=True)
    response = model.generate_response(prompt=PROMPT)
    usage = getattr(response.raw, "usage", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = (
        getattr(completion_details, "reasoning_tokens", None)
        if completion_details is not None
        else None
    )

    print("Text")
    print("----")
    print(response.text.strip())
    print()
    print("Metadata")
    print("--------")
    print(f"reasoning_content: {bool(response.reasoning_content)}")
    print(f"reasoning_tokens: {reasoning_tokens}")
    print(f"thinking_blocks: {len(response.thinking_blocks)}")
    print(f"images: {len(response.images)}")
    print(f"audio: {bool(response.audio)}")
    print(f"output_items: {len(response.output_items)}")
    if response.reasoning_content:
        print()
        print("Reasoning")
        print("---------")
        print(response.reasoning_content)


if __name__ == "__main__":
    main()
