"""Ollama example returning normalized response metadata.

Reasoning is a first-class control for thinking-capable Ollama models
(gemma4, deepseek-r1, qwen3, gpt-oss, ...). Datafast forwards `thinking` and
`reasoning_effort` to Ollama's `think` parameter (LiteLLM handles the mapping)
and normalizes the thinking trace into `reasoning_content`. For gpt-oss the
effort level (low/medium/high) tunes the trace; other thinking models treat any
level as on/off. Models without a thinking mode have no reasoning control (see
the unsupported-params example).
"""

from dotenv import load_dotenv

from datafast import ollama


MODEL_ID = "gemma4:12b"
PROMPT = (
    "A train travels 60 miles per hour for 2.5 hours. "
    "Work it out carefully, then give the final answer in one short sentence."
)


def print_metadata(title: str, response) -> None:
    print(title)
    print("-" * len(title))
    print(f"text: {response.text.strip()}")
    print(f"reasoning_content: {bool(response.reasoning_content)}")
    if response.reasoning_content:
        print(f"reasoning_preview: {response.reasoning_content[:200]}")
    print(f"thinking_blocks: {len(response.thinking_blocks)}")
    print(f"images: {len(response.images)}")
    print(f"audio: {bool(response.audio)}")
    print(f"output_items: {len(response.output_items)}")
    print()


def main() -> None:
    load_dotenv()

    model = ollama(MODEL_ID, temperature=0, reasoning_effort="high")
    print_metadata("Reasoning enabled", model.generate_response(prompt=PROMPT))


if __name__ == "__main__":
    main()
