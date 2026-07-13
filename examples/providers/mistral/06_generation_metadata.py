"""Mistral example returning normalized response metadata.

Reasoning is opt-in on Mistral: mistral-medium-3-5 and mistral-small (and the
magistral family) support a `reasoning_effort` parameter. With
reasoning_effort="high" the API returns a thinking chunk (the reasoning trace)
before the final text, which Datafast normalizes into `reasoning_content`. This
example requests reasoning so those fields populate, showing the full metadata
surface that Datafast exposes uniformly across providers.
"""

from dotenv import load_dotenv

from datafast import mistral


MODEL_ID = "mistral-medium-3-5"
PROMPT = (
    "A train travels 60 miles per hour for 2.5 hours. "
    "Work it out carefully, then give the final answer in one short sentence."
)


def main() -> None:
    load_dotenv()

    # Reasoning mode samples, so avoid temperature=0 (Mistral rejects greedy
    # sampling while reasoning); a small positive temperature keeps it focused.
    model = mistral(MODEL_ID, temperature=0.3, reasoning_effort="high")
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
    reasoning = response.reasoning_content or ""
    print(f"reasoning_content: {bool(reasoning)} ({len(reasoning)} chars)")
    print(f"reasoning_tokens: {reasoning_tokens}")
    print(f"thinking_blocks: {len(response.thinking_blocks)}")
    print(f"images: {len(response.images)}")
    print(f"audio: {bool(response.audio)}")
    print(f"output_items: {len(response.output_items)}")


if __name__ == "__main__":
    main()
