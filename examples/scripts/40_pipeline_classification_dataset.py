"""Classification dataset pipeline: Seeds → Generate examples → Verify labels.

Demonstrates: Generating labeled text data and verifying with a Classify step.
"""

from datafast import Seed, LLMStep, FlatMap, Classify, Filter, Sink, openrouter

import litellm
litellm.suppress_debug_info = True

# model = ollama("gemma3:4b")
model = openrouter("mistralai/ministral-14b-2512", temperature=0.8)

labels = ["positive", "negative", "neutral"]

pipeline = (
    Seed.product(
        Seed.values("sentiment", labels),
        Seed.values("domain", ["restaurant reviews", "product reviews"]),
    )

    # Generate 2 example texts per label+domain
    >> LLMStep(
        prompt=(
            "Generate 2 diverse, realistic {domain} that clearly express "
            "{sentiment} sentiment. Return a JSON object with a 'texts' field "
            "containing a list of 2 strings."
        ),
        input_columns=["sentiment", "domain"],
        output_columns=["texts"],
        model=model,
        parse_mode="json",
    ).as_step("generate")

    # Explode: one record per text
    >> FlatMap(lambda r: [
        {"text": text, "intended_label": r["sentiment"], "domain": r["domain"]}
        for text in (r["texts"] if isinstance(r["texts"], list) else [r["texts"]])
    ]).as_step("explode")

    # Verify: does the LLM agree with the intended label?
    >> Classify(
        labels=labels,
        input_columns=["text"],
        output_column="verified_label",
        llm=model,
    ).as_step("verify")

    # Keep only records where generation matches verification
    >> Filter(fn=lambda r: r.get("verified_label") == r.get("intended_label")).as_step("filter_consistent")

    >> Sink.jsonl("examples/outputs/40_classification_dataset.jsonl")
)

records = pipeline.run(batch_size=3)
