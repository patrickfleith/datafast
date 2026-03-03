"""Rewrite step: paraphrase mode with multiple variations.

Demonstrates: Rewrite with mode="paraphrase" and num_variations.
Each input record produces num_variations output records.
"""

from datafast_v2 import Source, Rewrite, Sink, openrouter
import litellm
litellm.suppress_debug_info = True

# model = ollama("gemma3:4b")
model = openrouter("mistralai/ministral-14b-2512", temperature=0.8)

pipeline = (
    Source.list([
        {"text": "Machine learning algorithms learn patterns from data to make predictions."},
        {"text": "The water cycle involves evaporation, condensation, and precipitation."},
    ])
    >> Rewrite(
        input_column="text",
        llm=model,
        mode="paraphrase",
        num_variations=2,
    )
    >> Sink.jsonl("v2_examples/outputs/31_rewrite_paraphrase.jsonl")
)

records = pipeline.run()
