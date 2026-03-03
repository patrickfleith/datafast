"""Classify step: LLM-based single-label classification.

Demonstrates: Classify with labels, include_explanation, include_confidence.
"""

from datafast_v2 import Source, Classify, Sink, openrouter
import litellm
litellm.suppress_debug_info = True

# model = ollama("gemma3:4b")
model = openrouter("mistralai/ministral-14b-2512", temperature=0.3)

pipeline = (
    Source.list([
        {"review": "Absolutely love this product! Best purchase I've ever made."},
        {"review": "Terrible quality. Broke after two days. Total waste of money."},
        {"review": "It's okay, nothing special. Does the job but nothing more."},
        {"review": "Exceeded my expectations! The design is beautiful and it works perfectly."},
    ])
    >> Classify(
        labels=["positive", "negative", "neutral"],
        input_columns=["review"],
        output_column="sentiment",
        llm=model,
        include_explanation=True,
        include_confidence=True,
    )
    >> Sink.jsonl("v2_examples/outputs/27_classify_llm.jsonl")
)

records = pipeline.run()
