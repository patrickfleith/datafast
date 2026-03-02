"""LLMStep with skip_if to conditionally skip records.

Demonstrates: skip_if=lambda to bypass LLM calls for records that don't need them.
Records that are skipped still appear in output but without the generated column.
"""

from datafast_v2 import Source, LLMStep, Sink, openrouter

# model = ollama("gemma3:4b")
model = openrouter("mistralai/ministral-14b-2512", temperature=0.7)

pipeline = (
    Source.list([
        {"id": 1, "text": "A thorough explanation of quantum entanglement and its implications"},
        {"id": 2, "text": "Hi"},
        {"id": 3, "text": "The theory of relativity reshaped our understanding of space and time"},
        {"id": 4, "text": "Ok"},
    ])
    >> LLMStep(
        prompt="Summarize this text in one sentence: {text}",
        input_columns=["text"],
        output_column="summary",
        model=model,
        skip_if=lambda r: len(r["text"]) < 20,
    )
    >> Sink.jsonl("v2_examples/outputs/25_llm_skip_if.jsonl")
)

records = pipeline.run()
