"""LLMStep with num_outputs: generate multiple variations per record.

Demonstrates: num_outputs=3 — each input record produces 3 output records.
2 records × 3 outputs = 6 total records.
"""

from datafast_v2 import Source, LLMStep, Sink, openrouter
import litellm
litellm.suppress_debug_info = True

# model = ollama("gemma3:4b")
model = openrouter("mistralai/ministral-14b-2512", temperature=0.9)

pipeline = (
    Source.list([
        {"topic": "space exploration"},
        {"topic": "deep sea creatures"},
    ])
    >> LLMStep(
        prompt="Write a creative one-sentence hook for an article about {topic}.",
        input_columns=["topic"],
        output_column="hook",
        model=model,
        num_outputs=3,
    )
    >> Sink.jsonl("v2_examples/outputs/21_llm_num_outputs.jsonl")
)

records = pipeline.run()
