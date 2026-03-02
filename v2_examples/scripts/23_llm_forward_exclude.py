"""LLMStep with forward_columns and exclude_columns to control output schema.

Demonstrates:
  - forward_columns: only keep specified input columns in output
  - exclude_columns: drop specified columns from output
"""

from datafast_v2 import Source, LLMStep, Sink, openrouter
import litellm
litellm.suppress_debug_info = True


# model = ollama("gemma3:4b")
model = openrouter("mistralai/ministral-14b-2512", temperature=0.7)

data = [
    {"id": 1, "topic": "gravity", "raw_notes": "F=ma, 9.8m/s^2, Newton", "lang": "en"},
    {"id": 2, "topic": "evolution", "raw_notes": "Darwin, natural selection", "lang": "en"},
]

# forward_columns: output only has id, topic, and the generated column
pipeline_fwd = (
    Source.list(data)
    >> LLMStep(
        prompt="Write a one-sentence fact about {topic}.",
        input_columns=["topic"],
        output_column="fact",
        model=model,
        forward_columns=["id", "topic"],
    )
    >> Sink.jsonl("v2_examples/outputs/23_forward.jsonl")
)
pipeline_fwd.run()

# exclude_columns: output has everything EXCEPT raw_notes and _model
pipeline_exc = (
    Source.list(data)
    >> LLMStep(
        prompt="Write a one-sentence fact about {topic}.",
        input_columns=["topic"],
        output_column="fact",
        model=model,
        exclude_columns=["raw_notes", "_model"],
    )
    >> Sink.jsonl("v2_examples/outputs/23_exclude.jsonl")
)
pipeline_exc.run()
