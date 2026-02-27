"""LLMStep with multiple models: each record processed by every model.

Demonstrates: model=[...] list — output includes _model field.
2 records × 2 models = 4 output records.
"""

from datafast_v2 import Source, LLMStep, Sink, openrouter
import litellm
litellm.suppress_debug_info = True

# model_a = ollama("gemma3:4b")
# model_b = ollama("llama3:8b")
model_a = openrouter("mistralai/ministral-14b-2512", temperature=0.7)
model_b = openrouter("mistralai/ministral-8b-2512", temperature=0.7)

pipeline = (
    Source.list([
        {"topic": "renewable energy"},
        {"topic": "quantum computing"},
    ])
    >> LLMStep(
        prompt="Write a brief fact about {topic}.",
        input_columns=["topic"],
        output_column="fact",
        model=[model_a, model_b],
    )
    >> Sink.jsonl("v2_examples/outputs/19_llm_multi_model.jsonl")
)

records = pipeline.run()
