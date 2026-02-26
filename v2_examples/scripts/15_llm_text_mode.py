"""Basic LLMStep with text parse mode (default).

Demonstrates: LLMStep generating free-form text from seed topics.
"""

from datafast_v2 import Source, LLMStep, Sink, openrouter
import litellm
litellm.suppress_debug_info = True

# model = ollama("gemma3:4b")
model = openrouter("mistralai/ministral-14b-2512", temperature=0.7)

pipeline = (
    Source.list([
        {"topic": "black holes"},
        {"topic": "photosynthesis"},
        {"topic": "the Roman Empire"},
    ])
    >> LLMStep(
        prompt="Write a concise one-paragraph summary about {topic}.",
        input_columns=["topic"],
        output_column="summary",
        model=model,
        parse_mode="text",
    )
    >> Sink.jsonl("v2_examples/outputs/15_llm_text.jsonl")
)

records = pipeline.run()
