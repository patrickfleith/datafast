"""LLMStep with JSON parse mode for structured output.

Demonstrates: parse_mode="json" with multiple output_columns.
The LLM is instructed to return JSON; the parser extracts named fields.
"""

from datafast_v2 import Source, LLMStep, Sink, openrouter
import litellm
litellm.suppress_debug_info = True

# model = ollama("gemma3:4b")
model = openrouter("mistralai/ministral-14b-2512", temperature=0.7)

pipeline = (
    Source.list([
        {"text": "The mitochondria is the powerhouse of the cell."},
        {"text": "Newton's third law states every action has an equal and opposite reaction."},
        {"text": "The French Revolution began in 1789."},
    ])
    >> LLMStep(
        prompt="Based on this text, generate a question and its answer.\n\nText: {text}",
        input_columns=["text"],
        output_columns=["question", "answer"],
        model=model,
        parse_mode="json",
    )
    >> Sink.jsonl("v2_examples/outputs/16_llm_json.jsonl")
)

records = pipeline.run()
