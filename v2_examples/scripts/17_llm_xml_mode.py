"""LLMStep with XML parse mode for tag-based structured output.

Demonstrates: parse_mode="xml" with multiple output_columns.
The LLM wraps each field in XML tags; the parser extracts by tag name.
"""

from datafast_v2 import Source, LLMStep, Sink, openrouter
import litellm
litellm.suppress_debug_info = True

# model = ollama("gemma3:4b")
model = openrouter("mistralai/ministral-14b-2512", temperature=0.7)

pipeline = (
    Source.list([
        {"concept": "gravity"},
        {"concept": "evolution"},
        {"concept": "supply and demand"},
    ])
    >> LLMStep(
        prompt="Explain the concept of {concept}. Provide a definition and a real-world example.",
        input_columns=["concept"],
        output_columns=["definition", "example"],
        model=model,
        parse_mode="xml",
    )
    >> Sink.jsonl("v2_examples/outputs/17_llm_xml.jsonl")
)

records = pipeline.run()
