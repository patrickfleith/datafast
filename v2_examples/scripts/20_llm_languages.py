"""LLMStep with language expansion: generate in multiple languages.

Demonstrates: language={"code": "Name"} dict — each record × each language.
2 records × 3 languages = 6 output records. Output includes _language field.
"""

from datafast_v2 import Source, LLMStep, Sink, openrouter
import litellm
litellm.suppress_debug_info = True

# model = ollama("gemma3:4b")
model = openrouter("mistralai/ministral-14b-2512", temperature=0.7)

pipeline = (
    Source.list([
        {"topic": "climate change"},
        {"topic": "artificial intelligence"},
    ])
    >> LLMStep(
        prompt="Write a short paragraph about {topic} in {language_name}.",
        input_columns=["topic"],
        output_column="paragraph",
        model=model,
        language={"en": "English", "fr": "French", "de": "German"},
    )
    >> Sink.jsonl("v2_examples/outputs/20_llm_languages.jsonl")
)

records = pipeline.run()
