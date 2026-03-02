"""LLMStep with a system prompt to set persona/behaviour.

Demonstrates: system_prompt parameter for controlling LLM persona.
"""

from datafast_v2 import Source, LLMStep, Sink, openrouter
import litellm
litellm.suppress_debug_info = True


# model = ollama("gemma3:4b")
model = openrouter("mistralai/ministral-14b-2512", temperature=0.7)

pipeline = (
    Source.list([
        {"topic": "black holes"},
        {"topic": "the water cycle"},
        {"topic": "blockchain technology"},
    ])
    >> LLMStep(
        prompt="Explain {topic} to me.",
        input_columns=["topic"],
        output_column="explanation",
        model=model,
        system_prompt=(
            "You are a enthusiastic science educator who explains concepts "
            "using fun analogies and everyday language. Keep answers under 3 sentences."
        ),
    )
    >> Sink.jsonl("v2_examples/outputs/22_llm_system_prompt.jsonl")
)

records = pipeline.run()
