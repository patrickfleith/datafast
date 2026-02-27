"""LLMStep with multiple prompts: each record is processed by every prompt.

Demonstrates: prompt=[...] list — exhaustive prompt expansion.
2 records × 2 prompts = 4 output records. Output includes _prompt_index.
"""

from datafast_v2 import Source, LLMStep, Sink, openrouter
import litellm
litellm.suppress_debug_info = True

# model = ollama("gemma3:4b")
model = openrouter("mistralai/ministral-14b-2512", temperature=0.7)

prompt_explain = "Explain {concept} in simple terms for a beginner."
prompt_quiz = "Create a multiple-choice quiz question about {concept} with 4 options."

pipeline = (
    Source.list([
        {"concept": "photosynthesis"},
        {"concept": "machine learning"},
    ])
    >> LLMStep(
        prompt=[prompt_explain, prompt_quiz],
        input_columns=["concept"],
        output_column="response",
        model=model,
    )
    >> Sink.jsonl("v2_examples/outputs/18_llm_multi_prompt.jsonl")
)

records = pipeline.run()
