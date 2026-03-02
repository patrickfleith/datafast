"""Sample objects inside LLMStep params for prompt/model sampling.

Demonstrates:
  - Sample(...) as prompt param: randomly select N prompts per record
  - Sample(...).pick() as model param: select once upfront for all records
"""

from datafast_v2 import Source, LLMStep, Sample, Sink, openrouter

# model = ollama("gemma3:4b")
model_a = openrouter("mistralai/ministral-14b-2512", temperature=0.7)
model_b = openrouter("google/gemini-2.0-flash-001", temperature=0.7)

prompts = [
    "Explain {concept} for a 5-year-old.",
    "Explain {concept} using a sports analogy.",
    "Explain {concept} in exactly three bullet points.",
    "Explain {concept} as if you were a pirate.",
]

pipeline = (
    Source.list([
        {"concept": "gravity"},
        {"concept": "inflation"},
        {"concept": "DNA replication"},
    ])
    >> LLMStep(
        # Sample 2 random prompts per record (from 4 available)
        prompt=Sample(prompts, n=2, seed=42),
        input_columns=["concept"],
        output_column="explanation",
        # Pick one model upfront for all records
        model=Sample([model_a, model_b], n=1, seed=42).pick(),
    )
    >> Sink.jsonl("v2_examples/outputs/26_llm_sample_config.jsonl")
)

records = pipeline.run()
