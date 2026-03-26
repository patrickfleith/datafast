"""Execution controls: limit, stop_after, and llm_strategy.

Demonstrates:
  - limit: process only first N source records (for testing)
  - stop_after: stop pipeline after a named step
  - llm_strategy: control LLM call ordering (by_model, round_robin, by_record)
"""

from datafast import Seed, LLMStep, Score, Filter, Sink, openrouter

import litellm
litellm.suppress_debug_info = True

# model = ollama("gemma3:4b")
model = openrouter("mistralai/ministral-14b-2512", temperature=0.7)

pipeline = (
    Seed.product(
        Seed.values("topic", ["gravity", "photosynthesis", "DNA", "black holes", "evolution"]),
        Seed.values("style", ["formal", "casual"]),
    )
    >> LLMStep(
        prompt="Write a {style} one-sentence fact about {topic}.",
        input_columns=["topic", "style"],
        output_column="fact",
        model=model,
    ).as_step("generate")
    >> Score(
        input_columns=["fact"],
        output_column="quality",
        score_range=(1, 5),
        llm=model,
        criteria="accuracy and clarity",
    ).as_step("score")
    >> Filter(where={"quality": {"$gte": 3}}).as_step("filter")
    >> Sink.jsonl("examples/outputs/37_execution.jsonl")
)

# Test with only 3 records (fast iteration)
records_limited = pipeline.run(limit=3)

# Stop after the "generate" step (skip scoring and filtering)
# records_stopped = pipeline.run(limit=3, stop_after="generate")

# Use round_robin strategy for multi-model pipelines
# records_rr = pipeline.run(limit=3, llm_strategy="round_robin", batch_size=2)
