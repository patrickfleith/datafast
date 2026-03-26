"""Generate → Score → Filter pipeline: keep only high-quality outputs.

Demonstrates: Classic pattern of LLM generation, quality scoring, and filtering.
"""

from datafast import Source, LLMStep, Score, Filter, Sink, openrouter

import litellm
litellm.suppress_debug_info = True

# model = ollama("gemma3:4b")
model = openrouter("mistralai/ministral-14b-2512", temperature=0.8)

pipeline = (
    Source.list([
        {"topic": "renewable energy sources"},
        {"topic": "the human immune system"},
        {"topic": "artificial neural networks"},
        {"topic": "the history of writing"},
        {"topic": "ocean currents"},
    ])

    # Generate a fact
    >> LLMStep(
        prompt="Write an interesting and accurate one-paragraph fact about {topic}.",
        input_columns=["topic"],
        output_column="fact",
        model=model,
    ).as_step("generate")

    # Score quality
    >> Score(
        input_columns=["fact"],
        output_column="quality",
        score_range=(1, 10),
        llm=model,
        criteria="accuracy, informativeness, and clarity",
        include_explanation=True,
    ).as_step("score")

    # Keep only high-quality records
    >> Filter(where={"quality": {"$gte": 7}}).as_step("filter")

    >> Sink.jsonl("examples/outputs/39_generate_score_filter.jsonl")
)

records = pipeline.run(batch_size=2)
