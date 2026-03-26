"""Capstone: preference dataset with branching, scoring, and filtering.

Demonstrates: Seeds → LLMStep → Branch → JoinBranches → Score × 2 → Filter → Sink.
Combines many features into a realistic preference data generation workflow.
"""

from datafast import (
    Seed, LLMStep, Branch, JoinBranches,
    Score, Filter, Sink, openrouter,
)

import litellm
litellm.suppress_debug_info = True

# model = ollama("gemma3:4b")
model = openrouter("mistralai/ministral-14b-2512", temperature=0.7)

pipeline = (
    Seed.values("topic", ["gravity", "photosynthesis", "machine learning"])

    # Generate a question per topic
    >> LLMStep(
        prompt="Generate a thoughtful educational question about {topic}.",
        input_columns=["topic"],
        output_column="question",
        model=model,
    ).as_step("generate_question")

    # Branch: generate a good response (chosen) and a weak response (rejected)
    >> Branch(
        chosen=LLMStep(
            prompt=(
                "You are an expert. Answer this question thoroughly and accurately.\n\n"
                "Question: {question}"
            ),
            input_columns=["question"],
            output_column="response",
            model=model,
            system_prompt="You are a world-class educator. Be precise, detailed, and helpful.",
        ),
        rejected=LLMStep(
            prompt="Answer briefly and vaguely: {question}",
            input_columns=["question"],
            output_column="response",
            model=model,
        ),
    ).as_step("branch_responses")
    >> JoinBranches()

    # Score the chosen response
    >> Score(
        input_columns=["question", "response_chosen"],
        output_column="score_chosen",
        score_range=(1, 10),
        llm=model,
        criteria="helpfulness, accuracy, and completeness",
    ).as_step("score_chosen")

    # Score the rejected response
    >> Score(
        input_columns=["question", "response_rejected"],
        output_column="score_rejected",
        score_range=(1, 10),
        llm=model,
        criteria="helpfulness, accuracy, and completeness",
    ).as_step("score_rejected")

    # Keep only records with a clear preference margin
    >> Filter(fn=lambda r: r.get("score_chosen", 0) - r.get("score_rejected", 0) >= 2).as_step("filter_margin")

    >> Sink.jsonl("examples/outputs/42_preference_dataset.jsonl")
)

records = pipeline.run(
    checkpoint_dir="examples/checkpoints/42_preference",
    batch_size=2,
)
