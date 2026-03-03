"""Branch + JoinBranches: generate preference data (chosen vs rejected).

Demonstrates: Branch to run parallel LLMStep paths, JoinBranches to merge.
Output has response_chosen and response_rejected columns.
"""

from datafast_v2 import Source, LLMStep, Branch, JoinBranches, Sink, openrouter

import litellm
litellm.suppress_debug_info = True

# model = ollama("gemma3:4b")
model_strong = openrouter("mistralai/ministral-14b-2512", temperature=0.7)
model_weak = openrouter("mistralai/ministral-14b-2512", temperature=1.2)

pipeline = (
    Source.list([
        {"question": "What causes the seasons on Earth?"},
        {"question": "How does a vaccine work?"},
        {"question": "Why is the sky blue?"},
    ])
    >> Branch(
        chosen=LLMStep(
            prompt=(
                "You are an expert educator. Answer this question thoroughly "
                "and accurately with clear explanations.\n\n"
                "Question: {question}"
            ),
            input_columns=["question"],
            output_column="response",
            model=model_strong,
            system_prompt="You are a knowledgeable and precise science educator.",
        ),
        rejected=LLMStep(
            prompt="Answer briefly: {question}",
            input_columns=["question"],
            output_column="response",
            model=model_weak,
        ),
    )
    >> JoinBranches()
    >> Sink.jsonl("v2_examples/outputs/34_branch_preference.jsonl")
)

records = pipeline.run()
