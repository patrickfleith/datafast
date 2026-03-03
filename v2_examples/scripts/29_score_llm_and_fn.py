"""Score step: LLM-based scoring with rubric and function-based scoring.

Demonstrates:
  Part A — LLM scoring with criteria, rubric, and explanation
  Part B — Function-based scoring (no LLM)
"""

from datafast_v2 import Source, Score, Sink, openrouter
import litellm
litellm.suppress_debug_info = True

# --- Part A: LLM-based scoring with rubric ---
# model = ollama("gemma3:4b")
model = openrouter("mistralai/ministral-14b-2512", temperature=0.3)

pipeline_llm = (
    Source.list([
        {"question": "What is gravity?", "answer": "Gravity is the force that pulls things down."},
        {"question": "What is gravity?", "answer": "Gravity is the attractive force between masses, proportional to their mass and inversely proportional to the square of distance."},
        {"question": "What is gravity?", "answer": "idk lol"},
    ])
    >> Score(
        input_columns=["question", "answer"],
        output_column="quality",
        score_range=(1, 5),
        llm=model,
        criteria="accuracy, completeness, and clarity of the answer",
        rubric={
            1: "Completely wrong or irrelevant",
            2: "Partially correct but major gaps",
            3: "Mostly correct, some missing details",
            4: "Correct and well-explained",
            5: "Excellent: precise, complete, and clear",
        },
        include_explanation=True,
    )
    >> Sink.jsonl("v2_examples/outputs/29_score_llm.jsonl")
)
pipeline_llm.run()

# --- Part B: Function-based scoring (no LLM) ---
pipeline_fn = (
    Source.list([
        {"text": "Short."},
        {"text": "A medium-length text with some detail."},
        {"text": "A very detailed explanation covering multiple aspects in depth with examples."},
    ])
    >> Score(
        input_columns=["text"],
        output_column="length_score",
        score_range=(0, 100),
        fn=lambda r: min(len(r["text"]), 100),
    )
    >> Sink.list()
)
pipeline_fn.run()
