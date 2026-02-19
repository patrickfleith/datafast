"""Load a previously saved seed dataset, enrich with new dimensions, and generate with LLM.

This example demonstrates the full flow:
  1. Load records from a JSONL file (produced by the seed showcase script)
  2. Enrich records with new dimensions using FlatMap (cross-join)
  3. Filter to a subset
  4. Sample for manageable LLM costs
  5. Generate text with LLMStep
  6. Save the final dataset

Prerequisites:
  - Run seed_dataset_showcase.py first to produce the input file
  - Set OPENROUTER_API_KEY environment variable (or swap for ollama)
"""

from datafast_v2.sources.source import Source
from datafast_v2.transforms.data_ops import FlatMap, Filter
from datafast_v2.transforms.sample import Sample
from datafast_v2.transforms.llm_step import LLMStep
from datafast_v2.sinks.sink import Sink
from datafast_v2.llm.provider import ollama

# ---- Models (pick one or use multiple) ----
# model = openrouter("openai/gpt-4o-mini", temperature=0.8)
model = ollama("gemma3:4b", temperature=0.8)

# ---------------------------------------------------------------------------
# Pipeline: file → enrich → filter → sample → LLM → save
# ---------------------------------------------------------------------------
pipeline = (
    # 1. Load the seed dataset we created earlier
    Source.jsonl("v2_examples/outputs/showcase_product_sampled.jsonl")

    # 2. Cross-join with new dimensions using FlatMap
    #    Each input record gets duplicated for each new "style" value
    >> FlatMap(lambda r: [
        {**r, "style": style, "audience": audience}
        for style in ["formal", "conversational"]
        for audience in ["beginner", "expert"]
    ]).as_step("enrich_with_style_and_audience")

    # 3. Filter to only keep records we care about
    >> Filter(where={
        "difficulty_score": {"$gte": 3},
        "language": "en",
    }).as_step("filter_english_hard")

    # 4. Sample a manageable number for LLM generation
    >> Sample(n=4, strategy="uniform", seed=42).as_step("sample_for_llm")

    # 5. Generate a question using LLM
    >> LLMStep(
        prompt=(
            "You are an expert {persona} creating educational content.\n\n"
            "Write a {style} question about {subtopic} (part of {topic}) "
            "for a {audience} audience.\n\n"
            "The question should be at difficulty level {difficulty_score}/5."
        ),
        input_columns=["persona", "topic", "subtopic", "difficulty_score", "style", "audience"],
        output_column="question",
        model=model,
        parse_mode="text",
    ).as_step("generate_question")

    # 6. Generate an answer for the question
    >> LLMStep(
        prompt=(
            "Answer the following question in a {style} tone, "
            "appropriate for a {audience} audience.\n\n"
            "Question: {question}\n\n"
            "Provide a clear, accurate answer."
        ),
        input_columns=["question", "style", "audience"],
        output_column="answer",
        model=model,
        parse_mode="text",
        exclude_columns=["_model", "_language"],
    ).as_step("generate_answer")

    # 7. Save final dataset
    >> Sink.jsonl("v2_examples/outputs/qa_from_seeds.jsonl")
)

records = pipeline.run(
    checkpoint_dir="v2_examples/checkpoints/qa_from_seeds",
    batch_size=2,
)
