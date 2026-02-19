"""Test pipeline: Generate Q&A pairs from topics using OpenRouter."""

import sys
sys.path.insert(0, ".")
import litellm

litellm.suppress_debug_info = True

from datafast_v2 import (
    Source,
    Sample,
    LLMStep,
    Seed,
    Sink,
    openrouter,
)

# Define topics and question types
topics = [
    "machine learning",
    "climate change",
    "ancient history",
    "cooking techniques",
    "space exploration",
]

question_types = [
    "factual",
    "conceptual",
    "analytical",
]

# Create seed data: all combinations of topics × question types
seed_data = Seed.product(
    Seed.values("topic", topics),
    Seed.values("question_type", question_types),
)

# Create the model
model = openrouter("nvidia/nemotron-3-nano-30b-a3b")

# Build the pipeline
pipeline = (
    seed_data
    # Sample 5 random combinations
    >> Sample(n=10, seed=42)
    # Generate a question
    >> LLMStep(
        prompt=(
            "Generate a {question_type} question about {topic}. "
            "Return only the question, nothing else."
        ),
        input_columns=["topic", "question_type"],
        output_column="question",
        parse_mode="text",
        model=model,
    )
    .as_step("generate_question")
    # Generate an answer
    >> LLMStep(
        prompt=(
            "Answer this question concisely:\n\n{question}"
        ),
        input_columns=["question"],
        output_column="answer",
        parse_mode="text",
        model=model,
    )
    .as_step("generate_answer")
    # Save to JSONL
    >> Sink.jsonl("output/qa_dataset.jsonl")
)

if __name__ == "__main__":
    print("=" * 60)
    print("Test Pipeline: Q&A Generation")
    print("=" * 60)
    print(f"Topics: {topics}")
    print(f"Question types: {question_types}")
    print("Model: google/gemini-3-flash-preview")
    print()
    # Run with checkpointing
    results = pipeline.run(
        checkpoint_dir="./checkpoints/qa_test",
        batch_size=2,
        llm_strategy="by_model",
    )
