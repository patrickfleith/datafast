"""End-to-end QA generation pipeline: Seeds → Question → Answer → Sink.

Demonstrates: Full pipeline combining seeds, LLMStep (JSON mode), and checkpointing.
"""

from datafast import Seed, LLMStep, Sample, Sink, openrouter

import litellm
litellm.suppress_debug_info = True

# model = ollama("gemma3:4b")
model = openrouter("mistralai/ministral-14b-2512", temperature=0.7)

pipeline = (
    Seed.product(
        Seed.expand("domain", "topic", {
            "Science": ["Genetics", "Quantum Physics", "Climate Change"],
            "History": ["Ancient Egypt", "World War II", "Industrial Revolution"],
        }),
        Seed.values("difficulty", ["easy", "hard"]),
    )
    >> Sample(n=6, strategy="uniform", seed=42).as_step("sample")

    # Generate question (JSON mode for structured output)
    >> LLMStep(
        prompt=(
            "Generate an educational {difficulty} question about {topic} "
            "(a subtopic of {domain}).\n\n"
            "Return a JSON object with a 'question' field."
        ),
        input_columns=["domain", "topic", "difficulty"],
        output_columns=["question"],
        model=model,
        parse_mode="json",
    ).as_step("generate_question")

    # Generate answer
    >> LLMStep(
        prompt=(
            "Answer the following question accurately and concisely.\n\n"
            "Question: {question}"
        ),
        input_columns=["question"],
        output_column="answer",
        model=model,
        exclude_columns=["_model"],
    ).as_step("generate_answer")

    >> Sink.jsonl("examples/outputs/38_qa_generation.jsonl")
)

records = pipeline.run(
    checkpoint_dir="examples/checkpoints/38_qa",
    batch_size=3,
)
