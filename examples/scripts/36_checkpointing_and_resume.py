"""Pipeline checkpointing and resume after interruption.

Demonstrates: checkpoint_dir, resume=True, named steps with .as_step().
Run once normally, then uncomment the resume block to continue from checkpoint.
"""

from datafast import Seed, LLMStep, Sample, Sink, openrouter

import litellm
litellm.suppress_debug_info = True

# model = ollama("gemma3:4b")
model = openrouter("mistralai/ministral-14b-2512", temperature=0.7)

pipeline = (
    Seed.product(
        Seed.values("topic", ["gravity", "photosynthesis", "plate tectonics"]),
        Seed.values("level", ["beginner", "advanced"]),
    )
    >> Sample(n=4, strategy="uniform", seed=42).as_step("sample")
    >> LLMStep(
        prompt="Write a one-paragraph explanation of {topic} for a {level} audience.",
        input_columns=["topic", "level"],
        output_column="explanation",
        model=model,
    ).as_step("generate")
    >> Sink.jsonl("examples/outputs/36_checkpointed.jsonl")
)

# First run: creates checkpoints in the specified directory
records = pipeline.run(
    checkpoint_dir="examples/checkpoints/36_demo",
    batch_size=2,
)

# If the pipeline was interrupted, resume from where it left off:
records = pipeline.run(
    checkpoint_dir="examples/checkpoints/36_demo",
    resume=True,
    batch_size=2,
)
