# Datafast

Datafast is a composable pipeline library for synthetic data generation.

The old dataset-class API has been removed. The supported path is to build a pipeline from steps and run it with the built-in runner.

```python
from datafast import LLMStep, Seed, Sink, openrouter

pipeline = (
    Seed.product(
        Seed.values("topic", ["robotics", "materials"]),
        Seed.values("style", ["brief", "technical"]),
    )
    >> LLMStep(
        prompt=(
            "Write one {style} question about {topic}. "
            "Return JSON with fields question and answer."
        ),
        input_columns=["topic", "style"],
        output_columns=["question", "answer"],
        parse_mode="json",
        model=openrouter("z-ai/glm-4.6"),
    )
    >> Sink.jsonl("examples/outputs/home_example.jsonl")
)

pipeline.run(batch_size=4)
```

## What Changed

- `datafast` is now step-based and pipeline-first
- legacy dataset/config/prompt modules are gone
- examples and docs are centered on pipelines, not preset dataset classes

## Start Here

- Read [Concepts](concepts.md) for the execution model
- Read [Building Pipelines](guides/building_pipelines.md) for sources, transforms, and sinks
- Read [LLM Steps](guides/llm_steps.md) for generation and evaluation steps
- Read [Checkpointing](guides/checkpointing.md) for resume and execution controls
- Read [Langfuse Tracing](guides/langfuse_tracing.md) for optional observability setup
