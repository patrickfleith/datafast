# Datafast

Datafast is a Python library for generating synthetic datasets with
LLMs.

You describe what your dataset should cover, compose synthetic data generation steps,
and run the pipeline. What you get back is a dataset carrying full traceability from
seeds to metadata and models' outputs.

```python
from datafast import LLMStep, Seed, Sink, openai

pipeline = (
    Seed.product(
        Seed.values("topic", ["photosynthesis", "plate tectonics", "vaccines"]),
        Seed.values("level", ["beginner", "advanced"]),
    )
    >> LLMStep(
        prompt="Write one {level} exam question about {topic}, with its answer. "
               "Return JSON with fields question and answer.",
        input_columns=["topic", "level"],
        output_columns=["question", "answer"],
        parse_mode="json",
        model=openai(),
    )
    >> Sink.jsonl("questions.jsonl")
)

pipeline.run()
```

Three topics and two levels produce six rows — the seed expands the combinations, the
LLM step fills each one in, and the sink writes the result.

## Why pipelines

A synthetic dataset is rarely one prompt. It is a set of axes you want covered, a
generation step, usually a filter or a scoring pass, and somewhere to put the output.
Datafast makes each of those a step, composed with `>>`:

- **Coverage is declarative.** `Seed.product` expands the combinations instead of you
  writing nested loops.
- **Runs are resumable.** LLM calls are checkpointed per call, so an interrupted run
  resumes instead of being paid for twice.
- **Providers are interchangeable.** One configuration surface covers OpenAI,
  Anthropic, Gemini, Mistral, OpenRouter, Ollama and any OpenAI-compatible server.
- **Mistakes surface before the spend.** `Pipeline.compile()` validates structure and
  column references before a single call is made.

## Start here

- Follow the [Quickstart](quickstart.md) — install to first dataset, one page.
- Read [Concepts](concepts.md) for the execution model.
- Read [Sources & seed](reference/sources_and_seed.md) and [Data
  ops](reference/data_ops.md) for sources, transforms and sinks.
- Read [LLM step](reference/llm_step.md) for generation and evaluation steps.
- Read [Pipelines & execution](guides/pipelines_and_execution.md) for resume and
  execution controls.
- Read [Served models](llms.md) for provider configuration, reasoning and multimodal
  input.
- Browse the [Cookbook](cookbook/index.md) for complete recipes.
- Look up anything in the [API reference](api.md).
