# What's in v1

datafast 1.0.0 is the first stable release. This page says what the library does now,
what shipped, what is deliberately not here, and where the rough edges are.

Everything before 1.0.0 was experimental and is unsupported. There is no migration
guide, because v1 is the starting point rather than a transition.

## What datafast does

You describe how a dataset should be built, as a chain of steps, and run it.

```python
from datafast import LLMStep, Sink, Source, openai

pipeline = (
    Source.list([{"topic": "fusion power"}])
    >> LLMStep(
        prompt="Write one short question about {topic}",
        input_columns=["topic"],
        output_column="question",
        model=openai(),
    )
    >> Sink.jsonl("questions.jsonl")
)

pipeline.run()
```

Records flow through the chain. Sources and seeds create them, transforms reshape them,
LLM steps generate or judge content, and sinks write them out. The runner handles
batching, checkpointing and resume.

Read [Concepts](concepts.md) for the model behind that, or the
[Quickstart](quickstart.md) to run one yourself.

## What shipped

**One pipeline model.** Steps chain with `>>`. Every step takes records and returns
records, so they compose in any order the validator accepts. `compile()` checks the
shape before a single call is made — the source comes first, nothing follows the sinks,
every `Branch` has a matching `JoinBranches`, and every column a step names exists.

**Seven ways to reach a model.** `openai`, `anthropic`, `gemini`, `mistral`,
`openrouter`, `ollama` and `openai_compatible`. Each one builds a *served model*: a
model, the server that serves it, and the settings for both. Rate limits, timeouts,
retries and sampling settings live there, never on the runner — so two steps sharing a
model share one limit. See [Served models](reference/served_models.md).

**Generation and evaluation steps.** `LLMStep` is the general one. `Classify`, `Score`,
`Compare`, `Rewrite` and `Extract` cover the common jobs without a hand-written prompt.
See [LLM step](reference/llm_step.md) and
[Specialized LLM steps](reference/llm_specialized.md).

**Branching.** `Branch` sends the same records down several paths and `JoinBranches`
merges them back. This is what makes preference data possible — two answers per record,
scored and compared. See [Branching](reference/branching.md).

**Checkpointing and resume.** Each step writes its output before the next one starts,
and LLM steps checkpoint per call. A run interrupted at call 501 of 1000 resumes there
instead of paying for the first 500 again. See
[Pipelines & execution](guides/pipelines_and_execution.md).

**Multimodal input.** Images, audio, video and files, gated by what each served model
declares it accepts. See [Multimodal input](guides/multimodal_input.md).

**Optional extras.** `datafast[parquet]` for Parquet files, `datafast[hub]` for
HuggingFace datasets, `datafast[all]` for both. A base install declares five
dependencies and resolves to 50 packages, down from 107. See
[Installation](installation.md).

**Type hints.** The package ships a `py.typed` marker, so type checkers read datafast's
annotations in your own code.

## What is deliberately not here

- **No migration path from pre-1.0.** The old dataset-class API is gone.
- **No async API.** Batching and concurrency are handled inside the runner and the
  served model. You call `run()` and it blocks.
- **No progress bar.** Progress is reported through the logger at `INFO`.
- **No nested branching.** A `Branch` cannot sit inside another branch path.
- **No provider-enforced JSON.** Steps ask the model for JSON in the prompt and parse
  the reply. Guaranteed schemas are reachable only by calling a served model directly.
  See [Structured output](guides/structured_output.md).

## Known rough edges

Real behaviour worth knowing before you spend money on a long run. Each one is
documented in more detail on the page that covers it.

### LLM steps

- When a call fails, that record is dropped and the run continues, so **count your
  output records** against what you expected. Set `on_parse_error="raise"` on the step
  to stop on the first failure instead.
- If a model returns only some of the columns you asked for, the missing ones arrive
  empty rather than raising. An empty column is not reported as an error.

### Resume and checkpoints

- Changing a prompt, a served model or the body of a transform does **not** invalidate a
  checkpoint. Resume will continue as if nothing changed. Point a changed pipeline at a
  fresh checkpoint directory.

### Scoring

- `Score` always returns a number inside the range you gave it. A model that ignores the
  instruction still produces a score, usually the lowest one. Look at the spread before
  you filter on it.

### Branching

- If two branch paths write the same column, only one of the two survives the join.
- `JoinBranches(how="outer")` does not fill the missing side with empty values.

### Data ops and sinks

- If your records do not all carry the same columns, CSV raises and Parquet quietly
  drops the extra column.

### Models

- On Anthropic, Gemini and OpenRouter, a model that is not in the catalog falls back to
  one default set of capabilities. Reasoning may be off when you expect it on. Check the
  provider page for your model.

[Error handling & troubleshooting](guides/troubleshooting.md) covers what each error
means and what survives a crash.

## Where to go next

- [Quickstart](quickstart.md) — install to a stored dataset, one page.
- [Concepts](concepts.md) — record, step, pipeline, runner.
- [Guides](guides/index.md) — one page per job.
- [Cookbook](cookbook/index.md) — complete recipes, and every example script.
- [Contributing](contributing.md) — the repository, the tests, and how to add a step.
