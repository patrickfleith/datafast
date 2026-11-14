# What's in v1

datafast 1.0.0 is the first stable release. This page covers what the library does,
what shipped, what is left out on purpose, and the rough edges to know about.

Versions before 1.0.0 were experimental and are no longer supported. There is no
migration guide: v1 is a fresh start.

## What datafast does

You describe how to build a dataset as a chain of steps, then run the chain.

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
batching, checkpoints and resume.

Read [Key Concepts](concepts.md) for how the pieces fit, or the
[Quickstart](quickstart.md) to run a pipeline yourself.

## What shipped

**One pipeline model.** Steps chain with `>>`. Every step takes records and returns
records, so steps combine freely within a few structural rules. `compile()` checks those
rules before the first call is made. The source comes first, nothing follows the sinks,
every `Branch` has a matching `JoinBranches`, and every column a step reads exists.

**Seven ways to reach a model.** `openai`, `anthropic`, `gemini`, `mistral`,
`openrouter`, `ollama` and `openai_compatible`. Each one builds a *served model*: a
model, the server that serves it, and the settings for both. Rate limits, timeouts,
retries and sampling settings live on the served model, not on the runner. Two steps
that share a served model share one rate limit. See
[Served models](reference/served_models.md).

**Generation and evaluation steps.** `LLMStep` is the general step. `Classify`, `Score`,
`Compare`, `Rewrite` and `Extract` do common jobs without a hand-written prompt. See
[LLM step](reference/llm_step.md) and
[Specialized LLM steps](reference/llm_specialized.md).

**Branching.** `Branch` sends the same records down several paths and `JoinBranches`
merges them back. You need it for preference data: two answers per record, scored and
compared. See [Branching](reference/branching.md).

**Checkpoints and resume.** Each step saves its output before the next one starts, and
LLM steps save each call as it returns. A run that dies after 700 of 1,000 calls resumes
at call 701, so you do not pay for the first 700 again. See
[Pipelines & execution](guides/pipelines_and_execution.md).

**Multimodal input.** Images, audio, video and files, limited to what each served model
declares it accepts. See [Multimodal input](guides/multimodal_input.md).

**Optional extras.** `datafast[parquet]` adds Parquet files, `datafast[hub]` adds
Hugging Face datasets, and `datafast[all]` adds both. The base install declares five
dependencies and pulls in 50 packages, down from 107. See
[Installation](installation.md).

**Type hints.** The package ships a `py.typed` marker, so your type checker reads
datafast's annotations.

## What is left out on purpose

- **No migration path from pre-1.0.** The old dataset-class API is gone.
- **No async API.** The runner and the served model handle batching and concurrency.
  You call `run()` and it blocks until the run ends.
- **No progress bar.** The logger reports progress at `INFO` level.
- **No nested branching.** A `Branch` cannot sit inside another branch path.
- **No provider-enforced JSON.** Steps ask for JSON in the prompt and parse the reply.
  To get a schema the provider guarantees, call a served model directly. See
  [Structured output](guides/structured_output.md).

## Known rough edges

Read these before you spend money on a long run. The page linked for each topic covers
it in more detail.

### LLM steps

- When a call fails, datafast drops that record and the run continues. **Count your
  output records** against what you expected. Set `on_parse_error="raise"` on the step
  to stop at the first failure instead.
- If a model returns only some of the columns you asked for, the missing ones arrive
  empty. datafast does not report an empty column as an error.

### Resume and checkpoints

- Changing a prompt, a served model or the body of a transform does **not** invalidate a
  checkpoint. Resume carries on as if nothing changed. Point a changed pipeline at a
  fresh checkpoint directory.
- An LLM step saves its first progress file after `checkpoint_every` calls, 100 by
  default. If the run dies before that, resume runs the whole step again. Lower
  `checkpoint_every` for slow or expensive steps.

### Scoring

- `Score` always returns a number inside the range you gave it. A model that ignores the
  instruction still produces a score, usually the lowest one. Look at the spread of
  scores before you filter on them.

### Branching

- If two branch paths write the same column, only one of the two values survives the
  join.
- `JoinBranches(how="outer")` does not fill the missing side with empty values.

### Data ops and sinks

- If your records do not all have the same columns, `Sink.csv` raises, and
  `Sink.parquet` drops the extra column without a warning.

### Models

- On Anthropic, Gemini and OpenRouter, a model missing from the catalog gets the
  provider's default capabilities. Reasoning may be off when you expect it on. Check the
  provider page for your model.

[Error handling & troubleshooting](guides/troubleshooting.md) explains each error and
what survives a crash.

## Where to go next

- [Quickstart](quickstart.md): from install to a stored dataset, on one page.
- [Key Concepts](concepts.md): every concept, how it works and where it fits.
- [Guides](guides/index.md): one page per job.
- [Cookbook](cookbook/index.md): complete recipes, and every example script.
- [Contributing](contributing.md): the repository, the tests, and how to add a step.
