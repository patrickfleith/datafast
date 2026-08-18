# Datafast

Datafast is a pipeline-first Python library for generating synthetic datasets with
LLMs.

You describe the axes your dataset should cover, compose the steps that fill it in, and
run the pipeline. What you get back is a dataset — a JSONL or CSV file, a Parquet file,
a Hugging Face Hub repo, or records in memory — with every row still carrying the seed
values and the model that produced it.

Documentation: **[patrickfleith.github.io/datafast](https://patrickfleith.github.io/datafast/)**
· [Quickstart](https://patrickfleith.github.io/datafast/quickstart/)
· [Cookbook](https://patrickfleith.github.io/datafast/cookbook/)
· [API reference](https://patrickfleith.github.io/datafast/api/)

## Installation

```bash
pip install datafast
```

The base install covers every step, all six provider factories (plus
`openai_compatible`), and the JSONL, CSV and in-memory sinks. Optional file
formats and Hub I/O ship as extras:

| Extra | Enables | Pulls in |
|-------|---------|----------|
| `datafast[parquet]` | `Source.parquet(...)`, `ParquetSink` | `pyarrow` |
| `datafast[hub]` | `HuggingFaceSource`, `HubSink` | `datasets`, `huggingface-hub` |
| `datafast[langfuse]` | Langfuse tracing | `langfuse` |
| `datafast[all]` | `parquet` + `hub` | — |

```bash
pip install "datafast[hub]"
```

## Quick Start

```python
from datafast import LLMStep, Seed, Sink, openai

pipeline = (
    Seed.product(
        Seed.values("topic", ["photosynthesis", "plate tectonics", "vaccines"]),
        Seed.values("level", ["beginner", "advanced"]),
    )
    >> LLMStep(
        prompt=(
            "Write one {level} exam question about {topic}, with its answer. "
            "Return JSON with fields question and answer."
        ),
        input_columns=["topic", "level"],
        output_columns=["question", "answer"],
        parse_mode="json",
        model=openai(),
    )
    >> Sink.jsonl("questions.jsonl")
)

pipeline.run()
```

Three topics and two levels produce six rows: the seed expands the combinations, the
LLM step fills each one in, and the sink writes the result. The full walkthrough is in
the [Quickstart](https://patrickfleith.github.io/datafast/quickstart/).

## Why pipelines

- **Coverage is declarative.** `Seed.product` expands the combinations instead of you
  writing nested loops.
- **Runs are resumable.** LLM calls are checkpointed per call, so an interrupted run
  resumes instead of being paid for twice.
- **Providers are interchangeable.** One configuration surface covers OpenAI,
  Anthropic, Gemini, Mistral, OpenRouter, Ollama and any OpenAI-compatible server.
- **Mistakes surface before the spend.** `Pipeline.compile()` validates structure and
  column references before a single call is made.

## Main Building Blocks

- `Source`: load records from Python lists, files, or Hugging Face datasets
- `Seed`: generate record combinations declaratively
- `AddUUID`, `Map`, `FlatMap`, `Filter`, `Group`, `Pair`, `Concat`, `Join`: data operations
- `Sample`: draw a subset by one of nine strategies
- `LLMStep`: free-form generation
- `Classify`, `Score`, `Compare`, `Rewrite`, `Extract`: higher-level LLM transforms
- `Branch` and `JoinBranches`: multi-path pipelines
- `Sink`: write JSONL, CSV, Parquet, Hub datasets, or collect records in memory
- `Runner` and `RunConfig`: execution, batching, checkpoints, resume

## Served Models

A **served model** is a provider and a model together, with its configuration — the
object you construct and pass to a step. Build one with a provider factory:

`openai`, `anthropic`, `gemini`, `mistral`, `openrouter`, `ollama`, `openai_compatible`

```python
from datafast import openai

model = openai("gpt-5.4-mini", temperature=0.7)
```

The factories are the public entry points; `ServedModel` is exported for annotations.
Each reads its own API key from the environment — `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
and so on — or from a `.env` file, loaded once when the first served model is built.

## Optional Langfuse Tracing

With the `langfuse` extra installed, Datafast enables Langfuse tracing through LiteLLM
automatically. Put the credentials in `.env`:

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

Tracing then switches on when you create a served model — no code change:

```python
from datafast import openai

model = openai()  # traced if the Langfuse credentials are present
```

To enable it explicitly instead, call `configure_langfuse_tracing()` at startup:

```python
from datafast import configure_langfuse_tracing

configure_langfuse_tracing()
```

## Repo Layout

- `datafast/`: canonical source package
- `examples/scripts/`: runnable pipeline examples
- `examples/providers/`: direct provider usage examples
- `docs/`: pipeline-first documentation, published at
  [patrickfleith.github.io/datafast](https://patrickfleith.github.io/datafast/)

## Running Tests

`pytest` is not on PATH in this repo. Use the project virtualenv:

```bash
.venv/bin/pytest
```

Tests under `tests/live/` call real providers. They self-skip unless you opt in with
`--run-live`, and you can deselect them outright with `-m "not live"`.
