# Datafast

Datafast is a python library for synthetic data generation using llms.

The old dataset-class API has been removed. The canonical package is `datafast`, and the primary model is:

- create records with `Source` or `Seed`
- transform them with composable steps
- call LLMs with `LLMStep`, `Classify`, `Score`, `Compare`, `Rewrite`, or `Extract`
- persist results with `Sink`

## Installation

```bash
pip install datafast
```

Optional Langfuse tracing:

```bash
pip install "datafast[langfuse]"
```

## Quick Start

```python
from datafast import LLMStep, Seed, Sink, openrouter

model = openrouter("z-ai/glm-4.6")

pipeline = (
    Seed.product(
        Seed.values("topic", ["robotics", "energy storage"]),
        Seed.values("audience", ["beginner", "expert"]),
    )
    >> LLMStep(
        prompt=(
            "Write one short {audience} question about {topic}. "
            "Return JSON with fields question and answer."
        ),
        input_columns=["topic", "audience"],
        output_columns=["question", "answer"],
        parse_mode="json",
        model=model,
    )
    >> Sink.jsonl("examples/outputs/quickstart.jsonl")
)

pipeline.run(batch_size=4)
```

## Main Building Blocks

- `Source`: load records from Python lists, files, or Hugging Face datasets
- `Seed`: generate record combinations declaratively
- `Map`, `FlatMap`, `Filter`, `Group`, `Pair`, `Concat`, `Join`: data operations
- `LLMStep`: free-form generation
- `Classify`, `Score`, `Compare`, `Rewrite`, `Extract`: higher-level LLM transforms
- `Branch` and `JoinBranches`: multi-path pipelines
- `Sink`: write JSONL, CSV, Parquet, Hub datasets, or collect records in memory
- `Runner` and `RunConfig`: execution, batching, checkpoints, resume

## Providers

The package keeps direct provider coverage for:

- `OpenAIProvider`
- `AnthropicProvider`
- `GeminiProvider`
- `MistralProvider`
- `OpenRouterProvider`
- `OllamaProvider`

Top-level factory helpers are also available: `openai`, `anthropic`, `gemini`, `mistral`, `openrouter`, `ollama`.

## Optional Langfuse Tracing

If you install the optional extra, Datafast can auto-enable Langfuse tracing through LiteLLM.

`.env`

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

Once those values are present, tracing is enabled automatically when you create a provider.

```python
from datafast import LLMStep, Seed, openrouter

model = openrouter("z-ai/glm-4.6")
```

If you prefer an explicit startup call, use:

```python
from datafast import configure_langfuse_tracing

configure_langfuse_tracing()
```

## Repo Layout

- `datafast/`: canonical source package
- `examples/scripts/`: runnable pipeline examples
- `docs/`: pipeline-first documentation
- `datafast_new_design_document.md`: retained design reference

## Running Tests

`pytest` is not on PATH in this repo. Use the project virtualenv:

```bash
.venv/bin/pytest
```
