# Quickstart

Install datafast, generate a small dataset, and look at what it wrote. You need about
five minutes and one API key.

## Install

```bash
pip install datafast
```

The base install has everything this page uses: every step, every provider factory,
and the JSONL, CSV and in-memory sinks. Parquet files and the Hugging Face Hub need an
extra: `pip install "datafast[parquet]"`, `"datafast[hub]"`, or `"datafast[all]"` for
both. [Installation](installation.md) lists every extra and every environment variable
datafast reads.

## Set an API key

datafast reaches every provider through LiteLLM. You only need a key for the provider
you call. This page uses OpenAI:

```bash
export OPENAI_API_KEY="sk-..."
```

A `.env` file in your working directory works too. datafast reads it once, the first
time you build a served model. A variable already set in your shell wins over the file.

## Generate a dataset

Save this as `quickstart.py`:

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
    >> Sink.jsonl("quickstart.jsonl")
)

pipeline.run()
```

Then run it:

```bash
python quickstart.py
```

The script makes six calls to OpenAI's default model and writes six records.

## What just happened

The pipeline has three steps, and the `>>` operator connects them. All datafast
pipelines have this structure:

1. A source or a seed makes the first records.
2. One or more steps change the records.
3. A sink writes the records.

**`Seed.product` makes the first records.** It makes one record for each combination of
the values in its dimensions. The `topic` dimension has three values and the `level`
dimension has two values. The result is six records (3 × 2 = 6). You write the values
for each dimension, and the seed makes all the combinations. Most datasets in datafast
start with a seed.

**`LLMStep` sends one request to the model for each record.** The step puts the
record's `topic` and `level` values into the `{topic}` and `{level}` placeholders of the
prompt. With `parse_mode="json"`, the step reads the reply as a JSON object. The step
then divides the reply into the two columns in `output_columns`: `question` and
`answer`.

**`Sink.jsonl` writes the records to a file.** A sink does not remove the records from
the pipeline, so `run()` gives the records back to you. If you add a second sink, the
pipeline writes the same records to two files.

## What it wrote

`quickstart.jsonl` holds one JSON object per line. The first one looks like this:

```json
{
  "topic": "photosynthesis",
  "level": "beginner",
  "question": "What process do plants use to convert light into chemical energy?",
  "answer": "Photosynthesis, which occurs in the chloroplasts.",
  "_model": "gpt-6-luna"
}
```

The seed columns are still there. `topic` and `level` sit beside the generated columns,
so each record shows what produced it.

datafast also added `_model`, the id of the model that wrote the record. Columns that
start with an underscore are metadata. A step adds more of them when you give it several
prompts or languages.

## Where to go next

- [Key Concepts](concepts.md): records, steps, pipelines and the runner, and how they
  fit.
- [Source](reference/source.md) and [Seed](reference/seed.md): every way to start a
  pipeline. [Data ops](reference/data_ops.md) covers the steps that reshape records.
- [LLM step](reference/llm_step.md): several models, several languages, JSON output,
  and the steps that score and judge.
- [Pipelines & execution](guides/pipelines_and_execution.md): resume a long run instead
  of paying for it twice.
- [Cookbook](cookbook/index.md): complete recipes that build real datasets.
