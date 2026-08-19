# Quickstart

Install datafast, generate your first dataset, and look at what it wrote. About five
minutes, one API key.

## Install

```bash
pip install datafast
```

That is everything you need for this page: every step, every provider factory, and the
JSONL, CSV and in-memory sinks come with the base install. Parquet and Hugging Face Hub
I/O ship as extras — `pip install "datafast[parquet]"`, `"datafast[hub]"`, or
`"datafast[all]"` for both. See [Installation](installation.md) for the full list and
every environment variable datafast reads.

## Set an API key

datafast reaches every provider through LiteLLM, so you only need a key for the
provider you actually call. This page uses OpenAI:

```bash
export OPENAI_API_KEY="sk-..."
```

A `.env` file in your working directory works too — datafast loads it once, the first
time you construct a served model. Real environment variables take precedence, so an
exported key always wins over the file.

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

## What just happened

The three steps are the shape of every datafast pipeline: a **source**, one or more
**transforms**, and a **sink**, composed with `>>`.

`Seed.product` takes the cartesian product of its dimensions, so three topics × two
levels produces **six seed records** — you wrote five values and got six rows. This is
where datasets come from in datafast: you describe the axes you want covered, and the
seed expands them.

`LLMStep` then calls the model once per record, filling `{topic}` and `{level}` from
each one. `parse_mode="json"` parses the reply and splits it across the two
`output_columns`, so `question` and `answer` arrive as real fields rather than one blob
of text.

`Sink.jsonl` writes the records to disk. Sinks pass their records through, so `run()`
still returns them, and you can chain a second sink to write the same dataset to two
places at once.

## What it wrote

`quickstart.jsonl` holds one JSON object per line. The first row looks like this:

```json
{
  "topic": "photosynthesis",
  "level": "beginner",
  "question": "What process do plants use to convert light into chemical energy?",
  "answer": "Photosynthesis, which occurs in the chloroplasts.",
  "_model": "gpt-5.5"
}
```

Two things worth noticing. The seed columns **survive** into the output — `topic` and
`level` are still there beside the generated fields, so every row records what produced
it. And datafast added `_model`, which names the served model that generated the row.
Underscore-prefixed columns are metadata like this; you get more of them when a step
fans out across several models, prompts or languages.

## Where to go next

- [Concepts](concepts.md) — the record → step → pipeline → runner model.
- [Sources & seed](reference/sources_and_seed.md) — the full set of sources, and
  [Data ops](reference/data_ops.md) for the transforms and sinks.
- [LLM step](reference/llm_step.md) — multiple models, languages, structured output and
  the evaluation steps.
- [Pipelines & execution](guides/pipelines_and_execution.md) — resume a long run instead
  of paying for it twice.
- [Cookbook](cookbook/index.md) — complete recipes that produce real datasets.
