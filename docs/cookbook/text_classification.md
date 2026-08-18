# Text classification

Build a labelled, multilingual text-classification dataset from nothing but a list of
labels.

The problem this solves is common: you want to train or evaluate a classifier, you know
what the classes are, and you have no data. Writing examples by hand is slow and the
result is narrow — the same handful of phrasings for every class. Here the classes are
declared up front, every example is generated to fit exactly one of them, and the label
is known before the text exists rather than guessed afterwards.

The worked example is trail-condition reports: short comments a hiker might leave about a
path, sorted into four categories a trail-maintenance team would act on.

- **Script:** `examples/scripts/45_cookbook_text_classification.py`
- **Prompt:** [`text_classification_generation.txt`](assets/text_classification_generation.txt)
- **Output:** `examples/outputs/45_text_classification_cookbook.jsonl`
- **Checkpoints:** `examples/checkpoints/45_text_classification_cookbook`
- **96 rows** by default, from 4 labels

## The pipeline

Five steps, and every one of them is named:

```python
from datafast import AddUUID, LLMStep, Map, Seed, SeedDimension, Sink

pipeline = (
    Seed.product(
        SeedDimension(columns=["label", "label_description"], values=LABELS),
        Seed.values("trail_type", TRAIL_TYPES),
        Seed.values("style", STYLES),
    ).as_step("seed_trail_report_grid")
    >> LLMStep(
        prompt=PROMPT_PATH,
        input_columns=["label", "label_description", "trail_type", "style"],
        output_column="text",
        parse_mode="text",
        model=MODELS,
        language=LANGUAGES,
    ).as_step("generate_trail_reports")
    >> Map(keep_output_fields).as_step("keep_output_fields")
    >> AddUUID(column="id", overwrite=True).as_step("add_uuid")
    >> Sink.jsonl(OUTPUT_PATH)
)
```

Record counts along the way:

| Step | In | Out | Why |
|---|---|---|---|
| `seed_trail_report_grid` | 0 | 24 | 4 labels × 3 trail types × 2 styles |
| `generate_trail_reports` | 24 | 96 | × 2 languages × 2 models |
| `keep_output_fields` | 96 | 96 | drops the columns not meant for publication |
| `add_uuid` | 96 | 96 | adds `id` |
| `JSONLSink` | 96 | 96 | writes the file, passes records through |

## Designing the seed

The seed is the whole design of the dataset. Everything the model is asked to vary is
declared here, so you can count and inspect it before spending anything.

`Seed.product` crosses the dimensions it is given: every label against every trail type
against every style. Three axes were chosen, and each one exists for a different reason.

**`label` — what the dataset is for.** Four trail-condition classes: `trail_obstruction`,
`infrastructure_issues`, `hazards`, `positive_conditions`. This is the target column, so
it has to be balanced by construction: every label gets the same number of rows because
the product gives it the same number of combinations.

**`trail_type` — content variety.** A mountain trail, a coastal path, a forest walk. A
classifier trained only on mountain-trail language learns the setting, not the class.

**`style` — surface variety.** A brief social media post, or a hiking review. Same
meaning, very different sentence shape. This is what stops every row in a class from
reading like the same person wrote it.

### Why the label and its description are one dimension

`label` and `label_description` vary **together**. A label is meaningless with another
label's definition, so they must never be crossed:

```python
from datafast import Seed, SeedDimension

LABELS = [
    {"label": "hazards", "label_description": "The trail has immediate safety risks..."},
    {"label": "positive_conditions", "label_description": "The report highlights clear..."},
]

dimension = SeedDimension(columns=["label", "label_description"], values=LABELS)
assert len(dimension) == 2  # two values in one dimension, not two dimensions
```

`SeedDimension` is the general form of a dimension: any number of columns that move
together, given as a list of dicts. [`Seed.expand`](../reference/sources_and_seed.md) is
the two-column shorthand for the same idea.

Getting this wrong is the classic seeding mistake. Two separate `Seed.values` dimensions
would produce 16 label-and-description pairs where 12 of them are contradictions.

### Counting before spending

The script computes the total itself and prints it before the run:

```python
LABELS, TRAIL_TYPES, STYLES = range(4), range(3), range(2)
LANGUAGES, MODELS = {"en": "English", "fr": "French"}, range(2)

expected_rows = len(LABELS) * len(TRAIL_TYPES) * len(STYLES) * len(LANGUAGES) * len(MODELS)
assert expected_rows == 96
```

`len(seed)` gives the same number for the seed itself, before the LLM step multiplies it.
Do this every time: one careless dimension turns 96 calls into 960.

## The generation step

One `LLMStep` does all the work, and it multiplies its input three ways at once.

### Languages

`language=LANGUAGES` takes a dict of code to name, and runs every record once per entry:

```python
LANGUAGES = {"en": "English", "fr": "French"}
```

Both halves reach the prompt, as two different placeholders:

| Placeholder | Value | Use |
|---|---|---|
| `{language}` | `en` | when the code reads better |
| `{language_name}` | `English` | what you ask the model to write in |

The prompt uses `{language_name}` — models follow "in English" far more reliably than "in
en". The code is what lands in the `_language` column.

### Models

`model=MODELS` takes a list, and every record is generated once by each:

```python
MODEL_IDS = [
    "nvidia/nemotron-3-super-120b-a12b:nitro",
    "mistralai/ministral-14b-2512",
]
```

Two models from **different families** is the point. Generating with one model bakes its
habits into the dataset, and a classifier trained on it learns that model as much as the
task. Two families, one API key: both are reached through
[OpenRouter](../reference/providers/openrouter.md).

`temperature=0.8` is set on each served model — high enough for varied phrasing, not so
high that the comment stops matching its label. Sampling settings live on the served
model, never on the step.

The model that produced each row is recorded in `_model`, so you can compare them
afterwards or drop one without regenerating everything.

### The prompt

The prompt is a file, not a string:

```python
PROMPT_PATH = "docs/cookbook/assets/text_classification_generation.txt"
```

Passing a path keeps a long prompt out of the pipeline definition and under version
control on its own. It opens by naming the task and the language, then gives the category
and its definition, then a list of constraints — the setting, the style, a length limit,
and several things *not* to do:

```text
- Do not sound like an official report, safety bulletin, or structured form.
- Do not mention the category name directly.
```

That second line matters more than it looks. Without it the model writes "This is a
hazard on the trail", and every row contains its own label. A classifier trained on that
learns to find the word, and scores well on data no real user would write.

**`input_columns` is a whitelist, not a note.** Only the columns listed there are
available to the prompt. A placeholder for a column that exists in the record but is
missing from `input_columns` stops the run with a `KeyError` — which is the good outcome,
because it happens before any call is made.

`parse_mode="text"` means the whole reply, stripped, becomes the `text` column. There is
nothing to parse: one prompt, one answer, one column.

## Shaping the output

Two steps stand between generation and the file, and both exist for the same reason: the
columns that were useful *during* the run are not the columns you want to publish.

```python
def keep_output_fields(record: dict) -> dict:
    """Keep only the fields meant for publication."""
    return {
        "label": record["label"],
        "trail_type": record["trail_type"],
        "style": record["style"],
        "language": record.get("_language", ""),
        "model": record.get("_model", ""),
        "text": record["text"],
    }
```

`Map` returns a **new** dict rather than editing the old one, so anything left out is
dropped. `label_description` goes here — it was written for the model, and republishing a
paragraph of instructions on every row helps nobody. The same step renames the two
metadata columns from `_language` and `_model` to `language` and `model`, and fixes the
column order.

`AddUUID(column="id", overwrite=True)` runs **after** the `Map`, so the id belongs to the
published row.

### A row

```json
{
  "label": "trail_obstruction",
  "trail_type": "mountain trail",
  "style": "a brief social media post",
  "language": "en",
  "model": "nvidia/nemotron-3-super-120b-a12b:nitro",
  "text": "Bridge over the creek is out, had to wade across. Cold.",
  "id": "3295d639-6aee-493f-bfa9-96b9c074c8b7"
}
```

Every column except `text` and `id` came from the seed or from the run. Nothing was
inferred, so nothing can be wrong.

## Running it

```bash
python examples/scripts/45_cookbook_text_classification.py
```

You need `OPENROUTER_API_KEY` in a `.env` file at the repository root. Hugging Face
authentication is only needed if you publish.

```python
records = pipeline.run(
    batch_size=4,
    checkpoint_dir=CHECKPOINT_DIR,
    resume=True,
)
```

`resume=True` with a `checkpoint_dir` makes the command safe to repeat. Run it again after
an interruption and it picks up where it stopped instead of paying twice.

Naming the steps pays off here. The checkpoint directory holds one file per step, named
after it:

```text
manifest.json
step_000_seed_trail_report_grid.jsonl
step_001_generate_trail_reports.jsonl
step_002_keep_output_fields.jsonl
step_003_add_uuid.jsonl
step_004_JSONLSink.jsonl
```

Those are the names `resume_from` wants, and the reason `.as_step(...)` is worth the
keystrokes — the fifth step was left unnamed and shows up as `JSONLSink`. To rebuild the
output after editing `keep_output_fields`, without regenerating a single comment:

```python
records = pipeline.run(
    checkpoint_dir=CHECKPOINT_DIR,
    resume=True,
    resume_from="keep_output_fields",
)
```

One warning about resume: the checkpoint fingerprint covers step names and classes only.
Change the prompt, the labels or a model and resume will happily continue with the old
records. Use a fresh `checkpoint_dir` when you change what is generated. See
[Error handling & troubleshooting](../guides/troubleshooting.md).

## Publishing

The Hub push is deliberately outside the pipeline:

```python
if os.getenv("DATAFAST_PUSH_TO_HUB") == "1":
    list(
        Sink.hub(
            repo_id=HF_REPO_ID,
            private=False,
            train_size=0.8,
            seed=SEED,
            shuffle=True,
            commit_message="Publish cookbook 45 classification dataset",
        ).process(records)
    )
```

A sink in the chain runs on every run. Publishing is not something you want to happen
because you re-ran a script to check an output column, so it sits behind an environment
variable:

```bash
DATAFAST_PUSH_TO_HUB=1 python examples/scripts/45_cookbook_text_classification.py
```

`train_size=0.8` splits the dataset 80/20 into `train` and `test`. The rows come out of
the run grouped by model, so the split has to mix them: `shuffle=True` shuffles before
splitting, and the fixed `seed` is what makes the result the same every time.
`private=False` publishes openly; **change `HF_REPO_ID` to a repository you own before running this.**

`Sink.hub` is a step like any other, so calling `.process(records)` runs it directly. It
is a generator, which is why the call is wrapped in `list(...)` — without that, nothing
happens.

## Making it yours

| To change | Edit | Effect on the row count |
|---|---|---|
| the classes | `LABELS` | linear |
| the settings | `TRAIL_TYPES` | linear |
| the voices | `STYLES` | linear |
| the languages | `LANGUAGES` | linear |
| the models | `MODEL_IDS` | linear |
| where it publishes | `HF_REPO_ID` | none |

Every axis multiplies with every other, so adding one value to each of five dimensions
does not add five rows.

Two things worth adding once the shape is right:

- **A verification pass.** Add a [`Classify`](../reference/llm_specialized.md) step that
  reads `text` and predicts the label, then a `Filter` that keeps only the rows where the
  prediction matches the seeded label. Generation quality varies by label; this measures
  it instead of assuming it.
- **A deduplication step.** High temperature and 24 seed combinations still produce near
  duplicates. A `Map` that normalizes `text`, followed by `Group`, will find them.

## Where to go next

- [Sources & Seed](../reference/sources_and_seed.md) — every dimension and combiner.
- [LLM steps](../guides/llm_steps.md) — the rest of what `LLMStep` takes.
- [Pipelines & execution](../guides/pipelines_and_execution.md) — checkpoints and run controls.
- [Sinks](../reference/sinks.md) — `Sink.jsonl`, `Sink.hub` and the others.
- [Persona generation](persona_generation.md) — a cookbook that starts from real text instead of a seed.
