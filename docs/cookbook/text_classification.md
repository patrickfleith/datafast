# Text classification

Make a text classification dataset in 2 languages. The only input that you write is a
list of labels.

Use this procedure when you must train or evaluate a classifier and you have no data.
You know the classes, but you have no examples. If you write examples manually, the work
is slow. Also, the examples are not different: each class gets the same small set of
phrases. In this procedure, you specify the classes first. The LLM then generates each
text to agree with 1 class. Thus you know the label before the LLM writes the text.

The example is a set of trail reports. Each report is a short comment from a hiker about
a trail. Each report has 1 of 4 categories. A trail maintenance team does a different
action for each category.

- **Script:** `examples/scripts/45_cookbook_text_classification.py`
- **Prompt template:** [`text_classification_generation.txt`](assets/text_classification_generation.txt)
- **Output:** `examples/outputs/45_text_classification_cookbook.jsonl`
- **Checkpoints:** `examples/checkpoints/45_text_classification_cookbook`
- **96 records** by default, from 4 labels

## The pipeline

The pipeline has 5 steps. Each step has a name:

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

The record count after each step:

| Step | In | Out | Why |
|---|---|---|---|
| `seed_trail_report_grid` | 0 | 24 | 4 labels × 3 trail types × 2 styles |
| `generate_trail_reports` | 24 | 96 | × 2 languages × 2 models |
| `keep_output_fields` | 96 | 96 | removes the columns that you do not publish |
| `add_uuid` | 96 | 96 | adds `id` |
| `JSONLSink` | 96 | 96 | writes the file and sends the records through |

## Design the seed

The seed is the design of the dataset. You specify each dimension that the LLM must
change in the seed. Thus you can count and examine the records before you make LLM calls.

`Seed.product` makes all the combinations of its dimensions. Each label goes with each
trail type and each style. This seed has 3 dimensions, and each dimension has a different
function.

**`label`: the purpose of the dataset.** There are 4 trail condition classes:
`trail_obstruction`, `infrastructure_issues`, `hazards` and `positive_conditions`. This is
the target column, so the classes must have equal quantities of records. The product
gives each label the same number of combinations, thus each label gets the same number of
records.

**`trail_type`: different content.** The trail types are a mountain trail, a coastal path
and a forest walk. If you train a classifier only on reports about mountain trails, it
learns the location and not the class.

**`style`: different sentence shapes.** The styles are a short social media post and a
hiking review. The meaning stays the same, but the sentences are different. Thus
the records in a class do not all read as if 1 person wrote them.

### Why the label and its description are 1 dimension

`label` and `label_description` change **together**. A label with the description of a
different label has no meaning. Thus you must not make combinations of the 2 columns:

```python
from datafast import Seed, SeedDimension

LABELS = [
    {"label": "hazards", "label_description": "The trail has immediate safety risks..."},
    {"label": "positive_conditions", "label_description": "The report highlights clear..."},
]

dimension = SeedDimension(columns=["label", "label_description"], values=LABELS)
assert len(dimension) == 2  # two values in one dimension, not two dimensions
```

`SeedDimension` is the general form of a dimension. It contains 1 or more columns that
change together, as a list of dicts. [`Seed.expand`](../reference/seed.md) is the short
form of the same function for 2 columns.

This is a frequent seed error. If you use 2 different `Seed.values` dimensions, you get
16 pairs of a label and a description. Of these 16 pairs, 12 are incorrect.

### Count the records before you make LLM calls

The script calculates the total and shows it before the run starts:

```python
LABELS, TRAIL_TYPES, STYLES = range(4), range(3), range(2)
LANGUAGES, MODELS = {"en": "English", "fr": "French"}, range(2)

expected_rows = len(LABELS) * len(TRAIL_TYPES) * len(STYLES) * len(LANGUAGES) * len(MODELS)
assert expected_rows == 96
```

`len(seed)` gives the record count of the seed, before the LLM step multiplies it. Do
this calculation each time. If 1 dimension is incorrect, 96 calls can become 960 calls.

## The generation step

One `LLMStep` does all the generation. This step multiplies its input in 3 ways.

### Languages

`language=LANGUAGES` takes a dict of language codes and names. The step runs each record
1 time for each entry:

```python
LANGUAGES = {"en": "English", "fr": "French"}
```

The prompt template gets the 2 parts as 2 different placeholders:

| Placeholder | Value | Use |
|---|---|---|
| `{language}` | `en` | when the code is better |
| `{language_name}` | `English` | the language for the model to write in |

The prompt template uses `{language_name}`. Models obey "in English" better than "in en".
The step writes the code in the `_language` column.

### Models

`model=MODELS` takes a list. Each model generates each record 1 time:

```python
MODEL_IDS = [
    "nvidia/nemotron-3-super-120b-a12b:nitro",
    "mistralai/ministral-14b-2512",
]
```

The 2 models are from **different families**. If 1 model generates all the data, the
dataset gets the habits of that model. A classifier that you train on that data learns
the model as well as the task. One API key gives access to the 2 models, through
[OpenRouter](../reference/providers/openrouter.md).

Each served model has `temperature=0.8`. This value gives different phrases, but the
comments continue to agree with their labels. Put the sampling settings on the served
model. Do not put them on the step.

The `_model` column records the model that generated each record. Thus you can compare
the models later, or remove the records of 1 model without a new generation.

### The prompt template

The prompt template is in a file:

```python
PROMPT_PATH = "docs/cookbook/assets/text_classification_generation.txt"
```

With a path, the long prompt template stays out of the pipeline, and it has its own
version history. The prompt template first gives the task and the language. Then it
gives the category and its description. Then it gives a list of limits: the location,
the style, a maximum length and some things that the model must not do:

```text
- Do not sound like an official report, safety bulletin, or structured form.
- Do not mention the category name directly.
```

The second line is important. Without it, the model writes "This is a hazard on the
trail", and each record contains its label. A classifier learns to find the word. It
then gets good scores on data that is not similar to the text of real users.

**Only the columns in `input_columns` go to the prompt template.** A record can contain a
column that is not in `input_columns`. If the prompt template has a placeholder for that
column, the run stops with a `KeyError`. This result is good, because it occurs before
the first LLM call.

`parse_mode="text"` puts the full answer, without spaces at the two ends, in the `text`
column. The step does not parse the answer. There is 1 prompt, 1 answer and 1 column.

## Prepare the output

Two steps come between the generation and the file. They have the same purpose: the
columns that you use during the run are not the columns that you publish.

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

`Map` returns a **new** dict. It does not change the old dict, so it removes each column
that the function does not name. This step removes `label_description`, because the text
is an instruction for the model, and it has no use in each published record. The same
step changes the names of 2 metadata columns from `_language` and `_model` to `language`
and `model`. It also sets the column order.

`AddUUID(column="id", overwrite=True)` runs **after** `Map`, so the `id` identifies the
published record.

### A record

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

All the columns, other than `text` and `id`, come from the seed or from the run. The
pipeline does not guess these values, so they are correct.

## Run the script

```bash
python examples/scripts/45_cookbook_text_classification.py
```

Put `OPENROUTER_API_KEY` in a `.env` file at the root of the repository. You must log in
to Hugging Face only if you publish.

```python
records = pipeline.run(
    batch_size=4,
    checkpoint_dir=CHECKPOINT_DIR,
    resume=True,
)
```

With `resume=True` and a `checkpoint_dir`, you can do the same command again safely. If
the run stops, start it again. It continues from the point where it stopped, and it does
not make the same LLM calls again.

The step names are important here. The checkpoint directory has 1 file for each step,
with the name of the step:

```text
manifest.json
step_000_seed_trail_report_grid.jsonl
step_001_generate_trail_reports.jsonl
step_002_keep_output_fields.jsonl
step_003_add_uuid.jsonl
step_004_JSONLSink.jsonl
```

`resume_from` uses these names. This is the reason to use `.as_step(...)`. The fifth step
has no name, so its file has the name `JSONLSink`. If you change `keep_output_fields`,
you can make the output again without a new generation:

```python
records = pipeline.run(
    checkpoint_dir=CHECKPOINT_DIR,
    resume=True,
    resume_from="keep_output_fields",
)
```

!!! warning "Use a new `checkpoint_dir` when you change the generation"

    The checkpoint fingerprint includes only the step names and classes. If you change
    the prompt template, the labels or a model, resume continues with the old records.
    Refer to [Error handling & troubleshooting](../guides/troubleshooting.md).

## Publish the dataset

The push to the Hub is outside the pipeline:

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

A sink in the pipeline runs each time that the pipeline runs. Do not publish
each time that you run the script to examine a column. Thus an environment variable
controls the push:

```bash
DATAFAST_PUSH_TO_HUB=1 python examples/scripts/45_cookbook_text_classification.py
```

`train_size=0.8` divides the dataset into `train` (80%) and `test` (20%). The run gives
the records in groups by model, so the split must mix them. `shuffle=True` mixes the
records before the split. The fixed `seed` gives the same result each time.
`private=False` publishes the dataset for all users.

**Before you run this script, change `HF_REPO_ID` to a repository that you own.**

`Sink.hub` is a step, so `.process(records)` runs it directly. It is a generator. Thus
the call is in `list(...)`. Without `list(...)`, the push does not occur.

## Change the dataset

| To change | Edit | Effect on the record count |
|---|---|---|
| the classes | `LABELS` | linear |
| the locations | `TRAIL_TYPES` | linear |
| the styles | `STYLES` | linear |
| the languages | `LANGUAGES` | linear |
| the models | `MODEL_IDS` | linear |
| the destination of the push | `HF_REPO_ID` | none |

Each dimension multiplies with all the other dimensions. If you add 1 value to each of 5
dimensions, you get many more than 5 new records.

When the dataset is correct, you can add these steps:

- **A verification step.** Add a [`Classify`](../reference/llm_specialized.md) step. It
  reads `text` and predicts the label. Then add a `Filter` that keeps only the records
  where the prediction agrees with the seed label. The quality of the generation is
  different for each label. This step measures the quality.
- **A step that finds duplicates.** A high temperature and 24 seed combinations can
  still give records that are almost the same. Use a `Map` that normalizes `text`, then
  a `Group`, to find them.

## Related pages

- [Seed](../reference/seed.md): all the dimensions and combiners.
- [LLM step](../reference/llm_step.md): the other arguments of `LLMStep`.
- [Pipelines & execution](../guides/pipelines_and_execution.md): checkpoints and run controls.
- [Sinks](../reference/sinks.md): `Sink.jsonl`, `Sink.hub` and the other sinks.
- [Persona generation](persona_generation.md): a cookbook that starts from real text, not from a seed.
