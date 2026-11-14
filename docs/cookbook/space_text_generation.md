# Space engineering text generation

Generate a text corpus about spacecraft engineering, with titles and body text. The only
inputs that you write are 3 lists of words.

For many tasks, text without labels is the first data that is necessary. You can use it to
continue the pre-training of a model on a domain, to make a retrieval corpus, or to add
labels later. Real technical text about space engineering exists. But it is in many
locations, it has different licences, and it is mostly for 1 type of reader. This
procedure generates the text. You specify the dimensions first: the document type, the
subject and the reader. Thus the seed divides the corpus equally across them.

This dataset has no labels. This is the difference from
[text classification](text_classification.md). The pipeline has the same shape, but the
seed columns tell the model *how* to write the text. They do not give a class.

- **Script:** `examples/scripts/44_cookbook_space_text_generation.py`
- **Prompt template:** [`space_text_generation.txt`](assets/space_text_generation.txt), at
  `docs/cookbook/assets/space_text_generation.txt`
- **Output:** `examples/outputs/44_space_text_generation_cookbook.jsonl`
- **Checkpoints:** `examples/checkpoints/44_space_text_generation_cookbook`
- **144 records** by default, with 1 LLM call for each record

## The pipeline

The pipeline has 5 steps. Four steps have a name:

```python
from datafast import AddUUID, LLMStep, Map, Seed, Sink

pipeline = (
    Seed.product(
        Seed.values("document_type", DOCUMENT_TYPES),
        Seed.values("topic", TOPICS),
        Seed.values("expertise_level", EXPERTISE_LEVELS),
    ).as_step("seed_space_text_grid")
    >> LLMStep(
        prompt=PROMPT_PATH,
        input_columns=["document_type", "topic", "expertise_level"],
        output_columns=["title", "text"],
        parse_mode="json",
        model=make_models(),
        language=LANGUAGES,
        num_outputs=NUM_OUTPUTS,
        on_parse_error="raise",
    ).as_step("generate_space_text")
    >> Map(finalize_record).as_step("finalize_record")
    >> AddUUID(column="id", overwrite=True).as_step("add_uuid")
    >> Sink.jsonl(OUTPUT_PATH)
)
```

The record count after each step:

| Step | In | Out | Why |
|---|---|---|---|
| `seed_space_text_grid` | 0 | 72 | 3 document types × 8 topics × 3 expertise levels |
| `generate_space_text` | 72 | 144 | × 2 languages × 1 model × `num_outputs` of 1 |
| `finalize_record` | 144 | 144 | removes the internal columns and changes 2 names |
| `add_uuid` | 144 | 144 | adds `id` |
| `JSONLSink` | 144 | 144 | writes the file and sends the records through |

## Design the seed

You design the corpus in the seed. You specify each dimension that the text must have.
`Seed.product` then makes all the combinations: each document type with each topic and
each expertise level.

**`DOCUMENT_TYPES`: the register.** The types are a textbook, a design justification
document and a personal blog. Most corpora do not have this dimension. A textbook and a
blog about the same subject use the same words, but they are different in all other
ways. A model that you train on 1 register can write only that register.

**`TOPICS`: the subject.** There are 8 effects of the space environment: microgravity,
vacuum, heavy ions, thermal extremes, atomic oxygen, debris impact, electrostatic
charging and propellant boil-off. Each topic has a name in the list. If you ask 8 times
for "space engineering text", the model writes about the same topic each time.

**`EXPERTISE_LEVELS`: the reader.** The readers are executives, senior engineers and PhD
candidates. For 3 different readers, the same fact has a different depth, different
assumptions and different sentence lengths.

These 3 dimensions do not change together, so 3 usual `Seed.values` dimensions are
correct. Some columns *must* change together, for example a label and its description.
Put these columns in 1 dimension. Refer to [text classification](text_classification.md)
for an example.

### Count the records before you make LLM calls

The script calculates its record count and shows it before the run starts:

```python
DOCUMENT_TYPES, TOPICS, EXPERTISE_LEVELS = range(3), range(8), range(3)
LANGUAGES, MODEL_IDS, NUM_OUTPUTS = {"en": "English", "fr": "French"}, ["one-model"], 1

def expected_row_count(model_count: int | None = None) -> int:
    model_total = len(MODEL_IDS) if model_count is None else model_count
    return (
        len(DOCUMENT_TYPES)
        * len(TOPICS)
        * len(EXPERTISE_LEVELS)
        * len(LANGUAGES)
        * NUM_OUTPUTS
        * model_total
    )

assert expected_row_count() == 144
assert expected_row_count(3) == 432   # what three models would cost
```

Each record is 1 LLM call. Give a number of models to the function to calculate the
effect of a change before you make it. If you add 2 models to `MODEL_IDS`, you get 432
calls, not 146.

## The prompt template

The prompt template is 1 line in a file:

```text
Write one {document_type} excerpt about {topic} for {expertise_level} in {language_name}.
```

The [text classification prompt template](assets/text_classification_generation.txt) is
a full page of limits. The 2 datasets have different purposes. In a classification
dataset, the text must *agree with its label*, so the prompt template gives the model
many limits. A corpus without labels must have a large range. Thus the prompt template
is as short as possible, and the seed columns supply the differences.

Three of the 4 placeholders are seed columns, and `input_columns` contains them.
`{language_name}` is not a seed column. It comes from `language=LANGUAGES`, which adds 2
placeholders to each prompt:

| Placeholder | Value | Use |
|---|---|---|
| `{language}` | `en` | the code, which the step writes in the `_language` column |
| `{language_name}` | `English` | the language for the model to write in |

Use the name in the prompt template. Models obey "in English" better than "in en".

`prompt` takes the path, not the text. Keep a prompt template in a file, also if it is 1
line. You can then edit it, compare versions and review it without a change to the
pipeline. To try a different prompt template, change `PROMPT_PATH`.

**Only the columns in `input_columns` go to the prompt template.** A record can contain a
column that is not in `input_columns`. If the prompt template has a placeholder for that
column, the step raises `KeyError`. This occurs before the first LLM call, so the result
is good.

### The model

`model=make_models()` takes a list of served models. Each served model generates each
seed record 1 time. The script makes 1 served model:

```python
from datafast import openrouter

MODEL_IDS = ["nvidia/nemotron-3-super-120b-a12b:nitro"]

def make_models():
    return [openrouter(model_id, temperature=0.7) for model_id in MODEL_IDS]
```

One model is a start for a corpus of this size. It is not a recommendation. Add IDs to
`MODEL_IDS`, and the list becomes longer. A second model gives 2 times the records and 2
times the calls. The `_model` column records the model of each record. Thus you can
compare the models, or remove the records of 1 model later.

Set `temperature=0.7` on the served model. Do not set it on the step: sampling settings
are part of the served model. One API key gives access to all the models through
[OpenRouter](../reference/providers/openrouter.md).

## Two columns from 1 call

`parse_mode="json"` with `output_columns=["title", "text"]` gets 2 columns from 1 answer.
The step adds a JSON instruction at the end of the prompt:

```text
Respond with valid JSON containing these fields: {"title": "<title value>", "text": "<text value>"}
Return only the JSON object, no additional text or markdown code fences.
```

The prompt file does not contain this text. The step makes it from `output_columns`, so
the file and the columns always agree. Know these 3 facts about the answer:

- **The step removes code fences.** If the answer is in a ` ```json ` fence, the parse is
  correct. The instruction tells the model not to use fences, but the step accepts them.
- **Each parsed column is a string.** If a model answers `"text": ["a", "b"]`, you get the
  string `['a', 'b']`.
- **A missing field becomes an empty string, without an error.** If the answer contains
  `title` but not `text`, the step keeps the record with `text` set to `""`. The step
  writes a warning to the log. This is not a parse error, so `on_parse_error` does not
  see it.

The last fact is the most important for this dataset. The `text` column is the purpose of
the dataset. An empty `text` column uses 1 call and looks like a correct record. Before
you publish, find `"text": ""` in the output.

The step sets `on_parse_error="raise"`. But in `Pipeline.run()`, the error does not stop
the run. The runner writes the error to the log and removes the record. If an answer is
not JSON, you get 1 record less than 144, not a traceback. Count the records that you
get. Refer to [Error handling & troubleshooting](../guides/troubleshooting.md).

### `num_outputs`

`num_outputs` makes *n* different calls for each combination of prompt, model and
language. It gives *n* records. Use it to get more text from the same instruction. Here
it is 1, because this seed already has 72 combinations.

If you increase it, know this: the extra records have no column that identifies the
output number. Two records from the same seed are different only in their generated text
and their `id`. If you must find the related records of a record, add that column in the
`Map`.

## Prepare the output

`Map` runs a usual function that returns a **new** dict. Thus it removes each column that
the function does not name:

```python
def finalize_record(record: dict) -> dict:
    """Keep the columns meant for publication."""
    return {
        "document_type": record["document_type"],
        "topic": record["topic"],
        "expertise_level": record["expertise_level"],
        "language": record.get("_language", ""),
        "model": record.get("_model", ""),
        "title": record["title"],
        "text": record["text"],
    }
```

This step removes the underscore from `_language` and `_model`. The underscore at the
start identifies a column that the run adds. The data does not contain it. During the
run, these columns are useful. In a published dataset, they have no use. The same step
sets the column order.

`AddUUID(column="id", overwrite=True)` runs **after** `Map`, so the `id` identifies the
published record, and `Map` does not remove it.

### A record

```json
{
  "document_type": "space engineering textbook",
  "topic": "Microgravity",
  "expertise_level": "executives",
  "language": "en",
  "model": "nvidia/nemotron-3-super-120b-a12b:nitro",
  "title": "Operating Without Weight",
  "text": "In orbit a spacecraft and everything inside it fall together...",
  "id": "9d6ba485-0ee8-4eb2-b68e-64896add9258"
}
```

You know 5 of the 8 columns before the call. The model generates only `title` and `text`.
`AddUUID` adds `id` after the call. Thus the metadata is always correct for the text.

## Run the script

```bash
python examples/scripts/44_cookbook_space_text_generation.py
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

A run of 144 calls takes a long time, so prepare for a stop. With `resume=True` and a
`checkpoint_dir`, you can do the same command again safely. The checkpoint has 1 file for
each step, with the name of the step:

```text
manifest.json
step_000_seed_space_text_grid.jsonl
step_001_generate_space_text.jsonl
step_002_finalize_record.jsonl
step_003_add_uuid.jsonl
step_004_JSONLSink.jsonl
```

`.as_step(...)` gives these names. `resume_from` also uses these names. The sink has no
name, so its file has the name `JSONLSink`.

If you run the completed script a second time, it makes **no** LLM calls. You get the
same 144 records, the same IDs and the same file. The next section uses this fact to
publish in a different step.

!!! warning "Use a new `checkpoint_dir` when you change the generation"

    The checkpoint fingerprint includes only the step names and classes. If you change
    the prompt template, the topics or the model, resume continues with the old records.

## Publish the dataset after the run

The push to the Hub is a function. The script calls it after the run, and an environment
variable controls it:

```python
def push_records_to_hub(records: list[dict]) -> None:
    list(
        Sink.hub(
            repo_id=HF_REPO_ID,
            private=True,
            train_size=0.8,
            seed=SEED,
            shuffle=True,
            commit_message=f"Publish cookbook 44 text dataset with {', '.join(MODEL_IDS)}",
        ).process(records)
    )

if os.getenv("DATAFAST_PUSH_TO_HUB") == "1":
    push_records_to_hub(records)
```

```bash
DATAFAST_PUSH_TO_HUB=1 python examples/scripts/44_cookbook_space_text_generation.py
```

A sink in the pipeline runs each time that the pipeline runs. This is correct for a file
on the disk. It is not correct for a public dataset: if you run the script again to
examine a column, you must not publish. Thus this procedure keeps the push out of the
pipeline, and an environment variable controls it.

Use this procedure:

1. Run the script.
2. Read the JSONL file.
3. Run the same command again with the variable set.

The second run makes no LLM calls. Resume runs the full pipeline from the checkpoint and
pushes the result.

The 2 methods have different effects:

| | Sink in the pipeline | Push after the run |
|---|---|---|
| When it runs | at each run | only when you ask |
| In the manifest | yes, as a step | no |
| Checkpoints and resume | yes | no |
| `compile()` examines it | yes | no |
| Must use `list(...)` | no | yes, because `process` is a generator |

The [persona cookbook](persona_generation.md) uses the other method and puts
`Sink.jsonl >> Sink.hub` in the pipeline. That method is correct for that cookbook,
because its repository is private and it must publish at each run. The 2 methods are
both correct. Select the method from this question: is an accidental push a problem?

`train_size=0.8` divides the dataset into `train` and `test`. `SEED` is a fixed random
seed. It is not related to `Seed.product`. It gives the same split each time.
`private=True` publishes to a repository that only you can read.

**Before you run this script, change `HF_REPO_ID` to a repository that you own.**

## Change the dataset

| To change | Edit | Effect on the record count |
|---|---|---|
| the registers | `DOCUMENT_TYPES` | linear |
| the subjects | `TOPICS` | linear |
| the readers | `EXPERTISE_LEVELS` | linear |
| the languages | `LANGUAGES` | linear |
| the models | `MODEL_IDS` | linear |
| the number of texts for each combination | `NUM_OUTPUTS` | linear |
| the instruction | `PROMPT_PATH` | none |
| the destination of the push | `HF_REPO_ID` | none |

Each dimension multiplies with all the other dimensions. If you add 1 value to each of 6
dimensions, you get many more than 6 new records.

When the dataset is correct, you can add these steps:

- **A length filter.** Add a `Filter` on `len(record["text"])`. It finds the empty `text`
  columns, and the answers that are only 1 sentence.
- **A step that finds duplicates.** With 8 topics and a prompt template of 1 line, many
  texts start with the same words. Use a `Map` that normalizes `text`, then a `Group`, to
  find how many.

## Related pages

- [Seed](../reference/seed.md): all the dimensions and combiners.
- [LLM step](../reference/llm_step.md): the other arguments of `LLMStep`.
- [Structured output](../guides/structured_output.md): parse modes, and the schemas that a provider can apply.
- [Pipelines & execution](../guides/pipelines_and_execution.md): checkpoints and run controls.
- [Sinks](../reference/sinks.md): `Sink.jsonl`, `Sink.hub` and the other sinks.
- [Text classification](text_classification.md): the same shape, with labels.
