# Space engineering text generation

Generate a domain text corpus — titles and body text about spacecraft engineering — from
nothing but three lists of words.

Unlabelled text is what you need before you need anything else: to continue pre-training a
model on a domain, to build a retrieval corpus, or to have something to label later. Real
technical text about space engineering exists, but it is scattered, differently licensed,
and mostly written for one kind of reader. This recipe generates it instead, and makes the
axes you care about — the document type, the subject, the reader — explicit up front, so
the corpus is spread evenly across them by construction.

There are no labels here. That is the difference from
[text classification](text_classification.md): the same shape of pipeline, but the seed
columns describe *how* the text should be written rather than what it is an example of.

- **Script:** `examples/scripts/44_cookbook_space_text_generation.py`
- **Prompt:** [`space_text_generation.txt`](assets/space_text_generation.txt), at
  `docs/cookbook/assets/space_text_generation.txt`
- **Output:** `examples/outputs/44_space_text_generation_cookbook.jsonl`
- **Checkpoints:** `examples/checkpoints/44_space_text_generation_cookbook`
- **144 rows** by default, one LLM call each

## The pipeline

Five steps, four of them named:

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

Record counts along the way:

| Step | In | Out | Why |
|---|---|---|---|
| `seed_space_text_grid` | 0 | 72 | 3 document types × 8 topics × 3 expertise levels |
| `generate_space_text` | 72 | 144 | × 2 languages × 1 model × `num_outputs` of 1 |
| `finalize_record` | 144 | 144 | drops the internal columns, renames two |
| `add_uuid` | 144 | 144 | adds `id` |
| `JSONLSink` | 144 | 144 | writes the file, passes records through |

## Designing the seed

The seed is where the corpus is designed. Every axis you want the text to cover is
declared here, and `Seed.product` crosses them — every document type against every topic
against every expertise level.

**`DOCUMENT_TYPES` — the register.** A textbook, a design justification document, a
personal blog. This is the axis most corpora are missing. Text about the same subject
written for a textbook and written for a blog share vocabulary and share almost nothing
else, and a model trained on one register only writes that register.

**`TOPICS` — the subject matter.** Eight space environment effects: microgravity, vacuum,
heavy ions, thermal extremes, atomic oxygen, debris impact, electrostatic charging,
propellant boil-off. Naming them individually, rather than asking for "space engineering
text" eight times, is what stops the model from writing about its favourite topic every
time.

**`EXPERTISE_LEVELS` — the reader.** Executives, senior engineers, PhD candidates. The
same fact stated for three audiences differs in depth, in how much is assumed, and in
sentence length.

Nothing here co-varies, so three plain `Seed.values` dimensions are right. When two
columns *must* move together — a label and its definition — they belong in one dimension
instead; see [text classification](text_classification.md) for that case.

### Counting before spending

The script computes its own row count and prints it before the run starts:

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

Every one of those rows is one LLM call. Passing a model count lets you price a change
before making it: adding two models to `MODEL_IDS` is 432 calls, not 146.

## The prompt

One line, in a file:

```text
Write one {document_type} excerpt about {topic} for {expertise_level} in {language_name}.
```

Compare this with the [text-classification prompt](assets/text_classification_generation.txt),
which is a page of constraints. The difference is what the two datasets are for. A
classification dataset needs the text to *stay inside its label*, so the prompt fences the
model in. A raw corpus wants range, so the prompt says as little as possible and lets the
seed columns do the varying.

Three of the four placeholders are seed columns, listed in `input_columns`.
`{language_name}` is not — it comes from `language=LANGUAGES`, which adds two placeholders
to every prompt:

| Placeholder | Value | Use |
|---|---|---|
| `{language}` | `en` | the code, which lands in the `_language` column |
| `{language_name}` | `English` | what you ask the model to write in |

Use the name in the prompt. Models follow "in English" far more reliably than "in en".

`prompt` takes the path, not the text. Even a one-line prompt is worth keeping in a file:
it can be edited, diffed and reviewed without touching the pipeline, and swapping
`PROMPT_PATH` is how you try a different one.

**`input_columns` is a whitelist.** Only the columns named there reach the prompt. A
placeholder for a column that exists in the record but was left out of `input_columns`
raises `KeyError` — before any call is made, which is the good outcome.

### The model

`model=make_models()` takes a list of served models, and every seed record is generated
once by each. The script builds one:

```python
from datafast import openrouter

MODEL_IDS = ["nvidia/nemotron-3-super-120b-a12b:nitro"]

def make_models():
    return [openrouter(model_id, temperature=0.7) for model_id in MODEL_IDS]
```

One model is a deliberate starting point for a corpus this size, not a recommendation.
Add ids to `MODEL_IDS` and the list grows itself — a second one doubles the corpus and
doubles the cost, and the model that wrote each row is recorded in `_model` so you can
compare them or drop one later.

`temperature=0.7` is set on the served model, never on the step: sampling settings belong
to the thing being called. Both models are reached through
[OpenRouter](../reference/providers/openrouter.md) on one API key.

## Two columns out of one call

`parse_mode="json"` with `output_columns=["title", "text"]` gets two columns from one
response. The step asks for JSON by appending an instruction to the prompt it just
formatted:

```text
Respond with valid JSON containing these fields: {"title": "<title value>", "text": "<text value>"}
Return only the JSON object, no additional text or markdown code fences.
```

That text is not in the prompt file. It is generated from `output_columns`, so the file
and the columns can never disagree. Three things worth knowing about what comes back:

- **Code fences are stripped.** A reply wrapped in ` ```json ` parses fine, despite the
  instruction asking for none.
- **Every parsed column is a string.** A model that answers `"text": ["a", "b"]` gives you
  the string `['a', 'b']`.
- **A missing field becomes an empty string, silently.** If the reply carries `title` but
  no `text`, the row is kept with `text` set to `""`. It is logged as a warning, and it is
  not a parse error — so `on_parse_error` never sees it.

That last one is the failure mode to watch on this recipe. The whole point of the dataset
is the `text` column, and an empty one costs a call and looks like a row. Grep the output
for `"text": ""` before publishing.

`on_parse_error="raise"` is set on the step, and under `Pipeline.run()` it does not raise
— the runner logs the failure and drops the record. A reply that is not JSON at all costs
you a row from the 144, not a traceback. Count what you got. See
[Error handling & troubleshooting](../guides/troubleshooting.md).

### `num_outputs`

`num_outputs` asks for *n* separate calls per prompt, model and language combination, and
yields *n* rows. It is the knob for "same instruction, more text", and it is set to 1 here
because this seed already has 72 combinations.

Raise it and note what you get: the extra rows carry no column saying which output they
were. Two rows from the same seed differ only in their generated text and their `id`, so
if you need to trace a row back to its sibling, add that column yourself in the `Map`.

## Shaping the output

`Map` runs a plain function that returns a **new** dict, so anything it does not name is
dropped:

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

This is where `_language` and `_model` lose their underscore. The leading underscore marks
a column the run added rather than the data carrying, which is useful in flight and noise
in a published dataset. The same step fixes the column order.

`AddUUID(column="id", overwrite=True)` runs **after** the `Map`, so the id belongs to the
published row and survives it.

### A row

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

Five of the eight columns were known before the call. Only `title` and `text` were
generated, and only `id` was invented afterwards — so the metadata cannot be wrong about
the text it describes.

## Running it

```bash
python examples/scripts/44_cookbook_space_text_generation.py
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

144 calls is long enough that an interruption is worth planning for. `resume=True` with a
`checkpoint_dir` makes the command safe to repeat: the checkpoint holds one file per step,
named after it.

```text
manifest.json
step_000_seed_space_text_grid.jsonl
step_001_generate_space_text.jsonl
step_002_finalize_record.jsonl
step_003_add_uuid.jsonl
step_004_JSONLSink.jsonl
```

Those names are `.as_step(...)` paying for itself — they are also the values `resume_from`
takes. The sink was left unnamed and shows up as `JSONLSink`.

Run the finished script a second time and it costs **nothing**: zero calls, the same 144
records, the same ids, the same file. That is worth knowing before the next section,
because it is what makes publishing a separate step.

One warning about resume: the checkpoint fingerprint covers step names and classes only.
Change the prompt, the topics or the model and resume will happily continue with the old
records. Use a fresh `checkpoint_dir` when you change what is generated.

## Publishing, and why it is not a step

The Hub push is a function, called after the run, behind an environment variable:

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

A sink inside the pipeline runs on every run. That is the right behaviour for a file on
disk and the wrong one for a public artefact: re-running a script to check a column should
not publish. So this recipe keeps the push out of the chain and gates it, and the free
re-run above is what makes that pleasant — generate first, read the JSONL, then run the
same command again with the variable set. Resume replays the whole pipeline from the
checkpoint at zero cost and pushes the result.

The trade-off is real, and it is worth being clear about it:

| | Sink in the pipeline | Push after the run |
|---|---|---|
| When it runs | every run | only when you ask |
| In the manifest | yes, as a step | no |
| Checkpointed and resumable | yes | no |
| Checked by `compile()` | yes | no |
| Needs `list(...)` | no | yes — `process` is a generator |

The [persona cookbook](persona_generation.md) makes the opposite choice and chains
`Sink.jsonl >> Sink.hub`, which is a good fit there because its repository is private and
it is meant to publish on every run. Neither is more correct; pick by whether an
accidental push would matter.

`train_size=0.8` splits the dataset into `train` and `test`, and `SEED` — a fixed random
seed, nothing to do with `Seed.product` — makes that split reproducible. `private=True` publishes to a repository only you can read —
**change `HF_REPO_ID` to a repository you own before running this.**

## Making it yours

| To change | Edit | Effect on the row count |
|---|---|---|
| the registers | `DOCUMENT_TYPES` | linear |
| the subject matter | `TOPICS` | linear |
| the readers | `EXPERTISE_LEVELS` | linear |
| the languages | `LANGUAGES` | linear |
| the models | `MODEL_IDS` | linear |
| how many texts per combination | `NUM_OUTPUTS` | linear |
| what is asked for | `PROMPT_PATH` | none |
| where it publishes | `HF_REPO_ID` | none |

Every axis multiplies with every other. Six dimensions gaining one value each does not add
six rows.

Two things worth adding once the shape is right:

- **A length check.** A `Filter` on `len(record["text"])` catches both the empty `text`
  described above and the model that answered in one sentence.
- **A deduplication pass.** Eight topics and one prompt line will produce openings that
  repeat. A `Map` that normalizes `text`, then `Group`, will show you how much.

## Where to go next

- [Sources & Seed](../reference/sources_and_seed.md) — every dimension and combiner.
- [LLM steps](../guides/llm_steps.md) — the rest of what `LLMStep` takes.
- [Structured output](../guides/structured_output.md) — parse modes, and what a provider could enforce instead.
- [Pipelines & execution](../guides/pipelines_and_execution.md) — checkpoints and run controls.
- [Sinks](../reference/sinks.md) — `Sink.jsonl`, `Sink.hub` and the others.
- [Text classification](text_classification.md) — the same shape, with labels.
