# Persona generation

Turn a corpus of real articles into a set of personas, then expand each persona into
someone connected to them.

Synthetic data is only as varied as the thing that seeds it. A list of personas written by
hand converges fast — they all sound like the person who wrote them. This recipe seeds the
personas from **real text** instead: for each article, infer one plausible reader. Real
writing carries a range of subjects, registers and assumptions that no hand-written list
matches, so the personas inherit that range for free.

Then it does it again, one step out. From each inferred persona, the pipeline infers a
second person in a close relationship with the first. One article becomes two connected
people.

The approach comes from the Persona Hub paper — its Text-to-Persona and
Persona-to-Persona methods — reimplemented here with datafast. No Persona Hub code is
reused; the paper says its published prompts are simplified rather than the exact
experiment strings, so these are adaptations.

- **Script:** `examples/scripts/43_cookbook_persona_generation.py`
- **Prompts:** [asset index](assets/index.md) — three variants per step
- **Output:** `examples/outputs/43_persona_cookbook.jsonl` **and** a private Hub dataset
- **Checkpoints:** `examples/checkpoints/43_persona_cookbook`
- **10 rows** by default, and two LLM calls per row

## The pipeline

Twelve steps, in four stages: prepare the corpus, infer a persona, expand it, publish.

```python
from datafast import AddUUID, Filter, LLMStep, Map, Sample, Sink, Source, openrouter

model = openrouter(MODEL_ID, temperature=0.7)

pipeline = (
    Source.huggingface("xsum", split="validation", columns=["id", "document", "summary"])
    >> Map(add_word_count).as_step("add_word_count")
    >> Filter(fn=lambda r: 300 <= r["word_count"] <= 500).as_step("filter_word_count")
    >> Sample(n=10, strategy="first").as_step("take_first_10")
    >> Map(assign_life_stage).as_step("assign_life_stage")
    >> LLMStep(
        prompt=Sample(TEXT_TO_PERSONA_PROMPTS, n=1),
        input_columns=["document", "life_stage"],
        output_columns=["persona_description"],
        model=model,
        parse_mode="json",
        on_parse_error="raise",
    ).as_step("text_to_persona")
    >> Map(assign_related_life_stage).as_step("assign_related_life_stage")
    >> LLMStep(
        prompt=Sample(PERSONA_TO_PERSONA_PROMPTS, n=1),
        input_columns=["persona_description", "related_life_stage"],
        output_columns=["relationship_type", "related_persona_description"],
        model=model,
        parse_mode="json",
        on_parse_error="raise",
    ).as_step("persona_to_persona")
    >> Map(keep_output_fields).as_step("keep_output_fields")
    >> AddUUID(column="id", overwrite=True).as_step("add_uuid")
    >> Sink.jsonl(OUTPUT_PATH)
    >> Sink.hub(HF_REPO_ID, private=True)
)
```

Unlike a seeded dataset, the record count does not multiply. Each stage keeps or reduces
what it is given:

| Step | Effect on the count |
|---|---|
| `add_word_count` | unchanged — adds `word_count` |
| `filter_word_count` | drops articles outside 300–500 words |
| `take_first_10` | cuts to `n=10` |
| `assign_life_stage` | unchanged — adds `life_stage` |
| `text_to_persona` | unchanged — one call per record |
| `assign_related_life_stage` | unchanged — adds `related_life_stage` |
| `persona_to_persona` | unchanged — one call per record |
| `keep_output_fields` | unchanged — drops columns |
| `add_uuid` | unchanged — adds `id` |
| `Sink.jsonl`, `Sink.hub` | unchanged — both write and pass through |

## Preparing the corpus

The source is [XSum](https://huggingface.co/datasets/xsum), BBC articles with
single-sentence summaries, loaded from the Hub:

```python
from datafast import Source

source = Source.huggingface(
    "xsum",
    split="validation",
    columns=["id", "document", "summary"],
)
```

`columns=` keeps only what is needed. The dataset's own `id` is kept deliberately —
without it there is no way back from a generated persona to the article it came from.

Three steps then narrow the corpus, and the order matters.

**Measure, then filter.** `Filter` reads columns; it does not compute them. So a `Map`
adds `word_count` first:

```python
def add_word_count(record: dict) -> dict:
    return {**record, "word_count": len(record["document"].split())}
```

**Filter on length.** 300 to 500 words. Below that there is not enough signal to infer a
specific reader and the model produces generic personas; above it the article usually
covers several topics and the persona blurs. It is also the largest lever on cost, since
every article becomes prompt tokens.

**Then cut to size.** `Sample(n=10, strategy="first")` takes the first ten survivors.
`"first"` is not random: the same ten articles come out on every run, which makes prompt
changes comparable. Raise `n` once the output looks right.

A step's name is not a label. It becomes the checkpoint file name and the value
`resume_from` takes, so `take_first_10` is what you will see on disk as
`step_003_take_first_10.jsonl`. Keep the name and the number it claims in step.

Filtering before sampling is the cheap order. Ten articles chosen first, then filtered,
could leave you with three.

For a local corpus, swap the source and keep everything after it:

```python
from datafast import Map, Source

pipeline = Source.file("data/articles.jsonl") >> Map(lambda r: {**r, "document": r["text"]})
```

The rest of the pipeline only needs a `document` column.

## Two LLM steps

Both are plain `LLMStep`s with the same shape: read a column, ask for JSON, write
named columns.

### Text-to-Persona

```python
from datafast import LLMStep, Sample

LLMStep(
    prompt=Sample(TEXT_TO_PERSONA_PROMPTS, n=1),
    input_columns=["document", "life_stage"],
    output_columns=["persona_description"],
    model=model,
    parse_mode="json",
)
```

The prompt asks for **a reader of the text, not its subject** — an article about a flood
should give a council officer or a local resident, not the mayor it quotes. It also asks
for one specific person rather than a category, forbids quoting the source, and forbids
referring to the article at all, so the persona stands on its own.

### Persona-to-Persona

```python
from datafast import LLMStep, Sample

LLMStep(
    prompt=Sample(PERSONA_TO_PERSONA_PROMPTS, n=1),
    input_columns=["persona_description", "related_life_stage"],
    output_columns=["relationship_type", "related_persona_description"],
    model=model,
    parse_mode="json",
)
```

This step reads only `persona_description`. The article is gone — the second person is
derived from the first, not from the text. It returns **two** columns: the relationship
and the new persona.

The relationship is what makes the pair useful. Two unrelated personas are just two rows;
a persona and their coach can be given the same question and disagree in a way that has a
reason behind it.

### Randomizing the prompt

Both steps pass a `Sample` where a prompt normally goes:

```python
from datafast import Sample

TEXT_TO_PERSONA_PROMPTS = [
    "docs/cookbook/assets/text_to_persona_v1.txt",
    "docs/cookbook/assets/text_to_persona_v2.txt",
    "docs/cookbook/assets/text_to_persona_v3.txt",
]

picker = Sample(TEXT_TO_PERSONA_PROMPTS, n=1)
assert len(picker.pick()) == 1
```

`Sample` does two different jobs depending on how it is built. Given `items`, it is a
picker you can hand to `prompt`, `model` or `language`, and it is drawn **once per
record**. Given no items, it is a pipeline step that samples records — which is what
`Sample(n=10, strategy="first")` is, four lines higher in the same pipeline.

Three prompt variants ask for the same thing in different words. One prompt applied to a
thousand articles produces a thousand personas with the same sentence structure; the
variants break that up. It is the same idea as varying the model, applied to the
instruction instead.

The life stages are randomized the same way, but in plain Python:

```python
import random

LIFE_STAGES = ["a teenager", "a young adult", "an adult (30s/40s)"]

def assign_life_stage(record: dict) -> dict:
    return {**record, "life_stage": random.choice(LIFE_STAGES)}
```

Two separate draws, one before each LLM step, so a teenager's related persona is not
forced to be another teenager. The prompt asks the model to reflect the stage without
naming an age, so the text does not fill up with "42-year-old".

**Random draws are not checkpoint-safe.** These `Map` steps give a different answer every
time they run. That is fine while the pipeline runs forward, because each step's output is
saved, but it means a resumed run cannot reproduce the draws it did not save. See
[Error handling & troubleshooting](../guides/troubleshooting.md).

### How the JSON actually happens

Neither prompt file mentions JSON. `parse_mode="json"` appends the instruction, built from
`output_columns`:

```text
Respond with valid JSON containing these fields: {"relationship_type": "<relationship_type value>", "related_persona_description": "<related_persona_description value>"}
Return only the JSON object, no additional text or markdown code fences.
```

Do not write your own — you would get both. This is a *prompted* schema, not an enforced
one: nothing stops the model returning prose, and if it does, parsing fails. See
[Structured output & parse modes](../guides/structured_output.md).

Both steps set `on_parse_error="raise"`, so a reply the parser cannot read stops the
run rather than quietly shortening the dataset. Leave it at the default `"skip"` and a
bad reply drops that record instead — ten rows in, nine rows out.

## The output

`keep_output_fields` selects and renames, and `AddUUID` adds the row id last:

```python
def keep_output_fields(record: dict) -> dict:
    return {
        "source_id": record["id"],
        "summary": record["summary"],
        "document": record["document"],
        "word_count": record["word_count"],
        "life_stage": record["life_stage"],
        "persona_description": record["persona_description"],
        "relationship_type": record["relationship_type"],
        "related_life_stage": record["related_life_stage"],
        "related_persona_description": record["related_persona_description"],
    }
```

The dataset's `id` is renamed to `source_id` here, which is what frees the name `id` for
the row's own UUID a step later. The article and its summary are kept, so every persona
can be traced to the text it came from and judged.

| Column | Where it comes from |
|---|---|
| `id` | `AddUUID` |
| `source_id` | the XSum record |
| `summary`, `document` | the XSum record |
| `word_count` | `add_word_count` |
| `life_stage` | a random draw |
| `persona_description` | Text-to-Persona |
| `relationship_type` | Persona-to-Persona |
| `related_life_stage` | a second random draw |
| `related_persona_description` | Persona-to-Persona |

## Two sinks, one run

The pipeline ends in two sinks:

```python
from datafast import Sink, Source

pipeline = (
    Source.list([{"persona_description": "A council officer."}])
    >> Sink.jsonl("personas.jsonl")
    >> Sink.hub("your-name/personas", private=True)
)
pipeline.compile()
```

A sink writes its records **and yields them through unchanged**, so a second sink receives
the same records and writes them somewhere else. One run, two destinations, no second pass
and no hand-carrying of records between scripts.

`compile()` allows any number of sinks in a row and rejects anything after them — a
transform following a sink would mean writing records before the pipeline had finished
shaping them.

`private=True` matters here. The output contains article text and inferred descriptions of
people; a public default would be the wrong one. **Set `HF_REPO_ID` to a repository you
own before running**, or the push fails on the author's:

```python
HF_REPO_ID = "<your-username-or-org>/new-persona-cookbook-dataset"
```

Unlike the other cookbooks, this push is not behind an environment variable — it is part
of the pipeline, so it runs every time.

## Running it

```bash
python examples/scripts/43_cookbook_persona_generation.py
```

You need `OPENROUTER_API_KEY`, and Hugging Face authentication through `HF_TOKEN` or a
cached `huggingface_hub` login — for reading XSum as well as for the push.

```python
build_pipeline().run(
    batch_size=1,
    checkpoint_dir=CHECKPOINT_DIR,
    resume=False,
)
```

`resume=False` starts fresh every time. Checkpoints are still written, so the work is
recoverable, but you have to opt in to reusing it by passing `resume=True`. That is the
right default while the prompts are still moving: the fingerprint that guards a checkpoint
covers step names and classes only, so an edited prompt file would resume against records
generated by the old one.

`batch_size=1` sends one call at a time. The articles are 300–500 words each, so the
prompts are large, and this pipeline is short — the safe setting costs little here.

Named steps become the checkpoint file names, which is what makes a partial rerun
possible:

```text
step_003_take_first_10.jsonl
step_005_text_to_persona.jsonl
step_007_persona_to_persona.jsonl
```

To rework the second stage while keeping the personas you already paid for:

```python
records = build_pipeline().run(
    checkpoint_dir=CHECKPOINT_DIR,
    resume=True,
    resume_from="assign_related_life_stage",
)
```

## Making it yours

| To change | Edit |
|---|---|
| the corpus | the `Source` step |
| how many rows | `n` in `Sample(n=10, strategy="first")` |
| article length | the bounds in `filter_word_count` |
| the age spread | `LIFE_STAGES` |
| persona style | the prompt files under `docs/cookbook/assets/` |
| the model | `MODEL_ID` |
| where it publishes | `HF_REPO_ID` |

Worth adding once it runs:

- **A third hop.** Persona-to-Persona can be chained again for a small network of related
  people around one article.
- **A quality filter.** A [`Score`](../reference/llm_specialized.md) step rating how
  specific each persona is, then a `Filter` on the score, removes the generic ones that
  short articles produce.
- **Stratified sampling.** With a corpus that has source metadata,
  `Sample(n=250, strategy="stratified", by="source_file")` stops one file dominating.

## Where to go next

- [Sources & Seed](../reference/sources_and_seed.md) — `Source.huggingface` and its options.
- [Data ops](../reference/data_ops.md) — `Map`, `Filter` and `AddUUID`.
- [Sample](../reference/sample.md) — every strategy, and both of its jobs.
- [Structured output & parse modes](../guides/structured_output.md) — what `parse_mode="json"` guarantees.
- [Sinks](../reference/sinks.md) — `Sink.jsonl`, `Sink.hub` and chaining them.
- [Text classification](text_classification.md) — the same shape, seeded instead of sourced.
