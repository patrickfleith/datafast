# Persona generation

Get personas from a corpus of real articles. Then, for each persona, get a second person
who has a relation to the first.

The seed of synthetic data controls how different the data is. If you write a list of
personas manually, the personas become similar quickly: they all sound like the writer.
This procedure starts from **real text**. For each article, the LLM gives 1 possible
reader. Real articles have many different subjects, registers and assumptions. A list
that you write manually does not have this range. The personas get the same range as the
articles.

Then the pipeline does the same procedure 1 more time. For each persona, the pipeline
gets a second person who has a close relation to the first. Thus 1 article gives 2
related people.

This procedure comes from the Persona Hub paper and its Text-to-Persona and
Persona-to-Persona methods. This page does the procedure again with datafast. It does not
use code from Persona Hub. The paper tells you that its published prompts are simplified
versions of the prompts in the experiment. Thus the prompt templates on this page are
adaptations.

- **Script:** `examples/scripts/43_cookbook_persona_generation.py`
- **Prompt templates:** [asset index](assets/index.md), 3 variants for each step
- **Output:** `examples/outputs/43_persona_cookbook.jsonl` **and** a private Hub dataset
- **Checkpoints:** `examples/checkpoints/43_persona_cookbook`
- **10 records** by default, with 2 LLM calls for each record

## The pipeline

The pipeline has 12 steps in 4 stages: prepare the corpus, get a persona, get a related
persona, publish.

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

A pipeline with a seed multiplies the record count. This pipeline does not. Each step
keeps the record count, or makes it smaller:

| Step | Effect on the count |
|---|---|
| `add_word_count` | no change, adds `word_count` |
| `filter_word_count` | removes the articles that are not 300 to 500 words long |
| `take_first_10` | keeps `n=10` records |
| `assign_life_stage` | no change, adds `life_stage` |
| `text_to_persona` | no change, 1 call for each record |
| `assign_related_life_stage` | no change, adds `related_life_stage` |
| `persona_to_persona` | no change, 1 call for each record |
| `keep_output_fields` | no change, removes columns |
| `add_uuid` | no change, adds `id` |
| `Sink.jsonl`, `Sink.hub` | no change, the 2 sinks write the records and send them through |

## Prepare the corpus

The source is [XSum](https://huggingface.co/datasets/EdinburghNLP/xsum). XSum contains
BBC articles, each with a summary of 1 sentence. The source loads it from the Hub:

```python
from datafast import Source

source = Source.huggingface(
    "xsum",
    split="validation",
    columns=["id", "document", "summary"],
)
```

`columns=` keeps only the necessary columns. The source keeps the `id` of the dataset.
Without the `id`, you cannot find the article that a persona comes from.

Then 3 steps make the corpus smaller. The order of these steps is important.

**Measure, then filter.** `Filter` reads columns. It does not calculate them. Thus a
`Map` adds `word_count` first:

```python
def add_word_count(record: dict) -> dict:
    return {**record, "word_count": len(record["document"].split())}
```

**Filter on length.** Keep articles of 300 to 500 words. A shorter article does not give
sufficient data for a specific reader, and the model gives general personas. A longer
article usually has many topics, and the persona becomes unclear. The length also has
the largest effect on the token count, because each article becomes prompt tokens.

**Then keep a small number.** `Sample(n=10, strategy="first")` keeps the first 10
records that the filter keeps. `"first"` is not random. Each run gives the same 10
articles, so you can compare the results when you change a prompt template. When the
output is correct, increase `n`.

A step name is not only a label. It becomes the name of the checkpoint file and a value
for `resume_from`. Thus the file for `take_first_10` on the disk is
`step_003_take_first_10.jsonl`. If you change the number, change the name too.

Filter first, then sample. This order uses fewer articles. If you sample 10 articles and
then filter them, only 3 can stay.

For a local corpus, replace the source and keep all the other steps:

```python
from datafast import Map, Source

pipeline = Source.file("data/articles.jsonl") >> Map(lambda r: {**r, "document": r["text"]})
```

The other steps use only a `document` column.

## Two LLM steps

The 2 steps are usual `LLMStep`s with the same shape. Each step reads a column, asks for
JSON and writes columns with names.

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

The prompt template asks for **a reader of the text, not its subject**. For an article
about a flood, the persona must be a council officer or a local resident. It must not be
the mayor that the article quotes. The prompt template also asks for 1 specific person,
not a category. It tells the model not to quote the article and not to refer to the
article. Thus the persona has no dependency on the article.

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

This step reads only `persona_description`. It does not read the article. The second
person comes from the first person, not from the text. The step gives **2** columns: the
relation and the new persona.

The relation makes the pair useful. Two personas without a relation are only 2 records.
You can ask a persona and their coach the same question. They can then give different
answers, and each answer has a reason.

### Change the prompt template at random

Each of the 2 steps gives a `Sample` to `prompt`, not a prompt template:

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

`Sample` has 2 different functions:

- With `items`, it selects values. You can give it to `prompt`, `model` or `language`. It
  selects 1 time **for each record**.
- Without `items`, it is a pipeline step that samples records.
  `Sample(n=10, strategy="first")` in the same pipeline is this type.

The 3 variants of the prompt template ask for the same thing with different words. If 1
prompt template goes to 1000 articles, you get 1000 personas with the same sentence
structure. The variants prevent this. A list of models does the same thing for the model.

The life stages also change at random, but with usual Python:

```python
import random

LIFE_STAGES = ["a teenager", "a young adult", "an adult (30s/40s)"]

def assign_life_stage(record: dict) -> dict:
    return {**record, "life_stage": random.choice(LIFE_STAGES)}
```

There are 2 different random selections, 1 before each LLM step. Thus the related
persona of a teenager can be a different life stage. The prompt template tells the model
to show the life stage without an age. Thus the text does not contain "42-year-old" many
times.

!!! warning "Random selections are not safe with checkpoints"

    These `Map` steps give a different result each time that they run. When the pipeline
    runs from start to end, this is not a problem, because the runner records the output
    of each step. But if you resume a run, the runner cannot make the same random
    selections again for the steps that it did not record. Refer to
    [Error handling & troubleshooting](../guides/troubleshooting.md).

### How the step asks for JSON

The 2 prompt files do not contain the word JSON. `parse_mode="json"` adds the instruction
at the end of the prompt. The step makes the instruction from `output_columns`:

```text
Respond with valid JSON containing these fields: {"relationship_type": "<relationship_type value>", "related_persona_description": "<related_persona_description value>"}
Return only the JSON object, no additional text or markdown code fences.
```

Do not write a JSON instruction in the prompt file. If you do, the model gets 2
instructions. This schema is only an instruction in the prompt. The provider does not
apply it. The model can return text that is not JSON, and then the parse fails. Refer to
[Structured output & parse modes](../guides/structured_output.md).

The 2 steps set `on_parse_error="raise"`. If the step cannot parse an answer, the run
stops. The dataset does not become smaller without a message. With the default value
`"skip"`, the step removes the record of a bad answer: 10 records in, 9 records out.

## The output

`keep_output_fields` selects columns and changes their names. Then `AddUUID` adds the
record `id`:

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

This step changes the name of the dataset `id` to `source_id`. Thus the next step can
use the name `id` for the UUID of the record. The step keeps the article and its summary.
Thus you can find the text that each persona comes from, and you can judge the persona.

| Column | Origin |
|---|---|
| `id` | `AddUUID` |
| `source_id` | the XSum record |
| `summary`, `document` | the XSum record |
| `word_count` | `add_word_count` |
| `life_stage` | a random selection |
| `persona_description` | Text-to-Persona |
| `relationship_type` | Persona-to-Persona |
| `related_life_stage` | a second random selection |
| `related_persona_description` | Persona-to-Persona |

## Two sinks in 1 run

The pipeline has 2 sinks at the end:

```python
from datafast import Sink, Source

pipeline = (
    Source.list([{"persona_description": "A council officer."}])
    >> Sink.jsonl("personas.jsonl")
    >> Sink.hub("your-name/personas", private=True)
)
pipeline.compile()
```

A sink writes its records **and sends them through with no change**. Thus a second sink
gets the same records and writes them to a different location. One run writes to 2
destinations. A second run is not necessary, and you do not move records between
scripts.

`compile()` accepts many sinks in a sequence. It does not accept a step after them. A
transform after a sink causes the sink to write records before the pipeline has the
final records.

`private=True` is important here. The output contains article text and descriptions of
people. A public dataset is not a correct default for this data.

**Before you run the script, set `HF_REPO_ID` to a repository that you own.** If you do
not, the push to the repository of the author fails:

```python
HF_REPO_ID = "<your-username-or-org>/new-persona-cookbook-dataset"
```

In the other cookbooks, an environment variable controls the push. In this cookbook, the
push is a step in the pipeline, so it occurs at each run.

## Run the script

```bash
python examples/scripts/43_cookbook_persona_generation.py
```

You must have `OPENROUTER_API_KEY`. You must also log in to Hugging Face with `HF_TOKEN`
or with a `huggingface_hub` login. The script uses this login to read XSum and to push
the dataset.

```python
build_pipeline().run(
    batch_size=1,
    checkpoint_dir=CHECKPOINT_DIR,
    resume=False,
)
```

With `resume=False`, each run starts again from the beginning. The runner continues to
write checkpoints, so you can get the work back. To use the checkpoints, set
`resume=True`. While you change the prompt templates, `resume=False` is the correct
default. The checkpoint fingerprint includes only the step names and classes. Thus a run
with an edited prompt file can resume with records from the old prompt file.

`batch_size=1` sends 1 call at a time. Each article has 300 to 500 words, so the prompts
are large. The pipeline is short, so this safe setting does not make it much slower.

The step names become the names of the checkpoint files. Thus you can run only a part of
the pipeline again:

```text
step_003_take_first_10.jsonl
step_005_text_to_persona.jsonl
step_007_persona_to_persona.jsonl
```

To change the second stage and keep the personas that you have:

```python
records = build_pipeline().run(
    checkpoint_dir=CHECKPOINT_DIR,
    resume=True,
    resume_from="assign_related_life_stage",
)
```

## Change the dataset

| To change | Edit |
|---|---|
| the corpus | the `Source` step |
| the number of records | `n` in `Sample(n=10, strategy="first")` |
| the length of the articles | the limits in `filter_word_count` |
| the range of ages | `LIFE_STAGES` |
| the style of the personas | the prompt files in `docs/cookbook/assets/` |
| the model | `MODEL_ID` |
| the destination of the push | `HF_REPO_ID` |

When the script runs correctly, you can add these steps:

- **A third persona.** Add 1 more Persona-to-Persona step. You then get a small group of
  related people for each article.
- **A quality filter.** Add a [`Score`](../reference/llm_specialized.md) step that gives
  a score for how specific each persona is. Then add a `Filter` on the score. This
  removes the general personas that short articles give.
- **Stratified sampling.** If your corpus has metadata about the source, use
  `Sample(n=250, strategy="stratified", by="source_file")`. This prevents too many
  records from 1 file.

## Related pages

- [Source](../reference/source.md): `Source.huggingface` and its options.
- [Data ops](../reference/data_ops.md): `Map`, `Filter` and `AddUUID`.
- [Sample](../reference/sample.md): all the strategies, and the 2 functions of `Sample`.
- [Structured output & parse modes](../guides/structured_output.md): what `parse_mode="json"` does and does not do.
- [Sinks](../reference/sinks.md): `Sink.jsonl`, `Sink.hub` and how to put sinks in a sequence.
- [Text classification](text_classification.md): the same shape, with a seed and not a source.
