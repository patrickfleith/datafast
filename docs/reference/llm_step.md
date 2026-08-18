# LLMStep

`LLMStep` is the general LLM step. For each record it fills a prompt template, sends it
to a served model, and writes the answer back as one or more new columns.

The [specialized LLM steps](llm_specialized.md) write the prompt for you. `LLMStep` is
the one where you write it yourself.

## At a glance

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `prompt` | `str \| Path \| list \| Sample` | required | prompt template, or a path to a file holding one |
| `input_columns` | `list[str]` | required | columns injected into the template |
| `model` | `ServedModel \| list \| Sample` | required | the served model(s) to call |
| `output_column` | `str` | `"generated"` | column the answer is written to (`text` mode) |
| `output_columns` | `list[str] \| None` | `None` | columns the answer is split into (`json`/`xml` mode) |
| `parse_mode` | `str` | `"text"` | `"text"`, `"json"` or `"xml"` |
| `num_outputs` | `int` | `1` | answers per prompt × model × language |
| `language` | `str \| list \| dict \| Sample \| None` | `None` | language(s) to generate in |
| `forward_columns` | `list[str] \| None` | `None` | keep only these input columns |
| `exclude_columns` | `list[str] \| None` | `None` | drop these input columns |
| `skip_if` | `Callable[[Record], bool] \| None` | `None` | return `True` to skip a record |
| `system_prompt` | `str \| None` | `None` | system message put before the prompt |
| `temperature` | `float \| None` | `None` | accepted, but not applied — see below |
| `max_tokens` | `int \| None` | `None` | accepted, but not applied — see below |
| `on_parse_error` | `str` | `"skip"` | `"skip"` or `"raise"` when parsing fails |

Everything after `model` is keyword-only.

## The prompt

The prompt template uses `{column}` placeholders. Each column you name in it must be
listed in `input_columns`, and each column in `input_columns` is looked up on the record
and injected.

```python
from datafast import LLMStep, openai

step = LLMStep(
    prompt="Write one short question about {topic}",
    input_columns=["topic"],
    output_column="question",
    model=openai(),
)
```

A placeholder that is not in `input_columns` raises `KeyError` when the step runs. A
literal brace in the prompt must be doubled — `{{` and `}}` — or it is read as a
placeholder.

`prompt` also accepts a file path, as a `str` or a `Path`. The file is read on first use
and cached.

```python
from pathlib import Path

from datafast import LLMStep, openai

step = LLMStep(
    prompt=Path("prompts/summarize.txt"),
    input_columns=["text"],
    output_column="summary",
    model=openai(),
)
```

The path is only treated as a file **if that file exists**. If it does not, the value is
used as the prompt text itself, so a mistyped path becomes a prompt that reads
`prompts/summarize.txt`. Check the path before you run.

## The output

`parse_mode` decides how one raw answer becomes columns.

| `parse_mode` | What it does | Columns to declare |
|---|---|---|
| `"text"` | writes the whole answer, stripped, to one column | `output_column` |
| `"json"` | reads the answer as a JSON object and takes one column per key | `output_columns` |
| `"xml"` | reads `<name>...</name>` tags and takes one column per tag | `output_columns` |

`json` and `xml` require `output_columns`; without it the constructor raises
`ValueError`. Both modes also append format instructions to the end of your prompt,
naming the columns they expect, so you do not have to describe the format yourself.

```python
from datafast import LLMStep, openai

step = LLMStep(
    prompt="Write a question and its answer about {topic}",
    input_columns=["topic"],
    output_columns=["question", "answer"],
    parse_mode="json",
    model=openai(),
)
```

In `json` mode a missing key gives an empty string in that column, with a warning. In
`xml` mode a missing tag does the same; tag matching ignores case. Answers wrapped in
markdown code fences are unwrapped before the JSON is read.

`output_column` and `output_columns` are not alternatives to each other: if you pass
`output_columns` in `text` mode, the answer goes into its **first** entry and the rest are
never written, while `output_column` is ignored.

## How many LLM calls

A step fans out. For every input record it makes one call per combination of:

```text
prompts × served models × languages × num_outputs
```

Each call produces one output record, so the step does not preserve record counts —
it multiplies them.

| Records in | Prompts | Served models | Languages | `num_outputs` | Calls and records out |
|---|---|---|---|---|---|
| 100 | 1 | 1 | — | 1 | 100 |
| 100 | 2 | 1 | — | 1 | 200 |
| 100 | 2 | 2 | — | 1 | 400 |
| 100 | 2 | 2 | 3 | 2 | 2400 |

No `language` counts as one, not zero. Multiply before you run: this is the number you
pay for.

```python
from datafast import LLMStep, openai

step = LLMStep(
    prompt=["Summarize: {text}", "List the key points of: {text}"],
    input_columns=["text"],
    output_column="result",
    model=[openai(), openai("gpt-5.4-mini")],
    num_outputs=2,
)
```

Eight calls per record: 2 prompts × 2 served models × 1 language × 2 outputs.

## Languages

`language` adds a language axis to that fan-out and gives the template two extra
placeholders: `{language}` for the code and `{language_name}` for the name.

| Value | Meaning |
|---|---|
| `None` | no language axis, no placeholders |
| `"fr"` | one language; code and name are both `"fr"` |
| `["en", "fr"]` | two languages; code and name are the same for each |
| `{"en": "English", "fr": "French"}` | codes as keys, names as values |

```python
from datafast import LLMStep, openai

step = LLMStep(
    prompt="Write one sentence about {topic} in {language_name}",
    input_columns=["topic"],
    output_column="sentence",
    model=openai(),
    language={"en": "English", "fr": "French"},
)
```

Use the dict form when the prompt should name the language, since a model reads
`French` better than `fr`.

## Which columns come through

By default every column of the input record is copied to the output record, then the
generated columns are added on top.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `forward_columns` | `list[str] \| None` | `None` | keep only these input columns |
| `exclude_columns` | `list[str] \| None` | `None` | keep everything except these |

```python
from datafast import LLMStep, openai

step = LLMStep(
    prompt="Summarize: {text}",
    input_columns=["text"],
    output_column="summary",
    model=openai(),
    forward_columns=["doc_id"],
)
```

`forward_columns` wins if both are given; `exclude_columns` is then ignored. A generated
column with the same name as an input column overwrites it.

## Skipping records and system prompts

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `skip_if` | `Callable[[Record], bool] \| None` | `None` | called per record; `True` means no call |
| `system_prompt` | `str \| None` | `None` | sent as a system message before the prompt |

```python
from datafast import LLMStep, openai

step = LLMStep(
    prompt="Summarize: {text}",
    input_columns=["text"],
    output_column="summary",
    model=openai(),
    skip_if=lambda record: len(record["text"]) < 100,
    system_prompt="You are a concise technical writer.",
)
```

A skipped record is **dropped**, not passed through. If you want to keep it, split the
pipeline with `Filter` and `Concat` instead.

## Metadata columns

Every output record carries the step's own columns, all starting with `_`.

| Column | Written when | Value |
|---|---|---|
| `_model` | always | the model id of the served model that answered |
| `_prompt_index` | `prompt` is a list | position of the prompt used, from `0` |
| `_language` | `language` is set | the language code |

With a single prompt and no language, the only added column is `_model`. All three names
are reserved: `compile()` treats them as present downstream, so do not use them for your
own data.

## Temperature and max tokens

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `temperature` | `float \| None` | `None` | stored on the step, never sent |
| `max_tokens` | `int \| None` | `None` | stored on the step, never sent |

These two are accepted by the constructor but **not applied**. The step calls
`ServedModel.generate()`, which takes no such arguments, so the served model's own
settings decide temperature and length. Set them where the served model is built — see
[Served models](served_models.md).

## Errors

`on_parse_error` is `"skip"` or `"raise"`; anything else raises `ValueError` at
construction.

It only takes effect when the step runs its own `process()`. Under `Pipeline.run()` the
runner logs a failed call and carries on regardless, so a run with
`on_parse_error="raise"` and unparseable answers finishes with fewer records instead of
raising. Compare the record count with the number you expected.

## Things worth knowing

- **Count the calls before you run.** Records × prompts × served models × languages ×
  `num_outputs`. Three small choices multiply into a large bill.
- **A mistyped prompt path becomes the prompt.** No file, no error — the string is used
  as-is.
- **`json` and `xml` add format instructions** to the end of your prompt. Do not write
  your own on top of them.
- **A missing key or tag is an empty string**, not a failure. The record is still yielded.
- **`output_columns` in `text` mode** writes to its first entry only.
- **`skip_if` drops records** rather than letting them pass.
- **`_prompt_index` is conditional.** With one prompt it never appears; with a `Sample`
  of prompts it appears on some records and not others. Pass a list for a stable schema.
- **`temperature` and `max_tokens` do nothing here.** Configure the served model.

## Where to go next

- [Specialized LLM steps](llm_specialized.md) — `Classify`, `Score`, `Compare`, `Rewrite`, `Extract`.
- [LLM Steps guide](../guides/llm_steps.md) — the steps in context.
- [Served models](served_models.md) — building the models this step calls.
- [Sources & Seed](sources_and_seed.md) — where the records come from.
- [Checkpointing](../guides/checkpointing.md) — resuming a run without paying twice.
- [Glossary](../glossary.md) — record, column, parse mode, served model.
- [API reference](../api.md) — the generated signature.
