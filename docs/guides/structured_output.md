# Structured output & parse modes

Two different things in datafast turn an LLM reply into named columns, and they are easy
to confuse because both involve JSON.

**Parse mode** is the step-level choice of how one raw LLM response is split into named
columns (`text`, `json`, `xml`). It runs *after* the reply comes back.

**Structured output** is the provider constraining a whole response to a schema
(`json_schema`, `json_object`, or `prompted_json`). It is a capability of the served
model, and it acts *before* the reply is written.

| | Parse mode | Structured output |
|---|---|---|
| Where you set it | `parse_mode=` on a step | `response_format=` on a served-model call |
| When it acts | after the reply arrives | while the reply is generated |
| What it does | splits text into columns | constrains what the model may write |
| What it gives you | a dict of strings | a validated Pydantic object |
| If the model misbehaves | parsing fails | the provider prevents it |

The short version: **inside a pipeline you use parse mode; calling a served model
directly is how you reach structured output.** The rest of this page explains both and
where they meet.

## Parse mode

Every LLM step takes `parse_mode`. There are three, and `"text"` is the default.

### `parse_mode="text"`

The whole reply, stripped of surrounding whitespace, goes into one column named by
`output_column` (default `"generated"`).

```python
from datafast import LLMStep, Source, openai

step = LLMStep(
    prompt="Summarize: {text}",
    input_columns=["text"],
    output_column="summary",
    model=openai(),
)
```

Nothing is added to your prompt and nothing can fail to parse. Use it whenever you want
one piece of prose per record.

### `parse_mode="json"`

The reply is read as a JSON object, and each name in `output_columns` becomes a column.

```python
from datafast import LLMStep, openai

step = LLMStep(
    prompt="Write a question and answer about: {topic}",
    input_columns=["topic"],
    output_columns=["question", "answer"],
    parse_mode="json",
    model=openai(),
)
```

`output_columns` is required here — building the step without it raises `ValueError`
straight away.

Three details decide whether this works in practice:

- **Markdown fences are stripped.** A reply wrapped in ```` ```json ... ``` ```` parses
  fine.
- **A missing key becomes an empty string**, with a warning in the log. It does not fail
  the record.
- **Non-string values are converted to strings.** `{"score": 4}` gives `"4"`. Columns
  from a parse mode are always strings.

If the reply is not valid JSON at all, parsing raises and the record is handled according
to `on_parse_error`, which defaults to `"skip"` and drops the record. See
[Error handling & troubleshooting](troubleshooting.md).

### `parse_mode="xml"`

Each name in `output_columns` is read out of a matching tag pair.

```python
from datafast import LLMStep, openai

step = LLMStep(
    prompt="Write a question and answer about: {topic}",
    input_columns=["topic"],
    output_columns=["question", "answer"],
    parse_mode="xml",
    model=openai(),
)
```

The step looks for `<question>...</question>` and `<answer>...</answer>`. Tag matching
**ignores case** and spans newlines, and text outside the tags is ignored — so a model
that adds "Here you go:" before the tags still parses.

A missing tag becomes an empty string with a warning, exactly like a missing JSON key.
Nothing about XML mode can raise, which makes it the forgiving choice for models that
struggle to emit clean JSON: you get partial results instead of a dropped record.

### The instructions added to your prompt

`json` and `xml` mode append format instructions to every prompt they send. You do not
write them and you cannot see them in your own prompt text, so it is worth knowing they
are there.

For `output_columns=["question", "answer"]` in `json` mode, the model receives your
prompt followed by:

```text
Respond with valid JSON containing these fields: {"question": "<question value>", "answer": "<answer value>"}
Return only the JSON object, no additional text or markdown code fences.
```

In `xml` mode it receives a tag skeleton instead. In `text` mode nothing is added.

This is also why you should not repeat the format request in your own prompt. Saying it
twice does not help, and giving field names that disagree with `output_columns` is a good
way to get empty columns.

## Structured output

`ServedModel.generate()` takes a `response_format` — a Pydantic model. You get back an
instance of that model rather than a string:

```python
from pydantic import BaseModel

from datafast import openai


class Question(BaseModel):
    question: str
    answer: str


model = openai()
result = model.generate("Ask me something about physics.", response_format=Question)
# result.question, result.answer  — a validated object, not a dict of strings
```

What actually reaches the provider depends on the served model's declared capability.

### The four modes

| Mode | What datafast sends | What you can rely on |
|---|---|---|
| `json_schema` | your Pydantic model as the request's schema | the provider enforces the shape |
| `json_object` | `{"type": "json_object"}` plus JSON instructions in the prompt | valid JSON, but not your fields |
| `prompted_json` | JSON instructions in the prompt, and a warning | nothing — the model may still stray |
| `none` | nothing; the call raises `ValueError` | structured output is unavailable |

In all four cases datafast validates the reply against your Pydantic model afterwards, so
you either get a correct object or a `ValueError` that shows the validation error and the
first 200 characters received. The mode decides how likely the failure is, not whether it
is caught.

`prompted_json` also raises a `UserWarning` naming the provider and model, so a served
model quietly falling back to prompting is visible rather than silent.

### What your model gets

Every provider datafast ships a profile for declares `json_schema`:

| Served model | Mode |
|---|---|
| `openai()`, `anthropic()`, `gemini()`, `mistral()`, `openrouter()`, `ollama()` | `json_schema` |
| `openai_compatible()` with `provider_id="vllm"` or `"llamacpp"` | `json_schema` |
| `openai_compatible()` with any other `provider_id` | `prompted_json` |

The last row is the conservative default: a self-hosted server datafast has no profile
for is assumed to support only what the OpenAI wire format itself guarantees. If your
server does enforce schemas, pass your own capabilities rather than living with prompted
JSON — see [Served models](../reference/served_models.md).

`none` is not used by any shipped profile. You will only meet it if you declare it
yourself.

### The Responses endpoint is stricter

OpenAI reasoning models use the Responses API, where there is no prompt-level fallback.
Structured output there requires `json_schema`; anything else raises `ValueError` rather
than quietly degrading.

## Where the two meet

They mostly do not, and that is the thing to know: **no step in datafast passes
`response_format`.** `LLMStep` sends your prompt plus format instructions and parses what
comes back. The specialized steps (`Classify`, `Score`, `Compare`, `Rewrite`, `Extract`)
ask for JSON in their own prompts and read it themselves.

So a pipeline against a `json_schema` provider still relies on the model choosing to
answer in JSON. The provider could have guaranteed it, and datafast does not ask.

In practice this matters most with a small local model and `parse_mode="json"`. Two ways
to reduce the pain:

- Use `parse_mode="xml"`, which never raises and gives you partial columns.
- Do the strict part yourself: call the served model directly with `response_format=`,
  and put that call in a `Map` step.

## Choosing

| You want | Use |
|---|---|
| one piece of prose per record | `parse_mode="text"` |
| several fields, capable model | `parse_mode="json"` |
| several fields, small or local model | `parse_mode="xml"` |
| a guarantee, not a request | a direct call with `response_format=` |

## Things worth knowing

- **Parse-mode columns are always strings.** Cast numbers in a `Map` step afterwards.
- **`json` and `xml` need `output_columns`.** `text` uses `output_column`, singular.
- **Missing fields are empty, not fatal** — in both `json` and `xml` mode.
- **Only invalid JSON raises.** XML mode has no failure path.
- **Format instructions are appended for you.** Do not write your own as well.
- **A `prompted_json` warning is worth reading.** It means the schema is a request.

## Where to go next

- [LLM Steps](llm_steps.md) — every other option on an LLM step.
- [Served models](../reference/served_models.md) — declaring capabilities yourself.
- [LLM step reference](../reference/llm_step.md) — the full parameter list.
- [Error handling & troubleshooting](troubleshooting.md) — what to do when a reply will not parse.
- [Glossary](../glossary.md) — the exact wording of parse mode and structured output.
