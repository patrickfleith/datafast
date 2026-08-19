# Calling a served model directly

A served model is a provider and a model together, with its configuration — the thing you
construct and call. Most of the time a pipeline calls it for you. Sometimes you want to
call it yourself.

```python
from datafast import openai

model = openai()
answer = model.generate("Name three moons of Saturn.")
print(answer)
```

## When to call directly

| You want | Use |
|---|---|
| a dataset — many records, written out | an [LLM step](../reference/llm_step.md) in a pipeline |
| one answer, right now | `generate()` |
| the reasoning trace, or the provider's raw reply | `generate_response()` |
| a quick check that a model and key work | `generate()` |

Inside a pipeline, an LLM step calls these same methods and adds what a dataset needs:
batching, checkpointing and resume, the `_model` column, and a prompt template filled from
each record. Calling directly gives you none of that, which is exactly the point when you
only want one answer.

## The four methods

| Method | Takes | Returns |
|---|---|---|
| `generate()` | a prompt, or messages | the text (or your Pydantic object) |
| `generate_batch()` | a list of message lists | a list of the same |
| `generate_response()` | a prompt, or messages | a `NormalizedResponse` |
| `generate_batch_response()` | a list of message lists | a list of `NormalizedResponse` |

The split is along two lines: **one input or many**, and **the text or everything**.

### `generate()`

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `prompt` | `str \| list[str] \| None` | `None` | one prompt, or several |
| `messages` | `Messages \| list[Messages] \| None` | `None` | a conversation, or several |
| `response_format` | `type[BaseModel] \| None` | `None` | a schema to fill; see [Structured output](structured_output.md) |
| `metadata` | `dict \| None` | `None` | forwarded to the provider for tracing |
| `previous_response_id` | `str \| None` | `None` | continue an earlier response (Responses endpoint) |

Pass `prompt` **or** `messages`, never both — and never neither. Either mistake raises
`ValueError` at once, before any request goes out.

What you get back mirrors what you put in:

| You pass | You get |
|---|---|
| `generate("one prompt")` | a string |
| `generate(["a", "b"])` | a list of two strings |
| `generate(messages=[{...}])` | a string |
| `generate(messages=[[{...}], [{...}]])` | a list of two strings |

A single conversation is a list of message dicts; a batch is a list of those. Datafast
tells them apart by looking inside, so you do not declare which you meant.

```python
from datafast import openai

model = openai()

one = model.generate("Summarize photosynthesis in one sentence.")
many = model.generate(["Define entropy.", "Define enthalpy."])
chat = model.generate(
    messages=[
        {"role": "system", "content": "You answer in one short sentence."},
        {"role": "user", "content": "Why is the sky blue?"},
    ]
)
```

### `generate_batch()`

The same as passing a list to `generate()`, but it only takes pre-built message lists and
always returns a list — including an empty one for empty input.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `messages` | `list[Messages]` | required | one conversation per item |
| `response_format` | `type[BaseModel] \| None` | `None` | a schema to fill |
| `metadata` | `list \| dict \| None` | `None` | one entry per item, or one shared |
| `previous_response_ids` | `list[str \| None] \| None` | `None` | one per item |

Order is preserved: result `i` answers message list `i`. `metadata` and
`previous_response_ids` must be the same length as `messages` or you get a `ValueError`.

Reach for this over `generate()` when you already have message lists in hand and want the
list-in, list-out shape without thinking about the single-input special case.

### `generate_response()`

Same inputs as `generate()`, minus `response_format`. Instead of the text it returns a
`NormalizedResponse`, which carries the text *and* everything else the provider sent.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `prompt` | `str \| list[str] \| None` | `None` | one prompt, or several |
| `messages` | `Messages \| list[Messages] \| None` | `None` | a conversation, or several |
| `metadata` | `dict \| None` | `None` | forwarded to the provider for tracing |
| `previous_response_id` | `str \| None` | `None` | continue an earlier response |

```python
from datafast import anthropic

model = anthropic(thinking=True)
response = model.generate_response("Is 91 prime? Think it through.")

print(response.text)
print(response.reasoning_content)
```

There is no `response_format` here. You can have a validated Pydantic object or the
response metadata, not both in one call.

### `generate_batch_response()`

`generate_batch()` and `generate_response()` combined: a list of message lists in, a list
of `NormalizedResponse` out.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `messages` | `list[Messages]` | required | one conversation per item |
| `metadata` | `list \| dict \| None` | `None` | one entry per item, or one shared |
| `previous_response_ids` | `list[str \| None] \| None` | `None` | one per item |

## `NormalizedResponse`

Providers disagree about where they put things. This dataclass is datafast's single shape
for a reply.

| Field | Type | Holds |
|---|---|---|
| `text` | `str` | the answer |
| `raw` | `Any` | the provider's own response object, untouched |
| `reasoning_content` | `str \| None` | the reasoning trace, as text |
| `thinking_blocks` | `list[dict]` | the trace as structured blocks |
| `images` | `list[dict]` | images the model produced |
| `audio` | `dict \| None` | audio the model produced |
| `output_items` | `list[dict]` | the Responses API's own item list |

Two fields depend on which endpoint the served model uses:

| Field | Chat endpoint | Responses endpoint |
|---|---|---|
| `thinking_blocks` | filled when present | always empty |
| `output_items` | always empty | filled |

So an empty `thinking_blocks` does not mean the model did not reason — check
`reasoning_content` too. Everything a field cannot express is still in `raw`, which is the
provider's object exactly as it arrived.

## Batching is not always batching

When you pass several inputs, what happens depends on the served model:

- Providers whose profile declares native batching (`anthropic()`, `gemini()`,
  `mistral()`, `openrouter()`, and OpenAI's chat models) send one batched request.
- Everything else runs **bounded parallel single requests**, capped by the served model's
  `max_concurrent`, and raises a `UserWarning` saying so.

That warning is informational, not a failure — you still get every answer, in order. It
does fire on OpenAI's default model, which is a reasoning model on the Responses endpoint
where native batching is unavailable.

Batching is per call, not per second. Rate limiting is a separate setting on the served
model; see [Pipelines & execution](pipelines_and_execution.md).

## Errors

Every one of the four methods follows the same contract:

- **`ValueError`** means you asked for something impossible — both `prompt` and
  `messages`, a modality the model does not accept, a reply that failed schema
  validation. It reaches you unchanged, with a message naming the problem.
- **`RuntimeError`** wraps everything else — a network failure, an authentication error, a
  provider rejecting the request. The message names the provider and includes the original
  traceback.

Retries come first, but only for failures worth retrying: a rate limit, a connection
error, a timeout, or the provider's own 5xx. Those are attempted `retry_limit` more times
with a growing delay before the error reaches you. Anything else — a bad key, a malformed
request — raises on the first failure, because trying it again would fail the same way.

## Things worth knowing

- **Construction does not check your key.** A missing key surfaces on the first call.
- **`metadata` is for tracing.** It is forwarded to the provider and is what Langfuse
  reads; it does not change the answer.
- **Order is always preserved** for every batch method.
- **An empty batch is not an error.** `generate_batch([])` returns `[]` without calling
  anything.
- **`raw` is your escape hatch.** Anything datafast does not normalize is still there.
- **`previous_response_id` only means something on the Responses endpoint.**

Runnable versions of all of this ship with the source, under `examples/providers/` — one
directory per provider, covering prompts, batches, structured output, image input,
response metadata, timeouts and rate limits. They make real calls, so read before running.

## Where to go next

- [LLM step](../reference/llm_step.md) — the same calls, inside a pipeline.
- [Structured output](structured_output.md) — `response_format` in full.
- [Multimodal input](multimodal_input.md) — sending images, audio and files.
- [Served models](../reference/served_models.md) — every construction setting.
- [Glossary](../glossary.md) — served model, provider, transport, capabilities.
