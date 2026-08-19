# Error handling & troubleshooting

Failures in datafast come in two kinds, and the second kind is the expensive one.

**Loud failures** raise. A pipeline that is built wrong, a missing API key, a prompt
placeholder with no matching column — these stop before or during the run and tell you
what happened.

**Quiet failures** do not raise. A record whose LLM call failed is logged and dropped,
and the run finishes normally with fewer records than you asked for. Nothing in the
return value says so.

This page covers both, starting with the errors you can read and ending with the ones you
have to go looking for.

## The two exceptions datafast defines

| Exception | Raised by | Means |
|---|---|---|
| `PipelineValidationError` | `Pipeline.compile()` | the pipeline is built wrong; nothing ran |
| `PipelineChangedError` | `Pipeline.run(resume=True)` | the checkpoint belongs to a different pipeline |

Everything else you will see is a plain `ValueError`, `RuntimeError`, `KeyError` or
`ImportError`.

Both are on the top-level package:

```python
from datafast import PipelineChangedError, PipelineValidationError
```

## Before the run: `compile()`

`compile()` checks the shape of a pipeline without running it. `run()` calls it for you,
so these errors arrive before a single LLM call is paid for. Call it yourself to check a
pipeline you are still writing:

```python
from datafast import Map, Sink, Source

pipeline = Source.list([{"text": "hello"}]) >> Map(lambda r: r) >> Sink.list()
pipeline.compile()
```

It raises on the first problem it finds:

| Message begins | Cause | Fix |
|---|---|---|
| `Pipeline is empty.` | no steps | build the chain with `>>` |
| `Pipeline must start with a source` | the first step is a transform | put a `Source` or a `Seed` combiner first |
| `Source '...' at position N discards upstream records` | a second source mid-chain | use `Concat` to combine two sources |
| `Step '...' comes after the sink at position N` | a transform after a sink | move it before the sinks; several sinks may be chained, but nothing may follow them |
| `Step '...' references column(s) [...] that are not available` | a step reads a column no earlier step produces | check the spelling against the *Available columns* the message lists |
| `Branch at position N is never closed by a JoinBranches` | a `Branch` with no `JoinBranches` | close it |
| `JoinBranches at position N has no matching Branch` | the reverse | open a `Branch` first |
| `nesting a Branch inside a branch path is not supported` | a `Branch` inside a branch path | close the outer `Branch` first |
| `Sink '...' is not allowed inside Branch path '...'` | a sink inside a branch path | put the sink after the `JoinBranches` |
| `Concat source N must start with a source` | a `Concat` argument that is not a source | give each one its own source |

Column checking is deliberately conservative. `Map`, `FlatMap`, `Group`, `Pair`, `Join`
and `Concat` reshape records in ways `compile()` cannot read, so after one of them the
schema is unknown and no column is reported missing. A clean `compile()` means the
structure is right, not that every column reference is.

## During the run: what stops and what does not

This is the most important distinction on the page.

| What fails | What happens |
|---|---|
| a normal step raises (`Map`, `Filter`, a sink) | the run stops and the error propagates |
| building the prompt for a record fails | the run stops — a `KeyError` naming the missing column |
| **an LLM call fails** | **logged as a warning, the record is dropped, the run continues** |
| **parsing an LLM reply fails** | **the same: logged, dropped, the run continues** |
| you press Ctrl-C | the run stops; the checkpoint keeps what finished |

A prompt placeholder with no matching column is a loud failure, which is worth knowing
because it looks like the quiet kind:

An `LLMStep` whose prompt is `"Write about {topic}"` while the records only carry a
`text` column stops the run with `KeyError: 'topic'`. This happens while the calls are
being built, so nothing has been spent.

### One failing call takes its batch with it

The runner sends LLM calls in batches, grouped by served model. If any call in a group
raises, the **whole group** is abandoned: every record in it is logged as failed and
dropped, including the calls that had not been attempted yet.

`batch_size` therefore sets how much one transient failure costs. With eight records and
a single failure on the second call:

| `batch_size` | Calls attempted | Records kept |
|---|---|---|
| `1` | 8 | 7 |
| `4` (the default) | 6 | 4 |
| `8` | 2 | 0 |

If a step keeps losing records to occasional provider errors, lower `batch_size`. It
costs some throughput and saves the rest of the batch.

Note that the log prints one `LLM call failed` line per record in the group, all with the
same error. Only one of them really failed.

### `on_parse_error` and the records that disappear

Every LLM step takes `on_parse_error`, and it has two values:

| Value | Behaviour |
|---|---|
| `"skip"` | **the default.** Log a warning, drop the record, keep going |
| `"raise"` | re-raise the error |

Two things about it are not obvious.

**It is not only about parsing.** The `try` block around each record covers the whole
call — building the messages, the provider request, and the parse. A rate limit, a
timeout or a bad key is treated exactly like a malformed reply: with the default, the
record is dropped and the run continues.

**`"raise"` does not raise under `Pipeline.run()`.** The runner catches the error itself,
logs it, and moves to the next call. `on_parse_error="raise"` only raises when you call
`step.process(records)` directly. Under `run()` both values behave the same way.

So the way to find out whether a run went well is to count:

```python
from datafast import LLMStep, ListSink, Source, openai

records = [{"text": "a"}, {"text": "b"}, {"text": "c"}]

pipeline = (
    Source.list(records)
    >> LLMStep(
        prompt="Summarize: {text}",
        input_columns=["text"],
        output_column="summary",
        model=openai(),
    )
    >> ListSink()
)

results = pipeline.run()
if len(results) < len(records):
    print(f"{len(records) - len(results)} records were dropped")
```

If the count is short, the log holds one `LLM call failed` line per lost record, with the
model, the call id and the error.

## Warnings

datafast raises five Python warnings. None of them stops a run, and each means something
different.

| Warning | Means | What to do |
|---|---|---|
| `Parameter 'x' is not supported by resolved served model p/m and will be omitted.` | you set something this served model cannot do | remove it, or accept that it is dropped |
| `LiteLLM rejected one or more request parameters as unsupported. Retrying once with drop_params=True` | the provider refused a parameter datafast thought was fine | the call still succeeds; the profile is out of date |
| `p/m does not expose native batching for this endpoint. Falling back to bounded parallel single requests.` | batching is being emulated with concurrent single calls | nothing — it is informational |
| `p/m has no declared native schema support. Using prompted JSON plus Pydantic validation.` | the schema is a request, not a guarantee | expect occasional invalid replies |
| `Langfuse tracing requested but ...` | tracing is on and Langfuse is missing | `pip install langfuse`, or turn tracing off |

The first one names the parameter **as it goes on the wire**, not as you set it. Setting
`thinking=True` on a served model with no reasoning support warns about
`reasoning_effort`, because that is the request field it would have filled.

### Choosing how strict to be

`unsupported_params` on any served model decides what the first warning does:

| Value | Behaviour |
|---|---|
| `"warn"` | the default: warn and drop the parameter |
| `"fail"` | raise `ValueError` instead of dropping it |
| `"quiet"` | drop it silently |

```python
from datafast import anthropic

model = anthropic(unsupported_params="fail")
```

Use `"fail"` while you are working out what a served model supports. It turns a silent
difference between two providers into an error you cannot miss.

## Checkpoints and resume

A checkpoint is the on-disk state of a run: a `manifest.json` plus one JSONL file per
step. Resume reads it and skips the steps that finished.

### `PipelineChangedError`

The manifest holds a fingerprint of the pipeline. If it does not match, datafast will not
reuse the checkpoint:

```
PipelineChangedError: Pipeline structure has changed since checkpoint.
Use resume=False to start fresh.
```

With `resume=False` there is no error — the old checkpoint is cleared and the run starts
over, with `Pipeline changed, clearing old checkpoint` in the log.

**The fingerprint only sees step names and step classes.** Change a prompt, swap the
served model, or rewrite the function inside a `Map`, and the fingerprint is identical.
Resume will continue as if nothing changed and mix old records with new ones. Point a
changed pipeline at a fresh `checkpoint_dir`.

### `resume_from` and `stop_after` errors

`resume_from` re-runs one step and everything after it. All three of its failure modes
are `ValueError`, raised before anything executes:

| Message | Cause |
|---|---|
| `resume_from requires checkpoint_dir to be set.` | no `checkpoint_dir` |
| `resume_from='X' requires an existing checkpoint in ...` | the directory has no checkpoint |
| `resume_from step 'X' not found. Steps: ...` | the name is not a step; the message lists the real ones |

`stop_after` takes the same kind of value and fails the same way:

| Message | Cause |
|---|---|
| `stop_after step 'X' not found. Steps: ...` | the name is not a step |
| `stop_after step N is out of range. The pipeline has M steps.` | the index is not a step position |

The names are step class names — `ListSource`, `LLMStep`, `JSONLSink` — not variable
names, unless you set one with `as_step`.

### What actually survives a crash

Completed steps always survive. A **partly finished LLM step** is the hard case, and the
answer depends on `checkpoint_every`, which defaults to `100`.

Progress inside an LLM step is only recorded every `checkpoint_every` calls. A run that
dies at call 60 on the default has no progress file, so resume re-runs all of them and
you pay twice. Lower it for slow or expensive steps:

```python
from datafast import Map, Sink, Source

pipeline = Source.list([{"text": "hello"}]) >> Map(lambda r: r) >> Sink.list()
pipeline.run(checkpoint_dir="./checkpoints", checkpoint_every=10)
```

There is a cost to that setting too. Output records are written one at a time, but the
list of finished calls is only rewritten every `checkpoint_every` calls. Anything
completed after the last save is in the records file without being marked done, so resume
runs those calls again and **the records they produced appear twice**. Crashing at call 6
with `checkpoint_every=10` gives you an extra copy of the records from calls 5 and 6.

Deduplicate after a resumed run if exact counts matter:

```python
records = [{"id": 1, "text": "a"}, {"id": 1, "text": "a"}, {"id": 2, "text": "b"}]

seen = set()
unique = [r for r in records if not (r["id"] in seen or seen.add(r["id"]))]
assert len(unique) == 2
```

## Provider errors

Calls to a served model follow one rule: **`ValueError` means you asked for something
impossible; everything else is wrapped in `RuntimeError` naming the provider.**

| Error | Means |
|---|---|
| `ValueError: OPENAI_API_KEY environment variable not set.` | no key in the environment and none passed as `api_key` |
| `ValueError: Either prompt or messages must be provided` | the call had neither |
| `ValueError: Provide either prompt or messages, not both` | the call had both |
| `ValueError: p/m does not support structured output` | `response_format` on a served model whose mode is `none` |
| `RuntimeError: Error generating response with <provider>: ...` | the provider rejected or failed the call |

The `RuntimeError` message carries the full traceback of the original error, so the
provider's own message — the 401, the 429, the model-not-found — is at the bottom of it.
Read from the end.

Only five kinds of failure are retried: rate limits, connection errors, timeouts,
internal server errors and service-unavailable errors. A bad key or a malformed request
fails on the first attempt, whatever `retry_limit` says.

Inside a pipeline you will not see these at all, because the runner catches them. They
appear as `LLM call failed` in the log and as missing records in the result.

## Things that fail silently

The list worth re-reading when a run finishes and the output looks wrong.

- **Dropped records.** The default `on_parse_error="skip"` swallows every exception, not
  just parse failures. Compare the output count with the input count.
- **One failure drops a whole batch.** Records you never see attempted are still counted
  as failures.
- **A mistyped prompt file path becomes the prompt.** `prompt=Path("prompts/typo.txt")`
  sends the literal text `prompts/typo.txt` to the model when the file does not exist. No
  error, real spend. Check the file exists before the run.
- **The checkpoint fingerprint ignores prompts and served models.** Only step names and
  classes are hashed.
- **Resume can duplicate records** completed since the last progress save.
- **`temperature` and `max_tokens` on an LLM step are ignored.** Set them on the served
  model instead.
- **A parse mode never fails loudly in XML.** A missing tag gives an empty string.

## Reading the log

datafast logs through [loguru](https://loguru.readthedocs.io/). The lines that matter:

| Line | Meaning |
|---|---|
| `Step N (name): X → Y records` | one per step; a drop from X to Y is where records were lost |
| `LLM call failed \| Model: ... \| Call: ... \| Error: ...` | one per lost record |
| `LLMStep complete: N outputs, M errors` | the per-step tally |
| `JSON parse error: ... \| Content: ...` | the reply that would not parse, quoted |
| `Pipeline changed, clearing old checkpoint` | the checkpoint was thrown away |
| `Resuming LLM step: N/M calls already done` | how much resume actually recovered |

## Where to go next

- [Pipelines & execution](pipelines_and_execution.md) — checkpoints, resume and the run controls in full.
- [LLM step](../reference/llm_step.md) — every option on a step, including `on_parse_error`.
- [Structured output & parse modes](structured_output.md) — why a reply failed to parse.
- [Calling a served model directly](calling_a_served_model.md) — the error contract outside a pipeline.
- [Served models](../reference/served_models.md) — declaring capabilities yourself.
