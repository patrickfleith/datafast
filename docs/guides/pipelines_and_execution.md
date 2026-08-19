# Pipelines & execution

A pipeline is an ordered chain of steps. This page covers the two halves of getting one
to run: **building** it with `>>` and the check `compile()` runs before anything
executes, then **running** it — every control `run()` accepts, how checkpoints work, and
what to reach for when a long run stops halfway.

## Building a pipeline

`>>` joins steps into a pipeline:

```python
from datafast import Map, Sink, Source

pipeline = (
    Source.list([{"text": "hello"}, {"text": "world"}])
    >> Map(lambda record: {**record, "length": len(record["text"])})
    >> Sink.jsonl("out.jsonl")
)
```

The chain is linear. A pipeline is itself a step, so you can build one in pieces and
join the pieces:

```python
from datafast import Map, Source

prepare = Source.list([{"text": "hello"}]) >> Map(lambda r: {**r, "n": 1})
finish = Map(lambda r: {**r, "n": r["n"] + 1})

pipeline = prepare >> finish
print(len(pipeline.steps))  # 3 — the parts are flattened, not nested
```

Give a step a name with `as_step`. The name shows up in logs, in checkpoint file names,
and is what `resume_from` and `stop_after` match on:

```python
from datafast import Map

step = Map(lambda record: record).as_step("clean_text")
print(step.name)  # clean_text
```

Without a name, a step is called after its class — `Map`, `LLMStep`, `ListSource`. Two
unnamed steps of the same class share a name, which is legal but makes logs harder to
read.

## What `compile()` checks

`Pipeline.compile()` validates the pipeline without running it. `run()` calls it for
you, so you rarely call it yourself — but calling it directly is a free way to catch a
mistake before you spend anything on LLM calls:

```python
from datafast import Map, Sink, Source

pipeline = Source.list([{"a": 1}]) >> Map(lambda r: r) >> Sink.jsonl("out.jsonl")
pipeline.compile()  # returns the pipeline, so you can chain .run()
```

It raises `PipelineValidationError` on the **first** problem it finds, so fixing one
error can reveal the next. Here is what it looks for:

| Rule | Example message |
|---|---|
| A pipeline starts with a source | `Pipeline must start with a source; step 0 is 'Map' (Map).` |
| Only the first step may be a source | `Source 'ListSource' at position 1 discards upstream records; a source may only be the first step.` |
| Nothing may follow the sinks | `Step 'Map' at position 2 comes after the sink at position 1; sinks must be the last steps.` |
| Every `Branch` is closed by a `JoinBranches` | `Branch at position 1 is never closed by a JoinBranches.` |
| Every `JoinBranches` has a `Branch` | `JoinBranches at position 1 has no matching Branch.` |
| A step reads only columns that exist | `Step 'LLMStep' references column(s) ['b'] that are not available. Available columns: ['a'].` |

Two rules read as if they were one but are not. **Several sinks may be chained** — they
pass their records through, so writing the same dataset to a file and to the Hub is one
run. What is rejected is a step *after* a sink.

### Column checking is deliberately partial

`compile()` tracks the columns a record has, starting from what the source declares. Some
steps reshape records in ways it cannot predict — `Map`, `FlatMap`, `Group`, `Pair`,
`Join`, `Concat`, `Branch` and the specialized LLM steps. After one of those, the schema
becomes unknown and column checking stops for the rest of the pipeline.

This is on purpose: a missed error is an inconvenience, but a *wrong* error would stop a
correct pipeline from running. So treat a clean `compile()` as "no mistake found", not
"no mistake possible".

### Sub-pipelines

A `Branch` path, a `Concat` source and a `Join` right side each hold their own chain of
steps, and `compile()` checks inside them too. The rules differ by what feeds them:

| Sub-pipeline | Gets records from | Must start with a source | May contain a sink |
|---|---|---|---|
| `Branch` path | the records entering the branch | no — a source would throw them away | no |
| `Concat` source | nothing | yes | no |
| `Join` right side | nothing | yes | no |

A `Branch` inside a branch path is rejected outright. The inner branch would overwrite
the outer branch's metadata, and `JoinBranches` would then drop every record. Close the
outer branch first.

## Running a pipeline

```python
records = pipeline.run()
```

`run()` returns every record from the last step as a list. Sinks pass records through, so
a pipeline that writes to a file still returns its records.

The runner executes one step at a time and **materializes the whole result** before
starting the next step. That is what makes checkpointing, resume and LLM batching
possible: each step has a clean boundary where the full set of records exists.

### Run controls

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `checkpoint_dir` | `str \| None` | `None` | Directory for checkpoint files. `None` disables checkpointing. |
| `resume` | `bool` | `False` | Continue from an existing checkpoint instead of starting over. |
| `resume_from` | `str \| None` | `None` | Re-run from this step name, discarding it and every later step. |
| `stop_after` | `int \| str \| None` | `None` | Stop once this step finishes. A step index or a step name. |
| `limit` | `int \| None` | `None` | Keep only the first N records from the source. |
| `batch_size` | `int` | `4` | LLM calls per batch. |
| `llm_strategy` | `str` | `"by_model"` | Order of LLM calls: `"by_model"`, `"round_robin"` or `"by_record"`. |
| `checkpoint_every` | `int` | `100` | Save LLM progress every N completed calls. |

```python
records = pipeline.run(
    checkpoint_dir="checkpoints/my_run",
    resume=True,
    batch_size=8,
    limit=50,
)
```

`limit` applies **after the source has read everything**. It truncates the source's
output, so a 100,000-row file is still read in full and then cut to N. It is a way to
keep a trial run cheap in LLM calls, not a way to read less from disk.

`stop_after` accepts either form:

```python
pipeline.run(stop_after=1)          # after the step at index 1
pipeline.run(stop_after="clean_text")  # after the step named clean_text
```

A name that matches no step, or an index outside the pipeline, raises `ValueError` before
the first step runs — the same treatment `resume_from` gives a name it cannot find.

### `RunConfig` and `run_pipeline`

`run()` is a thin wrapper. It packs its arguments into a `RunConfig` and hands it to the
runner. The eight controls above are exactly the fields of `RunConfig` — there is no
hidden ninth. You can build one yourself when you want to reuse a configuration:

```python
from datafast import Map, RunConfig, Runner, Source

pipeline = Source.list([{"a": 1}]) >> Map(lambda record: record)
config = RunConfig(batch_size=8, llm_strategy="round_robin")
records = Runner(pipeline, config).execute()
```

`run_pipeline(pipeline, ...)` is the same thing as a function, and takes the same
arguments as `run()`. Note that `run()` and `run_pipeline()` call `compile()` for you,
while constructing a `Runner` directly does not.

Anything `run()` does not name in its own signature is passed on to `RunConfig`, so
`checkpoint_every` works as a keyword — and a typo raises `TypeError` immediately rather
than being ignored.

## Execution strategies

When one LLM step has several served models, `llm_strategy` decides the order the calls
go out in. With three records and two models:

| Strategy | Order of calls |
|---|---|
| `by_model` (default) | all of model A, then all of model B |
| `round_robin` | A, B, A, B, A, B |
| `by_record` | record 1 on every model, then record 2, ... |

`round_robin` and `by_record` produce the same order in the common case, and differ only
when the number of calls per record is uneven.

The order is not cosmetic. Records come out in the order their calls completed, so the
strategy is also the order of the output list. `by_model` groups every row from one model
together; `by_record` keeps each record's rows side by side, which is easier to eyeball
when you are comparing models.

`by_model` is the default because it sends consecutive calls to the same provider, which
batches better and keeps one provider's rate limit in one place.

## Checkpoints and resume

Set `checkpoint_dir` and the runner writes its progress to disk:

```python
pipeline.run(checkpoint_dir="checkpoints/my_run")
```

The directory then holds one JSONL file per step plus a manifest:

```
checkpoints/my_run/
├── manifest.json
├── step_000_ListSource.jsonl
└── step_001_LLMStep.jsonl
```

`manifest.json` records the state of the run: each step's index, name, status
(`pending`, `in_progress` or `complete`) and record counts, plus a fingerprint of the
pipeline.

Run again with `resume=True` and the runner picks up at the first step that is not
complete, loading the previous step's records from disk:

```python
pipeline.run(checkpoint_dir="checkpoints/my_run", resume=True)
```

If every step is already complete, `resume=True` returns the saved records without
running anything.

### Resuming inside an LLM step

LLM steps checkpoint per call, not per step. Each completed call's record is appended as
it arrives, and the list of completed call ids is saved every `checkpoint_every` calls.
A run that dies after 700 of 1,000 calls resumes at call 701 — the 700 you already paid
for are not repeated.

Lower `checkpoint_every` for expensive calls, where losing 100 of them hurts. Raise it
for cheap ones, where the disk writes cost more than the calls.

### When the checkpoint no longer matches

The fingerprint covers the **structure** of the pipeline: each step's name and class, and
the same information for anything nested inside a `Branch` path.

- If the structure changed and `resume=True`, the run raises `PipelineChangedError`
  rather than mixing results from two different pipelines.
- If the structure changed and `resume=False`, the runner clears the old checkpoint and
  starts fresh, with a warning in the log.

The fingerprint does **not** cover what the steps actually do. Changing a prompt, a
served model, a temperature or the function inside a `Map` leaves it identical, so resume
will happily continue a run under the new settings and nothing warns you.

That makes one rule important: **a step must produce the same records for the same input
every time it runs.** Calls are matched to records by position, so a step that shuffles,
samples randomly or reads a changing file can attach a resumed LLM result to the wrong
record. If you changed what a step does, either start fresh or use `resume_from`.

Two more things worth knowing before you point `checkpoint_dir` somewhere:

- **Give each pipeline its own directory.** Clearing a stale checkpoint deletes every
  file in that directory, not only the ones datafast wrote.
- **Naming steps makes checkpoints readable.** Files are named
  `step_<index>_<step name>.jsonl`, so two unnamed `Map` steps give you
  `step_001_Map.jsonl` and `step_002_Map.jsonl`.

### Re-running one step: `resume_from`

`resume_from` names a step to redo. That step and every step after it are discarded;
everything before it is reused from the checkpoint:

```python
pipeline.run(checkpoint_dir="checkpoints/my_run", resume_from="score_answers")
```

This is the tool for "the generation was fine, the scoring prompt was wrong". It needs
`checkpoint_dir` and an existing checkpoint, and raises `ValueError` if the name is not a
step in the pipeline — the message lists the names it does know.

## Throughput lives on the served model

Nothing on the runner controls how fast you call a provider. Rate limiting, concurrency,
timeouts and retries are configured on the **served model**:

| Setting | Default | What it does |
|---|---|---|
| `rpm_limit` | `None` | Requests per minute. The model waits rather than exceeding it. |
| `max_concurrent` | `4` | Calls made in parallel inside one batch. |
| `timeout` | `None` | Seconds to wait for one response. |
| `retry_limit` | `None` → 3 | Retries after a failed call. |

```python
from datafast import openai

model = openai(model_id="gpt-5.5", rpm_limit=500, max_concurrent=8, timeout=60)
```

This split is deliberate. A rate limit belongs to the provider's account, not to a step:
if two steps share one served model, they must share one limit, and a limit set per run
could not do that.

`batch_size` and `max_concurrent` are easy to confuse. `batch_size` is how many calls the
runner hands over at a time; `max_concurrent` is how many of those actually run in
parallel. Setting `batch_size=64` with `max_concurrent=4` still makes four calls at a
time — it only means the runner checkpoints and logs less often.

## Things worth knowing

- **`compile()` runs automatically.** `run()` calls it first, so an invalid pipeline
  fails before it spends anything.
- **Steps are materialized, not streamed.** Every record from a step exists in memory
  before the next step starts. That is what resume and batching are built on, and it is
  also the memory ceiling on a very large run.
- **`limit` truncates, it does not skip reading.** The source still loads everything.
- **A clean `compile()` is not a proof.** Column checking stops after any step that
  reshapes records opaquely.
- **Resume trusts you on determinism.** The fingerprint sees structure, not behaviour.
- **`run()` returns records even when a sink wrote them.** Sinks pass records through.

## Where to go next

- [Concepts](../concepts.md) — the record → step → pipeline → runner model.
- [Data ops](../reference/data_ops.md) — the transforms and sinks you chain together.
- [LLM step](../reference/llm_step.md) — the steps that make the calls this page schedules.
- [Branching](../reference/branching.md) — `Branch` and `JoinBranches` in full.
- [Served models](../reference/served_models.md) — every rate limit, timeout and retry
  setting.
- [Glossary](../glossary.md) — the exact meaning of step, runner, checkpoint and manifest.
