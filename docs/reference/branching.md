# Branching

Sometimes one record should be processed in more than one way. `Branch` sends every
record down several named **branch paths**, and `JoinBranches` merges the results back
into one record.

The usual reason is comparison: the same question answered by two prompts, or by two
served models, ending up side by side in one record. Preference data is built this way.

## At a glance

| Piece | What it does |
|---|---|
| `Branch(name=step, ...)` | runs every path on every record and tags the outputs |
| `JoinBranches()` | groups the tagged records back into one record per input |
| `_branch_id`, `_branch_name`, `_branch_input_keys` | the columns `Branch` adds for `JoinBranches` to read |
| default suffix | `_{path name}`, on every column a path added |
| `how="inner"` | drop an input record unless every path produced something |
| `how="outer"` | keep it; the missing path adds no columns |

A `Branch` must be followed by a `JoinBranches`. `compile()` rejects a pipeline where it
is not.

## Branch

### `Branch(**paths)`

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `paths` | `**Step` | at least two | named steps, given as keyword arguments |

Each keyword is a path name, and each value is a step — or a whole pipeline, built with
`>>`. Every input record is sent down every path.

```python
from datafast import Branch, JoinBranches, Map, Sink, Source

pipeline = (
    Source.list([{"text": "hello"}, {"text": "goodbye"}])
    >> Branch(
        upper=Map(lambda r: {**r, "shout": r["text"].upper()}),
        length=Map(lambda r: {**r, "shout": str(len(r["text"]))}),
    )
    >> JoinBranches()
    >> Sink.list()
)

records = pipeline.run()
# [{'text': 'hello', 'shout_upper': 'HELLO', 'shout_length': '5'}, ...]
```

Fewer than two paths raises `ValueError` at construction.

Paths run **one after another**, not at the same time. Each path gets a deep copy of the
records, so one path cannot change what another path sees.

### The tagging columns

`Branch` adds three columns so that `JoinBranches` can put the records back together:

| Column | Value |
|---|---|
| `_branch_id` | the position of the input record, shared by every path |
| `_branch_name` | the name of the path that produced this record |
| `_branch_input_keys` | the column names the record had **before** the branch |

```python
from datafast import Branch, Map

branch = Branch(
    upper=Map(lambda r: {**r, "shout": r["text"].upper()}),
    length=Map(lambda r: {**r, "shout": str(len(r["text"]))}),
)
tagged = list(branch.process(iter([{"text": "hello"}])))
# [{'text': 'hello', '_branch_id': 0, '_branch_input_keys': ['text'],
#   'shout': 'HELLO', '_branch_name': 'upper'}, ...]
```

`_branch_input_keys` is what tells `JoinBranches` which columns are old and which ones
the path added. A path that rebuilds a record from scratch and drops `_branch_id` loses
its records at the join; a warning is logged when that happens.

`JoinBranches` removes all three columns, so they do not reach the sink.

## JoinBranches

### `JoinBranches(suffixes=None, how="inner")`

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `suffixes` | `dict[str, str] \| None` | `None` | per-path column suffix; `_{path name}` for any path not listed |
| `how` | `str` | `"inner"` | `"inner"` or `"outer"`; what to do when a path produced nothing |

Any other value of `how` raises `ValueError` at construction.

Records are grouped by `_branch_id`, and each group becomes one output record:

- Columns that existed **before** the branch are copied over once, with no suffix.
- Columns a path **added** get that path's suffix.
- The three `_branch_*` columns are dropped.

```python
from datafast import Branch, JoinBranches, Map, Sink, Source

pipeline = (
    Source.list([{"q": "hi"}])
    >> Branch(
        a=Map(lambda r: {**r, "answer": "short"}),
        b=Map(lambda r: {**r, "answer": "long"}),
    )
    >> JoinBranches(suffixes={"a": "_first"})
    >> Sink.list()
)

records = pipeline.run()
# [{'q': 'hi', 'answer_first': 'short', 'answer_b': 'long'}]
```

Path `a` uses the suffix given; path `b` was not listed, so it falls back to `_b`.

### The cartesian join

A path may produce more than one record per input — an LLM step with several outputs per
record, or a `FlatMap`. When that happens the merge is a **cartesian product**: every
record from one path is paired with every record from the others.

```python
from datafast import Branch, FlatMap, JoinBranches, Sink, Source

pipeline = (
    Source.list([{"n": 1}])
    >> Branch(
        a=FlatMap(lambda r: [{**r, "a": 1}, {**r, "a": 2}]),
        b=FlatMap(lambda r: [{**r, "b": 10}, {**r, "b": 20}]),
    )
    >> JoinBranches()
    >> Sink.list()
)

records = pipeline.run()
assert len(records) == 4  # 2 x 2
```

Two paths of three records each give nine output records for one input record. This
grows fast, so keep it in mind before asking two paths for several outputs apiece.

### Missing paths: `inner` and `outer`

A path can also produce **no** record for an input — a `Filter` inside the path dropped
it, or an LLM call failed.

- `how="inner"` (the default) drops that `_branch_id` entirely. Every output record is
  complete.
- `how="outer"` keeps it. The missing path simply contributes no columns, so the output
  record has fewer columns than the others.

```python
from datafast import Branch, Filter, JoinBranches, Map, Sink, Source

pipeline = (
    Source.list([{"n": 1}, {"n": 2}])
    >> Branch(
        keep=Map(lambda r: {**r, "a": r["n"]}),
        picky=Filter(lambda r: r["n"] == 1) >> Map(lambda r: {**r, "b": r["n"]}),
    )
    >> JoinBranches(how="outer")
    >> Sink.list()
)

records = pipeline.run()
# [{'n': 1, 'a_keep': 1, 'b_picky': 1}, {'n': 2, 'a_keep': 2}]
```

Note that the second record has no `b_picky` key at all. It is absent, not `None`. Code
reading that column must use `record.get("b_picky")`.

## Inside the runner

When a pipeline is run, the runner drives the paths itself instead of leaving `Branch`
to do it. That matters for three reasons.

**Nested batching.** An LLM step inside a path gets the same batching and execution
strategy as one at the top level. Four records against one path go out as one batch of
four calls, not as four separate batches.

**Checkpoint files per path.** The whole `Branch` is one step in the manifest, but each
path writes its own checkpoint file, keyed by a dotted name:

```text
step_001_Branch.jsonl              the tagged output of the whole step
step_001_Branch.a.jsonl            path 'a'
step_001_Branch.b.1_LLMStep.jsonl  the LLM step at position 1 of path 'b'
```

The name of the `Branch` step is in every one of those file names, so
`Branch(...).as_step("compare")` renames them too.

**Resume.** A path that finished before the crash is read back from its file rather than
run again, and a path that crashed part way through only re-runs the LLM calls it had
not completed. This is the same per-call resume LLM steps get at the top level; see
[Checkpointing](../guides/checkpointing.md).

### Path steps must be deterministic

Non-LLM steps inside a path are **not** checkpointed. On resume they run again, to
rebuild the records that the path's LLM step was working on.

A completed LLM call is matched to its record by **position**. So if a step inside the
path yields the records in a different order the second time, or yields a different
number of them, the saved results are attached to the wrong records — quietly.

In practice: give any `Sample` inside a path a `seed`, and keep the functions you pass to
`Map` and `Filter` free of randomness and of clocks.

The fingerprint that protects a checkpoint records the path names and the step **classes**
inside them. Changing the function inside a `Map` does not change it, so nothing will warn
you. Changing which steps a path contains does change it, and resuming then raises
`PipelineChangedError`.

## What compile() rejects

`compile()` checks the branch structure before anything runs, and raises
`PipelineValidationError`:

| Shape | Why |
|---|---|
| a `Branch` with no `JoinBranches` after it | the tagging columns would reach the sink |
| a `JoinBranches` with no `Branch` before it | there is nothing to merge |
| a second `Branch` opened before the first is closed | same as the first case |
| a `Branch` inside a branch path | the inner branch overwrites the outer tagging columns, and every record is then dropped at the join |

A branch path may not contain a source (it would throw away the records the branch fed
it) and may not contain a sink. Both are rejected the same way.

To branch twice, close the first branch first:

```python
from datafast import Branch, JoinBranches, Map, Sink, Source

pipeline = (
    Source.list([{"n": 1}])
    >> Branch(a=Map(lambda r: {**r, "x": 1}), b=Map(lambda r: {**r, "x": 2}))
    >> JoinBranches()
    >> Branch(c=Map(lambda r: {**r, "y": 1}), d=Map(lambda r: {**r, "y": 2}))
    >> JoinBranches()
    >> Sink.list()
)
pipeline.compile()
```

## Things worth knowing

- **A path that changes an existing column loses the change.** A column that existed
  before the branch is copied from the first path that produced a record, and is never
  suffixed. If two paths rewrite `text`, only the first path's `text` survives the join.
  Write to a new column instead.
- **Suffixes are appended, not inserted.** `response` in path `chosen` becomes
  `response_chosen`. Columns an LLM step adds are suffixed too, so `_model` becomes
  `_model_chosen`.
- **`inner` hides failures.** One path failing on a record removes that record from the
  dataset with only a log line. Compare the record count in and out if that matters.
- **Paths are sequential.** Two paths take about as long as running them one after the
  other, because that is what happens. Branching is for shaping the data, not for speed.
- **Two paths double the cost.** Each path makes its own LLM calls on every record.
- **Path names become column names.** Keep them short and valid as a suffix.

## Where to go next

- [Concepts](../concepts.md) — records, steps, the runner, and checkpoints.
- [Glossary](../glossary.md) — the precise meaning of branch path, record and column.
- [Sources & Seed](sources_and_seed.md) — the steps a pipeline starts with.
- [Building Pipelines](../guides/building_pipelines.md) — `Map`, `Filter` and the other
  transforms a path is made of.
- [Checkpointing](../guides/checkpointing.md) — resume, and the per-call progress a path
  relies on.
- [API reference](../api.md) — generated signatures for both steps.
