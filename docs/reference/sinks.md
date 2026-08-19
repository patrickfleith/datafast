# Sinks

A **sink** is a step that writes records out: to a file, to a dataset on the Hugging
Face Hub, or to a list in memory.

A sink yields every record it receives, unchanged. Writing is a side effect, not a
transformation, so `run()` still gives you the records, and several sinks can be chained
to write the same dataset to several places in one run.

## At a glance

| Constructor | Writes | Returns | Needs |
|---|---|---|---|
| `Sink.jsonl(path)` | one JSON object per line | `JSONLSink` | — |
| `Sink.csv(path)` | one row per record | `CSVSink` | — |
| `Sink.parquet(path)` | one row per record | `ParquetSink` | `datafast[parquet]` |
| `Sink.hub(repo_id)` | a dataset on the Hugging Face Hub | `HubSink` | `datafast[hub]` |
| `Sink.list()` | nothing to disk; keeps records in memory | `ListSink` | — |

Each class can also be imported and constructed directly — `JSONLSink("out.jsonl")` is
exactly what `Sink.jsonl("out.jsonl")` returns, with the same parameters. The factory is
just the shorter way to write it.

### `Sink.jsonl(path)`

One JSON object per line. The usual format for a generated dataset.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `path` | `str \| Path` | required | file to write |

```python
from datafast import Sink, Source

records = (Source.list([{"text": "hello"}]) >> Sink.jsonl("out/data.jsonl")).run()
```

Missing parent directories are created. An existing file is overwritten, not appended
to, so re-running a pipeline replaces the previous output.

A value that JSON cannot represent is written as its `str()` form instead of raising —
a `datetime` becomes `"2020-01-01"`. Non-ASCII text is written as-is, not escaped.

### `Sink.csv(path)`

One row per record, with a header row of column names.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `path` | `str \| Path` | required | file to write |

```python
from datafast import Sink, Source

records = (Source.list([{"text": "hello", "label": "greeting"}]) >> Sink.csv("out.csv")).run()
```

**The column names come from the first record only.** If a later record has a column the
first one did not, the step raises `ValueError`. If a later record is missing one, the
cell is left empty. Records of different shapes are common after a `Branch` or a partly
failed LLM step, so give the sink records with the same columns — a `Map` step that
rebuilds each record is the usual fix.

### `Sink.parquet(path)`

One row per record, in a columnar file. Smaller and faster to read back than JSONL for
large datasets.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `path` | `str \| Path` | required | file to write |

```python
from datafast import Sink

sink = Sink.parquet("out/data.parquet")
```

Requires `pip install "datafast[parquet]"`. Without it the step raises an `ImportError`
naming the extra.

The columns come from the first record here too, but the failure is quieter than CSV's: a
column that only later records have is **dropped without a warning**, and a missing one
becomes `null`.

### `Sink.hub(repo_id, token=None, private=True, train_size=None, seed=42, shuffle=True, commit_message=None)`

Pushes the records to the Hugging Face Hub as a dataset.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `repo_id` | `str` | required | dataset id, e.g. `"username/my-dataset"` |
| `token` | `str \| None` | `None` | Hub token; when `None` the `HF_TOKEN` environment variable is used, then the cached login |
| `private` | `bool` | `True` | create the dataset as private |
| `train_size` | `float \| None` | `None` | share of records in the train split; no split when `None` |
| `seed` | `int` | `42` | random seed for the shuffle and the split |
| `shuffle` | `bool` | `True` | shuffle the records before splitting |
| `commit_message` | `str \| None` | `None` | commit message; `"Upload dataset via datafast"` when `None` |

```python
from datafast import Sink

sink = Sink.hub("username/my-dataset", train_size=0.9, private=False)
```

**Datasets are private by default.** Pass `private=False` to publish one.

**The token is never a parameter you must pass.** `Sink.hub` looks at `token` first, then
at the `HF_TOKEN` environment variable. If both are empty it pushes with no token and the
Hub falls back to your cached login from `huggingface-cli login`.

With `train_size` set, the dataset is pushed as two splits, `train` and `test`. It must
be strictly between 0.0 and 1.0; anything else raises `ValueError`. Without it, the whole
dataset is pushed as one split.

`shuffle=True` shuffles the records before they are pushed, using `seed`. The same
`seed` is used for the split, so the same records give the same train and test splits
every run.

After the push, datafast prepends a `datafast` tag to the dataset's `README.md` in a
second commit, so datasets made with the library are findable on the Hub. Any README text
already there is kept below the tag block, and a later push that finds the tag leaves the
file alone.

Requires `pip install "datafast[hub]"`. Without it the step raises an `ImportError`
naming the extra.

### `Sink.list()`

Collects the records in memory, on the sink's `records` attribute. Useful in tests and
in notebooks, where you want the records in hand without a file.

Takes no parameters.

```python
from datafast import Sink, Source

collected = Sink.list()
(Source.list([{"text": "hello"}]) >> collected).run()

print(collected.records)
```

`records` holds one run. A second `run()` on the same pipeline replaces the list rather
than appending to it, so what you read is always the last run.

## Things worth knowing

- **A sink is not the end of the data.** Records pass through unchanged, so `run()`
  returns them whatever the last step is.
- **Chain sinks to write once to several places.** `>> Sink.jsonl(...) >> Sink.hub(...)`
  is one run, not two.
- **Sinks must come last.** `compile()` rejects any step placed after a sink, and rejects
  a sink inside a branch path.
- **An empty run writes almost nothing.** With zero records, `Sink.jsonl` still creates an
  empty file, while `Sink.csv`, `Sink.parquet` and `Sink.hub` write nothing at all and log
  a warning. A missing CSV file usually means the pipeline produced no records.
- **Files are overwritten, never appended to.** Two runs to the same path leave you with
  the second run's records only.
- **`Sink.hub` shuffles what it pushes, not what it returns.** The records `run()` gives
  you stay in pipeline order.
- **`seed` here is a random seed**, unrelated to `Seed` and its dimensions.

## Where to go next

- [Sources & Seed](sources_and_seed.md) — the other end of the pipeline.
- [Concepts](../concepts.md) — how sinks fit the record → step → pipeline model.
- [Glossary](../glossary.md) — the precise meaning of sink, record, column and step.
- [Installation](../installation.md) — the `parquet` and `hub` extras.
- [API reference](../api.md) — generated signatures for everything on this page.
