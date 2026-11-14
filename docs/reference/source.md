# Source

A **source** starts a pipeline by bringing in records that already exist — a Python
list, a local file, a Hugging Face dataset.

The other way to start is a [seed](seed.md), which builds records that do not exist yet
from dimensions you declare. Most synthetic datasets start with a seed; sources are how
you bring in existing material to transform or augment.

A source must come **first** in the pipeline. `compile()` raises
`PipelineValidationError` for a source anywhere else, and for a pipeline that starts
with anything else.

## At a glance

| Constructor | Returns | Needs |
|---|---|---|
| `Source.list(records)` | records from a Python list | — |
| `Source.file(path)` | records from a local file, format from the extension | — |
| `Source.jsonl(path)` | one record per JSON line | — |
| `Source.csv(path)` | one record per row | — |
| `Source.tsv(path)` | one record per tab-separated row | — |
| `Source.txt(path)` | one record per line of text | — |
| `Source.parquet(path)` | one record per row | `datafast[parquet]` |
| `Source.huggingface(name)` | records from a Hub dataset | `datafast[hub]` |

## `Source.list(records)`

Records you already have in memory.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `records` | `list[dict]` | required | the records to yield |

```python
from datafast import Source

source = Source.list([
    {"text": "Document 1", "category": "science"},
    {"text": "Document 2", "category": "history"},
])
```

The list is yielded as-is: no copying, no validation, no schema check. Whatever
dictionaries you pass become the pipeline's records.

## `Source.file(path, format=None, **kwargs)`

Reads a local file, choosing the reader from the extension.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `path` | `str \| Path` | required | file to read |
| `format` | `str \| None` | `None` | `"jsonl"`, `"csv"`, `"tsv"`, `"txt"` or `"parquet"`; auto-detected when `None` |
| `**kwargs` | | | passed to the underlying reader |

Extensions map like this:

| Extension | Format |
|---|---|
| `.jsonl`, `.json` | `jsonl` |
| `.csv` | `csv` |
| `.tsv` | `tsv` |
| `.txt` | `txt` |
| `.parquet`, `.pq` | `parquet` |

Note that **`.json` is read as JSONL** — one JSON object per line, not a single JSON
array. A conventional `.json` array file will not load; convert it or pass the records
through `Source.list`.

Any other extension raises `ValueError` at construction, before the pipeline runs, and
asks you to pass `format` explicitly.

```python
from datafast import Source

source = Source.file("data.csv")            # format inferred
other = Source.file("export.dat", format="jsonl")   # format forced
```

## `Source.jsonl(path, **kwargs)`

One JSON object per line. Blank lines are skipped.

**A malformed line does not stop the run.** It is logged as a warning and skipped, so a
partially corrupt file loads the records it can. If you need the opposite, validate the
file before the pipeline rather than expecting the source to fail.

## `Source.csv(path, **kwargs)` and `Source.tsv(path, **kwargs)`

One record per row, with the header row supplying the column names. `**kwargs` go to
`csv.DictReader`.

**Every value is a string.** These readers do no type inference, so a CSV column of
digits arrives as `"1"`, not `1`. Convert with a `Map` step if you need numbers:

```python
from datafast import Map, Source

pipeline = Source.csv("scores.csv") >> Map(lambda r: {**r, "score": int(r["score"])})
```

## `Source.txt(path, text_column="text", **kwargs)`

One record per line, wrapping each line in a single column. Blank lines are skipped.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `path` | `str \| Path` | required | file to read |
| `text_column` | `str` | `"text"` | the column each line is stored under |

## `Source.parquet(path, **kwargs)`

One record per row. `**kwargs` go to `pyarrow.parquet.read_table`.

Requires `pip install "datafast[parquet]"`. Without it, the step raises an `ImportError`
naming the extra.

## `Source.huggingface(dataset_name, split="train", subset=None, columns=None, trust_remote_code=False, streaming=False)`

Loads a dataset from the Hugging Face Hub.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `dataset_name` | `str` | required | dataset id, e.g. `"squad"` |
| `split` | `str` | `"train"` | which split to load |
| `subset` | `str \| None` | `None` | config name, e.g. `"20220301.en"` |
| `columns` | `list[str] \| None` | `None` | keep only these columns; all when `None` |
| `trust_remote_code` | `bool` | `False` | allow the dataset's own loading script to run |
| `streaming` | `bool` | `False` | stream lazily instead of downloading first |

```python
from datafast import Source

source = Source.huggingface(
    "squad",
    split="validation",
    columns=["question", "context"],
)
```

Requires `pip install "datafast[hub]"`. Private datasets need `HF_TOKEN` — see
[Installation](../installation.md).

`streaming=True` avoids downloading the whole dataset, which matters for large ones.
Note that the runner materializes each step in full regardless, so streaming saves the
download, not the memory; pair it with `limit` on `run()` when you only want a sample.

## Things worth knowing

- **A source only goes first.** Anywhere else, `compile()` rejects it rather than
  letting it discard the records upstream of it.
- **`.json` means JSONL**, not a JSON array.
- **CSV and TSV values are strings.** Cast them in a `Map` step.
- **Malformed JSONL lines are skipped**, with a warning rather than an error.
- **An unknown extension fails at construction**, not mid-run.

## Where to go next

- [Seed](seed.md) — the other way to start a pipeline, building records instead of
  loading them.
- [Key Concepts](../concepts.md) — how sources fit the record → step → pipeline model.
- [Full Glossary](../glossary.md) — the precise meaning of source, record and column.
- [Data ops](data_ops.md) — the transforms that come next.
- [API reference](../api.md) — generated signatures for everything on this page.
