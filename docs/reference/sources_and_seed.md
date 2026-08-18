# Sources & Seed

Every pipeline starts with records, and there are exactly two ways to get them.

**`Source`** loads records that already exist — a Python list, a local file, a Hugging
Face dataset. **`Seed`** builds records that do not exist yet, from dimensions you
declare. Most synthetic datasets start with a seed; sources are how you bring in
existing material to transform or augment.

Both produce a step that must come **first** in the pipeline. `compile()` rejects a
source anywhere else, and rejects a pipeline that starts with anything else.

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
| `Seed.values(column, values)` | a **dimension**, not a source | — |
| `Seed.range(column, start, end)` | a **dimension** of numbers | — |
| `Seed.expand(parent, child, mapping)` | a **dimension** of nested pairs | — |
| `Seed.product(*dimensions)` | a source: every combination | — |
| `Seed.zip(*dimensions)` | a source: dimensions aligned by position | — |

## Source

### `Source.list(records)`

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

### `Source.file(path, format=None, **kwargs)`

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

### `Source.jsonl(path, **kwargs)`

One JSON object per line. Blank lines are skipped.

**A malformed line does not stop the run.** It is logged as a warning and skipped, so a
partially corrupt file loads the records it can. If you need the opposite, validate the
file before the pipeline rather than expecting the source to fail.

### `Source.csv(path, **kwargs)` and `Source.tsv(path, **kwargs)`

One record per row, with the header row supplying the column names. `**kwargs` go to
`csv.DictReader`.

**Every value is a string.** These readers do no type inference, so a CSV column of
digits arrives as `"1"`, not `1`. Convert with a `Map` step if you need numbers:

```python
from datafast import Map, Source

pipeline = Source.csv("scores.csv") >> Map(lambda r: {**r, "score": int(r["score"])})
```

### `Source.txt(path, text_column="text", **kwargs)`

One record per line, wrapping each line in a single column. Blank lines are skipped.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `path` | `str \| Path` | required | file to read |
| `text_column` | `str` | `"text"` | the column each line is stored under |

### `Source.parquet(path, **kwargs)`

One record per row. `**kwargs` go to `pyarrow.parquet.read_table`.

Requires `pip install "datafast[parquet]"`. Without it, the step raises an `ImportError`
naming the extra.

### `Source.huggingface(dataset_name, split="train", subset=None, columns=None, trust_remote_code=False, streaming=False)`

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

## Seed

Seeding is a two-stage vocabulary: you declare **dimensions**, then **combine** them
into a source.

A dimension on its own is not a step. `Seed.values(...) >> LLMStep(...)` raises
`TypeError` — only `Seed.product` and `Seed.zip` return something a pipeline can start
with. This is the most common mistake with seeds.

### Declaring dimensions

#### `Seed.values(column, values)`

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `column` | `str` | required | the column this dimension fills |
| `values` | `list` | required | the values it takes |

```python
from datafast import Seed

dimension = Seed.values("language", ["en", "fr", "de"])
```

#### `Seed.range(column, start, end, step=1)`

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `column` | `str` | required | the column this dimension fills |
| `start` | `int` | required | first value, inclusive |
| `end` | `int` | required | last value, **inclusive** |
| `step` | `int` | `1` | gap between values |

`end` is inclusive, unlike Python's own `range`. `Seed.range("grade", 1, 12)` gives
twelve values, 1 through 12.

#### `Seed.expand(parent, child, mapping)`

One dimension holding a parent–child pair, for values that only make sense together.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `parent` | `str` | required | the outer column |
| `child` | `str` | required | the inner column |
| `mapping` | `dict[str, list]` | required | each parent value to its child values |

```python
from datafast import Seed

dimension = Seed.expand("topic", "subtopic", {
    "Physics": ["Quantum", "Relativity"],
    "Biology": ["Genetics", "Evolution"],
})
```

This is four values in **one** dimension — `(Physics, Quantum)`, `(Physics, Relativity)`,
`(Biology, Genetics)`, `(Biology, Evolution)` — not two dimensions of two. That is the
point: crossing topic and subtopic as separate dimensions would pair Physics with
Genetics.

### Combining dimensions

#### `Seed.product(*dimensions)`

Every combination — the cartesian product.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `dimensions` | `*SeedDimension` | none | the dimensions to cross |

```python
from datafast import Seed

seed = Seed.product(
    Seed.values("persona", ["student", "teacher"]),
    Seed.values("language", ["en", "fr"]),
)
```

Four records: each persona against each language. The record count is the product of the
dimension sizes, so it grows fast — three dimensions of ten are a thousand LLM calls.
`len(seed)` tells you the count before you run anything.

#### `Seed.zip(*dimensions)`

Dimensions aligned by position, like Python's `zip`.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `dimensions` | `*SeedDimension` | none | the dimensions to align; all must be the same length |

```python
from datafast import Seed

seed = Seed.zip(
    Seed.values("question", ["Q1", "Q2", "Q3"]),
    Seed.values("answer", ["A1", "A2", "A3"]),
)
```

Three records, pairing each question with the answer at the same index. Use this for
values that correspond, where a product would be meaningless.

All dimensions must be the same length; mismatched lengths raise `ValueError` naming the
lengths it got. Calling either combiner with no dimensions gives an empty source.

## Things worth knowing

- **A dimension is not a source.** Only `Seed.product` and `Seed.zip` start a pipeline.
- **Count before you spend.** `len(seed)` is the number of records, and usually the
  number of LLM calls the next step will make.
- **Overlapping columns: last one wins.** If two dimensions fill the same column, the
  later one overwrites the earlier, silently. Give each dimension its own column.
- **`.json` means JSONL**, not a JSON array.
- **CSV and TSV values are strings.** Cast them in a `Map` step.
- **Malformed JSONL lines are skipped**, with a warning rather than an error.
- **Seed never means a random seed.** The `seed` *parameter* on `Sample` and `Sink.hub`
  is the random one.

## Where to go next

- [Concepts](../concepts.md) — how sources fit the record → step → pipeline model.
- [Glossary](../glossary.md) — the precise meaning of seed, dimension, record and column.
- [Building Pipelines](../guides/building_pipelines.md) — the transforms that come next.
- [API reference](../api.md) — generated signatures for everything on this page.
