# Seed

A **seed** starts a pipeline by building records that do not exist yet, from dimensions
you declare. Most synthetic datasets start this way.

The other way to start is a [source](source.md), which loads records that already
exist — a Python list, a local file, a Hugging Face dataset.

Seeding is a two-stage vocabulary: you declare **dimensions**, then **combine** them
into a source. The combined seed must come **first** in the pipeline, like any other
source.

A dimension on its own is not a step. `Seed.values(...) >> LLMStep(...)` raises
`TypeError` — only `Seed.product` and `Seed.zip` return something a pipeline can start
with. This is the most common mistake with seeds.

## At a glance

| Constructor | Returns | Needs |
|---|---|---|
| `Seed.values(column, values)` | a **dimension**, not a source | — |
| `Seed.range(column, start, end)` | a **dimension** of numbers | — |
| `Seed.expand(parent, child, mapping)` | a **dimension** of nested pairs | — |
| `Seed.product(*dimensions)` | a source: every combination | — |
| `Seed.zip(*dimensions)` | a source: dimensions aligned by position | — |

## Declaring dimensions

### `Seed.values(column, values)`

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `column` | `str` | required | the column this dimension fills |
| `values` | `list` | required | the values it takes |

```python
from datafast import Seed

dimension = Seed.values("language", ["en", "fr", "de"])
```

### `Seed.range(column, start, end, step=1)`

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `column` | `str` | required | the column this dimension fills |
| `start` | `int` | required | first value, inclusive |
| `end` | `int` | required | last value, **inclusive** |
| `step` | `int` | `1` | gap between values |

`end` is inclusive, unlike Python's own `range`. `Seed.range("grade", 1, 12)` gives
twelve values, 1 through 12.

### `Seed.expand(parent, child, mapping)`

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

## Combining dimensions

### `Seed.product(*dimensions)`

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

### `Seed.zip(*dimensions)`

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
- **`Seed.range` ends inclusive**, unlike Python's `range`.
- **Seed never means a random seed.** The `seed` *parameter* on `Sample` and `Sink.hub`
  is the random one.

## Where to go next

- [Source](source.md) — the other way to start a pipeline, loading records instead of
  building them.
- [Key Concepts](../concepts.md) — how seeds fit the record → step → pipeline model.
- [Full Glossary](../glossary.md) — the precise meaning of seed, dimension, record and column.
- [Data ops](data_ops.md) — the transforms that come next.
- [API reference](../api.md) — generated signatures for everything on this page.
