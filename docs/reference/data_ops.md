# Data Operations

These steps reshape records without calling an LLM. They rename, drop, combine, split
and merge — everything between the source and the sink that costs nothing but CPU.

They are ordinary transforms: put them anywhere in a pipeline, except `Concat`, which
starts one.

## At a glance

| Step | What it does |
|---|---|
| `Map(fn)` | one record in, one record out |
| `FlatMap(fn)` | one record in, zero or more records out |
| `AddUUID()` | add a unique id column |
| `Filter(...)` | keep or drop records by condition |
| `Group(by=...)` | collapse records that share a key into one record |
| `Pair(n=2)` | combine records into pairs or larger tuples |
| `Concat(*sources)` | stack several pipelines end to end |
| `Join(right, on=...)` | merge two pipelines side by side on a key |

## `Map(fn)`

Rewrite each record. The record you return replaces the one that came in.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `fn` | `Callable[[Record], Record]` | required | takes a record, returns the new record |

```python
from datafast import Map

step = Map(lambda r: {**r, "length": len(r["text"])})
print(list(step.process(iter([{"text": "hello"}]))))
# [{'text': 'hello', 'length': 5}]
```

Nothing is merged for you. `Map(lambda r: {"length": len(r["text"])})` drops every other
column, which is sometimes what you want. Spread the record with `{**r, ...}` to keep it.

## `FlatMap(fn)`

Turn one record into many, or into none.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `fn` | `Callable[[Record], list[Record]]` | required | takes a record, returns a list of records |

```python
from datafast import FlatMap

step = FlatMap(lambda r: [{"q": q} for q in r["questions"]])
print(list(step.process(iter([{"questions": ["Q1", "Q2"]}]))))
# [{'q': 'Q1'}, {'q': 'Q2'}]
```

Returning `[]` drops the record. That makes `FlatMap` a filter that can also reshape.

## `AddUUID(column="id", overwrite=False)`

Give each record a unique id, so records stay traceable after later steps reshape them.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `column` | `str` | `"id"` | column the id is written to |
| `overwrite` | `bool` | `False` | replace a value that is already there |

```python
from datafast import AddUUID

record, = AddUUID(column="example_id").process(iter([{"text": "hi"}]))
print(sorted(record))
# ['example_id', 'text']
```

The id is a `uuid4` as a string. With `overwrite=False` a record that already has the
column passes through untouched — including when its value is `None`, because the check
is presence, not emptiness.

## `Filter(fn=None, where=None, keep=True)`

Keep or drop records. Give it either a function or a `where` condition, never both.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `fn` | `Callable[[Record], bool] \| None` | `None` | keep the records it returns `True` for |
| `where` | `dict \| None` | `None` | declarative condition, column to expected value |
| `keep` | `bool` | `True` | `True` keeps matching records, `False` drops them |

```python
from datafast import Filter

step = Filter(where={"score": {"$gte": 7}, "category": "science"})
print(list(step.process(iter([
    {"score": 9, "category": "science"},
    {"score": 3, "category": "science"},
]))))
# [{'score': 9, 'category': 'science'}]
```

Passing neither `fn` nor `where` raises `ValueError`, and so does passing both.

### Conditions

In `where`, a bare value tests equality. A dict applies operators. Every column in a
`where`, and every operator in one column's dict, must hold.

| Operator | Keeps a record when the column |
|---|---|
| `$eq` | equals the value — the same as a bare value |
| `$ne` | differs from the value |
| `$gt` | is greater than the value |
| `$gte` | is greater than or equal to the value |
| `$lt` | is less than the value |
| `$lte` | is less than or equal to the value |
| `$in` | is one of the listed values |
| `$nin` | is none of the listed values |
| `$contains` | contains the value — substring, or list member |
| `$all` | is a list, tuple or set holding every listed value |
| `$any` | is a list, tuple or set holding at least one listed value |
| `$startswith` | is a string starting with the value |
| `$endswith` | is a string ending with the value |
| `$regex` | is a string matching the pattern anywhere (`re.search`) |
| `$len_eq` | has that length |
| `$len_gt` | is longer than that |
| `$len_gte` | is that long or longer |
| `$len_lt` | is shorter than that |
| `$len_lte` | is that long or shorter |
| `$exists` | is present and non-null (`True`), or absent or null (`False`) |
| `$type` | is of that type: `"str"`, `"int"`, `"float"`, `"bool"`, `"list"`, `"dict"`, `"none"` |

The length operators work on anything with a length — a string, a list, a dict. The
string operators reject non-strings instead of failing.

### Logical forms

`$or` and `$and` take a list of conditions and sit at the top of `where`, not under a
column.

```python
from datafast import Filter

step = Filter(where={"$or": [{"category": "science"}, {"score": {"$gte": 9}}]})
print(len(list(step.process(iter([
    {"category": "history", "score": 9},
    {"category": "history", "score": 2},
])))))
# 1
```

### Dropping instead of keeping

`keep=False` inverts the whole condition: matching records are dropped.

```python
from datafast import Filter

step = Filter(where={"quality": {"$lt": 3}}, keep=False)
print(list(step.process(iter([{"quality": 1}, {"quality": 5}]))))
# [{'quality': 5}]
```

## `Group(by, collect=None, output_column=None, agg=None, min_per_group=None, max_per_group=None)`

Collapse the records sharing a key into one record — chunks back into a document,
reviews into a product.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `by` | `str \| list[str]` | required | column(s) whose shared values define a group |
| `collect` | `str \| list[str] \| None` | `None` | column(s) gathered into lists, named `{column}_list` |
| `output_column` | `str \| None` | `None` | name for the collected list; only when `collect` is one column |
| `agg` | `dict[str, str] \| None` | `None` | `{"new_column": "column:function"}` |
| `min_per_group` | `int \| None` | `None` | drop groups holding fewer records than this |
| `max_per_group` | `int \| None` | `None` | keep only the first N records of each group |

```python
from datafast import Group

step = Group(by="doc_id", collect="chunk", agg={"n": "chunk:count"})
print(list(step.process(iter([
    {"doc_id": 1, "chunk": "a"},
    {"doc_id": 1, "chunk": "b"},
]))))
# [{'doc_id': 1, 'chunk_list': ['a', 'b'], 'n': 2}]
```

**The output record holds only the `by` columns, the collected lists and the `agg`
columns.** Every other column is dropped. Name in `collect` or `agg` anything you need
downstream.

An `agg` value is written `"column:function"`, or `"column:concat:separator"` to choose
the separator for `concat` (the default is a newline). The nine functions:

| Function | Value |
|---|---|
| `count` | number of records in the group |
| `sum` | sum of the values, skipping nulls |
| `mean` | average of the values, skipping nulls; `None` if there are none |
| `min` | smallest value, skipping nulls |
| `max` | largest value, skipping nulls |
| `first` | value from the group's first record |
| `last` | value from the group's last record |
| `collect` | the values as a list |
| `concat` | the values joined into one string, skipping nulls |

A bad spec is caught when you build the step, not when the pipeline runs: a missing `:`
or an unknown function raises `ValueError` immediately.

## `Pair(n=2, strategy="random", within=None, across=None, output_format="columns", max_pairs=None, seed=None)`

Combine records into pairs or larger tuples — for preference data, comparisons, or
multi-chunk questions.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `n` | `int` | `2` | tuple size; 2 is a pair, 3 a triplet. Minimum 2 |
| `strategy` | `str` | `"random"` | how tuples are chosen; see below |
| `within` | `str \| list[str] \| None` | `None` | records must share these columns to be combined |
| `across` | `str \| list[str] \| None` | `None` | records must differ on these columns to be combined |
| `output_format` | `str` | `"columns"` | `"columns"` or `"list"` |
| `max_pairs` | `int \| None` | `None` | stop after this many tuples in total |
| `seed` | `int \| None` | `None` | random seed; only used by `"random"` |

| Strategy | Tuples |
|---|---|
| `"random"` | random draws, repeatable with `seed` |
| `"sequential"` | consecutive, non-overlapping: (0,1), (2,3), … |
| `"sliding"` | a sliding window: (0,1), (1,2), (2,3), … |
| `"all"` | every combination |

```python
from datafast import Pair

step = Pair(n=2, strategy="sliding", within="doc_id")
print(list(step.process(iter([
    {"doc_id": 1, "chunk": "a"},
    {"doc_id": 1, "chunk": "b"},
]))))
# [{'chunk_1_doc_id': 1, 'chunk_1_chunk': 'a', 'chunk_2_doc_id': 1, 'chunk_2_chunk': 'b'}]
```

`output_format="columns"` prefixes every column with `chunk_1_`, `chunk_2_` and so on.
`output_format="list"` instead writes a `chunks` column holding the whole records, plus
one `{column}_list` per column.

A group holding fewer than `n` records produces nothing. An invalid `strategy`,
`output_format`, or an `n` below 2 raises `ValueError` when you build the step.

**Always set `max_pairs` with `"random"`.** The random strategy has no natural end: left
alone it draws until it has made up to 100,000 tuples, per group, and the same tuple can
come out twice.

## `Concat(*sources)`

Stack pipelines end to end: run each one, yield all its records, then move to the next.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `sources` | `*Step` | at least one | the steps or pipelines to run in order |

```python
from datafast import Concat, Source

step = Concat(Source.list([{"text": "a"}]), Source.list([{"text": "b"}]))
print(list(step.process(iter([]))))
# [{'text': 'a'}, {'text': 'b'}]
```

`Concat` reads only its own sources, so it counts as a source itself: it must be the
**first** step of the pipeline, and `compile()` rejects it anywhere else. Records arriving
from upstream are discarded, not passed through.

Each source must be a complete pipeline of its own — it starts with its own source and
holds no sink. Building one with no sources raises `ValueError`.

## `Join(right, on, how="inner", suffixes=("_left", "_right"))`

Merge two pipelines side by side on a shared key. The left side is the upstream pipeline;
the right side is the one you pass in.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `right` | `Step` | required | the step or pipeline supplying the right records |
| `on` | `str \| list[str]` | required | column(s) forming the key |
| `how` | `str` | `"inner"` | `"inner"`, `"left"`, `"right"` or `"outer"` |
| `suffixes` | `tuple[str, str]` | `("_left", "_right")` | endings added to columns both sides share |

| How | Output |
|---|---|
| `"inner"` | only keys found on both sides |
| `"left"` | every left record, matched or not |
| `"right"` | every right record, matched or not |
| `"outer"` | every record from both sides |

```python
from datafast import Join, Source

right = Source.list([{"user_id": 1, "action": "click"}])
step = Join(right, on="user_id")
print(list(step.process(iter([{"user_id": 1, "name": "ada"}]))))
# [{'user_id': 1, 'name': 'ada', 'action': 'click'}]
```

Columns named on both sides get the suffixes; key columns never do. An unmatched record
carries only its own columns — the missing side is absent, not filled with `None`.

Keys are matched exactly, and a record missing a key column joins on `None`, which
matches every other record missing it. Both sides are held in memory, and a key present
several times on both sides yields one record per combination.

## Things worth knowing

- **`Group` throws away the columns you did not name.** Only `by`, `collect` and `agg`
  reach the output record.
- **`Pair(strategy="random")` without `max_pairs` will make up to 100,000 tuples.** Set
  the cap, and set `seed` if you want the same dataset twice.
- **`$or` and `$and` narrow their siblings.** In `{"$or": [...], "score": {"$gt": 5}}` a
  record must match the `$or` *and* score above 5. Every key in a `where` dict must hold.
- **A missing column is not an error in `Filter`.** It reads as `None`, so it fails most
  conditions but satisfies `$ne`, `$nin` and `$exists: False`.
- **`Concat` discards upstream records** and must be the first step.
- **`Group`, `Pair`, `Concat` and `Join` hold every record in memory** while they work.
  Filter first when the dataset is large.
- **`compile()` stops checking columns after `Map`, `FlatMap`, `Group`, `Pair`, `Concat`
  or `Join`**, because their output shape is not knowable in advance. It does still check
  that `by`, `within`, `across` and `on` name columns that exist.

## Where to go next

- [Pipelines & execution](../guides/pipelines_and_execution.md) — how these steps chain together.
- [Sources & Seed](sources_and_seed.md) — the steps that come first.
- [Concepts](../concepts.md) — the record → step → pipeline model.
- [Glossary](../glossary.md) — what record, column, step and transform mean here.
- [API reference](../api.md) — generated signatures for everything on this page.
