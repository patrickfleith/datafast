# Sample

`Sample` keeps some records and drops the rest.

Use it as a step to cut a large seed down before an LLM step spends money on every
record. Use it as a value inside an LLM step to choose which prompts or which served
models that step uses.

How it chooses is the **sampling strategy**. There are nine.

## At a glance

| Strategy | What it keeps | Requires | Random |
|---|---|---|---|
| `uniform` | any records, each equally likely (default) | — | yes |
| `first` | the first `n`, in order | — | no |
| `last` | the last `n`, in order | — | no |
| `systematic` | every `step`-th record | `step` | no |
| `top` | the highest values | `by` | no |
| `bottom` | the lowest values | `by` | no |
| `weighted` | records in proportion to a weight | `by` | yes |
| `stratified` | each group's share of the records | `by` | yes |
| `gaussian` | values near `center` | `by`, `center`, `std` | yes |

Every requirement is checked when you construct the `Sample`, not when the pipeline
runs. A missing `by`, `step`, `center` or `std` raises `ValueError` immediately.

## Sample as a step

Put a `Sample` between two steps and it thins the records flowing through.

```python
from datafast import Sample, Seed, Sink

pipeline = (
    Seed.product(
        Seed.values("persona", ["student", "teacher", "researcher"]),
        Seed.values("topic", ["physics", "biology", "history"]),
    )
    >> Sample(n=4, seed=42)
    >> Sink.list()
)
```

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `items` | `list \| None` | `None` | values to choose from; leave it out to sample the pipeline's records |
| `n` | `int \| None` | `None` | how many to keep |
| `frac` | `float \| None` | `None` | what fraction to keep, `0.0` to `1.0` |
| `strategy` | `str` | `"uniform"` | one of the nine sampling strategies |
| `by` | `str \| Callable \| list[float] \| None` | `None` | a column name, a function of the record, or one weight per record |
| `ascending` | `bool` | `False` | for `top` only: sort lowest first instead |
| `center` | `float \| None` | `None` | the `by` value `gaussian` favours |
| `std` | `float \| None` | `None` | how fast `gaussian` loses interest away from `center` |
| `step` | `int \| None` | `None` | the gap `systematic` walks with |
| `seed` | `int \| None` | `None` | random seed, for a repeatable result |
| `replace` | `bool` | `False` | allow the same record to be kept more than once |

`Sample` is a transform, not a source. It cannot start a pipeline; `compile()` rejects
that.

### How many: `n` and `frac`

`n` is a count, `frac` is a share. Give one or the other — both together raise
`ValueError`.

Give neither and **every record is kept**, whatever the strategy. That is easy to do by
accident with `Sample(strategy="top", by="score")`, which then only sorts.

```python
from datafast import Sample

records = [{"score": i} for i in range(10)]

half = list(Sample(frac=0.5).process(iter(records)))       # 5 records
best = list(Sample(n=3, strategy="top", by="score").process(iter(records)))
```

`frac` rounds down but never below 1: `frac=0.25` of 10 records keeps 2, and
`frac=0.01` of 10 keeps 1.

Asking for more records than exist gives you all of them, not an error — unless
`replace=True`, which repeats records to reach `n`.

### Repeating a run: `seed`

Without `seed`, the four random strategies give a different result every run. With
`seed`, the same input gives the same output.

```python
from datafast import Sample

records = [{"i": i} for i in range(100)]
first = list(Sample(n=5, seed=42).process(iter(records)))
again = list(Sample(n=5, seed=42).process(iter(records)))
assert first == again
```

`seed` does nothing for `first`, `last`, `systematic`, `top` and `bottom` — those pick
the same records every time anyway.

## The strategies

### `uniform` — a plain random subset

The default. Every record has the same chance. This is what you want to shrink a seed
without skewing it.

### `first`, `last` — take from one end

No randomness, no cost. `Sample(n=20, strategy="first")` is the fastest way to run a
pipeline on a small slice while you are still writing it.

### `systematic` — every `step`-th record

Walks the records taking one every `step`, starting at the first. With `step=3` you keep
records 0, 3, 6, 9 and so on. It spreads the choice across the whole input, which matters
when the records are ordered by something meaningful.

```python
from datafast import Sample

records = [{"i": i} for i in range(10)]
picked = [r["i"] for r in Sample(strategy="systematic", step=3).process(iter(records))]
assert picked == [0, 3, 6, 9]
```

`n` is optional here. Add it and the walk is cut short at `n` records.

### `top`, `bottom` — the best and the worst

Sort by `by`, then take `n`. Use them to keep the highest-scoring records after a `Score`
step, or to look at the worst ones.

```python
from datafast import Sample

records = [{"text": "a", "score": 3}, {"text": "b", "score": 9}, {"text": "c", "score": 1}]
best = [r["text"] for r in Sample(n=1, strategy="top", by="score").process(iter(records))]
assert best == ["b"]
```

`by` can also be a function of the record, which is how you sort on something the record
does not store:

```python
from datafast import Sample

records = [{"text": "short"}, {"text": "a much longer piece of text"}]
longest = list(Sample(n=1, strategy="top", by=lambda r: len(r["text"])).process(iter(records)))
```

`ascending=True` flips `top` to sort lowest first, which makes it identical to `bottom`.
`bottom` ignores `ascending` entirely.

### `weighted` — chance proportional to a number

Random, but records with a bigger `by` value are more likely. Use it when one column says
how much a record deserves to be kept.

`by` may be a column of numbers, a function, or a plain list of weights with one entry
per record. A weight list of the wrong length raises `ValueError` when the step runs.

```python
from datafast import Sample

records = [{"topic": "physics", "importance": 10}, {"topic": "trivia", "importance": 1}]
picked = list(Sample(n=1, strategy="weighted", by="importance", seed=0).process(iter(records)))
```

If every weight is 0 the strategy falls back to `uniform` rather than failing.

### `stratified` — keep the mix

Groups the records by `by`, then keeps roughly each group's share. A column that is 80%
English and 20% French stays about 80/20 after sampling, which plain `uniform` does not
guarantee for small samples.

```python
from datafast import Sample

records = [{"lang": "en"}] * 80 + [{"lang": "fr"}] * 20
kept = list(Sample(n=10, strategy="stratified", by="lang", seed=42).process(iter(records)))
assert sum(1 for r in kept if r["lang"] == "fr") >= 1
```

Every group gets at least one record. In a small sample that over-represents rare groups:
99 English records and 1 French one, sampled down to 3, gives 2 English and 1 French.

### `gaussian` — values near a target

Random, favouring records whose `by` value is close to `center`. `std` sets the width:
small `std` means only values very near `center` survive, large `std` behaves more like
`uniform`.

```python
from datafast import Sample

records = [{"length": n} for n in range(0, 1000, 10)]
medium = list(Sample(n=5, strategy="gaussian", by="length", center=500, std=50, seed=42).process(iter(records)))
```

A record whose `by` value is missing gets a weight of 0 and is never kept. If that is
every record, the strategy falls back to `uniform`.

## Sample as a value: `.pick()` and `.sample()`

This is the part that confuses people. A `Sample` is two things depending on where you
put it.

**As a step**, it takes the pipeline's records. You give it no `items`:

```python
from datafast import Sample

step = Sample(n=100, strategy="stratified", by="category", seed=42)
```

**As a value**, it takes `items` you hand it, and chooses among those. LLM steps accept
one wherever they accept a list — `prompt`, `model`, `language`:

```python
from datafast import LLMStep, Sample, ollama

prompts = [
    "Explain {topic} to a child.",
    "Explain {topic} with a sports analogy.",
    "Explain {topic} in three bullet points.",
]

step = LLMStep(
    prompt=Sample(prompts, n=1, seed=42),
    input_columns=["topic"],
    model=ollama("gemma3:4b"),
    output_column="explanation",
)
```

The difference in timing is what matters:

- **Pass the `Sample` itself** and the LLM step re-picks **for every record**. Each record
  gets its own prompt. That is how you get variety across a dataset.
- **Call `.pick()` yourself** and you get a plain list, chosen **once**. Every record then
  uses that same list.

```python
from datafast import Sample, ollama

models = [ollama("gemma3:4b"), ollama("qwen3:8b")]
chosen = Sample(models, n=1, seed=42).pick()   # a list, decided now
```

`.sample()` is another name for `.pick()` with no arguments. `.pick(n)` overrides `n`
for that one call. Both raise `ValueError` on a `Sample` built without `items` — a step
has no values to pick from until the pipeline runs.

A `Sample` with `items` also supports `len()`, iteration, and `.items`, so you can count
or read the values without choosing any.

## Things worth knowing

- **No `n` and no `frac` keeps everything.** The strategy still runs, so `top` sorts and
  keeps every record.
- **`n` and `frac` together raise `ValueError`**, at construction.
- **Missing requirements fail at construction, not at run time.** `by`, `step`, `center`
  and `std` are checked the moment you write the `Sample`.
- **`ascending` only affects `top`.** `bottom` always sorts lowest first.
- **A string `by` is a column read.** `compile()` checks that column exists by the time
  the step runs, and refuses the pipeline if it does not.
- **`replace=True` can return the same record twice**, and is the only way `n` may exceed
  the number of records available.
- **`stratified` gives every group at least one record**, so rare groups are
  over-represented in a small sample.
- **`weighted` and `gaussian` fall back to `uniform`** when all the weights come out 0,
  rather than raising.
- **Reusing one `Sample` value keeps drawing from the same random stream.** Two `.pick()`
  calls on the same object give different results even with a `seed`; the `seed` makes
  the whole sequence repeatable, not each call identical.
- **`Sample` reads every record before yielding any.** It has to, to choose. Nothing
  downstream starts until the input is exhausted.

## Where to go next

- [Sources & Seed](sources_and_seed.md) — the steps that produce the records you sample.
- [Building Pipelines](../guides/building_pipelines.md) — the other transforms.
- [LLM Steps](../guides/llm_steps.md) — where `.pick()` gets used.
- [Glossary](../glossary.md) — the precise meaning of record, column, step and sampling strategy.
- [API reference](../api.md) — the generated signature.
