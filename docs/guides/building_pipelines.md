# Building Pipelines

## Minimal Pipeline

```python
from datafast import AddUUID, Map, Sink, Source

pipeline = (
    Source.list([{"text": "hello"}])
    >> Map(lambda r: {**r, "length": len(r["text"])})
    >> AddUUID()
    >> Sink.list()
)

rows = pipeline.run()
```

## Sources

Use `Source` when the input already exists:

- `Source.list(...)`
- `Source.file(...)`
- `Source.jsonl(...)`
- `Source.csv(...)`
- `Source.parquet(...)`
- `Source.huggingface(...)`

Use `Seed` when you want to generate the initial combinations declaratively.

```python
from datafast import Seed

seed = Seed.product(
    Seed.values("topic", ["robotics", "energy"]),
    Seed.values("language", ["en", "fr"]),
)
```

## Core Data Operations

- `AddUUID`: add a UUID field to each record
- `Map`: one record in, one record out
- `FlatMap`: one record in, many records out
- `Filter`: keep or drop records
- `Group`: aggregate records
- `Pair`: pair records together
- `Concat`: concatenate multiple sources or pipelines
- `Join`: relational-style joins between datasets

## Sinks

Use `Sink` to persist results:

- `Sink.jsonl(...)`
- `Sink.csv(...)`
- `Sink.parquet(...)`
- `Sink.hub(...)`
- `Sink.list()`
