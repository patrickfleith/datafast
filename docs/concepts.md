# Concepts

## Records

A record is a Python dictionary. Every step reads records and yields records.

```python
{"topic": "battery chemistry", "language": "en"}
```

## Steps

Every pipeline node is a `Step`.

- sources create records
- transforms modify records
- sinks write records out

Steps are chained with `>>`.

## Pipelines

```python
pipeline = (
    Source.list([{"text": "hello"}])
    >> Map(lambda r: {**r, "length": len(r["text"])})
    >> Sink.list()
)
```

`Pipeline.run(...)` executes the chain and returns the final records.

## Seeds and Sampling

`Seed` is the main way to generate initial combinations declaratively.

```python
Seed.product(
    Seed.values("topic", ["robotics", "fusion"]),
    Seed.values("audience", ["general", "expert"]),
)
```

`Sample` lets you select from lists or record streams with controlled strategies.

## LLM Steps

`LLMStep` is the generic generation primitive.

Specialized steps build on the same model:

- `Classify`
- `Score`
- `Compare`
- `Rewrite`
- `Extract`

## Branching

Use `Branch` to fan out a record through multiple paths and `JoinBranches` to merge the results again.

This is useful for preference-style data, rewrite comparisons, or multi-strategy generation.

## Execution

The runner materializes step outputs one step at a time. That enables:

- checkpoints after each step
- resuming interrupted runs
- per-step inspection
- LLM call batching and ordering strategies
