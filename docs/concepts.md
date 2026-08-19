# Concepts

Datafast has one execution model, and it is small enough to hold in your head:

**Records** flow through **steps**. A **pipeline** is a chain of steps. The **runner**
executes that chain one step at a time, saving a **checkpoint** after each one.

Everything else — seeds, sinks, LLM steps, branching — is a kind of step. Every term
below is defined precisely in the [Glossary](glossary.md).

## Records

A record is a plain Python dictionary. It is the only thing that moves through a
pipeline:

```python
{"topic": "battery chemistry", "level": "beginner"}
```

Its keys are **columns**. Steps declare the columns they read and write, which is what
lets `compile()` catch a typo before the run starts.

Records are not a class, and there is no schema object. A step that adds a column
returns a dict with one more key.

## Steps

A step takes an iterable of records and yields records:

```python
def process(self, records: Iterable[Record]) -> Iterable[Record]: ...
```

That signature is the entire contract, and every part of datafast satisfies it. Steps
come in three roles:

- **Sources** bring records in from outside the pipeline — a list, a file, a Hugging
  Face dataset. A **seed** is the other way to start, building records instead of
  loading them.
- **Transforms** read records and yield reshaped ones. `Map`, `Filter` and `Sample` are
  transforms, and so is every LLM step.
- **Sinks** write records out and yield them through unchanged.

Because a sink yields its records, it is not a dead end: several sinks can be chained to
write one dataset to two places in a single run.

## Pipelines

Steps compose with `>>`:

```python
from datafast import AddUUID, Map, Sink, Source

pipeline = (
    Source.list([{"text": "hello"}, {"text": "goodbye"}])
    >> Map(lambda r: {**r, "length": len(r["text"])})
    >> AddUUID()
    >> Sink.list()
)

records = pipeline.run()
```

A pipeline is **linear by construction**. `Branch` fans out into named paths and
`JoinBranches` merges them back, but the chain itself never forks — there is no graph to
reason about, and steps always run in the order you wrote them.

A pipeline is itself a step, so a pipeline can be composed into a larger one. That is
how `Concat` takes several pipelines as its inputs.

## Compile

`Pipeline.compile()` is a static check that runs before any execution:

```python
from datafast import Map, Sink, Source

pipeline = Source.list([{"text": "hi"}]) >> Map(lambda r: r) >> Sink.list()
pipeline.compile()
```

It checks that the pipeline starts with a source or a seed, that nothing follows the
sinks, that every `Branch` is matched by a `JoinBranches`, and that every column a step
reads will exist by the time it runs. On the first problem it raises
`PipelineValidationError` naming the step and the position.

`run()` calls `compile()` for you, so a structural mistake fails immediately rather than
after the first few LLM calls have been paid for. Calling it yourself is worth it when
the pipeline is expensive and you want to check it without starting it.

Compile is a static check, not code generation — nothing is compiled in the usual sense.

## The runner

The runner materializes each step's output **in full** before starting the next one:

```python
records = list(step.process(iter(records)))
```

This is a deliberate trade. Streaming end-to-end would use less memory, but there would
be no boundary at which to write a checkpoint, batch a set of LLM calls, or report
progress. Step boundaries give all three:

- a checkpoint is written after every completed step,
- an LLM step can see all its records at once and batch its calls,
- the log reports records in and records out, per step.

Datasets in this library are typically thousands of records, not millions, so holding
one step's output in memory is affordable and being able to resume is not.

## Checkpoints and the manifest

Pass a `checkpoint_dir` and the runner saves its progress:

```python
records = pipeline.run(checkpoint_dir="./checkpoints", resume=True)
```

A checkpoint is each completed step's records plus a **manifest** — an index recording
every step's name, position, status and record counts, along with a fingerprint of the
pipeline itself.

That fingerprint is what makes resuming safe. If you edit the pipeline and resume
against an old checkpoint, the fingerprint no longer matches and the runner raises
`PipelineChangedError` instead of stitching together records from two different
pipelines.

Inside an LLM step, checkpointing is **per call** rather than per step. A run
interrupted halfway through a thousand generations resumes at call 501, so the first
five hundred are not paid for twice. See
[Pipelines & execution](guides/pipelines_and_execution.md) for the execution controls.

Steps are named for the checkpoint by their class, and you can name them yourself when
one class appears more than once:

```python
from datafast import Map

step = Map(lambda r: r).as_step("normalize")
print(step.name)  # normalize
```

## Two ways to start

Every pipeline begins with records, and there are two ways to get them:

- **`Source`** loads records that already exist: `Source.list`, `Source.file`,
  `Source.jsonl`, `Source.csv`, `Source.tsv`, `Source.txt`, `Source.parquet` and
  `Source.huggingface`.
- **`Seed`** builds records that do not exist yet, from **dimensions** you declare with
  `Seed.values`, `Seed.range` or `Seed.expand`, combined by `Seed.product` or
  `Seed.zip`:

```python
from datafast import Seed

seed = Seed.product(
    Seed.values("topic", ["robotics", "fusion"]),
    Seed.values("audience", ["general", "expert"]),
)
```

`Seed.product` takes the cartesian product of its dimensions, so two topics and two
audiences produce four records. This is what makes coverage declarative: you describe
the axes the dataset should span, and the seed expands them, rather than writing nested
loops.

Note that `Seed` never means a random seed. The `seed` *parameter* on `Sample` and
`Sink.hub` is the random one.

## Metadata columns

Steps add columns prefixed with an underscore to record how a row was produced: `_model`
names the served model that generated it, `_prompt_index` and `_language` appear when a
step fans out across several prompts or languages, `_variation` when it produces several
outputs per record, and `_branch_name` and `_branch_input_keys` record which branch path
a record came from.

They are ordinary columns — nothing strips them — so they land in the written dataset
alongside the generated fields, and every row carries its own provenance.

## Where to go next

- [Glossary](glossary.md) — the precise definition of every term used here.
- [Quickstart](quickstart.md) — the shortest path from install to a stored dataset.
- [Data ops](reference/data_ops.md) — the transforms and sinks available.
- [LLM step](reference/llm_step.md) — generation and the evaluation steps.
- [Served models](llms.md) — how a provider and a model are configured together.
- [API reference](api.md) — every class and every parameter.
