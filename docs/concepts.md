# Key Concepts

datafast has one execution model, and it is small enough to hold in your head.

**Records** flow through **steps**. A **pipeline** is a chain of steps. The **runner**
executes that chain one step at a time and saves a **checkpoint** after each one.

Sources, seeds, sinks and LLM steps are all kinds of step. This page takes each concept
in the order you meet it when you build a dataset. For each one it says what it is, how
it works, and where it fits. The last sections cover how a pipeline is checked and run.

The [Full Glossary](glossary.md) has the one-line definition of every term in bold
below, plus the words datafast avoids for it.

## Record

A **record** is a plain Python dictionary. It is the only thing that moves through a
pipeline:

```python
{"topic": "battery chemistry", "level": "beginner"}
```

Its keys are **columns**. Steps declare the columns they read and write. That is how
`compile()` catches a typo in a column name before the run starts.

Records are not a class, and there is no schema object. A step that adds a column
returns a dict with one more key.

## Step

A **step** takes an iterable of records and yields records:

```python
def process(self, records: Iterable[Record]) -> Iterable[Record]: ...
```

That signature is the whole contract, and every part of datafast meets it. Steps come in
three roles:

- **Sources** bring records in from outside the pipeline. A **seed** is the other way to
  start: it builds records instead of loading them.
- **Transforms** read records and yield reshaped ones. `Map`, `Filter` and `Sample` are
  transforms, and so is every LLM step.
- **Sinks** write records out and pass them on unchanged.

A sink passes its records on, so it is not a dead end. You can chain several sinks to
write one dataset to two places in a single run.

## Source

A **source** starts a pipeline with records that already exist: a Python list, a local
file, or a Hugging Face dataset.

```python
from datafast import Source

source = Source.huggingface("squad", split="validation", columns=["question"])
```

There are eight: `Source.list`, `Source.file`, `Source.jsonl`, `Source.csv`,
`Source.tsv`, `Source.txt`, `Source.parquet` and `Source.huggingface`. Use a source when
you have material to transform or augment.

A source only goes first. `compile()` rejects a source anywhere else in the chain.

See [Source](reference/source.md) for every way to load records, and the options of each.

## Seed

A **seed** starts a pipeline by building records that do not exist yet. Most synthetic
datasets begin here, because there is nothing to load.

You declare **dimensions**, the axes the dataset should span, and combine them:

```python
from datafast import Seed

seed = Seed.product(
    Seed.values("topic", ["robotics", "fusion"]),
    Seed.values("audience", ["general", "expert"]),
)
```

`Seed.product` takes the cartesian product of its dimensions, so two topics and two
audiences produce four records. You describe the axes and the seed expands them. No
nested loops.

Dimensions come from `Seed.values`, `Seed.range` and `Seed.expand`. `Seed.product` and
`Seed.zip` combine them. A dimension on its own is not a step: only the combiners return
something a pipeline can start with.

The record count is the product of the dimension sizes, so it grows fast. `len(seed)`
tells you how many records you will get before you run anything. For a step that makes
one LLM call per record, that is also the number of calls.

`Seed` never means a random seed. The `seed` *parameter* on `Sample` and `Sink.hub` is
the random one.

See [Seed](reference/seed.md) for every dimension and combiner.

## Served model

Three words keep apart things that other libraries blur:

- The **model** is the LLM itself, the weights being served. `"gpt-6-luna"` is a model
  name.
- The **provider** is the server that serves it. OpenAI, Anthropic, Gemini, Mistral and
  OpenRouter are cloud providers. Ollama, vLLM and llama.cpp run on your own machine.
- The **served model** is a provider and a model together, with their configuration. It
  is the object you construct and hand to an LLM step.

```python
from datafast import openai

model = openai("gpt-6-luna", temperature=0.2)
```

Seven factories build one: `openai`, `anthropic`, `gemini`, `mistral`, `openrouter`,
`ollama` and `openai_compatible`. Each is a **provider factory**: a module-level function
that knows its provider's defaults, its API key variable, and how to reach it. The
`ServedModel` class is exported for type annotations. You build instances through the
factories.

Configuration lives on the served model: sampling settings, timeouts, retries, rate
limits and concurrency. The runner has none of these. Two steps that share a served
model share one rate limit, because the limit belongs to the model and not to the step.

Constructing a served model does not check the API key. A missing key shows up on the
first call.

The **transport** is the wire protocol used to reach a provider: chat completions or the
OpenAI Responses API, plus the routing prefix datafast hands to LiteLLM. It is separate
from the provider. A vLLM server speaks the OpenAI protocol, but OpenAI is not its
provider:

```python
from datafast import openai_compatible

model = openai_compatible(
    "qwen3",
    provider_id="vllm",
    api_base_url="http://localhost:8000/v1",
)
print(model.provider_id)  # vllm
```

datafast keeps the word **target** for the pipeline sense: a target audience, a target
length, a target column. It never means the provider and model pair. That is a served
model.

See [Served models](reference/served_models.md) for configuration, and
[Model defaults](models.md) for what each factory resolves to.

## Capabilities

A served model knows its **capabilities**: what this model on this provider can do. They
cover which parameters it accepts, which modalities it takes as input, which transports
it speaks, how it does structured output and batching, and how its reasoning switches on
and off.

datafast declares capabilities up front and does not discover them at call time. That
is why datafast can reject a request the model cannot serve with a clear message, before
the provider returns an error of its own. Send an image to a text-only served model and
datafast raises instead of dropping the image.

Capabilities are resolved once, at construction, in this order:

1. Capabilities you pass explicitly.
2. The **served-model catalog**, a table of known provider and model pairs.
3. A name match for OpenAI, Mistral and Ollama, so an uncatalogued reasoning model still
   resolves to a reasoning profile.
4. The provider's default.
5. A generic OpenAI-compatible profile, when you gave an `api_base_url`.
6. A cautious profile for an unknown model: text only, chat only, JSON asked for in the
   prompt.

```python
from datafast import openai

model = openai("gpt-6-luna")
print(model.capabilities.structured_output.value)  # json_schema
print(model.endpoint_mode.value)  # responses
```

Each entry in that chain is a **capability profile**: a named, reusable record shared by
served models that behave alike. A model datafast has never heard of still works. It
falls to a profile that assumes less.

Anthropic, Gemini and OpenRouter have no name match. An uncatalogued model there gets the
provider default, so reasoning may be off when you expect it on. Check the provider page
for your model.

A parameter the model does not accept is handled by the served model's
`unsupported_params` setting: `warn` drops it and logs, `fail` raises, `quiet` drops it
silently.

See [Served models](reference/served_models.md#how-capabilities-are-resolved) for the
full resolution rules.

## LLM step

An **LLM step** is where generation happens. For each record it fills a prompt template,
sends it to a served model, and writes the answer back as one or more new columns:

```python
from datafast import LLMStep, openai

step = LLMStep(
    prompt="Write one short question about {topic} for a {audience} reader",
    input_columns=["topic", "audience"],
    output_column="question",
    model=openai(),
)
```

The **prompt template** is the prompt text with `{column}` placeholders. Each placeholder
must be named in `input_columns`. Give the template inline, or as a path to a `.txt` or
`.md` file so long prompts live outside your code.

`LLMStep` is the general step, where you write the prompt. Five **specialized steps**
write it for you, so you declare the job instead of the wording:

| Step | The job |
|---|---|
| `Classify` | pick labels from a fixed set |
| `Score` | give a number inside a range |
| `Compare` | judge two columns against each other |
| `Rewrite` | rewrite text in a chosen way |
| `Extract` | pull structured values out of text |

An LLM step is still a transform: records in, records out. It can multiply records. Pass
several prompts, several served models, several languages or `num_outputs` above one,
and each input record makes one call per combination:

```text
prompts × served models × languages × num_outputs
```

Two prompts on two served models make four records out of each record in. Multiply
before you run, because that is the number of calls you pay for.

The runner treats LLM steps differently from other steps. It batches their calls,
orders them, and checkpoints them call by call.

See [LLM step](reference/llm_step.md) and
[Specialized LLM steps](reference/llm_specialized.md).

## Parse mode and structured output

Two separate mechanisms turn a model's answer into columns. They are easy to confuse.

**Parse mode** is a setting on the step. It decides how datafast splits one answer that
has already come back:

| `parse_mode` | What it does |
|---|---|
| `"text"` | the whole answer goes into one column |
| `"json"` | the answer is read as a JSON object, one column per key |
| `"xml"` | the answer is read as `<name>` tags, one column per tag |

```python
from datafast import LLMStep, openai

step = LLMStep(
    prompt="Write a question and its answer about {topic}",
    input_columns=["topic"],
    output_columns=["question", "answer"],
    parse_mode="json",
    model=openai(),
)
```

In `json` and `xml` mode, datafast adds the format instructions to your prompt. A key
the model leaves out arrives as an empty string with a warning, not an error.

**Structured output** is a capability of the served model. It is the provider forcing
the whole answer to match a schema before it comes back. There are four levels, from
strongest to weakest: `json_schema`, `json_object`, `prompted_json` and `none`.

In v1, steps rely on parse mode only. No step asks the provider for a schema. To get a
guaranteed schema, call a served model directly with `response_format`.

See [Structured output](guides/structured_output.md) for both, side by side.

## Sample

`Sample` keeps some records and drops the rest. It has two uses.

**As a step**, it thins the records flowing through. The most useful place is just
before an LLM step, so a large seed does not turn into a large bill:

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

**As a value**, it goes inside an LLM step in place of a list of prompts, served models
or languages. The step then uses only the prompts, models or languages the `Sample` picks,
so a run can draw on a large pool without calling every one of them.

How it chooses is the **sampling strategy**, and there are nine: `uniform`, `first`,
`last`, `systematic`, `top`, `bottom`, `weighted`, `stratified` and `gaussian`.
`Sample` checks each strategy's requirements when you construct it, before the pipeline
runs. A sampling strategy has nothing to do with LLM sampling settings such as
temperature.

`Sample` is a transform. It cannot start a pipeline.

See [Sample](reference/sample.md) for every strategy and parameter.

## Sink

A **sink** writes records out to a file, to a dataset on the Hugging Face Hub, or to a
list in memory:

```python
from datafast import Sink, Source

records = (Source.list([{"text": "hello"}]) >> Sink.jsonl("out/data.jsonl")).run()
```

There are five: `Sink.jsonl`, `Sink.csv`, `Sink.parquet`, `Sink.hub` and `Sink.list`.

A sink yields every record it receives, unchanged. Writing is a side effect, so `run()`
still gives you the records back. Chain two sinks and the same dataset lands in two
places in one run.

See [Sinks](reference/sinks.md) for each destination and its options.

## Data operations

These steps reshape records without calling an LLM. Everything between the source and
the sink that costs CPU time and no money:

| Step | What it does | For example |
|---|---|---|
| `Map(fn)` | one record in, one record out | strip the whitespace off every generated answer |
| `FlatMap(fn)` | one record in, zero or more records out | split a reply holding five questions into five records |
| `AddUUID()` | add a unique id column | give every row a stable `id` before pushing to the Hub |
| `Filter(...)` | keep or drop records by condition | drop the answers a `Score` step rated below 3 |
| `Group(by=...)` | collapse records that share a key into one record | gather every answer to one question into a list |
| `Pair(n=2, ...)` | combine records into pairs or larger tuples | pair a high-scoring answer with a low-scoring one for preference data |
| `Concat(*sources)` | stack several pipelines end to end | pool the English and the French rows into one dataset |
| `Join(right, on=...)` | merge two pipelines side by side on a key | attach each question's score back onto the question |

They are ordinary transforms. Put them anywhere in a pipeline, except `Concat`, which
starts one.

See [Data ops](reference/data_ops.md) for every parameter.

## Branching

Sometimes one record should be processed in more than one way: the same question
answered by two prompts, or by two served models. `Branch` sends every record down
several named **branch paths**, and `JoinBranches` merges the results back into one
record per input.

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
print(records[0])  # {'text': 'hello', 'shout_upper': 'HELLO', 'shout_length': '5'}
```

Each path is a step or a whole sub-pipeline. Every path gets its own copy of every
record, and the paths run one after another. `Branch` tags each output with the path
that made it. `JoinBranches` reads the tags, groups the outputs by input record, and
suffixes every column a path added with the path's name.

This is how preference data is built: two answers per record, side by side, ready to
score and compare.

Three rules apply. A `Branch` needs a matching `JoinBranches`, and `compile()` rejects
one without the other. A `Branch` cannot sit inside another branch path. And if two
paths rewrite the same existing column, only one version survives the join, so write to
new columns inside a path.

See [Branching](reference/branching.md) for join modes and suffixes.

## Pipeline

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

A **pipeline** is linear by construction. `Branch` splits into named paths and
`JoinBranches` merges them back, but the chain itself stays one line. Steps always run
in the order you wrote them.

A pipeline is itself a step, so you can compose a pipeline into a larger one. That is
how `Concat` takes several pipelines as its inputs, and how a branch path can hold more
than one step.

## Compile

**Compile** is the static check `Pipeline.compile()` runs before anything executes:

```python
from datafast import Map, Sink, Source

pipeline = Source.list([{"text": "hi"}]) >> Map(lambda r: r) >> Sink.list()
pipeline.compile()
```

It checks four things. The pipeline starts with a source or a seed. Nothing follows the
sinks. Every `Branch` has a matching `JoinBranches`. Every column a step reads exists by
the time that step runs. On the first problem it raises `PipelineValidationError`,
naming the step and its position.

`run()` calls `compile()` for you, so a structural mistake fails before the first LLM
call is paid for. Call it yourself when you want to check an expensive pipeline without
starting it.

Nothing is compiled in the usual sense. There is no code generation.

## The runner

The **runner** materializes each step's output in full before it starts the next step:

```python
records = list(step.process(iter(records)))
```

This is a deliberate trade. Streaming end to end would use less memory, but it leaves
no boundary at which to write a checkpoint, batch LLM calls, or report progress. Step
boundaries give all three:

- a checkpoint is written after every completed step,
- an LLM step sees all its records at once and can batch its calls,
- the log reports records in and records out for each step.

Datasets built with datafast are usually thousands of records, not millions. Holding one
step's output in memory is cheap. Losing a half-finished run is not.

When an LLM step has several served models, the **execution strategy** decides the order
the calls go out in. Pass it to `run()` as `llm_strategy`:

| `llm_strategy` | Order of calls |
|---|---|
| `by_model` (default) | every call for model A, then every call for model B |
| `round_robin` | A, B, A, B, and so on |
| `by_record` | record 1 on every model, then record 2 |

Records come out in the order their calls completed, so the strategy also sets the
order of the output. `by_model` is the default because consecutive calls to one provider
batch better.

`batch_size` on `run()` sets how many calls go into one batch. How many of them run in
parallel is `max_concurrent` on the served model.

## Checkpoints and the manifest

Pass a `checkpoint_dir` and the runner saves its progress:

```python
records = pipeline.run(checkpoint_dir="./checkpoints", resume=True)
```

A **checkpoint** is each completed step's records, one JSONL file per step, plus a
**manifest**. The manifest records every step's name, position, status and record
counts, and a fingerprint of the pipeline.

Inside an LLM step, checkpointing works call by call. Each finished call's record and
its call id are written as soon as the call returns. The step also saves a progress
file every `checkpoint_every` calls, 100 by default, and resume needs that file. A run
that dies after 700 of 1,000 calls resumes at call 701, so you do not pay for the first
700 twice. A run that dies before the first progress file, at call 60 say, runs the
whole step again. Lower `checkpoint_every` for slow or expensive steps.

The fingerprint covers the pipeline's structure: each step's name and class, including
the steps inside branch paths. Change the structure and resume with `resume=True`, and
the runner raises `PipelineChangedError` instead of mixing records from two pipelines.

The fingerprint does not cover what a step does. Change a prompt, a served model or the
function inside a `Map`, and resume carries on as if nothing changed. Point a changed
pipeline at a fresh checkpoint directory, or rerun from the changed step with
`resume_from`.

Steps are named after their class in the checkpoint. Name one yourself when a class
appears more than once:

```python
from datafast import Map

step = Map(lambda r: r).as_step("normalize")
print(step.name)  # normalize
```

See [Pipelines & execution](guides/pipelines_and_execution.md) for every option `run()`
takes.

## Metadata columns

Steps add columns that start with an underscore to record how a row was produced:

| Column | Added when |
|---|---|
| `_model` | an LLM step generated the row; holds the served model's id |
| `_prompt_index` | the step's `prompt` is a list |
| `_language` | the step's `language` is set |
| `_variation` | `Rewrite` made several variations per record |

They are ordinary columns. Nothing strips them, so they land in the written dataset next
to the generated columns and every row carries its own provenance.

`Branch` adds three more, `_branch_id`, `_branch_name` and `_branch_input_keys`, for
`JoinBranches` to read. The join removes them.

## Where to go next

- [Quickstart](quickstart.md): the shortest path from install to a stored dataset.
- [Full Glossary](glossary.md): the one-line definition of every term on this page.
- [Source](reference/source.md) and [Seed](reference/seed.md): every way to start.
- [LLM step](reference/llm_step.md): generation, and the steps that judge it.
- [Served models](reference/served_models.md): configuration and capabilities.
- [Pipelines & execution](guides/pipelines_and_execution.md): running, resuming and
  ordering calls.
- [API reference](api.md): every class and every parameter.
