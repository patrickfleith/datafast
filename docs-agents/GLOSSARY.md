# Glossary

## Pipelines

**Branch path** — One named lane inside a `Branch`, run independently on every incoming record and tagged so `JoinBranches` can merge the lanes back into a single record. _avoid:_ arm, fork, leg, track

**Checkpoint** — The on-disk state of a run: each completed step's records plus a manifest, written so an interrupted run resumes instead of paying for the same LLM calls twice. _avoid:_ cache, snapshot, save file

**Column** — A named key on a record, and the unit steps declare that they read and write. _avoid:_ field, key, attribute

**Compile** — The static check `Pipeline.compile()` runs before execution: step order, branch structure, and whether every column a step reads will exist by the time it runs. Not code generation. _avoid:_ validate, lint, build

**Dimension** — One axis of variation in a seed: a column (or a parent–child column pair) and the values it takes. _avoid:_ axis, variable, factor, facet

**Execution strategy** — The order in which the runner issues LLM calls when a step has several served models: `by_model`, `round_robin`, or `by_record`. _avoid:_ scheduling, ordering mode

**LLM step** — A step whose work is LLM calls, which the runner batches, orders, and checkpoints per call rather than per step. `LLMStep` is the general one; `Classify`, `Score`, `Compare`, `Rewrite`, and `Extract` are the specialized ones. _avoid:_ generation step, model step

**Manifest** — The checkpoint's index: each step's name, position, status, and record counts, plus a fingerprint of the pipeline that invalidates the checkpoint once the pipeline changes. _avoid:_ metadata file, state file

**Parse mode** — The step-level choice of how one raw LLM response is split into named columns (`text`, `json`, `xml`). Distinct from structured output, which is the provider constraining the response in the first place. _avoid:_ output format, parser mode

**Pipeline** — An ordered chain of steps composed with `>>` and run as a unit by `Pipeline.run()`. Linear by construction: `Branch` fans out and `JoinBranches` merges back, but the chain itself never forks. _avoid:_ graph, DAG, flow, workflow

**Prompt template** — The prompt text an LLM step fills in per record, with `{column}` placeholders naming the columns to inject. Given inline or as a file path. _avoid:_ prompt string, template string

**Record** — One unit of the dataset in flight, as a plain Python dict. Every step takes records and yields records; nothing else moves through a pipeline. _avoid:_ row, sample, item, example, datapoint

**Runner** — The engine that executes a pipeline, materializing each step's output in full before starting the next so checkpointing, resume, and LLM batching all have a boundary to work on. _avoid:_ executor, scheduler, orchestrator

**Sampling strategy** — How a `Sample` step chooses what to keep: `uniform`, `first`, `last`, `systematic`, `top`, `bottom`, `weighted`, `stratified`, or `gaussian`. Unrelated to LLM sampling parameters such as temperature. _avoid:_ selection mode, sampling method

**Seed** — The declarative starting point of a pipeline: dimensions combined by `Seed.product` or `Seed.zip` into the initial records. Never a random seed — that is the `seed` parameter on `Sample` and `Sink.hub`. _avoid:_ fixture, matrix, config source

**Sink** — A terminal step that writes records out — JSONL, CSV, Parquet, a Hub dataset, or an in-memory list — and yields them through unchanged, so sinks can be chained. _avoid:_ writer, exporter, output step

**Source** — The step that starts a pipeline by bringing records in from outside it: a Python list, a local file, or a Hugging Face dataset. A seed is the other way to start, building records rather than loading them. _avoid:_ loader, reader, input step

**Step** — The unit a pipeline is built from: an object that takes an iterable of records and yields records. Sources, transforms, and sinks are all steps, and so is a pipeline itself. _avoid:_ node, stage, operator, block

**Transform** — A step between the source and the sink that reads records and yields reshaped ones. _avoid:_ operator, processor, mapper

## Models and providers

**Capabilities** — What a served model can actually do: the intersection of the model's own traits and the provider's features, declared by Datafast up front rather than discovered at call time. _avoid:_ features, support matrix

**Capability profile** — A named, reusable Capabilities record shared by served models that behave alike. _avoid:_ preset, template

**Model** — The LLM itself, the weights being served. Not the object you call — that's a served model. _avoid:_ engine

**Provider** — The server that serves LLMs, whether cloud (OpenAI, Anthropic, Mistral, OpenRouter) or local (Ollama, vLLM, llama.cpp). It serves one or many models, and is not the same thing as the wire protocol used to reach it. _avoid:_ backend, vendor, host

**Provider factory** — The module-level function that builds a served model for one provider: `openai`, `anthropic`, `gemini`, `mistral`, `openrouter`, `ollama`, `openai_compatible`. These are the public way to construct one; `ServedModel` is exported for annotations. _avoid:_ constructor, helper, shortcut

**Served model** — A provider and a model together, with its configuration: the thing you construct and call. _avoid:_ target, configured target, deployment, model instance

**Served-model catalog** — The table mapping known provider-and-model pairs to their capabilities, with per-provider and per-model-family fallbacks for pairs it doesn't list. _avoid:_ registry, model list

**Structured output** — The provider constraining a whole response to a schema (`json_schema`, `json_object`, or `prompted_json`), which is a capability of the served model. Distinct from parse mode, which splits a response that has already come back. _avoid:_ json mode, schema mode

**Target** — Reserved for the pipeline sense: what a transformation aims at, such as a target audience, target length, or target column. Never the provider-plus-model pair, which is a served model. _avoid:_ served model, model target

**Transport** — The wire protocol and route used to reach a provider: chat completions versus the Responses API, plus the routing prefix handed to LiteLLM. Independent of the provider, so an OpenAI-shaped transport does not imply OpenAI is the provider. _avoid:_ provider, backend, endpoint
