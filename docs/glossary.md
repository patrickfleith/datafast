# Glossary

Datafast uses these terms precisely and consistently, in the documentation, in error
messages, and in the API. This page is the canonical definition of each one.

Every entry ends with an **Avoid** line: terms datafast deliberately does *not* use for
that concept, usually because they are ambiguous or belong to a different idea. They are
listed so that searching for the word you expected brings you to the word datafast uses.

## Pipelines

**Branch path** — One named lane inside a `Branch`, run independently on every incoming record and tagged so `JoinBranches` can merge the lanes back into a single record. **Avoid:** arm, fork, leg, track

**Checkpoint** — The on-disk state of a run: each completed step's records plus a manifest, written so an interrupted run resumes instead of paying for the same LLM calls twice. **Avoid:** cache, snapshot, save file

**Column** — A named key on a record, and the unit steps declare that they read and write. **Avoid:** field, key, attribute

**Compile** — The static check `Pipeline.compile()` runs before execution: step order, branch structure, and whether every column a step reads will exist by the time it runs. Not code generation. **Avoid:** validate, lint, build

**Dimension** — One axis of variation in a seed: a column (or a parent–child column pair) and the values it takes. **Avoid:** axis, variable, factor, facet

**Execution strategy** — The order in which the runner issues LLM calls when a step has several served models: `by_model`, `round_robin`, or `by_record`. **Avoid:** scheduling, ordering mode

**LLM step** — A step whose work is LLM calls, which the runner batches, orders, and checkpoints per call rather than per step. `LLMStep` is the general one; `Classify`, `Score`, `Compare`, `Rewrite`, and `Extract` are the specialized ones. **Avoid:** generation step, model step

**Manifest** — The checkpoint's index: each step's name, position, status, and record counts, plus a fingerprint of the pipeline that invalidates the checkpoint once the pipeline changes. **Avoid:** metadata file, state file

**Parse mode** — The step-level choice of how one raw LLM response is split into named columns (`text`, `json`, `xml`). Distinct from structured output, which is the provider constraining the response in the first place. **Avoid:** output format, parser mode

**Pipeline** — An ordered chain of steps composed with `>>` and run as a unit by `Pipeline.run()`. Linear by construction: `Branch` fans out and `JoinBranches` merges back, but the chain itself never forks. **Avoid:** graph, DAG, flow, workflow

**Prompt template** — The prompt text an LLM step fills in per record, with `{column}` placeholders naming the columns to inject. Given inline or as a file path. **Avoid:** prompt string, template string

**Record** — One unit of the dataset in flight, as a plain Python dict. Every step takes records and yields records; nothing else moves through a pipeline. **Avoid:** row, sample, item, example, datapoint

**Runner** — The engine that executes a pipeline, materializing each step's output in full before starting the next so checkpointing, resume, and LLM batching all have a boundary to work on. **Avoid:** executor, scheduler, orchestrator

**Sampling strategy** — How a `Sample` step chooses what to keep: `uniform`, `first`, `last`, `systematic`, `top`, `bottom`, `weighted`, `stratified`, or `gaussian`. Unrelated to LLM sampling parameters such as temperature. **Avoid:** selection mode, sampling method

**Seed** — The declarative starting point of a pipeline: dimensions combined by `Seed.product` or `Seed.zip` into the initial records. Never a random seed — that is the `seed` parameter on `Sample` and `Sink.hub`. **Avoid:** fixture, matrix, config source

**Sink** — A terminal step that writes records out — JSONL, CSV, Parquet, a Hub dataset, or an in-memory list — and yields them through unchanged, so sinks can be chained. **Avoid:** writer, exporter, output step

**Source** — The step that starts a pipeline by bringing records in from outside it: a Python list, a local file, or a Hugging Face dataset. A seed is the other way to start, building records rather than loading them. **Avoid:** loader, reader, input step

**Step** — The unit a pipeline is built from: an object that takes an iterable of records and yields records. Sources, transforms, and sinks are all steps, and so is a pipeline itself. **Avoid:** node, stage, operator, block

**Transform** — A step between the source and the sink that reads records and yields reshaped ones. **Avoid:** operator, processor, mapper

## Models and providers

**Capabilities** — What a served model can actually do: the intersection of the model's own traits and the provider's features, declared by Datafast up front rather than discovered at call time. **Avoid:** features, support matrix

**Capability profile** — A named, reusable Capabilities record shared by served models that behave alike. **Avoid:** preset, template

**Model** — The LLM itself, the weights being served. Not the object you call — that's a served model. **Avoid:** engine

**Provider** — The server that serves LLMs, whether cloud (OpenAI, Anthropic, Mistral, OpenRouter) or local (Ollama, vLLM, llama.cpp). It serves one or many models, and is not the same thing as the wire protocol used to reach it. **Avoid:** backend, vendor, host

**Provider factory** — The module-level function that builds a served model for one provider: `openai`, `anthropic`, `gemini`, `mistral`, `openrouter`, `ollama`, `openai_compatible`. These are the public way to construct one; `ServedModel` is exported for annotations. **Avoid:** constructor, helper, shortcut

**Served model** — A provider and a model together, with its configuration: the thing you construct and call. **Avoid:** target, configured target, deployment, model instance

**Served-model catalog** — The table mapping known provider-and-model pairs to their capabilities, with per-provider and per-model-family fallbacks for pairs it doesn't list. **Avoid:** registry, model list

**Structured output** — The provider constraining a whole response to a schema (`json_schema`, `json_object`, or `prompted_json`), which is a capability of the served model. Distinct from parse mode, which splits a response that has already come back. **Avoid:** json mode, schema mode

**Target** — Reserved for the pipeline sense: what a transformation aims at, such as a target audience, target length, or target column. Never the provider-plus-model pair, which is a served model. **Avoid:** served model, model target

**Transport** — The wire protocol and route used to reach a provider: chat completions versus the Responses API, plus the routing prefix handed to LiteLLM. Independent of the provider, so an OpenAI-shaped transport does not imply OpenAI is the provider. **Avoid:** provider, backend, endpoint
