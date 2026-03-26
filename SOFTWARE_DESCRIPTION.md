# Software Description

## Overview

Datafast is a Python library for building synthetic-data pipelines from composable steps.

The repository has been refactored away from dataset-specific generator classes and toward a single pipeline execution model:

- sources and seeds create records
- transforms manipulate records
- LLM steps generate or evaluate content
- sinks persist output
- the runner handles batching, checkpoints, and resume

## Package Surface

The canonical package is `datafast`.

Top-level exports include:

- Core: `Record`, `Step`, `Pipeline`, `RunConfig`, `Runner`, `run_pipeline`
- Sources: `Source`, `Seed`
- Data ops: `Sample`, `Map`, `FlatMap`, `Filter`, `Group`, `Pair`, `Concat`, `Join`
- LLM transforms: `LLMStep`, `Classify`, `Score`, `Compare`, `Rewrite`, `Extract`
- Branching: `Branch`, `JoinBranches`
- Sinks: `Sink`, `JSONLSink`, `CSVSink`, `ListSink`, `ParquetSink`, `HubSink`
- Providers: `OpenAIProvider`, `AnthropicProvider`, `GeminiProvider`, `MistralProvider`, `OpenRouterProvider`, `OllamaProvider`

## Execution Model

Pipelines are chained with `>>` and executed with `pipeline.run(...)` or `run_pipeline(...)`.

Execution supports:

- batched LLM calls
- configurable LLM ordering
- step-by-step materialization
- checkpoint manifests
- resume after interruption

## Examples

Canonical example scripts live under `examples/scripts/`.

The repository also retains `datafast_new_design_document.md` as the architecture reference for the pipeline model.
