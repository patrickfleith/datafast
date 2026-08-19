# API Reference

Generated from the source docstrings, so it cannot drift from the code.

The recommended import surface is the top-level `datafast` package:

```python
from datafast import Source, LLMStep, Sink, openrouter
```

Everything on this page is exported from `datafast` directly. The served-model
internals (capability profiles, parsers, transport enums) live in `datafast.llm`
and are documented under [Served models](llms.md).

## Core

::: datafast.Record
::: datafast.Step
::: datafast.Pipeline
::: datafast.Runner
::: datafast.RunConfig
::: datafast.LLMExecutionStrategy
::: datafast.run_pipeline
::: datafast.CheckpointManager
::: datafast.PipelineChangedError
::: datafast.PipelineValidationError

## Sources and seeds

::: datafast.Source
::: datafast.HuggingFaceSource
::: datafast.Seed
::: datafast.SeedDimension

## Data operations

::: datafast.Sample
::: datafast.AddUUID
::: datafast.Map
::: datafast.FlatMap
::: datafast.Filter
::: datafast.Group
::: datafast.Pair
::: datafast.Concat
::: datafast.Join

## LLM operations

::: datafast.LLMStep
::: datafast.Classify
::: datafast.Score
::: datafast.Compare
::: datafast.Rewrite
::: datafast.Extract

## Branching

::: datafast.Branch
::: datafast.JoinBranches

## Sinks

::: datafast.Sink
::: datafast.JSONLSink
::: datafast.CSVSink
::: datafast.ParquetSink
::: datafast.ListSink
::: datafast.HubSink

## Served models

::: datafast.ServedModel
::: datafast.openai
::: datafast.anthropic
::: datafast.gemini
::: datafast.mistral
::: datafast.openrouter
::: datafast.ollama
::: datafast.openai_compatible

## Utilities

::: datafast.configure_logger
::: datafast.configure_langfuse_tracing
::: datafast.is_langfuse_tracing_enabled
::: datafast.get_version
