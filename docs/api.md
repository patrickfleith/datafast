# API Overview

## Top-Level Package

The recommended import surface is the top-level `datafast` package.

```python
from datafast import Source, LLMStep, Sink, openrouter
```

## Core Types

- `Record`
- `Step`
- `Pipeline`
- `RunConfig`
- `Runner`
- `run_pipeline`
- `CheckpointManager`

## Sources and Seeds

- `Source.list(...)`
- `Source.file(...)`
- `Source.jsonl(...)`
- `Source.csv(...)`
- `Source.parquet(...)`
- `Source.tsv(...)`
- `Source.txt(...)`
- `Source.huggingface(...)`
- `Seed.values(...)`
- `Seed.product(...)`
- `Seed.zip(...)`
- `Seed.range(...)`

## Data Operations

- `Sample`
- `Map`
- `FlatMap`
- `Filter`
- `Group`
- `Pair`
- `Concat`
- `Join`

## LLM Operations

- `LLMStep`
- `Classify`
- `Score`
- `Compare`
- `Rewrite`
- `Extract`
- `configure_langfuse_tracing(...)`
- `is_langfuse_tracing_enabled()`

## Branching and Sinks

- `Branch`
- `JoinBranches`
- `Sink.jsonl(...)`
- `Sink.csv(...)`
- `Sink.parquet(...)`
- `Sink.hub(...)`
- `Sink.list()`

## Providers

- `OpenAIProvider`
- `AnthropicProvider`
- `GeminiProvider`
- `MistralProvider`
- `OpenRouterProvider`
- `OllamaProvider`

Factory helpers:

- `openai(...)`
- `anthropic(...)`
- `gemini(...)`
- `mistral(...)`
- `openrouter(...)`
- `ollama(...)`
