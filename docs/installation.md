# Installation & environment

What to install, what to set, and where datafast looks for it.

## Install

```bash
pip install datafast
```

Datafast needs Python **3.10 or newer**.

The base install is deliberately small. It carries every step, every provider factory,
and the JSONL, CSV and in-memory sinks — five runtime dependencies, all of them
imported at module scope:

| Dependency | What it does |
|------------|--------------|
| `litellm` | the only transport to every LLM provider |
| `pydantic` | structured output schemas |
| `httpx` | HTTP |
| `loguru` | logging |
| `python-dotenv` | `.env` loading |

There is no provider SDK in that list. LiteLLM reaches OpenAI, Anthropic, Gemini and
the rest over its own HTTP transport, so installing datafast does not install four
vendor libraries you will never import.

### Extras

Anything that needs a heavier dependency ships as an extra:

| Extra | Enables | Pulls in |
|-------|---------|----------|
| `datafast[parquet]` | `Source.parquet(...)`, `ParquetSink` | `pyarrow` |
| `datafast[hub]` | `HuggingFaceSource`, `HubSink` | `datasets`, `huggingface-hub` |
| `datafast[langfuse]` | Langfuse tracing | `langfuse` |
| `datafast[all]` | `parquet` + `hub` | — |

```bash
pip install "datafast[all]"
```

Reach for one of these steps without its extra and datafast raises an `ImportError`
naming the extra to install, so you will not have to guess which package is missing.

## Environment variables

### Provider API keys

Each provider factory reads one variable. You only need keys for the providers you
actually call — there is no configuration step that requires all of them.

| Variable | Used by |
|----------|---------|
| `OPENAI_API_KEY` | `openai()` |
| `ANTHROPIC_API_KEY` | `anthropic()` |
| `GEMINI_API_KEY` | `gemini()` |
| `MISTRAL_API_KEY` | `mistral()` |
| `OPENROUTER_API_KEY` | `openrouter()` |

Two factories need no key at all: `ollama()` talks to a local daemon, and
`openai_compatible()` takes whatever `api_key=` you pass it, since a self-hosted server
may want none.

You can always pass `api_key=` directly to a factory, which takes precedence over the
environment. An **empty** variable counts as unset rather than being sent as an empty
string.

### Other variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `OLLAMA_API_BASE` | where the Ollama daemon listens | `http://localhost:11434` |
| `HF_TOKEN` | Hugging Face auth for `HubSink` and `HuggingFaceSource` | a cached `huggingface-cli login` is used instead |
| `LANGFUSE_PUBLIC_KEY` | Langfuse tracing (required pair) | — |
| `LANGFUSE_SECRET_KEY` | Langfuse tracing (required pair) | — |
| `LANGFUSE_HOST` | Langfuse endpoint | Langfuse Cloud |
| `DATAFAST_LITELLM_SUPPRESS_DEBUG_INFO` | hide LiteLLM's provider help text | `1` (hidden) |

Set `DATAFAST_LITELLM_SUPPRESS_DEBUG_INFO` to `0`, `false`, `no` or `off` to let
LiteLLM's own diagnostics through — useful when a provider call fails for a reason
datafast cannot explain.

## Using a `.env` file

Put the variables in a `.env` file in your working directory and datafast picks them
up:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
HF_TOKEN=hf_...
```

Two details worth knowing:

- **It is loaded once**, the first time you construct a served model — not at import,
  and not again afterwards. Writing to `.env` after that point has no effect on the
  running process.
- **Real environment variables win.** The file never overrides a variable that is
  already set, so an exported key beats the file.

## Langfuse tracing

Install the extra and set the key pair:

```bash
pip install "datafast[langfuse]"
```

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

Tracing then switches itself on when you build a served model. Both keys are required:
with only one set, the automatic path stays quietly off, while an explicit
`configure_langfuse_tracing()` call raises `ValueError` and tells you which pair it
wants. Calling it with `enabled=False` turns tracing off. See
[Langfuse Tracing](guides/langfuse_tracing.md) for the full surface.

## Verifying the install

```python
import datafast

print(datafast.get_version())
```

Building a served model does **not** verify the key. Construction resolves whatever it
can find and stores `None` if there is nothing, so a missing key surfaces on the first
actual call, not at startup:

```python
from datafast import openai

model = openai()
print(model.api_key is None)  # True when OPENAI_API_KEY is unset
```

That is worth checking before a long run, since the pipeline will otherwise seed,
expand and reach the LLM step before anything complains.

## Logging

Datafast logs through loguru and reports per-step progress at `INFO`. Change the level
globally:

```python
from datafast import configure_logger

configure_logger(level="WARNING")
```

Logging is process-wide, so there is no per-run logging setting — `RunConfig` exposes
only what the runner actually reads.
