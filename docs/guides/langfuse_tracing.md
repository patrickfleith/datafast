# Langfuse Tracing

Datafast supports optional Langfuse tracing through LiteLLM's native `langfuse` callback.

This integration is designed to be low-friction:

- install one optional extra
- put your Langfuse keys in `.env`
- create a provider as usual

If the Langfuse environment variables are present, Datafast enables tracing automatically when a provider is initialized.

## Install

```bash
pip install "datafast[langfuse]"
```

## Configure `.env`

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

`LANGFUSE_HOST` is optional if you use Langfuse Cloud, but it is useful for self-hosted deployments.

## Zero-Code Setup

Once the variables above exist, no extra tracing code is required.

```python
from datafast import LLMStep, Seed, Sink, openrouter

pipeline = (
    Seed.values("topic", ["robotics", "energy storage"])
    >> LLMStep(
        prompt="Write one short question about {topic}.",
        input_columns=["topic"],
        output_column="question",
        model=openrouter("z-ai/glm-4.6"),
    )
    >> Sink.list()
)

pipeline.run()
```

## Explicit Setup

If you prefer to make tracing opt-in in your script entrypoint, call:

```python
from datafast import configure_langfuse_tracing

configure_langfuse_tracing()
```

You can also pass credentials directly:

```python
from datafast import configure_langfuse_tracing

configure_langfuse_tracing(
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
    host="https://cloud.langfuse.com",
)
```

## Disable Tracing Explicitly

If Langfuse keys are present in the environment but you want to suppress tracing for a given run:

```python
from datafast import configure_langfuse_tracing

configure_langfuse_tracing(enabled=False)
```

That disables the LiteLLM Langfuse callback and prevents Datafast's auto-enable path from turning it back on later in the same process.

## What Datafast Adds To Traces

Datafast attaches LiteLLM metadata so traces are easier to interpret in Langfuse. This includes:

- provider and model id
- Datafast version
- step name and step type
- record index
- prompt index
- output index
- pipeline call id
- run-scoped session id

This metadata is added automatically for pipeline-managed LLM calls.

## Public API

Top-level helpers:

- `configure_langfuse_tracing(...)`
- `is_langfuse_tracing_enabled()`

## Notes

- Langfuse is optional. If you do not install `langfuse`, Datafast still works normally.
- If Langfuse keys are set but the `langfuse` package is missing, Datafast does not fail in auto-config mode; it skips tracing and emits a warning.
- The integration uses LiteLLM's native callback path rather than a custom wrapper, so provider behavior stays aligned with LiteLLM.
