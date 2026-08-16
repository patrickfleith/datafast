# Served Models

A **served model** is a provider and a model together, with its configuration — the
object you construct and call. Datafast keeps direct provider support while using the
pipeline-first execution model.

## Available Providers

Each provider has a factory that returns a `ServedModel`:

- `openai`
- `anthropic`
- `gemini`
- `mistral`
- `openrouter`
- `ollama`
- `openai_compatible` — for self-hosted servers speaking the OpenAI wire format.
  Takes a required `provider_id` naming the server itself (`vllm`, `llamacpp`, …),
  since the wire format says nothing about which server is on the other end.

## Recommended Import Style

The factories are the public entry points; import them from the top level:

```python
from datafast import LLMStep, openrouter
```

Pass a model id, plus any configuration, to build a served model:

```python
from datafast import openai, ollama

model = openai("gpt-5.4-mini", temperature=0.7)
local = ollama("gemma4:12b")
```

`ServedModel` is exported for type annotations:

```python
from datafast import ServedModel

def build_step(model: ServedModel): ...
```

## Example

```python
from datafast import LLMStep, Source, Sink, openrouter

pipeline = (
    Source.list([{"topic": "robotics"}])
    >> LLMStep(
        prompt="Write one question about {topic}",
        input_columns=["topic"],
        output_column="question",
        model=openrouter("z-ai/glm-4.6"),
    )
    >> Sink.list()
)
```

## Provider-Specific Methods

Two providers expose a method beyond the shared `ServedModel` interface, because
the provider's API requires a step LiteLLM does not cover.

**Mistral — file uploads.** Mistral's chat API accepts a document only as an
uploaded file id; inline base64 is rejected. Upload it, pass the id as a file
part's `url`, and delete it when you are done. The upload is explicit so one id
can serve every request in a pipeline run, and the file stays in your account
until you remove it.

```python
from datafast import mistral
from datafast.llm import ContentPart

model = mistral()
file_id = model.upload_file("report.pdf")          # optional expiry=<int>
try:
    answer = model.generate(messages=[{"role": "user", "content": [
        ContentPart(type="file", url=file_id),
        ContentPart(type="text", text="Summarise this in one sentence."),
    ]}])
finally:
    model.delete_file(file_id)
```

**Ollama — capability probe.** Which Ollama model is pulled is a property of the
machine, not of the id, so Datafast resolves capabilities from name heuristics and
can only be approximately right. `probe_capabilities()` asks the daemon instead,
returning Ollama's own capability names — `completion`, `vision`, `audio`,
`thinking`, `tools`. It reaches the same daemon your generate calls do
(`api_base_url`, else `OLLAMA_API_BASE`, else `http://localhost:11434`), and
raises if the model is not pulled.

```python
from datafast import ollama

model = ollama("qwen3:0.6b")
if "vision" in model.probe_capabilities():
    ...  # only then is attaching an image worth doing
```

## Sampling Parameters

`temperature`, `top_p` and `frequency_penalty` are config fields on every
factory. Each is only sent to served models whose profile declares it; where the
profile omits it the value is dropped under your `unsupported_params` policy
(warn by default) rather than reaching the provider. OpenAI's reasoning models,
for instance, reject sampling controls outright, so datafast never forwards one.

```python
model = openai("gpt-4o-mini", temperature=0.7, top_p=0.85, frequency_penalty=0.2)
```

**Ollama does not take `frequency_penalty`.** It speaks its own API, where the
repetition control is `repeat_penalty` — a multiplier neutral at `1.0`, and
values *below* 1.0 reward repetition. LiteLLM renames `frequency_penalty` onto it
without rescaling, so an OpenAI-style `0.15` would arrive as strong repetition
encouragement and degenerate the output. Datafast therefore drops
`frequency_penalty` on Ollama and asks for `repeat_penalty` on its own scale,
which passes through as a provider parameter:

```python
model = ollama("gemma4:12b", top_p=0.85, repeat_penalty=1.2)
```

This is specific to Ollama. Self-hosted vLLM and llama.cpp servers are reached
over the OpenAI wire format through `openai_compatible`, where
`frequency_penalty` keeps its usual meaning.

## Environment Variables

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `MISTRAL_API_KEY`
- `OPENROUTER_API_KEY`
- `OLLAMA_API_BASE`
- `DATAFAST_LITELLM_SUPPRESS_DEBUG_INFO`

Ollama typically does not require an API key and instead uses the local API base.

`DATAFAST_LITELLM_SUPPRESS_DEBUG_INFO` defaults to enabled. Datafast sets
LiteLLM's `suppress_debug_info` flag when a served model is created so example runs do
not print LiteLLM provider help text such as the OpenRouter provider list banner.
Set `DATAFAST_LITELLM_SUPPRESS_DEBUG_INFO=0` if you want LiteLLM's debug/help
output back while troubleshooting.

## Optional Langfuse Tracing

Install the optional extra:

```bash
pip install "datafast[langfuse]"
```

Add the standard Langfuse variables to `.env`:

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

Datafast loads `.env` when a served model is created, and if the Langfuse keys are present it registers LiteLLM's native `langfuse` callback automatically.

If you want an explicit startup hook instead of auto-detection:

```python
from datafast import configure_langfuse_tracing

configure_langfuse_tracing()
```
