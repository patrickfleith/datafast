# LLM Providers

Datafast keeps direct provider support while using the pipeline-first execution model.

## Available Providers

- `OpenAIProvider`
- `AnthropicProvider`
- `GeminiProvider`
- `MistralProvider`
- `OpenRouterProvider`
- `OllamaProvider`

## Recommended Import Style

For pipelines, use top-level factories when convenient:

```python
from datafast import LLMStep, openrouter
```

For explicit provider classes:

```python
from datafast import OpenAIProvider, OllamaProvider
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
LiteLLM's `suppress_debug_info` flag when a provider is created so example runs do
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

Datafast loads `.env` when a provider is created, and if the Langfuse keys are present it registers LiteLLM's native `langfuse` callback automatically.

If you want an explicit startup hook instead of auto-detection:

```python
from datafast import configure_langfuse_tracing

configure_langfuse_tracing()
```
