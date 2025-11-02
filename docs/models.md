# Models

Datafast supports multiple LLM providers through a unified interface. Since model evolve fast, it is not uncommon for things to break.
Please find below a list of my favoriate models to use in `datafast` for each LLMProvider which provide a balance of cost, performance and stability.

See [LLM Providers](llms.md) for more details about supported arguments for each provider.

## Recommended Models by Provider

### OpenAI

**Default**: `gpt-5-mini-2025-08-07`

**Recommended Models**:

- **gpt-5-2025-08-07** - Most intelligent, capable, but also expensive. Only use for the most complex tasks.  
  *Pricing: \$1.25/\$10 per million I/O token*
  
- **gpt-5-mini-2025-08-07** - Intelligent, capable, and affordable.  
  *Pricing: \$0.25/\$2 per million I/O token*
  
- **gpt-5-nano-2025-08-07** - Tiny and cheap. Only use for simple tasks or testing.  
  *Pricing: \$0.05/\$0.4 per million I/O token*

```python
from datafast.llms import OpenAIProvider

# Using default model
llm = OpenAIProvider()

# Using a specific model
llm = OpenAIProvider(model_id="gpt-5-2025-08-07")
```

### Anthropic

**Default**: `claude-haiku-4-5-20251001`

**Recommended Models**:

- **claude-haiku-4-5-20251001** - Fast, efficient for most tasks.  
  *Pricing: \$1/\$5 per million I/O token*
  
- **claude-sonnet-4-5-20250929** - Most powerful model, but also most expensive.  
  *Pricing: \$3/\$15 per million I/O token*

```python
from datafast.llms import AnthropicProvider

# Using default model
llm = AnthropicProvider()

# Using a specific model
llm = AnthropicProvider(model_id="claude-sonnet-4-5-20251001")
```

### Google Gemini

**Recommended and default**: `gemini-2.5-flash-lite`

```python
from datafast.llms import GeminiProvider

# Using default model
llm = GeminiProvider()

# Using a specific model
llm = GeminiProvider(model_id="gemini-2.5-flash-lite")
```

### Ollama (Local Models)

**Recommended**: `gemma3:27b-it-qat`

Fast, capable, reliable, and does not take up too much vRAM.

```python
from datafast.llms import OllamaProvider

# Using recommended model
llm = OllamaProvider(model_id="gemma3:27b-it-qat")

# Custom API endpoint
llm = OllamaProvider(
    model_id="gemma3:27b-it-qat",
    api_base="http://localhost:11434"
)
```

### OpenRouter

There are many models available on OpenRouter, but here are some of our favorites:

- **qwen/qwen3-next-80b-a3b-instruct** - High capability
- **deepseek/deepseek-r1-0528** - Strong reasoning, cost-effective
- **z-ai/glm-4.6** - Balanced performance
- **meta-llama/llama-3.3-70b-instruct** - Versatile, open-source

```python
from datafast.llms import OpenRouterProvider

# Using a specific model
llm = OpenRouterProvider(model_id="deepseek/deepseek-r1-0528")

# Another example
llm = OpenRouterProvider(model_id="qwen/qwen3-next-80b-a3b-instruct")
```

!!! warning
    Avoid using `gpt-oss:20b` or `gpt-oss:120b` as they do not work well with structured output.

## More Details

For comprehensive information about LLM providers, API keys, generation methods, and advanced usage, see the [LLM Providers](llms.md) page.
