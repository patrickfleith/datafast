# LLM Providers in Datafast

## Available Providers

Datafast offers a unified interface for multiple LLM providers through the [LiteLLM](https://github.com/BerriAI/litellm) library:

- **OpenAI** - For accessing GPT models.
- **Anthropic** - For accessing Claude models.
- **Gemini** - For accessing Google's Gemini models.
- **Ollama** - For accessing locally hosted models.
- **OpenRouter** - For accessing almost any models (proprietary or open source) through OpenRouter.

### Importing

```python
from datafast.llms import OpenAIProvider, AnthropicProvider, GeminiProvider, OllamaProvider, OpenRouterProvider
```

### Instantiating a Provider

Each provider can be instantiated with default parameters:

```python
# OpenAI (default: gpt-5-mini-2025-08-07)
openai_llm = OpenAIProvider()

# Anthropic (default: claude-3-5-haiku-latest)
anthropic_llm = AnthropicProvider()

# Gemini (default: gemini-2.0-flash)
gemini_llm = GeminiProvider()

# Ollama (default: gemma3:4b)
ollama_llm = OllamaProvider()

# OpenRouter (default: openai/gpt-4.1-mini)
openrouter_llm = OpenRouterProvider()
```

### With Custom Parameters

```python
openai_llm = OpenAIProvider(
    model_id="gpt-5-mini-2025-08-07",  # Custom model
    temperature=0.2,         # Lower temperature for more deterministic outputs
    max_completion_tokens=100,  # Limit token generation
    top_p=0.9,               # Nucleus sampling parameter
    frequency_penalty=0.1    # Penalty for frequent tokens
)

# Ollama with custom API endpoint
ollama_llm = OllamaProvider(
    model_id="llama3.2:latest",
    api_base="http://localhost:11434" # <--- this is the default url
)

# OpenRouter with different models
openrouter_llm = OpenRouterProvider(
    model_id="z-ai/glm-4.6",  # Access glm-4.6 via OpenRouter
    temperature=0.7,
    max_completion_tokens=500
)

# You can access many models through OpenRouter
openrouter_deepseek = OpenRouterProvider(model_id="deepseek/deepseek-r1-0528")
openrouter_qwen = OpenRouterProvider(model_id="qwen/qwen3-next-80b-a3b-instruct")
```

## API Keys

By default, providers look for API keys in environment variables:

- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Gemini: `GEMINI_API_KEY`
- Ollama: Uses `OLLAMA_API_BASE` (typically doesn't require an API key)
- OpenRouter: `OPENROUTER_API_KEY`

You can also provide keys directly:

```python
openai_llm = OpenAIProvider(api_key="your-api-key")
openrouter_llm = OpenRouterProvider(api_key="your-openrouter-key")
```

**Note**: Ollama typically runs locally and doesn't require an API key. You can set `OLLAMA_API_BASE` to specify a custom endpoint (defaults to `http://localhost:11434`).

!!! warning
    Note that `gpt-oss:20b` or `gpt-oss:120b` do not work well with structured output. Therefore we recommend you not to use them with datafast.

## About OpenRouter

[OpenRouter](https://openrouter.ai/) provides access to a wide variety of LLM models through a single API key. Model IDs follow the format `provider/model-name` (e.g., `deepseek/deepseek-r1-0528`, `qwen/qwen3-next-80b-a3b-instruct`). Visit [OpenRouter's models page](https://openrouter.ai/models) for the complete list.

## Generation Methods

### Simple Text Generation

```python
# Using a text prompt
response = openai_llm.generate(prompt="What is the capital of France?")
```

### Using Message Format

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant that provides brief answers."},
    {"role": "user", "content": "What is the capital of France?"}
]

response = openai_llm.generate(messages=messages)
```

### Structured Output with Pydantic

Define a Pydantic model for structured output:

```python
from pydantic import BaseModel, Field

class SimpleResponse(BaseModel):
    reasoning: str = Field(description="The reasoning behind the answer")
    answer: str = Field(description="The answer to the question")

# Generate structured response
response = openai_llm.generate(
    prompt="What is the capital of France? Provide an answer and reasoning.",
    response_format=SimpleResponse
)

# Access structured data
print(response.reasoning)  # "The capital of France is Paris because..."
print(response.answer)     # "Paris"
```

Using structured output is what enables us to create reliable dataset creation pipelines.

## Error Handling

The `generate` method can raise:

- `ValueError`: If neither prompt nor messages is provided, or if both are provided
- `RuntimeError`: If there's an error during generation
