# LLM Providers in Datafast

## Available Providers

Datafast offers a unified interface for multiple LLM providers through the [LiteLLM](https://github.com/BerriAI/litellm) library:

- **OpenAI** - For accessing GPT models.
- **Anthropic** - For accessing Claude models.
- **Gemini** - For accessing Google's Gemini models.
- **Ollama** - For accessing locally hosted models.

### Importing

```python
from datafast.llms import OpenAIProvider, AnthropicProvider, GeminiProvider, OllamaProvider
```

### Instantiating a Provider

Each provider can be instantiated with default parameters:

```python
# OpenAI (default: gpt-4.1-mini-2025-04-14)
openai_llm = OpenAIProvider()

# Anthropic (default: claude-3-5-haiku-latest)
anthropic_llm = AnthropicProvider()

# Gemini (default: gemini-2.0-flash)
gemini_llm = GeminiProvider()

# Ollama (default: gemma3:4b)
ollama_llm = OllamaProvider()
```

### With Custom Parameters

```python
openai_llm = OpenAIProvider(
    model_id="gpt-4o-mini",  # Custom model
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
```

## API Keys

By default, providers look for API keys in environment variables:

- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Gemini: `GEMINI_API_KEY`
- Ollama: Uses `OLLAMA_API_BASE` (typically doesn't require an API key)

You can also provide keys directly:

```python
openai_llm = OpenAIProvider(api_key="your-api-key")
```

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
