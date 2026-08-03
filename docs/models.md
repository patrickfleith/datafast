# Model Defaults

These are the current default model IDs each provider factory uses when you
don't pass one.

## OpenAI

- `gpt-5.5`

## Anthropic

- `claude-haiku-4-5`

## Gemini

- `gemini-3.1-flash-lite`

## Mistral

- `mistral-small-2603`

## OpenRouter

- `openai/gpt-5.4-mini`

## Ollama

- `gemma3:4b`

Override any of these by passing a model id to the factory, positionally or as
`model_id=...`.
