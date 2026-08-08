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

- `gemma4:12b`

Unlike the hosted providers, this default is only a default: it is whatever you
have pulled that counts. `gemma4:12b` is a ~7.6 GB multimodal reasoning model,
which is what the Ollama examples use; on a smaller machine pass a lighter id such
as `gemma3:4b` (~3.3 GB, no reasoning) or `qwen3:0.6b` (~0.5 GB, reasoning).

Override any of these by passing a model id to the factory, positionally or as
`model_id=...`.
