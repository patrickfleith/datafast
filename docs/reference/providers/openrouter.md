# OpenRouter

```python
from datafast import openrouter

model = openrouter()                              # openai/gpt-5.4-mini
glm = openrouter("z-ai/glm-4.6", temperature=0.2)
```

| | |
|---|---|
| **Factory** | `openrouter(model_id="openai/gpt-5.4-mini", **config)` |
| **API key** | `OPENROUTER_API_KEY`, or `api_key=` |
| **Transport** | Chat Completions only |
| **Extra needed** | none — LiteLLM reaches OpenRouter over its own HTTP transport |

Every field on [Served models](../served_models.md) works here. This page covers what is
specific to OpenRouter.

## OpenRouter is a router

OpenRouter does not serve models itself. It sits in front of many upstream servers and
forwards your request to one of them. So a model id names two things: who made the
model, then the model. That is the vendor prefix.

| Model id | Vendor | Model |
|---|---|---|
| `openai/gpt-5.4-mini` | `openai` | `gpt-5.4-mini` |
| `google/gemma-4-31b-it` | `google` | `gemma-4-31b-it` |
| `z-ai/glm-4.6` | `z-ai` | `glm-4.6` |

The prefix is part of the id, not something datafast adds. Leave it out and OpenRouter
does not know which model you mean. Copy the id from OpenRouter's own model page.

The provider is still OpenRouter. `openai/gpt-5.4-mini` on OpenRouter and `gpt-5.4-mini`
on `openai()` are two different served models with two different profiles.

## How models resolve

No OpenRouter model is catalogued, and there is no name heuristic. Every model id
resolves to the same capability profile, `OPENROUTER_CHAT`:

| Model id | Profile |
|---|---|
| `openai/gpt-5.4-mini` | `OPENROUTER_CHAT` |
| `google/gemma-4-31b-it` | `OPENROUTER_CHAT` |
| `z-ai/glm-4.6` | `OPENROUTER_CHAT` |
| anything else | `OPENROUTER_CHAT` |

That is deliberate. OpenRouter carries thousands of models from dozens of vendors, and a
per-model table would be wrong within a week.

## The profile

| | `OPENROUTER_CHAT` |
|---|---|
| Transport | Chat Completions — the only one |
| Accepted parameters | `temperature`, `top_p`, `frequency_penalty`, `max_completion_tokens`, `timeout` |
| Structured output | `json_schema` |
| Batching | native LiteLLM batch |
| Modalities | text, image |
| Reasoning | no |

`endpoint_mode="responses"` raises: the profile declares chat only.

**Read this profile as a best guess, not a promise.** One profile has to describe every
model behind the router, so it says what a typical model there does, not what yours
does. A text-only model will still refuse an image, and a small model may still refuse a
JSON schema. Datafast cannot know; the upstream server answers with an error.

## One model id, many endpoints

This is the trap on this provider. A single model id is often served by many upstream
endpoints, and **they disagree with each other about what they support**. `gemma-4-31b-it`
is served by around nineteen of them. Datafast's live tests found that some accept a
JSON schema and some reject it with "response format is not supported for model", and
that some answer an image content part with a `405`, all under the same model id.

Practically: without a pin, two identical calls can land on two different servers and
behave differently. Nothing in your code changed.

Pin the endpoint through `provider_params`, which goes to OpenRouter unchecked:

```python
from datafast import openrouter

model = openrouter(
    "google/gemma-4-31b-it",
    provider_params={
        "provider": {"only": ["novita/bf16"], "allow_fallbacks": False}
    },
)
```

`allow_fallbacks: False` is the half that matters. Without it OpenRouter quietly reroutes
when the pinned endpoint is busy, and the pin buys you nothing. Pin whenever a run
depends on structured output, images, or reproducible behaviour.

## Files

The profile declares **text and image only**. There is no file modality here.

Send a file content part and datafast raises a `ValueError` before the request leaves
your machine — `Modality 'file' is not supported by openrouter/<model>`. It does not
warn and drop; it stops. Use a provider whose profile declares files, or paste the text
into the prompt.

## Reasoning

The profile declares no reasoning. `reasoning_effort` and `thinking=True` are not
accepted, so they are dropped with a warning under the default
`unsupported_params="warn"` policy.

`thinking=False` sends nothing at all, because the profile has no off value to send. On
a model that reasons by default, that means it keeps reasoning — and keeps billing for
it. If you need reasoning controlled on OpenRouter, send the upstream vendor's own
fields through `provider_params`, which nothing validates.

## Batching

OpenRouter uses LiteLLM's native batch path, so an LLM step batches without the
concurrency-fallback warning. `max_concurrent` (default 4) and `rpm_limit` still apply.

## Where to go next

- [Served models](../served_models.md) — every configuration field, shared by all providers.
- [Installation](../../installation.md) — `OPENROUTER_API_KEY` and the `.env` rules.
- [Served models guide](../../llms.md) — narrative introduction.
- [Glossary](../../glossary.md) — provider, served model, transport, capability profile.
- [API reference](../../api.md) — generated signatures.
