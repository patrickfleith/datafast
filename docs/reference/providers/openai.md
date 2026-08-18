# OpenAI

```python
from datafast import openai

model = openai()                                  # gpt-5.5
mini = openai("gpt-5.4-mini", reasoning_effort="low")
```

| | |
|---|---|
| **Factory** | `openai(model_id="gpt-5.5", **config)` |
| **API key** | `OPENAI_API_KEY`, or `api_key=` |
| **Transport** | Responses for reasoning models, Chat Completions otherwise |
| **Extra needed** | none — LiteLLM reaches OpenAI over its own HTTP transport |

Every field on [Served models](../served_models.md) works here. This page covers what is
specific to OpenAI.

## Supported models

Four models are catalogued exactly:

| Model | Profile | Transport |
|---|---|---|
| `gpt-5.5` | `OPENAI_RESPONSES` | Responses |
| `gpt-5.4` | `OPENAI_RESPONSES` | Responses |
| `gpt-5.4-mini` | `OPENAI_RESPONSES` | Responses |
| `gpt-5.4-nano` | `OPENAI_RESPONSES` | Responses |

Anything else resolves by **prefix**, so new models work without a catalog entry:

| Model id starts with | Resolves to |
|---|---|
| `gpt-5`, `o1`, `o3`, `o4` | `OPENAI_RESPONSES` — a reasoning model |
| anything else | `OPENAI_CHAT` — a plain chat model |

So `gpt-4o` and `gpt-4.1` get the chat profile, while an unreleased `gpt-5.6` would get
the reasoning profile the day it ships. The prefix rule is a guess, and a deliberately
conservative one: treating a reasoning model as a chat model would send it sampling
parameters it rejects.

## The two profiles

|  | `OPENAI_RESPONSES` | `OPENAI_CHAT` |
|---|---|---|
| Default transport | Responses | Chat Completions |
| Also supports | Chat | Responses |
| Reasoning | yes | no |
| Accepted parameters | `max_completion_tokens`, `timeout`, `thinking`, `reasoning_effort`, `reasoning_summary`, `previous_response_id` | `temperature`, `max_completion_tokens`, `timeout`, `top_p`, `frequency_penalty` |
| Structured output | `json_schema` | `json_schema` |
| Batching | fallback concurrency | native LiteLLM batch |
| Modalities | text, image, file | text, image, file |

The parameter lists are the important difference. **Reasoning models reject sampling
controls**, so `temperature` and `top_p` are not in the Responses set. Passing
`temperature` to `gpt-5.5` drops it with a warning under the default
`unsupported_params="warn"` policy — the call still happens, at the provider's own
temperature.

If a temperature sweep is the point of your run, use a chat model, or set
`unsupported_params="fail"` so the drop stops the run instead of quietly skewing it.

Override the transport with `endpoint_mode="chat"` or `"responses"` when you need the
non-default one; both profiles support both.

## Reasoning

`thinking=True` and `thinking=False` are the portable controls. On OpenAI, off is sent
as `reasoning_effort="none"` rather than by omitting the parameter.

**This matters, and it is the trap on this provider.** The default effort is per-model:
`none` for `gpt-5.4` and its mini and nano variants, but `medium` for `gpt-5.5`. A
`gpt-5.5` request with the parameter simply omitted would reason — and bill for it —
despite `thinking=False`. Every GPT-5.x model accepts `none`, so datafast sends it
explicitly.

```python
from datafast import openai

fast = openai("gpt-5.5", thinking=False)          # sends reasoning_effort="none"
careful = openai("gpt-5.5", reasoning_effort="high")
```

`reasoning_summary` asks for a written summary of the reasoning. Two things to know:
`reasoning_content` is a summary written after the fact, not the trace itself, and it
arrives only when you ask for one. Cheap reasoning returns an empty summary, so a
readable one needs effort `medium` or above and a prompt worth summarising.

## Batching

The chat profile uses LiteLLM's native batch path. The Responses profile has none, so
datafast falls back to bounded concurrency and says so in a warning — expected on
reasoning models, not a misconfiguration. Tune it with `max_concurrent` (default 4) and
`rpm_limit`.

## Where to go next

- [Served models](../served_models.md) — every configuration field, shared by all providers.
- [Installation](../../installation.md) — `OPENAI_API_KEY` and the `.env` rules.
- [Served models guide](../../llms.md) — narrative introduction.
- [API reference](../../api.md) — generated signatures.
