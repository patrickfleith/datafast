# OpenAI

```python
from datafast import openai

model = openai()                                  # gpt-6-luna, effort high
sol = openai("gpt-6.1-sol")                       # effort medium
```

| | |
|---|---|
| **Factory** | `openai(model_id="gpt-6-luna", **config)` |
| **API key** | `OPENAI_API_KEY`, or `api_key=` |
| **Transport** | Responses for reasoning models, Chat Completions otherwise |
| **Extra needed** | none — LiteLLM reaches OpenAI over its own HTTP transport |

Every field on [Served models](../served_models.md) works here. This page covers what is
specific to OpenAI.

## Recommended models

| Model | Effort | Use it for |
|---|---|---|
| `gpt-6-luna` | `high` | The default. Focused, high-volume generation. |
| `gpt-6.1-sol` | `medium` | Complex work that needs more than Luna. |

Datafast sends these efforts unless you set `thinking` or `reasoning_effort`.

`gpt-6.1-terra` and `gpt-6-astra` are supported but not recommended. Terra has poor
quality for its cost. Astra is very expensive: it costs five times as much as Sol.

## Supported models

Eight models are catalogued exactly:

| Model | Profile | Transport |
|---|---|---|
| `gpt-6-luna` | `OPENAI_GPT6_LUNA` | Responses |
| `gpt-6.1-sol` | `OPENAI_GPT6_SOL` | Responses |
| `gpt-6.1-terra` | `OPENAI_GPT6` | Responses |
| `gpt-6-astra` | `OPENAI_GPT6_ALWAYS_ON` | Responses |
| `gpt-5.5` | `OPENAI_RESPONSES` | Responses |
| `gpt-5.4` | `OPENAI_RESPONSES` | Responses |
| `gpt-5.4-mini` | `OPENAI_RESPONSES` | Responses |
| `gpt-5.4-nano` | `OPENAI_RESPONSES` | Responses |

Anything else resolves by **prefix**, so new models work without a catalog entry:

| Model id starts with | Resolves to |
|---|---|
| `gpt-5`, `gpt-6`, `o1`, `o3`, `o4` | `OPENAI_RESPONSES` — a reasoning model |
| anything else | `OPENAI_CHAT` — a plain chat model |

So `gpt-4o` and `gpt-4.1` get the chat profile, while an unreleased `gpt-6.2` would get
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
`temperature` to `gpt-6-luna` drops it with a warning under the default
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

fast = openai("gpt-6-luna", thinking=False)       # sends reasoning_effort="none"
careful = openai("gpt-6.1-sol", reasoning_effort="high")
```

The GPT-6 models accept different efforts:

| Model | Efforts | No setting | `thinking=True` | `thinking=False` |
|---|---|---|---|---|
| `gpt-6-luna` | `none` to `max` | `high` | `high` | sends `none` |
| `gpt-6.1-sol` | `low` to `max` | `medium` | `medium` | raises |
| `gpt-6.1-terra` | `none` to `max` | provider default | `low` | sends `none` |
| `gpt-6-astra` | `low` to `max` | provider default | `low` | raises |

The full scale is `none`, `low`, `medium`, `high`, `xhigh`, `max`. Sol and Astra have no
`none` effort, so datafast refuses `thinking=False` on them. An effort the model does not
accept raises `ValueError` before any request.

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
