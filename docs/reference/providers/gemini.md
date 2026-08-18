# Gemini

```python
from datafast import gemini

model = gemini()                                  # gemini-3.5-flash-lite
flash = gemini("gemini-3.7-flash", reasoning_effort="low")
```

| | |
|---|---|
| **Factory** | `gemini(model_id="gemini-3.5-flash-lite", **config)` |
| **API key** | `GEMINI_API_KEY`, or `api_key=` |
| **Transport** | Chat Completions |
| **Extra needed** | none — LiteLLM reaches Gemini over its own HTTP transport, so `google-generativeai` is not required |

Every field on [Served models](../served_models.md) works here. This page covers what is
specific to Gemini.

## Supported models

Four models are catalogued exactly:

| Model | Profile | Transport |
|---|---|---|
| `gemini-3.7-flash` | `GEMINI_NO_MINIMAL_CHAT` | Chat |
| `gemini-3.5-flash` | `GEMINI_CHAT` | Chat |
| `gemini-3.5-flash-lite` | `GEMINI_CHAT` | Chat |
| `gemini-3.1-flash-lite` | `GEMINI_CHAT` | Chat |

Any other model id gets `GEMINI_CHAT`, the provider default. There is no name matching
here, so a Gemini model datafast has not heard of is assumed to behave like the
catalogued 3.5 models. That is usually right — but read the reasoning section below
before you trust it.

## The two profiles

|  | `GEMINI_CHAT` | `GEMINI_NO_MINIMAL_CHAT` |
|---|---|---|
| Transport | Chat | Chat |
| Reasoning | on by default, can be asked down | always on |
| `thinking=False` | sends `reasoning_effort="none"` | raises |
| `thinking=True` | sends `reasoning_effort="low"` | sends `reasoning_effort="low"` |
| Accepted efforts | any value, forwarded unchecked | `low`, `medium`, `high` |
| Accepted parameters | `temperature`, `top_p`, `frequency_penalty`, `max_completion_tokens`, `timeout`, `reasoning_effort` | the same six |
| Structured output | `json_schema` | `json_schema` |
| Batching | native LiteLLM batch | native LiteLLM batch |
| Modalities | text, image, audio, video, file | text, image, audio, video, file |

Audio and video are accepted here, which is not true of every cloud provider.

`temperature` and `top_p` are deprecated on Gemini 3 and later, and Google warns against
any temperature below 1.0. They stay accepted because they still work, and datafast
sends neither unless you ask for one.

## Reasoning

**Gemini 3 models reason by default.** Leaving the reasoning parameter out does not turn
reasoning off — the model keeps reasoning at its own level, and you keep paying for
those tokens. Turning it off therefore has to be explicit. With `thinking=False`,
datafast sends `reasoning_effort="none"` rather than omitting the parameter.

Even that is not silence. Gemini 3 has no off switch: LiteLLM turns `none` into the
model's lowest thinking level and hides the trace. Those tokens are still billed. On
Gemini, `thinking=False` means *as little reasoning as this model allows*, not *no
reasoning*.

`gemini-3.7-flash` goes further. Its lowest level is `low`; it rejects the `minimal`
level that `none` maps to. There is no value datafast could send, so `thinking=False`
raises instead of quietly reasoning at the model's own default, which is `medium`. Ask
for a level by name instead:

```python
from datafast import gemini

cheap = gemini("gemini-3.7-flash", reasoning_effort="low")
careful = gemini("gemini-3.7-flash", reasoning_effort="high")
```

At effort `low` the reasoning is real but invisible: no `reasoning_content` and no
thinking blocks come back, only an opaque signature. A readable trace needs `medium` or
`high`.

An uncatalogued model can have the same per-model minimum. It falls back to
`GEMINI_CHAT`, where `thinking=False` sends `none` — which such a model would reject.
If that happens, pass `reasoning_effort` yourself.

## Batching

Both profiles use LiteLLM's native batch path. Tune throughput with `max_concurrent`
(default 4) and `rpm_limit`.

## Where to go next

- [Served models](../served_models.md) — every configuration field, shared by all providers.
- [Installation](../../installation.md) — `GEMINI_API_KEY` and the `.env` rules.
- [Served models guide](../../llms.md) — narrative introduction.
- [API reference](../../api.md) — generated signatures.
