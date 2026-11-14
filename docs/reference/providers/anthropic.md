# Anthropic

```python
from datafast import anthropic

model = anthropic()                                   # claude-haiku-5-5 at effort high
hardest_tasks = anthropic("claude-opus-5-5", reasoning_effort="medium")
```

| | |
|---|---|
| **Factory** | `anthropic(model_id="claude-haiku-5-5", **config)` |
| **API key** | `ANTHROPIC_API_KEY`, or `api_key=` |
| **Transport** | Chat Completions, the only one |
| **Extra needed** | none — LiteLLM reaches Anthropic over its own HTTP transport |

Every field on [Served models](../served_models.md) works here. This page covers what is
specific to Anthropic.

## Which model to use

| Use | Model | Effort | Input / output per MTok |
|---|---|---|---|
| Default | `claude-haiku-5-5` | `reasoning_effort="high"` | from $0.10 / $0.50 |
| Tasks that need maximum intelligence | `claude-opus-5-5` | `reasoning_effort="medium"` | $4 / $20 |

`claude-sonnet-5-5` and `claude-fable-5-1` are supported but not recommended. Sonnet has
a poor quality-to-cost ratio at $2 / $10 per MTok. Fable is very expensive at $10 / $50
per MTok.

A bare `anthropic()` is the recommended default. datafast sends `reasoning_effort="high"`
on `claude-haiku-5-5` unless you set `reasoning_effort` or `thinking=False` yourself.

## Supported models

Seven models are catalogued exactly:

| Model | Profile |
|---|---|
| `claude-haiku-5-5` | `ANTHROPIC_HIGH_EFFORT_CHAT` |
| `claude-opus-5-5` | `ANTHROPIC_ALWAYS_THINKING_CHAT` |
| `claude-sonnet-5-5` | `ANTHROPIC_BETWEEN_TOOLS_CHAT` |
| `claude-fable-5-1` | `ANTHROPIC_ALWAYS_THINKING_CHAT` |
| `claude-sonnet-5` | `ANTHROPIC_ADAPTIVE_CHAT` |
| `claude-sonnet-4-6` | `ANTHROPIC_CHAT` |
| `claude-haiku-4-5` | `ANTHROPIC_CHAT` |

Any other model id gets `ANTHROPIC_CHAT`, the provider default. There is no prefix rule:
a model gets a newer profile only because the catalog names it.

That matters for a model newer than your datafast version. If it reasons by default and
refuses a temperature, `ANTHROPIC_CHAT` does not know it. `thinking=False` would then send
nothing, and the model would reason, and bill for it, anyway.

## The profiles

`ANTHROPIC_HIGH_EFFORT_CHAT` is `ANTHROPIC_ADAPTIVE_CHAT` with one difference: it sends
effort `high` for `thinking=True` and when you set no effort. The table lists it under
the adaptive column.

|  | `ANTHROPIC_CHAT` | `ANTHROPIC_ADAPTIVE_CHAT` | `ANTHROPIC_BETWEEN_TOOLS_CHAT` | `ANTHROPIC_ALWAYS_THINKING_CHAT` |
|---|---|---|---|---|
| Models | `claude-haiku-4-5`, `claude-sonnet-4-6` | `claude-haiku-5-5`, `claude-sonnet-5` | `claude-sonnet-5-5` | `claude-opus-5-5`, `claude-fable-5-1` |
| Transport | Chat Completions | Chat Completions | Chat Completions | Chat Completions |
| Accepted parameters | `temperature`, `max_completion_tokens`, `timeout`, `thinking`, `reasoning_effort` | `max_completion_tokens`, `timeout`, `thinking`, `reasoning_effort` | same as adaptive | same as adaptive |
| Reasons by default | no | yes | yes | always |
| `thinking=False` | sends nothing | sends `thinking={"type": "disabled"}` | sends `thinking={"type": "between_tools"}` | raises `ValueError` |
| Accepted efforts | any value, forwarded unchecked | `low`, `medium`, `high`, `xhigh`, `max` | same as adaptive | same as adaptive |
| Structured output | `json_schema` | `json_schema` | `json_schema` | `json_schema` |
| Batching | native LiteLLM batch | native LiteLLM batch | native LiteLLM batch | native LiteLLM batch |
| Modalities | text, image, file | text, image, file | text, image, file | text, image, file |

## temperature

There are two different rules, and they are easy to mix up.

On `ANTHROPIC_CHAT`, `temperature` is accepted. Anthropic allows only `temperature=1`
while thinking is on, so datafast drops yours on a reasoning request and lets the
provider's own value apply. With reasoning off, yours is sent.

On the newer profiles, `temperature` is not accepted at all. These models reject
every value but their default, whether reasoning is on or off, so no request could carry
yours. Passing one drops it with a warning under the default `unsupported_params="warn"`
policy. Use `claude-haiku-4-5` if you need to set a temperature.

`top_p` and `frequency_penalty` are on no profile.

## Reasoning

`thinking=True` and `thinking=False` are the portable controls. `thinking=True` asks for
effort `low` on every profile except `ANTHROPIC_HIGH_EFFORT_CHAT`, where it asks for
`high`. `reasoning_effort=` picks a level yourself. With neither,
datafast sends `high` on `claude-haiku-5-5`. The other Claude 5.5 models use their own
default: `medium` on `claude-opus-5-5`, `high` on `claude-sonnet-5-5` and
`claude-fable-5-1`.

Off is where the profiles part, and this is the trap on this provider.

`claude-haiku-4-5` and `claude-sonnet-4-6` do not reason unless asked, so `thinking=False`
sends no parameter at all.

`claude-haiku-5-5` and `claude-sonnet-5` reason by default. Omitting the parameter is
therefore not off: the model reasons and bills for it. `reasoning_effort="none"` is not
off either. LiteLLM turns that value into dropping the parameter or the disable switch,
depending on the model. So datafast sends Anthropic's own switch instead:

```python
from datafast import anthropic

fast = anthropic("claude-haiku-5-5", thinking=False)   # sends thinking={"type": "disabled"}
careful = anthropic("claude-haiku-5-5", reasoning_effort="max")
```

`claude-sonnet-5-5` rejects that value, so datafast sends `thinking={"type": "between_tools"}`.

`claude-opus-5-5` and `claude-fable-5-1` always reason and have no off switch.
`thinking=False` raises a `ValueError` on them, because LiteLLM would otherwise drop the
setting and the model would reason anyway. Use `reasoning_effort="low"` for the least
reasoning.

The newer profiles refuse two effort values that read as harmless elsewhere.
`none` would look like off without being off, and `minimal` is silently mapped to `low`.
Both raise.

Two more things when reasoning is on with these models:

- **The trace needs an effort.** With no effort set, `thinking_blocks` arrive with empty
  text and `reasoning_content` is empty too, because Anthropic leaves out the written
  summary. Setting `reasoning_effort` or `thinking=True` makes LiteLLM request it.
- **Thinking spends the answer's budget.** Reasoning tokens and answer tokens share
  `max_completion_tokens`. A hard prompt capped at 10000 tokens sometimes returns no
  answer at all, so raise the cap when reasoning is on.

## Where to go next

- [Served models](../served_models.md) — every configuration field, shared by all providers.
- [Installation](../../installation.md) — `ANTHROPIC_API_KEY` and the `.env` rules.
- [Served models guide](../../llms.md) — narrative introduction.
- [API reference](../../api.md) — generated signatures.
