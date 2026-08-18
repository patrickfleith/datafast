# Anthropic

```python
from datafast import anthropic

model = anthropic()                                   # claude-haiku-4-5
sonnet = anthropic("claude-sonnet-5", thinking=False)
```

| | |
|---|---|
| **Factory** | `anthropic(model_id="claude-haiku-4-5", **config)` |
| **API key** | `ANTHROPIC_API_KEY`, or `api_key=` |
| **Transport** | Chat Completions, the only one |
| **Extra needed** | none — LiteLLM reaches Anthropic over its own HTTP transport |

Every field on [Served models](../served_models.md) works here. This page covers what is
specific to Anthropic.

## Supported models

Three models are catalogued exactly:

| Model | Profile |
|---|---|
| `claude-haiku-4-5` | `ANTHROPIC_CHAT` |
| `claude-sonnet-4-6` | `ANTHROPIC_CHAT` |
| `claude-sonnet-5` | `ANTHROPIC_ADAPTIVE_CHAT` |

Any other model id gets `ANTHROPIC_CHAT`, the provider default. There is no prefix rule:
`claude-sonnet-5` is on the second profile because the catalog names it, not because of
how it is spelled.

That matters for a model newer than your datafast version. If it behaves like
`claude-sonnet-5` — reasons by default, refuses a temperature — `ANTHROPIC_CHAT` does not
know it. `thinking=False` would then send nothing, and the model would reason, and bill
for it, anyway.

## The two profiles

|  | `ANTHROPIC_CHAT` | `ANTHROPIC_ADAPTIVE_CHAT` |
|---|---|---|
| Models | `claude-haiku-4-5`, `claude-sonnet-4-6` | `claude-sonnet-5` |
| Transport | Chat Completions | Chat Completions |
| Accepted parameters | `temperature`, `max_completion_tokens`, `timeout`, `thinking`, `reasoning_effort` | `max_completion_tokens`, `timeout`, `thinking`, `reasoning_effort` |
| Reasons by default | no | yes |
| `thinking=False` sends | nothing | `thinking={"type": "disabled"}` |
| Accepted efforts | any value, forwarded unchecked | `low`, `medium`, `high`, `xhigh`, `max` |
| Structured output | `json_schema` | `json_schema` |
| Batching | native LiteLLM batch | native LiteLLM batch |
| Modalities | text, image, file | text, image, file |

## temperature

There are two different rules, and they are easy to mix up.

On `ANTHROPIC_CHAT`, `temperature` is accepted. Anthropic allows only `temperature=1`
while thinking is on, so datafast drops yours on a reasoning request and lets the
provider's own value apply. With reasoning off, yours is sent.

On `ANTHROPIC_ADAPTIVE_CHAT`, `temperature` is not accepted at all. `claude-sonnet-5`
rejects every value but 1, whether reasoning is on or off, so no request could carry
yours. Passing one drops it with a warning under the default `unsupported_params="warn"`
policy.

`top_p` and `frequency_penalty` are on neither profile.

## Reasoning

`thinking=True` and `thinking=False` are the portable controls. `thinking=True` asks for
effort `low` on both profiles. `reasoning_effort=` picks a level yourself.

Off is where the profiles part, and this is the trap on this provider.

`claude-haiku-4-5` and `claude-sonnet-4-6` do not reason unless asked, so `thinking=False`
sends no parameter at all.

`claude-sonnet-5` reasons by default. Omitting the parameter is therefore not off: the
model reasons and bills for it. `reasoning_effort="none"` is not off either — LiteLLM
turns that value into dropping the parameter, which lands back on the same default. So
datafast sends Anthropic's own switch instead:

```python
from datafast import anthropic

fast = anthropic("claude-sonnet-5", thinking=False)   # sends thinking={"type": "disabled"}
careful = anthropic("claude-sonnet-5", reasoning_effort="high")
```

For the same reason `claude-sonnet-5` refuses two effort values that read as harmless
elsewhere. `none` would look like off while leaving the default in force, and `minimal`
is silently mapped to `low`. Both raise.

Two more things when reasoning is on with `claude-sonnet-5`:

- **The trace comes back empty.** `thinking_blocks` arrive with empty text, and
  `reasoning_content` is empty too. Anthropic leaves out the written summary unless
  asked for one, and datafast has no control that asks.
- **Thinking spends the answer's budget.** Reasoning tokens and answer tokens share
  `max_completion_tokens`. A hard prompt capped at 10000 tokens sometimes returns no
  answer at all, so raise the cap when reasoning is on.

## Where to go next

- [Served models](../served_models.md) — every configuration field, shared by all providers.
- [Installation](../../installation.md) — `ANTHROPIC_API_KEY` and the `.env` rules.
- [Served models guide](../../llms.md) — narrative introduction.
- [API reference](../../api.md) — generated signatures.
