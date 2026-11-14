# Ollama

```python
from datafast import ollama

model = ollama()                                  # gemma4:12b, on your own machine
small = ollama("qwen3:0.6b", thinking=False)
```

| | |
|---|---|
| **Factory** | `ollama(model_id="gemma4:12b", **config)` |
| **API key** | none — Ollama is a daemon on your machine and takes no key |
| **Base URL** | `OLLAMA_API_BASE`, or `api_base_url=`, else `http://localhost:11434` |
| **Transport** | Chat Completions only, over LiteLLM's `ollama_chat` route |
| **Extra needed** | the daemon running, with the model already pulled |

Every field on [Served models](../served_models.md) works here. This page covers what is
specific to Ollama.

## Supported models

The served-model catalog lists no Ollama model. On a hosted provider the model name tells
datafast what it is talking to. Here it does not: **the model is whatever you pulled onto
the machine**, and the name is only a label you chose when you pulled it. So every Ollama
model resolves by what its name *contains*:

| Model name contains | Resolves to |
|---|---|
| `deepseek-r1` | `OLLAMA_REASONING_CHAT` |
| `deepseek-v3.1` | `OLLAMA_REASONING_CHAT` |
| `qwen3` | `OLLAMA_REASONING_CHAT` |
| `qwq` | `OLLAMA_REASONING_CHAT` |
| `gpt-oss` | `OLLAMA_REASONING_CHAT` |
| `magistral` | `OLLAMA_REASONING_CHAT` |
| `gemma4` | `OLLAMA_REASONING_CHAT` |
| `-thinking` | `OLLAMA_REASONING_CHAT` |
| anything else | `OLLAMA_CHAT` |

Each name is a whole model family that thinks, never a family where only some models do.
A wrong guess here sends a reasoning parameter to a model that rejects it, so a family
gets added only when all of it thinks. A single thinking model whose name says nothing —
`llama3.2` say — gets the plain chat profile, and the probe below is how you find out.

Three models the rest of this page uses:

| Model | Profile | Size |
|---|---|---|
| `gemma4:12b` | `OLLAMA_REASONING_CHAT` | ~7.6 GB — the factory default |
| `gemma3:4b` | `OLLAMA_CHAT` | ~3.3 GB |
| `qwen3:0.6b` | `OLLAMA_REASONING_CHAT` | ~0.5 GB |

Note `gemma3` and `gemma4`: one digit apart, different profiles, because gemma4 thinks
and gemma3 does not.

**The default is a big model.** `gemma4:12b` is around 7.6 GB and has to fit on your
machine, which is not true of any other provider's default. On a smaller machine pass a
lighter id — `gemma3:4b` at ~3.3 GB, or `qwen3:0.6b` at ~0.5 GB if you want reasoning.

## The two profiles

|  | `OLLAMA_CHAT` | `OLLAMA_REASONING_CHAT` |
|---|---|---|
| Transport | Chat Completions | Chat Completions |
| Reasoning | no | yes |
| Accepted parameters | `temperature`, `top_p`, `max_completion_tokens`, `timeout` | the same, plus `reasoning_effort` |
| Structured output | `json_schema` | `json_schema` |
| Batching | fallback concurrency | fallback concurrency |
| Modalities | text, image | text, image |

`thinking=True` and `thinking=False` work on the reasoning profile too; they are the
portable controls, and each resolves to one of the values below.

**Image is declared for every Ollama model, and most cannot see.** The profiles have no
way to know which weights you pulled, so they declare image support for all of them.
Datafast will happily send an image to a text-only model and the daemon will reject it.
Vision models such as `gemma3`, `gemma4` and `llama3.2-vision` do work. If you are not
sure, probe.

## Ask the daemon: `probe_capabilities()`

Because the pulled model is a property of the machine, the profiles are a guess. The
daemon knows the exact answer, so `probe_capabilities()` asks it:

```python
from datafast import ollama

model = ollama("qwen3:0.6b")
# model.probe_capabilities() -> frozenset({"completion", "thinking", "tools"})
```

It returns Ollama's own capability names — `completion`, `vision`, `audio`, `thinking`,
`tools`, `embedding`, `insert` — not datafast's vocabulary. Use it to check the two
things the name cannot tell you: whether the model can really see, and whether it can
really think.

It is opt-in and nothing calls it for you. It makes an HTTP call to the daemon, at the
same address your generate calls use, and raises if the model is not pulled or no daemon
is listening.

## Reasoning

`thinking=True` sends `reasoning_effort="low"`. `gpt-oss` honours the level, so `low`,
`medium` and `high` mean something there; every other thinking model treats any level as
plain on.

`thinking=False` sends `think=False`, which is Ollama's own parameter rather than an
effort. Two reasons it is done that way. Leaving the parameter out would not turn
reasoning off — `qwen3` thinks by default. And LiteLLM's usual route, mapping the effort
to a string, sends that literal string for `gpt-oss`, which Ollama rejects. Passing
`think` directly avoids both.

## `frequency_penalty` is not offered

Ollama's repetition control is `repeat_penalty`: a multiplier that is neutral at `1.0`,
where values **below** 1.0 *reward* repetition. LiteLLM renames `frequency_penalty` onto
it without rescaling the number, so an OpenAI-style `0.15` would arrive as strong
encouragement to repeat. Datafast therefore leaves `frequency_penalty` out of both
profiles. Pass `repeat_penalty` on its own scale instead — it goes through as a provider
parameter:

```python
from datafast import ollama

model = ollama("gemma4:12b", top_p=0.85, repeat_penalty=1.2)
```

This is an Ollama quirk. A self-hosted vLLM or llama.cpp server reached through
`openai_compatible` keeps the usual meaning of `frequency_penalty`.

## Batching

Ollama has no batch endpoint, so datafast issues concurrent single calls and warns that
it is doing so. Expected here, not a misconfiguration. Tune it with `max_concurrent`
(default 4). One local daemon serves one model at a time, so raising it far above the
default buys little.

## Where to go next

- [Served models](../served_models.md) — every configuration field, shared by all providers.
- [Model defaults](../../models.md) — the default model id of every provider.
- [Installation](../../installation.md) — `OLLAMA_API_BASE` and the `.env` rules.
- [Served models guide](../../llms.md) — narrative introduction.
- [API reference](../../api.md) — generated signatures.
