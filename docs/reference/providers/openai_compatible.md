# OpenAI-compatible servers

```python
from datafast import openai_compatible

model = openai_compatible(
    "meta-llama/Llama-3.1-8B-Instruct",
    provider_id="vllm",
    api_base_url="http://localhost:8000/v1",
)
```

| | |
|---|---|
| **Factory** | `openai_compatible(model_id, *, provider_id, api_base_url=None, **config)` |
| **API key** | none by default — pass `api_key=` if your server wants one |
| **Transport** | Chat Completions, routed to LiteLLM as `openai/<model_id>` |
| **Extra needed** | none — the server already speaks the OpenAI wire format |

Every field on [Served models](../served_models.md) works here. This page covers what is
specific to this factory.

## When to use it

Reach for a named provider factory when one exists: `openai`, `anthropic`, `gemini`,
`mistral`, `openrouter`, `ollama`. Each knows its provider's models and quirks.

`openai_compatible` is for every other server that speaks the OpenAI wire format — vLLM,
llama.cpp, a gateway at work, a machine under your desk. You give it the address and the
name of the server, and it works.

## `provider_id` names the server

`provider_id` is required. Leave it out and Python raises a `TypeError`, because it is a
keyword-only argument with no default.

It names the **server** that serves the model: `vllm`, `llamacpp`, `tgi`, `my_gateway`.
It never names the wire format. These three values are rejected with a `ValueError`:

| Rejected | Why |
|---|---|
| `openai_compatible` | a wire format, not a server |
| `openai_api` | a wire format, not a server |
| `oai_compatible` | a wire format, not a server |

The rule exists because the wire format is only the shape of the request. Any number of
different servers speak it, and they do not do the same things. Datafast uses
`provider_id` to decide what the served model can do, and puts it in the trace name of
every request, so you can tell one server from another afterwards. `openai_compatible`
as a name would answer neither question.

## Provider ids are normalized

The name you pass is cleaned up first:

| Rule | Example |
|---|---|
| surrounding spaces are stripped | `" vllm "` → `vllm` |
| upper case becomes lower case | `"VLLM"` → `vllm` |
| hyphens become underscores | `"my-gateway"` → `my_gateway` |
| the llama.cpp spellings are folded into one | `"llama.cpp"`, `"llama_cpp"`, `"LLAMA-CPP"` → `llamacpp` |

Nothing else is changed, and apart from the wire-format names above, nothing is refused.
Ids stay free-form, so a server datafast has never heard of still works.

## The base URL and the key

`api_base_url` is the address of your server, including its path — usually something
ending in `/v1`. Pass it. It is optional in the signature, but a server datafast has no
profile for is treated even more cautiously without one.

No environment variable is read here. A self-hosted server often needs no key at all, so
none is required and none is invented. If yours wants one, pass `api_key="..."`, or pass
`env_key_name="MY_SERVER_KEY"` to read it from the environment instead.

## How capabilities are resolved

A provider id datafast has a profile for gets that profile. Everything else falls back to
`OPENAI_COMPATIBLE_CHAT`, the conservative profile:

| `provider_id` | Profile |
|---|---|
| `vllm` | `VLLM_CHAT` |
| `llamacpp` | `LLAMACPP_CHAT` |
| anything else | `OPENAI_COMPATIBLE_CHAT` |

The conservative profile assumes only what the wire format itself guarantees:

| | `OPENAI_COMPATIBLE_CHAT` |
|---|---|
| Default transport | Chat Completions, with Responses also allowed |
| Accepted parameters | `timeout` |
| Structured output | `prompted_json` |
| Batching | fallback concurrency |
| Modalities | text |

**The parameter row is the one that costs you.** `timeout` is the only one accepted.
`temperature`, `top_p`,
`frequency_penalty` and `max_completion_tokens` are all assumed absent, so under the
default `unsupported_params="warn"` policy they are dropped with a warning and the call
goes out without them. `prompted_json` means nothing is enforced: JSON is asked for in
the prompt and checked afterwards. Text only means an image part raises.

This is a floor, not a verdict. Your server probably does more. When it does, put the
extra parameters in `provider_params`, which go to the server unchecked:

```python
from datafast import openai_compatible

model = openai_compatible(
    "local-model",
    provider_id="my_gateway",
    api_base_url="http://localhost:8000/v1",
    provider_params={"temperature": 0.7},
)
```

## vLLM and llama.cpp

Both are reached through this factory, and both have a profile of their own:

| | `VLLM_CHAT` | `LLAMACPP_CHAT` |
|---|---|---|
| Transports | Chat, Responses | Chat |
| Accepted parameters | `temperature`, `top_p`, `frequency_penalty`, `max_completion_tokens`, `timeout` | same |
| Structured output | `json_schema` | `json_schema` |
| Batching | fallback concurrency | fallback concurrency |
| Modalities | text, image, video | text, image, audio, video, file |

```python
from datafast import openai_compatible

vllm = openai_compatible(
    "Qwen/Qwen3-8B",
    provider_id="vllm",
    api_base_url="http://localhost:8000/v1",
    temperature=0.7,
)
llamacpp = openai_compatible(
    "gemma-3-4b-it",
    provider_id="llama.cpp",
    api_base_url="http://localhost:8080/v1",
)
```

Because these servers take OpenAI-shaped requests, no parameter is renamed or rescaled on
the way out. `frequency_penalty` here means what it means on OpenAI. (Ollama is the
counter-example: it speaks its own API, so datafast handles it through `ollama()`
instead.)

## Where to go next

- [Served models](../served_models.md) — every configuration field, shared by all providers.
- [OpenAI](openai.md) — the hosted provider, for comparison.
- [Installation](../../installation.md) — API keys and the `.env` rules.
- [Served models guide](../../llms.md) — narrative introduction.
- [Glossary](../../glossary.md) — provider, transport, capability profile.
- [API reference](../../api.md) — generated signatures.
