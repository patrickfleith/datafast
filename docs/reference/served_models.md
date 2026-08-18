# Served models

A **served model** is a provider and a model together, with its configuration — the
object you construct and hand to a step. `openai("gpt-5.5", temperature=0.3)` is a
served model; `"gpt-5.5"` on its own is just a model name.

This page covers everything shared by all seven providers: how you configure one, how
datafast decides what it can do, and what happens when you ask for something it cannot.
The per-provider pages cover the model tables and the traps specific to each.

## The factories

Seven module-level functions build served models. They are the public way to construct
one — `ServedModel` is exported for type annotations, not for instantiation.

| Factory | Default model | API key |
|---|---|---|
| `openai()` | `gpt-5.5` | `OPENAI_API_KEY` |
| `anthropic()` | `claude-haiku-4-5` | `ANTHROPIC_API_KEY` |
| `gemini()` | `gemini-3.5-flash-lite` | `GEMINI_API_KEY` |
| `mistral()` | `mistral-small-2603` | `MISTRAL_API_KEY` |
| `openrouter()` | `openai/gpt-5.4-mini` | `OPENROUTER_API_KEY` |
| `ollama()` | `gemma4:12b` | none — local daemon |
| `openai_compatible()` | none — `model_id` is required | whatever you pass |

Every factory takes `model_id` first and then any `ServedModelConfig` field as a keyword:

```python
from datafast import openai

model = openai("gpt-5.4-mini", temperature=0.2, max_tokens=512)
```

Constructing a served model does **not** verify the API key. It resolves whatever it can
find and stores `None` if there is nothing, so a missing key surfaces on the first call.

## Configuration

Every field below is a keyword argument on every factory.

### Identity and routing

| Field | Type | Default | Meaning |
|---|---|---|---|
| `provider_id` | `str` | set by the factory | names the server, never a wire format |
| `model_id` | `str` | per-factory | the model to call |
| `litellm_route` | `str` | derived | routing prefix handed to LiteLLM |
| `env_key_name` | `str \| None` | per-provider | environment variable holding the key |
| `api_key` | `str \| None` | from the environment | overrides the environment |
| `api_base_url` | `str \| None` | `None` | point at a different host |
| `endpoint_mode` | `"auto" \| "chat" \| "responses"` | `"auto"` | transport; `auto` takes the profile's default |

### Generation

| Field | Type | Default | Meaning |
|---|---|---|---|
| `temperature` | `float \| None` | `None` | provider default when unset |
| `top_p` | `float \| None` | `None` | nucleus sampling |
| `frequency_penalty` | `float \| None` | `None` | repetition control |
| `max_completion_tokens` | `int \| None` | `None` | output cap; `max_tokens=` is accepted as an alias |
| `thinking` | `bool \| None` | `None` | reasoning on or off, resolved per served model |
| `reasoning_effort` | `str \| None` | `None` | explicit effort, where the model accepts one |
| `reasoning_summary` | `str \| None` | `None` | ask for a written summary of the reasoning |

`thinking=True` and `thinking=False` are the portable controls: each served model knows
its own on and off values, so the same code means the same thing across providers.
`reasoning_effort` is the escape hatch when you want a specific level.

### Reliability and throughput

| Field | Type | Default | Meaning |
|---|---|---|---|
| `retry_policy` | `RetryPolicy` | `RetryPolicy()` | retries with backoff |
| `timeout` | `float \| None` | `None` | per-request timeout in seconds |
| `rpm_limit` | `int \| None` | `None` | client-side requests per minute |
| `max_concurrent` | `int` | `4` | parallel in-flight requests |

`RetryPolicy` takes `max_retries` (3), `base_delay` (1.0), `max_delay` (30.0) and
`jitter` (0.25).

Throughput lives here, on the served model — never on the runner. `RunConfig` has no
rate-limit or concurrency field, because two steps sharing one served model must share
one limit.

### Behaviour when something is unsupported

| Field | Type | Default | Meaning |
|---|---|---|---|
| `unsupported_params` | `"fail" \| "warn" \| "quiet"` | `"warn"` | what to do with a parameter this model does not accept |
| `provider_params` | `dict` | `{}` | passed straight through, unchecked |

Datafast knows which parameters each served model accepts. Set one it does not, and the
policy decides:

- **`warn`** (default) — the parameter is dropped and logged. The call still happens.
- **`fail`** — raises, so a silently ignored setting cannot skew a whole run.
- **`quiet`** — dropped without a word.

`warn` is the default because dropping is usually right and stopping a long run over a
cosmetic parameter is usually not. Reach for `fail` when the parameter is the point of
the run — a temperature sweep that silently ran at the provider default is wasted spend.

`provider_params` is the unchecked escape hatch for anything datafast does not model:

```python
from datafast import ollama

model = ollama(provider_params={"repeat_penalty": 1.2})
```

Nothing validates these. They go to the provider as given.

## How capabilities are resolved

What a served model can do is decided once, at construction, in this order:

1. **Explicit capabilities**, if you passed some.
2. **The served-model catalog** — an exact `(provider, model)` match.
3. **Provider heuristics** — OpenAI, Mistral and Ollama match on the model name, so an
   uncatalogued reasoning model still resolves to a reasoning profile.
4. **The provider default** — a profile for the provider as a whole.
5. **The OpenAI-compatible profile**, when an `api_base_url` says this is a self-hosted
   server of some kind.
6. **The conservative unknown profile** — text only, chat only, prompted JSON.

Each step down is a step further from certainty, and the profiles get more cautious as
you go. A model datafast has never heard of still works; it is just assumed to do less.

A **capability profile** is a named record shared by served models that behave alike. It
declares the endpoint modes, the accepted parameters, the modalities, the structured
output mode, the batch mode, and the reasoning contract.

### Structured output

`structured_output` is one of four modes, in descending order of strength:

| Mode | What the provider guarantees |
|---|---|
| `json_schema` | the response matches your schema |
| `json_object` | the response is valid JSON, shape not enforced |
| `prompted_json` | nothing — JSON is only asked for in the prompt |
| `none` | no JSON support at all |

This is distinct from a step's `parse_mode`, which splits a response that has already
come back. Structured output constrains the response in the first place.

### Batching

`batch_mode` says how a batch of calls is issued:

- `litellm_batch` — the provider's own batch path.
- `fallback_concurrency` — datafast issues concurrent single calls, bounded by
  `max_concurrent`, and warns that it is doing so.
- `none` — one at a time.

Either way an LLM step batches; the mode only decides the mechanism.

### Modalities

`modalities` gates what content parts a served model accepts — text, image, audio,
video, file, document. Sending an image to a text-only served model raises rather than
silently dropping it.

## Provider-specific methods

Two methods exist on some served models only:

- `probe_capabilities()` on Ollama asks the local daemon what the pulled model can
  actually do, since that is a property of the machine rather than of the name.
- `upload_file()` and `delete_file()` on Mistral, whose API takes files by id rather
  than inline.

## Calling a served model directly

You do not need a pipeline to use one:

```python
from datafast import openai

model = openai()
```

- `generate(prompt=..., messages=..., response_format=...)` returns text, or your
  Pydantic model when `response_format` is set.
- `generate_batch(messages=[...])` takes a list of message lists.
- `generate_response(...)` and `generate_batch_response(...)` return
  `NormalizedResponse` instead of bare text — `text`, `raw`, `reasoning_content`,
  `thinking_blocks`, `images`, `audio` and `output_items`.

Reach for these when you want one answer rather than a dataset. Inside a pipeline,
`LLMStep` calls them for you and adds checkpointing, batching and the `_model` column.

## Where to go next

- [OpenAI](providers/openai.md) — the model table, transports and traps for one provider.
- [Served models guide](../llms.md) — the narrative version of this surface.
- [Glossary](../glossary.md) — served model, provider, model, transport, capabilities.
- [Installation](../installation.md) — every API key and environment variable.
- [API reference](../api.md) — generated signatures.
