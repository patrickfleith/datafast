# Mistral

```python
from datafast import mistral

model = mistral()                                 # mistral-small-2603
large = mistral("mistral-large-2512", temperature=0.2)
```

| | |
|---|---|
| **Factory** | `mistral(model_id="mistral-small-2603", **config)` |
| **API key** | `MISTRAL_API_KEY`, or `api_key=` |
| **Transport** | Chat Completions, always |
| **Extra needed** | none — LiteLLM reaches Mistral over its own HTTP transport |

Every field on [Served models](../served_models.md) works here. This page covers what is
specific to Mistral.

## Supported models

Six models are catalogued exactly:

| Model | Profile |
|---|---|
| `mistral-medium-3-5` | `MISTRAL_REASONING_CHAT` |
| `mistral-small-2603` | `MISTRAL_REASONING_CHAT` |
| `mistral-large-2512` | `MISTRAL_CHAT` |
| `ministral-14b-2512` | `MISTRAL_CHAT` |
| `ministral-8b-2512` | `MISTRAL_CHAT` |
| `ministral-3b-2512` | `MISTRAL_CHAT` |

Anything else resolves by **name matching**, so new models work without a catalog entry:

| Model id contains | Resolves to |
|---|---|
| `magistral` | `MISTRAL_REASONING_CHAT` — a reasoning model |
| `-reasoning` | `MISTRAL_REASONING_CHAT` — a reasoning model |
| neither | `MISTRAL_CHAT` — a plain chat model |

**Both patterns matter.** `magistral` was Mistral's reasoning family, and LiteLLM still
keys its `reasoning_effort` support off that name. Reasoning has since moved into the
mainline models, which say so in the id instead: Ministral 3 ships `-reasoning` variants
beside the instruct ones. Match only one pattern and an uncatalogued reasoning model
lands on `MISTRAL_CHAT`, the profile with reasoning switched off, where the
`reasoning_effort` you set is dropped with a warning.

## The two profiles

|  | `MISTRAL_REASONING_CHAT` | `MISTRAL_CHAT` |
|---|---|---|
| Transport | Chat Completions | Chat Completions |
| Reasoning | yes | no |
| Accepted parameters | `temperature`, `max_completion_tokens`, `timeout`, `top_p`, `frequency_penalty`, `reasoning_effort` | `temperature`, `max_completion_tokens`, `timeout`, `top_p`, `frequency_penalty` |
| Structured output | `json_schema` | `json_schema` |
| Batching | native LiteLLM batch | native LiteLLM batch |
| Modalities | text, image, file | text, image, file |
| File input | uploaded id only | uploaded id only |

`reasoning_effort` is the only difference. Set it on a `MISTRAL_CHAT` model and it is
dropped with a warning under the default `unsupported_params="warn"` policy; set
`unsupported_params="fail"` to make that stop the run instead.

## Reasoning

`thinking=True` and `thinking=False` are the portable controls. On Mistral, `True` sends
`reasoning_effort="high"` and `False` sends `reasoning_effort="none"`.

**Mistral accepts only those two values.** `low` and `medium` are rejected by the API
with an HTTP 400. Datafast checks the value while it builds the request, so asking for
one of them raises a `ValueError` before anything is sent:

```python
from datafast import mistral

careful = mistral("mistral-medium-3-5", thinking=True)      # reasoning_effort="high"
fast = mistral("mistral-medium-3-5", thinking=False)        # reasoning_effort="none"
strict = mistral("mistral-medium-3-5", reasoning_effort="high")
```

Magistral models reason natively. `mistral-medium` and `mistral-small` accept
`reasoning_effort` on the server, but the installed LiteLLM only forwards it for a
subset of models, so datafast adds it to `allowed_openai_params` to push it through.
There is nothing to configure.

## Files

Mistral's chat API takes a document only as the id of a file you uploaded first. Inline
bytes and a plain URL are both rejected — with an HTTP 422 that says little — so
datafast refuses them itself and raises instead of sending them.

Two methods exist on Mistral served models and no others:

- `upload_file(path, purpose="ocr", expiry=None)` uploads a document and returns its id.
- `delete_file(file_id)` removes it again.

```python
from datafast import mistral
from datafast.llm import ContentPart

model = mistral()
file_id = model.upload_file("report.pdf")
part = ContentPart(type="file", url=file_id)
model.delete_file(file_id)
```

Uploading is a separate step on purpose: one id can serve every request in a pipeline
run, and the upload stays visible instead of hiding inside a `generate()` call. The file
then stays in your Mistral account until you delete it — datafast does not track it.

`expiry` asks Mistral to expire the file on its own, which is worth setting for a
throwaway upload. It is forwarded as given: Mistral documents the field as an integer
but not its unit. Left out, your account's own retention applies.

## Where to go next

- [Served models](../served_models.md) — every configuration field, shared by all providers.
- [Installation](../../installation.md) — `MISTRAL_API_KEY` and the `.env` rules.
- [Served models guide](../../llms.md) — narrative introduction.
- [API reference](../../api.md) — generated signatures.
