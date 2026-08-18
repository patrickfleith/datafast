# Multimodal input

Some served models take more than text. You send an image, a sound file, a video or a
document alongside your prompt, and datafast turns it into whatever shape the provider
expects.

You build these with `ContentPart`. A message's `content` becomes a list of parts instead
of a string:

```python
from datafast import openai
from datafast.llm import ContentPart

messages = [
    {
        "role": "user",
        "content": [
            ContentPart(type="text", text="What is in this picture?"),
            ContentPart(type="image", url="https://example.com/cat.png"),
        ],
    }
]

model = openai()
answer = model.generate(messages=messages)
```

Multimodal input is a **direct-call feature**. Pipeline steps build their messages from a
prompt template, so an image goes in through `ServedModel.generate()` — see
[Calling a served model](../reference/served_models.md).

## `ContentPart`

One frozen dataclass covers every kind of attachment.

| Field | Type | Meaning |
|---|---|---|
| `type` | `str` | `"text"`, `"image"`, `"audio"`, `"video"`, `"file"` or `"document"` |
| `text` | `str \| None` | the text, for a text part |
| `url` | `str \| None` | a link to the media — or, for a file part, an uploaded file id |
| `data` | `str \| None` | base64 bytes, or a complete `data:` URI |
| `media_type` | `str \| None` | the MIME type, e.g. `"image/png"`, `"application/pdf"` |
| `media_id` | `str \| None` | a caching id, forwarded only where the provider supports it |
| `filename` | `str \| None` | required by OpenAI's Responses API beside inline file data |
| `provider_options` | `dict` | extra keys merged into the part, unchecked |

You never send a `ContentPart` as-is. Datafast rewrites it into the provider's own part
shape before the request goes out, so the same code works across providers.

## What each type needs

| Type | Accepts | Becomes |
|---|---|---|
| `text` | `text` | `{"type": "text", ...}` |
| `image` | `url` **or** `data` | `{"type": "image_url", ...}` |
| `audio` | `data` only | `{"type": "input_audio", ...}` |
| `video` | `url` **or** `data` | `{"type": "video_url", ...}` |
| `file` / `document` | `url` (as a file id) **or** `data` | `{"type": "file", ...}` |

### Images

Either a link or the bytes:

```python
from datafast.llm import ContentPart

from_link = ContentPart(type="image", url="https://example.com/cat.png")
from_bytes = ContentPart(type="image", data="<base64>", media_type="image/png")
```

Bytes need `media_type`, because datafast builds a `data:` URI from the two. Without it
you get a `ValueError` naming the missing field. If you already have a complete
`data:image/png;base64,...` string, pass it as `data` and leave `media_type` out.

Sending the bytes is often the more portable choice: a provider that fetches URLs
server-side cannot always reach your host.

### Audio

Audio is the one type with **no URL form**. Chat audio APIs take the bytes, so a part with
only a `url` raises `ValueError`.

```python
from datafast.llm import ContentPart

part = ContentPart(type="audio", data="<base64>", media_type="audio/wav")
```

Providers want a bare format rather than a MIME type here, so datafast turns
`"audio/wav"` into `"wav"` for you. Omit `media_type` entirely and it assumes `wav`.

### Video

Like images: a `url` or `data` plus `media_type`.

```python
from datafast.llm import ContentPart

part = ContentPart(type="video", url="https://example.com/clip.mp4")
```

### Files and documents

A file part has two very different forms, and `url` does double duty:

```python
from datafast.llm import ContentPart

uploaded = ContentPart(type="file", url="file-abc123")          # an id, not a link
inline = ContentPart(
    type="file",
    data="<base64>",
    media_type="application/pdf",
    filename="report.pdf",
)
```

With `data`, datafast sends the bytes inline. With `url`, it sends the string as a **file
id** — so a plain `https://` link is also passed through as an id, which the Responses
API accepts as a file URL and other providers may not.

`document` is an alias: it normalizes to a file part before anything else looks at it.

## What your served model accepts

Every served model declares which modalities it supports, and datafast checks your parts
against that declaration **before the request leaves your machine**:

| Served model | Accepts |
|---|---|
| `openai()`, `anthropic()`, `mistral()` | text, image, file |
| `gemini()` | text, image, audio, video, file |
| `ollama()`, `openrouter()` | text, image |
| `openai_compatible()` with no matching profile | text only |

Send an unsupported part and you get a `ValueError` naming the modality and the served
model:

```text
Modality 'image' is not supported by openai/gpt-5.5-nano
```

This is worth being precise about, because it works differently from the rest of the
capability system. An unsupported *parameter* is dropped or warned about according to the
`unsupported_params` policy. An unsupported *modality* always raises, and the policy has
no effect on it. The reasoning: a dropped `top_p` still gives you an answer, while a
dropped image gives you an answer to a question you did not ask.

The declaration describes the profile, not your exact model. A vision model reached
through a provider whose profile omits images will still be refused locally, and a
text-only model under a profile that allows them will be refused by the provider instead.
If you know better than the profile, pass your own capabilities to the factory.

## Two traps worth knowing

Both of these were found by sending real requests, and both cost an opaque provider error
before they were understood.

### Mistral takes files only as an uploaded id

Mistral's chat API cannot read an inline document. Inline bytes reach it as a malformed
part and come back as an unhelpful schema error, so datafast refuses them up front and
tells you what to do instead:

```python
from datafast import mistral
from datafast.llm import ContentPart

model = mistral()
file_id = model.upload_file("report.pdf")
part = ContentPart(type="file", url=file_id)
```

Uploading is a separate step on purpose. One id serves every request in a pipeline run,
and nothing is uploaded behind a `generate()` call. The file stays in your Mistral account
until you call `model.delete_file(file_id)` — datafast does not track it for you.

`upload_file()` and `delete_file()` exist **only on a Mistral served model**. They are not
part of the shared served-model surface, because no other provider datafast supports needs
a file uploaded before it can be read.

### OpenAI's Responses API needs a `filename`

Inline file data is rejected by the Responses API unless the part also carries a
`filename`. Datafast forwards the field but does not require it, so this one surfaces at
the provider rather than locally. Always set `filename` when you send file bytes:

```python
from datafast.llm import ContentPart

part = ContentPart(
    type="file",
    data="<base64>",
    media_type="application/pdf",
    filename="report.pdf",  # not optional in practice
)
```

## Things worth knowing

- **A typo in `type` is silently treated as text.** An unknown part type is neither
  rewritten nor refused — it is forwarded as you wrote it and counted as a text part by
  the modality check. The provider is what finally complains.
- **`Modality.DOCUMENT` is never used.** No served model declares it, and `document`
  parts are gated as files. It exists in the enum only.
- **`media_id` is usually dropped.** It is forwarded as a caching `uuid` only by a served
  model that declares support for it — vLLM, among the shipped profiles.
- **`provider_options` is an unchecked escape hatch.** Its keys are merged straight into
  the part, so `provider_options={"detail": "high"}` reaches the provider verbatim. Wrong
  keys are not caught here.
- **Parts are validated per message, not per request.** A bad part raises on the first
  message that holds it.
- **Plain string content still works.** Use a list of parts only when you need one.

## Where to go next

- [Served models](../reference/served_models.md) — declaring capabilities yourself, and
  the `unsupported_params` policy this page contrasts with.
- [Providers](../reference/providers/openai.md) — what each provider supports.
- [Structured output](structured_output.md) — asking for a schema alongside an image.
- [Glossary](../glossary.md) — capabilities, served model, provider, transport.
