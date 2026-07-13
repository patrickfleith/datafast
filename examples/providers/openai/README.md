# OpenAI Examples

Requirements:

- `OPENAI_API_KEY` set in your environment or `.env`

Notes:

- OpenAI's `gpt-5` models are reasoning models served on the Responses API. They
  do not accept `temperature`, so these examples leave it unset (example 08 shows
  how the unsupported-parameter policy handles it).
- Datafast suppresses LiteLLM's provider help banner by default for cleaner example
  output.
- Set `DATAFAST_LITELLM_SUPPRESS_DEBUG_INFO=0` if you want LiteLLM to print that
  extra provider/debug information while troubleshooting.

Run:

```bash
.venv/bin/python examples/providers/openai/01_simple_prompt.py
.venv/bin/python examples/providers/openai/02_batch_prompts.py
.venv/bin/python examples/providers/openai/03_messages_with_system_prompt.py
.venv/bin/python examples/providers/openai/04_structured_output.py
.venv/bin/python examples/providers/openai/05_batch_messages.py
.venv/bin/python examples/providers/openai/06_generation_metadata.py
.venv/bin/python examples/providers/openai/07_structured_batch.py
.venv/bin/python examples/providers/openai/08_unsupported_params_policies.py
.venv/bin/python examples/providers/openai/09_multimodal_image_input.py
.venv/bin/python examples/providers/openai/10_raw_vs_normalized_response.py
.venv/bin/python examples/providers/openai/11_timeout_and_rate_limit.py
```

Files:

- `01_simple_prompt.py`: one prompt, one response
- `02_batch_prompts.py`: a list of prompts sent through one `generate(...)` call
- `03_messages_with_system_prompt.py`: chat messages with a system instruction
- `04_structured_output.py`: validated Pydantic output
- `05_batch_messages.py`: a batch of independent message lists
- `06_generation_metadata.py`: `generate_response(...)` and normalized metadata with Responses-API reasoning
- `07_structured_batch.py`: batched structured responses
- `08_unsupported_params_policies.py`: `warn`, `quiet`, and `fail` handling for an unsupported parameter (`temperature`)
- `09_multimodal_image_input.py`: text plus image input using `ContentPart`, sending the bundled `sample_lion.jpg` as base64 bytes
- `10_raw_vs_normalized_response.py`: compare normalized fields with the underlying raw Responses payload
- `11_timeout_and_rate_limit.py`: three separate requests through one provider instance, with `rpm_limit=2` so the third request shows client-side throttling
