# OpenRouter Examples

Requirements:

- `OPENROUTER_API_KEY` set in your environment or `.env`

Notes:

- Datafast suppresses LiteLLM's provider help banner by default for cleaner example
  output.
- Set `DATAFAST_LITELLM_SUPPRESS_DEBUG_INFO=0` if you want LiteLLM to print that
  extra provider/debug information while troubleshooting.

Run:

```bash
.venv/bin/python examples/providers/openrouter/01_simple_prompt.py
.venv/bin/python examples/providers/openrouter/02_batch_prompts.py
.venv/bin/python examples/providers/openrouter/03_messages_with_system_prompt.py
.venv/bin/python examples/providers/openrouter/04_structured_output.py
.venv/bin/python examples/providers/openrouter/05_batch_messages.py
.venv/bin/python examples/providers/openrouter/06_generation_metadata.py
.venv/bin/python examples/providers/openrouter/08_structured_batch.py
```

Files:

- `01_simple_prompt.py`: one prompt, one response
- `02_batch_prompts.py`: a list of prompts sent through one `generate(...)` call
- `03_messages_with_system_prompt.py`: chat messages with a system instruction
- `04_structured_output.py`: validated Pydantic output
- `05_batch_messages.py`: a batch of independent message lists
- `06_generation_metadata.py`: `generate_response(...)` and normalized metadata
- `08_structured_batch.py`: batched structured responses
