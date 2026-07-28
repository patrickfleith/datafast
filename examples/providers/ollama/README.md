# Ollama Examples

Requirements:

- [Ollama](https://ollama.com) installed and its server running (defaults to
  `http://localhost:11434`)
- The model pulled locally: `ollama pull gemma4:12b` (~7.6 GB, multimodal).
  Example `08` needs a non-reasoning model to show unsupported-param handling —
  `ollama pull ministral-3:3b` for it.
- No API key — Ollama runs locally

Notes:

- `gemma4:12b` is a capable multimodal reasoning model but is heavy for a small
  machine. If it is too slow or too large, the same scripts work unchanged
  against a lighter model such as `gemma3:4b` — just change `MODEL_ID`.
- Google's recommended sampling for Gemma is `temperature=1.0`, `top_p=0.95`,
  `top_k=64`. The examples use `temperature=0` for stable, repeatable output.
  `top_k` is not one of Datafast's mapped parameters, so pass it through the
  escape hatch when you want it: `ollama(MODEL_ID, provider_params={"top_k": 64})`.
- Datafast's Ollama profile is deliberately conservative (chat endpoint, text +
  image). Thinking-capable models (gemma4, deepseek-r1, qwen3, gpt-oss, ...) get
  a mapped reasoning control via `thinking` / `reasoning_effort`. Features the
  profile does not map — `top_k`, thinking on models it doesn't recognize — are
  reachable via `provider_params`.
- To offload the model to a beefier machine, point at a remote host:
  `ollama(MODEL_ID, api_base="http://<host>:11434")`.
- Datafast suppresses LiteLLM's provider help banner by default for cleaner
  example output. Set `DATAFAST_LITELLM_SUPPRESS_DEBUG_INFO=0` to see it while
  troubleshooting.

Run:

```bash
.venv/bin/python examples/providers/ollama/01_simple_prompt.py
.venv/bin/python examples/providers/ollama/02_batch_prompts.py
.venv/bin/python examples/providers/ollama/03_messages_with_system_prompt.py
.venv/bin/python examples/providers/ollama/04_structured_output.py
.venv/bin/python examples/providers/ollama/05_batch_messages.py
.venv/bin/python examples/providers/ollama/06_generation_metadata.py
.venv/bin/python examples/providers/ollama/07_structured_batch.py
.venv/bin/python examples/providers/ollama/08_unsupported_params_policies.py
.venv/bin/python examples/providers/ollama/09_multimodal_image_input.py
.venv/bin/python examples/providers/ollama/10_raw_vs_normalized_response.py
.venv/bin/python examples/providers/ollama/11_timeout_and_rate_limit.py
```

Files:

- `01_simple_prompt.py`: one prompt, one response
- `02_batch_prompts.py`: a list of prompts through one `generate(...)` call; Ollama has no native batch, so Datafast falls back to bounded concurrency (`max_concurrent=1`)
- `03_messages_with_system_prompt.py`: chat messages with a system instruction
- `04_structured_output.py`: validated Pydantic output via schema-constrained decoding
- `05_batch_messages.py`: a batch of independent message lists (bounded-concurrency fallback)
- `06_generation_metadata.py`: `generate_response(...)` metadata for a thinking model, with first-class reasoning via `reasoning_effort="high"` normalized into `reasoning_content`
- `07_structured_batch.py`: batched structured responses
- `08_unsupported_params_policies.py`: `warn`, `quiet`, and `fail` handling for `reasoning_effort` on a non-reasoning model (`ministral-3:3b`), which does not map it
- `09_multimodal_image_input.py`: text plus image input using `ContentPart`, sending the bundled `sample_lighthouse.jpg` as base64 bytes (needs a vision-capable model)
- `10_raw_vs_normalized_response.py`: compare normalized fields with the raw payload, including Ollama's timing/token counters
- `11_timeout_and_rate_limit.py`: three separate requests through one provider instance, with `rpm_limit=2` so the third request shows client-side throttling
