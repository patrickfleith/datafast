# Tasks

<!-- Newest to do on top. Check off in place; move to Done when complete. -->

## To do

- [ ] Make file content parts symmetric with image parts — `_normalize_file_part` passes `data` straight to `file_data`, so raw base64 plus `media_type` goes out malformed with no client-side error, while images get wrapped into a `data:` URI by `_media_url_from_part`
- [ ] Add a live pipeline smoke test — source → `LLMStep` → list sink against one real provider; every live test today drives `ServedModel` directly, so nothing covers the path users actually take <!-- belongs in tests/live/ root, parametrized over providers rather than duplicated per provider -->
- [ ] Roadmap's "file / document input support" is stale for Anthropic chat — `tests/live/anthropic/test_multimodal.py` proves it works end to end; confirm the Responses shape and the other file-capable providers, then narrow or drop the roadmap item
- [ ] Mirror the Anthropic live tests for every other provider — generation + structured output <!-- tests/live/anthropic is the settled template (generation, structured output, reasoning, multimodal); openai (Responses endpoint, fallback-concurrency batching), gemini, mistral, openrouter, plus local ollama/vllm/llamacpp/openai_compatible which need a backend running. Reasoning and multimodal are where providers diverge most (mistral high/none allowlist, ollama think=false, gemini billing on thinking=False) — those two resist copy-paste and want the capability-driven parametrization from the roadmap -->
- [ ] Fix `43_cookbook_persona_generation.py`: it chains two sinks (`Sink.jsonl >> Sink.hub`), which `compile()` rejects — decide whether to split into two runs or let a pipeline end in several sinks
- [ ] Consider supporting nested `Branch` inside a branch path — needs a metadata stack; `compile()` rejects the shape today because the inner branch overwrites the outer `_branch_id`
- [X] Manually test all Gemini example scripts <!-- examples/providers/gemini -->
- [ ] Review the documentation and identify missing blocks before publishing the new version of datafast
- [ ] <task> <!-- optional (context) -->

## Done

- [X] Make sure the roadmap clearly outlines the release of the new version of datafast
- [X] Make a roadmap
- [X] Try out the OpenAI example scripts <!-- examples/providers/openai -->
- [X] Try out the Mistral example scripts <!-- examples/providers/mistral -->
- [X] Try out the Anthropic example scripts <!-- examples/providers/anthropic -->
- [X] Try out the Ollama example script <!-- examples/providers/ollama -->

