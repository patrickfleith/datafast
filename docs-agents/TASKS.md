# Tasks

<!-- Newest to do on top. Check off in place; move to Done when complete. -->

## To do

- [ ] Decide whether to expose reasoning summaries — `reasoning_content` is unreachable on openai through datafast's own API: OpenAI only returns a summary when asked via `summary: "auto"`, which no config field exposes, and `provider_params` replaces the whole `reasoning` key rather than merging into it <!-- tests/live/openai/test_reasoning.py works around it with provider_params; a `reasoning_summary` field would be the fix -->
- [ ] Retire the legacy per-provider `integration` suites in one sweep — `tests/test_openai.py`, `test_anthropic.py`, `test_gemini.py`, `test_mistral.py`, `test_openrouter.py`, `test_ollama.py`; none guard on a missing API key, so they fail rather than skip <!-- superseded by tests/live/ per provider; port the QASet/MCQSet/persona schema cases first if they're worth keeping -->
- [ ] Add a live pipeline smoke test — source → `LLMStep` → list sink against one real provider; every live test today drives `ServedModel` directly, so nothing covers the path users actually take <!-- belongs in tests/live/ root, parametrized over providers rather than duplicated per provider -->
- [ ] Mirror the live tests for the remaining providers — gemini, mistral, openrouter, then the local backends <!-- tests/live/anthropic and tests/live/openai are the settled templates; openai adds the Responses transport, fallback-concurrency batching and the OPENAI_CHAT profile. Local ollama/vllm/llamacpp/openai_compatible need a backend running. Reasoning and multimodal are where providers diverge most (mistral high/none allowlist, ollama think=false, gemini billing on thinking=False) — those two resist copy-paste and want the capability-driven parametrization from the roadmap -->
- [ ] Fix `43_cookbook_persona_generation.py`: it chains two sinks (`Sink.jsonl >> Sink.hub`), which `compile()` rejects — decide whether to split into two runs or let a pipeline end in several sinks
- [ ] Consider supporting nested `Branch` inside a branch path — needs a metadata stack; `compile()` rejects the shape today because the inner branch overwrites the outer `_branch_id`
- [X] Manually test all Gemini example scripts <!-- examples/providers/gemini -->
- [ ] Review the documentation and identify missing blocks before publishing the new version of datafast
- [ ] <task> <!-- optional (context) -->

## Done

- [X] Narrow the roadmap's "file / document input support" item to what is actually left <!-- Responses input_file shape now covered by a mocked test -->
- [X] Make file content parts symmetric with image parts <!-- `_data_uri_from_part` now shared by media and file parts -->
- [X] Make sure the roadmap clearly outlines the release of the new version of datafast
- [X] Make a roadmap
- [X] Try out the OpenAI example scripts <!-- examples/providers/openai -->
- [X] Try out the Mistral example scripts <!-- examples/providers/mistral -->
- [X] Try out the Anthropic example scripts <!-- examples/providers/anthropic -->
- [X] Try out the Ollama example script <!-- examples/providers/ollama -->

