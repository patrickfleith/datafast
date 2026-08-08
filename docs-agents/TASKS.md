# Tasks

<!-- Newest to do on top. Check off in place; move to Done when complete. -->

## To do

- [ ] Decide whether to expose reasoning summaries — `reasoning_content` is unreachable on openai through datafast's own API: OpenAI only returns a summary when asked via `summary: "auto"`, which no config field exposes, and `provider_params` replaces the whole `reasoning` key rather than merging into it <!-- tests/live/openai/test_reasoning.py works around it with provider_params; a `reasoning_summary` field would be the fix -->
- [ ] Retire the legacy per-provider `integration` suites in one sweep — `tests/test_openai.py`, `test_anthropic.py`, `test_gemini.py`, `test_mistral.py`, `test_openrouter.py`, `test_ollama.py`; none guard on a missing API key, so they fail rather than skip <!-- superseded by tests/live/ per provider; port the QASet/MCQSet/persona schema cases first if they're worth keeping -->
- [ ] Add a live pipeline smoke test — source → `LLMStep` → list sink against one real provider; every live test today drives `ServedModel` directly, so nothing covers the path users actually take <!-- belongs in tests/live/ root, parametrized over providers rather than duplicated per provider -->
- [ ] Mirror the live tests for the remaining providers — gemini, openrouter, then the local backends <!-- tests/live/{anthropic,openai,mistral} are the settled templates: openai adds the Responses transport, fallback-concurrency batching and the OPENAI_CHAT profile; mistral adds the reasoning allowlist and the Files upload path. Gemini is the pick next — it bills reasoning tokens unless thinking=False sends reasoning_effort='none', and it declares AUDIO/VIDEO, which no live test has ever exercised. Openrouter last of the hosted three: its profile retreads paths anthropic already covers and its default model routes to a third party, so a failure would not cleanly indict datafast. Local ollama/vllm/llamacpp/openai_compatible need a backend running and a reachability guard instead of an API-key skip (ollama sets no_api_key), plus host-local model ids -->
- [ ] Fix `43_cookbook_persona_generation.py`: it chains two sinks (`Sink.jsonl >> Sink.hub`), which `compile()` rejects — decide whether to split into two runs or let a pipeline end in several sinks
- [ ] Consider supporting nested `Branch` inside a branch path — needs a metadata stack; `compile()` rejects the shape today because the inner branch overwrites the outer `_branch_id`
- [X] Manually test all Gemini example scripts <!-- examples/providers/gemini -->
- [ ] Review the documentation and identify missing blocks before publishing the new version of datafast
- [ ] <task> <!-- optional (context) -->

## Done

- [X] Make `Modality.FILE` state which carrier a served model accepts, and give Mistral a way to reach it <!-- `files_require_file_id` on the profile plus `upload_file`/`delete_file` on the Mistral served model (LiteLLM has no Files support for mistral). Upload stays explicit so one id serves a whole pipeline run and nothing is uploaded behind a generate() call; the caller owns the file's lifetime. Added MISTRAL_CHAT so non-reasoning mistral ids carry the same constraint -->
- [X] Narrow the roadmap's "file / document input support" item to what is actually left <!-- Responses input_file shape now covered by a mocked test -->
- [X] Make file content parts symmetric with image parts <!-- `_data_uri_from_part` now shared by media and file parts -->
- [X] Make sure the roadmap clearly outlines the release of the new version of datafast
- [X] Make a roadmap
- [X] Try out the OpenAI example scripts <!-- examples/providers/openai -->
- [X] Try out the Mistral example scripts <!-- examples/providers/mistral -->
- [X] Try out the Anthropic example scripts <!-- examples/providers/anthropic -->
- [X] Try out the Ollama example script <!-- examples/providers/ollama -->

