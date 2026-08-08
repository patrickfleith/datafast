# Roadmap

## Shipped

- Capability-aware served-model layer: per-served-model capability resolution (provider + endpoint + model), one common config surface, `unsupported_params` policy (`fail`/`warn`/`quiet`).
- Provider factories: `openai`, `anthropic`, `gemini`, `mistral`, `openrouter`, `ollama`, `openai_compatible`.
- Chat and Responses endpoint modes, structured output (Pydantic), reasoning controls (`thinking` / `reasoning_effort`); first-class reasoning across anthropic, gemini, mistral, ollama.
- Multimodal **input** normalization: text, image, video, file/document content parts.
- Native batching with warned fallback concurrency; retries, backoff, jitter, timeout, client-side RPM throttling.
- Example suites (11 scripts each) for openai, anthropic, gemini, mistral, ollama, openrouter.
- Mocked contract/capability/adapter/reliability tests in `tests/test_served_model_contract.py` (reliability: bounded retries, backoff growth, jitter range, timeout forwarding, RPM throttling, batch-retry ordering).
- Pipeline execution controls: `limit` and `resume_from` implemented; dead `rate_limits` / runner-level `max_concurrent` removed (throughput lives on the provider). Covered by `tests/test_runner_execution.py` (limit, resume_from, stop_after, llm_strategy ordering, full + mid-LLM-step checkpoint resume).
- Pipeline pre-flight validation: `Pipeline.compile()` (`datafast/core/validation.py`) runs before execution and raises an actionable `PipelineValidationError` — source-first / sink-last, Branch↔JoinBranches pairing, and conservative column-reference checks (`tests/test_pipeline_validation.py`).
- Branch runner integration: the runner recurses into `Branch` paths (and nested sub-pipelines), so LLM steps inside a path get batching, `llm_strategy` ordering and per-call checkpoint/resume. Nested steps share the parent manifest entry and own checkpoint files keyed by dotted path name; the pipeline hash now covers branch-path structure (`tests/test_runner_branch.py`).
- Served-model vocabulary rename: `LLMProvider` → `ServedModel`, `TargetConfig` → `ServedModelConfig`, `TargetCapabilities` → `ServedModelCapabilities`, `_CATALOG` → `_SERVED_MODEL_CATALOG`; fields `provider` → `provider_id` and `litellm_provider` → `litellm_route`; the seven per-provider subclasses are private, leaving the lowercase factories as the only public entry points. `llm/provider.py` → `llm/served_model.py`, `datafast/llms.py` deleted, trace key → `datafast_provider_id`. Docs, README and the mocked suite (`tests/test_served_model_contract.py`, `tests/test_served_model_unit.py`) follow the settled `GLOSSARY.md` terms.
- `compile()` sub-pipeline coverage: validation recurses into Branch paths (inherited input — no source, no sink, column refs checked against the branch's incoming schema) and into Concat sources / Join right sides (self-contained — must start with a source, no sink). Errors name the location (`inside Branch path 'chosen'`). Also validates Join's `on` against the left schema and rejects Branch-inside-Branch, which silently drops every record (`tests/test_pipeline_validation.py`).

- Per-served-model reasoning controls: `thinking=True`/`False` now resolve to each
  served model's own on/off values instead of a hardcoded `reasoning_effort="low"` and
  a silent no-op. New `ServedModelCapabilities` fields `reasoning_effort_on`,
  `reasoning_off_param` and `reasoning_efforts`. Mistral reasoning models use
  `high`/`none` (the API rejects `low`/`medium` with a 400, now caught client-side with
  an actionable error); Ollama sends `think=false` directly, since omitting it leaves
  the model default (on for qwen3) and LiteLLM's `reasoning_effort` mapping would send
  a literal `"none"` string that Ollama rejects for gpt-oss; Gemini sends
  `reasoning_effort="none"`, as Gemini 3 models think by default and were billing
  reasoning tokens on `thinking=False`. Anthropic and OpenAI already default to no
  reasoning, so they are unchanged. Verified live on all three providers; regression
  tests in `tests/test_served_model_contract.py`.

- `provider_id` always names a server, never a wire format (DEC-003):
  `openai_compatible()` takes a required `provider_id` keyword and rejects wire-format
  values with an actionable error; the `backend` parameter is gone, matching the
  glossary's "avoid: backend". Provider ids are normalized (`llama.cpp` → `llamacpp`)
  and stay free-form, so any self-hosted server works — known ids get their profile,
  unknown ones fall back to the conservative OpenAI-compatible one.

## In progress

- Live provider test suites (`tests/live/`), one directory per provider, gated behind
  `--run-live` and self-skipping when the API key is absent. Anthropic and openai have
  landed (generation, structured output, reasoning, multimodal, plus openai's Responses
  transport, fallback-concurrency batching and `OPENAI_CHAT` profile); gemini, mistral,
  openrouter and the local backends remain. Shared image/PDF assets live in
  `tests/live/assets/`.

## Next up

Launch checklist, grouped by area. All pipeline-architecture items gating the
release have landed (see Shipped); what remains is provider hardening and
documentation.

### Provider hardening & tests

The served-model rename has landed (see Shipped); vocabulary is settled in
`docs-agents/GLOSSARY.md`. What remains:

- **Write a new provider test plan.** The old drafts (`llm_provider_test_plan.md`,
  `llm_provider_test_guide.md`, `llm_provider_requirements.md`, `llm_live_test_plan.md`)
  are deleted and not worth reviving — they predate the served-model vocabulary and the
  current capability layer. The replacement should define the test layers and their
  markers (contract, capability, adapter, reliability, live), how to mock LiteLLM and
  inject `_sleep`, and how to add a served model or a step. Uses the settled
  vocabulary; becomes the source for the Contributing guide below.
- **Capability-driven live test catalogue.** A curated served-model catalog plus one
  shared live suite parametrized over it, so adding a model is a single catalog entry.
  Replaces the ad-hoc per-provider `integration` tests and wires up the `live` marker.
  Depends on the new test plan (the two reasoning bugs are fixed — see Shipped).
- **Migrate anthropic to `claude-sonnet-5`.** Check support for `claude-sonnet-5` and
  add it in place of `claude-sonnet-4-6` (`_SERVED_MODEL_CATALOG`, examples, defaults); confirm
  capability parity (reasoning / batching / structured output) before removing the
  4.6 entry.

### Documentation (launch)

Bring the published docs (mkdocs, `docs/`) to release quality. The site today covers
Home, Concepts, a few Guides, three Cookbook recipes, Served models, Models, and API.
Gaps to close, roughly in priority order:

- **Step reference (largest gap).** One reference page per step family documenting
  every parameter and its non-obvious behavior:
  - Sources & Seed — list / file / huggingface; `Seed.values/expand/range/product/zip`.
  - Sinks — jsonl / csv / parquet / hub / list; Hub token, private, train/test split, dataset card.
  - Data ops — Map, FlatMap, AddUUID; Filter (full operator table: comparison,
    `$in`/`$nin`, string ops, `$len_*`, `$exists`, `$type`, `$all`/`$any`, `$or`/`$and`);
    Group (`col:func` aggregation spec, min/max_per_group); Pair (strategies,
    within/across, output formats, max_pairs); Concat; Join (how modes, suffixes).
  - Sample — all nine strategies, required `by`, `n`/`frac`, `seed`, `replace`, and
    the step-vs-config duality (`.pick()`).
  - LLM steps — LLMStep (expansion math prompt×model×language×num_outputs, parse
    modes text/json/xml, prompt-from-file, forward/exclude columns, skip_if,
    `{language}`/`{language_name}`, `_model`/`_prompt_index`/`_language` metadata);
    Classify / Score / Compare (llm-vs-fn dual mode, rubric/criteria, output modes,
    include_explanation/confidence); Rewrite (modes); Extract (custom fields vs
    predefined extractors, flatten).
  - Branch / JoinBranches — tagging, cartesian join, suffixes, inner/outer, and how
    the runner drives paths (nested batching + resume, determinism requirement for
    non-LLM path steps).
- **Execution & configuration guide.** Everything `run()` / `RunConfig` exposes after
  the Tier-1 cleanup, and exactly what each does: checkpoint_dir, resume, batch_size,
  llm_strategy, limit, stop_after, and where rate limiting actually lives (provider
  `rpm_limit` vs runner). Prevents a repeat of the dead-parameter confusion.
- **Provider and served-model guide.** Each factory (openai / anthropic / gemini /
  mistral / openrouter / ollama / openai_compatible), required API-key env vars,
  transports (chat / responses), the capability model and per-served-model resolution,
  the `unsupported_params` policy (fail/warn/quiet), reliability knobs
  (retries/backoff/jitter/timeout/rpm_limit), native batching, structured output, and
  reasoning controls. Written from the code, not from the deleted requirements draft.
- **Structured output guide.** `parse_mode` (text/json/xml) at the step level vs
  Pydantic `response_format` at the provider level — when to use which. Resolves the
  design-doc-vs-code divergence.
- **Multimodal input guide.** Passing image/video/file/document (and later audio)
  content parts, and capability gating per target. Grows with the modality features below.
- **Migration guide (v1 → v2).** Map each removed dataset class (Classification, MCQ,
  preference, instruction) to its pipeline equivalent and call out the breaking removal
  of the dataset-class API. Port Appendix C of the deleted
  `datafast_new_design_document.md`, recoverable from git history.
- **Environment & install reference.** All env vars (provider API keys, `HF_TOKEN`,
  `LANGFUSE_*`) and optional extras (datasets, pyarrow, huggingface_hub, langfuse),
  with a minimal end-to-end setup path.
- **Cookbook expansion.** Promote the flagship `examples/scripts/` into recipes:
  preference/DPO via Branch, multi-hop QA, instruction dataset, MCQ, text augmentation
  (Rewrite), LLM-as-judge scoring + filtering, and multilingual generation. Add an
  examples index mapping each script (01–45) to what it demonstrates.
- **Error handling & troubleshooting.** `on_parse_error` (skip/raise), partial
  results / skipped records, resuming after a crash, debugging parse failures, and
  common provider errors.
- **Changelog / release notes.** Populate `docs-agents/CHANGELOG.md` for the new
  version and write a public "what's new / breaking changes" page (dataset classes
  removed → pipelines).
- **Contributing & development guide.** Test markers and layers, how to add a served
  model or a step, and project layout. Draws on the new provider test plan (see Provider
  hardening & tests) rather than the deleted drafts.
- **Retire `SOFTWARE_DESCRIPTION.md`.** Fold its content into the docs above and
  generate the user manual (`docs-agents/SUM.md`) with the `write-manual` skill; delete
  `SOFTWARE_DESCRIPTION.md` once superseded.
- **README & API reference polish.** Release-quality README (feature list, doc links)
  and a complete auto-generated API page (mkdocstrings) over the full public surface;
  ensure `py.typed` ships.

### Provider feature expansion

- **Audio input support.**
  - Implement: enable audio content parts end-to-end for models/providers that declare `Modality.AUDIO` (already normalized to `input_audio`; verify per-target gating).
  - Test: mocked contract test (M03) + capability gating + one live test on an audio-capable model.
  - Example: `NN_audio_input.py` for at least one audio-capable provider.
- **File / document input support.** Implemented and covered: chat (`file.file_data`)
  and Responses (`input_file` with `file_data` / `file_url`) shapes, raw base64 wrapped
  into a `data:` URI like image parts, gating against `Modality.FILE`, mocked contract
  tests, and a live Anthropic test with a real PDF
  (`tests/live/anthropic/test_multimodal.py`) and openai
  (`tests/live/openai/test_multimodal.py`, which found that the Responses API rejects
  inline file data without a `filename` — now a first-class `ContentPart` field).
  What remains:
  - `OPENROUTER_CHAT` and `OLLAMA_CHAT` don't declare `Modality.FILE`, so they only need
    a mocked test that the gate rejects a file part.
  - Example: `NN_document_input.py` for at least one document-capable provider.
  - `Modality.DOCUMENT` is declared by no served model and `document` parts normalize to
    `file` before gating — decide whether to drop the enum member or give it real meaning.
- **Image output support (image-generation models).**
  - Implement: request-side selection + response normalization for image-generation-capable chat/Responses targets (M09); expose generated images on `NormalizedResponse.images`.
  - Test: mocked contract test for image-output path + one live test.
  - Example: `NN_image_output.py` for an image-generation-capable provider.

## Later / long term

- **Caching.** Full caching design from requirements: provider-native prompt caching, router/gateway caching, local prefix/KV reuse, optional client-side result cache; capability-aware cache keys/hints; cache tests (H01–H07). Only `cache_mode` metadata exists today.
- **vLLM support.** Delta live tests + example suite (needs a running server).
- **llama.cpp support.** Delta live tests + example suite (needs a running server).
- **openai-compatible generic backend.** Tests + example for the generic self-hosted path.
- Video input live coverage; `previous_response_id` continuation live scenario (E07); full-catalog live sweep (E08).

## Improvements & tech debt

Non-feature work: rework, refactor, performance, cleanup.

- Migrate existing per-provider `integration` tests onto the `live` marker and the shared catalogue once it lands; retire duplicated ad-hoc coverage.
- Unused markers (`multimodal`, `ollama`, `vllm`, `llamacpp`) are declared but not yet applied to tests.
