# Roadmap

## Shipped

- Capability-aware LLM provider layer: per-target capability resolution (provider + endpoint + model), one common config surface, `unsupported_params` policy (`fail`/`warn`/`quiet`).
- Provider factories: `openai`, `anthropic`, `gemini`, `mistral`, `openrouter`, `ollama`, `openai_compatible`.
- Chat and Responses endpoint modes, structured output (Pydantic), reasoning controls (`thinking` / `reasoning_effort`).
- Multimodal **input** normalization: text, image, video, file/document content parts.
- Native batching with warned fallback concurrency; retries, backoff, jitter, timeout, client-side RPM throttling.
- Example suites (11 scripts each) for openai, anthropic, mistral, ollama, openrouter.
- Mocked contract/capability/adapter tests (C*/K*/A* coverage in `tests/test_llm_provider_contract.py`).

## In progress

- LLM provider redesign hardening on branch `feat/implement-new-and-robust-llm-providers`.

## Next up

Launch checklist, grouped by area. Pipeline execution correctness and architecture
are the newest additions and gate the release; provider hardening and documentation
run alongside them.

### Pipeline execution correctness

- **Wire up or remove dead execution controls.** Several `RunConfig` / `run()`
  parameters are accepted and documented but have no effect — silent no-ops are worse
  than missing features. For each: implement it, or delete it and document the real path.
  - `limit` — currently ignored; must truncate source records at step 0. Highest
    priority: `examples/scripts/37_limit_stop_and_strategies.py` leads with
    `run(limit=3)` and today processes the full seed set (~20 LLM calls instead of 3).
  - `rate_limits` — stored in the manifest and documented on `run()`, but never
    throttles. Either apply it in the runner batch loop, or remove it and document
    that throttling lives on the provider (`rpm_limit`).
  - `resume_from` — documented ("resume from a named step, discard later steps"),
    referenced nowhere. Implement or remove.
  - `max_concurrent` — implies runner-level batch concurrency that does not exist
    (batches run sequentially; concurrency is provider-internal). Implement or remove.
- **Execution & resume tests.** Cover the headline features that are currently
  untested: checkpoint save/resume (including mid-LLM-step resume), `limit`,
  `stop_after`, and `llm_strategy` ordering (by_model / round_robin / by_record).

### Pipeline architecture

- **Branch runner integration.** LLM steps nested inside `Branch` execute via
  `step.process()` directly, bypassing the runner's batching, checkpoint/resume, rate
  limiting, and execution strategy — this hits the flagship preference-data pipeline
  (`examples/scripts/42`). Either have the runner recurse into branch paths, or clearly
  document the limitation and its cost (a crash mid-Branch re-runs every branch call
  on resume).
- **Pipeline validation / `compile()`.** No structural checks today; mistakes surface
  as deep runtime `KeyError`s or silent empty output. Add validation: source first /
  sink last, `input_columns` & `forward_columns` references exist, `by` columns exist
  (Sample/Group/Pair), and Branch↔JoinBranches pairing — all with actionable error
  messages raised before execution.

### Provider hardening & tests

- **Mocked reliability tests (R01–R07).** Cover retry/backoff/rate-limit code that already exists but is untested.
  - R01 retryable error triggers bounded retries; R02 non-retryable fails immediately.
  - R03 backoff grows across attempts; R04 jitter stays within range (inject `_sleep`, assert delays).
  - R05 timeout is forwarded and timeout failure surfaces clearly.
  - R06 `rpm_limit` throttles before dispatch (mocked clock/sleep, no live call).
  - R07 batch retry preserves output ordering (per-item batch failure re-runs through single path).
- **Gemini example suite.** Add `examples/providers/gemini/` mirroring the other providers (11 scripts + README + sample image).
- **Capability-driven live test catalogue (L01–L10).** A curated model catalog + shared live suite parametrized over it, so adding a model is one catalog entry. Replaces ad-hoc per-provider `integration` tests; wire the `live` marker.

### Documentation (launch)

Bring the published docs (mkdocs, `docs/`) to release quality. The site today covers
Home, Concepts, a few Guides, three Cookbook recipes, Providers, Models, and API.
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
  - Branch / JoinBranches — tagging, cartesian join, suffixes, inner/outer, runner caveat.
- **Execution & configuration guide.** Everything `run()` / `RunConfig` exposes after
  the Tier-1 cleanup, and exactly what each does: checkpoint_dir, resume, batch_size,
  llm_strategy, limit, stop_after, and where rate limiting actually lives (provider
  `rpm_limit` vs runner). Prevents a repeat of the dead-parameter confusion.
- **Provider user guide.** User-facing counterpart to the internal provider-doc
  consolidation: each factory (openai / anthropic / gemini / mistral / openrouter /
  ollama / openai_compatible), required API-key env vars, endpoint modes
  (chat / responses), the capability model and per-target resolution, the
  `unsupported_params` policy (fail/warn/quiet), reliability knobs
  (retries/backoff/jitter/timeout/rpm_limit), native batching, structured output, and
  reasoning controls. Absorbs `llm_provider_requirements.md`; remove it from the root
  once merged.
- **Structured output guide.** `parse_mode` (text/json/xml) at the step level vs
  Pydantic `response_format` at the provider level — when to use which. Resolves the
  design-doc-vs-code divergence.
- **Multimodal input guide.** Passing image/video/file/document (and later audio)
  content parts, and capability gating per target. Grows with the modality features below.
- **Migration guide (v1 → v2).** Map each removed dataset class (Classification, MCQ,
  preference, instruction) to its pipeline equivalent and call out the breaking removal
  of the dataset-class API. Port design-doc Appendix C.
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
- **Contributing & development guide.** Test markers (integration / live / multimodal /
  ollama / vllm / llamacpp) and layers (contract C*, capability K*, adapter A*,
  reliability R*, live L*), how to add a provider or a step, and project layout.
  Absorbs `llm_provider_test_plan.md` and `llm_provider_test_guide.md`; remove them
  from the root once merged.
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
- **File / document input support.**
  - Implement: enable file/document parts for models/providers that declare `Modality.FILE` / `Modality.DOCUMENT`, chat and Responses shapes.
  - Test: mocked contract test (M05) + capability gating + one live test on a document-capable model.
  - Example: `NN_document_input.py` for at least one document-capable provider.
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

- **Detailed LLMProvider test guide** — drafted at `llm_provider_test_guide.md` (how to mock LiteLLM, inject `_sleep`, cover each test layer, add reliability/multimodal tests, extend the model catalog); to be folded into the Contributing & development guide (see Documentation) and removed from the root.
- Migrate existing per-provider `integration` tests onto the `live` marker and the shared catalogue once (L*) lands; retire duplicated ad-hoc coverage.
- Unused markers (`multimodal`, `ollama`, `vllm`, `llamacpp`) are declared but not yet applied to tests.
