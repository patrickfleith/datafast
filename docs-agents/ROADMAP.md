# Roadmap

## Shipped

- Capability-aware LLM provider layer: per-target capability resolution (provider + endpoint + model), one common config surface, `unsupported_params` policy (`fail`/`warn`/`quiet`).
- Provider factories: `openai`, `anthropic`, `gemini`, `mistral`, `openrouter`, `ollama`, `openai_compatible`.
- Chat and Responses endpoint modes, structured output (Pydantic), reasoning controls (`thinking` / `reasoning_effort`); first-class reasoning across anthropic, gemini, mistral, ollama.
- Multimodal **input** normalization: text, image, video, file/document content parts.
- Native batching with warned fallback concurrency; retries, backoff, jitter, timeout, client-side RPM throttling.
- Example suites (11 scripts each) for openai, anthropic, gemini, mistral, ollama, openrouter.
- Mocked contract/capability/adapter/reliability tests in `tests/test_llm_provider_contract.py` (reliability: bounded retries, backoff growth, jitter range, timeout forwarding, RPM throttling, batch-retry ordering).
- Pipeline execution controls: `limit` and `resume_from` implemented; dead `rate_limits` / runner-level `max_concurrent` removed (throughput lives on the provider). Covered by `tests/test_runner_execution.py` (limit, resume_from, stop_after, llm_strategy ordering, full + mid-LLM-step checkpoint resume).
- Pipeline pre-flight validation: `Pipeline.compile()` (`datafast/core/validation.py`) runs before execution and raises an actionable `PipelineValidationError` — source-first / sink-last, Branch↔JoinBranches pairing, and conservative column-reference checks (`tests/test_pipeline_validation.py`).
- Branch runner integration: the runner recurses into `Branch` paths (and nested sub-pipelines), so LLM steps inside a path get batching, `llm_strategy` ordering and per-call checkpoint/resume. Nested steps share the parent manifest entry and own checkpoint files keyed by dotted path name; the pipeline hash now covers branch-path structure (`tests/test_runner_branch.py`).
- `compile()` sub-pipeline coverage: validation recurses into Branch paths (inherited input — no source, no sink, column refs checked against the branch's incoming schema) and into Concat sources / Join right sides (self-contained — must start with a source, no sink). Errors name the location (`inside Branch path 'chosen'`). Also validates Join's `on` against the left schema and rejects Branch-inside-Branch, which silently drops every record (`tests/test_pipeline_validation.py`).

## In progress

- Nothing in flight.

## Next up

Launch checklist, grouped by area. All pipeline-architecture items gating the
release have landed (see Shipped); what remains is provider hardening and
documentation.

### Provider hardening & tests

The provider layer's vocabulary doesn't match what the code does. `LLMProvider` is
the most-used public symbol, but an `LLMProvider` instance isn't a provider — it
holds a provider *and* a model *and* its settings. Meanwhile `provider` as a field
accepts non-provider values (`"openai_compatible"` is a wire format, not a server),
`litellm_provider` is really a transport prefix, and `target` is already taken by the
pipeline API (`target_audience`, `target_length`, "target column"). Settling this
before the docs push matters: every page we're about to write inherits whichever
words we pick, and with no users yet the rename is free.

Nomenclature: a **provider** is the server (cloud or local), a **model** is the LLM it
serves, a **served model** is the two together plus its configuration, **capabilities**
are what that served model can actually do, and **transport** is the wire protocol and
route used to reach it.

- **Rename the provider layer to the served-model vocabulary.** No aliases or
  deprecation shims — there are no users.
  - Types and classes: `TargetCapabilities` → `ServedModelCapabilities`,
    `TargetConfig` → `ServedModelConfig`, `LLMProvider` → `ServedModel`. The eight
    per-provider subclasses become private (`_OpenAIServedModel`, …) and drop out of
    every `__all__`, leaving the lowercase factories (`openai`, `anthropic`, `gemini`,
    `mistral`, `openrouter`, `ollama`, `openai_compatible`) as the only public entry
    points. Capability profile constants (`OPENAI_RESPONSES`, `ANTHROPIC_CHAT`, …)
    stay as they are; `_CATALOG` becomes `_SERVED_MODEL_CATALOG` to match the glossary.
  - Filenames: `datafast/llm/provider.py` → `served_model.py` and
    `tests/test_llm_provider_contract.py` → `test_served_model_contract.py`, since both
    now hold served-model code. `examples/providers/` keeps its name — those suites
    really are grouped by provider.
  - Fields: `provider` → `provider_id` (consistent with `model_id`),
    `litellm_provider` → `litellm_route`, `provider_name` → `provider_id`.
  - Ripple: `resolve_capabilities()`, `_get_model_string()`, the four
    `transforms/llm_*.py` consumers, exports in `datafast/__init__.py` and
    `datafast/llm/__init__.py`, and `datafast/llms.py` — a compat shim for names with
    no users, so decide whether to delete it. `provider_name` is duck-typed: it
    reaches `tracing.py` (and the `datafast_provider` metadata key) plus 14 stub
    definitions across 7 test files. Also sweep the "target" prose in docstrings and
    comments. Two things that must *not* be swept: `LLMStep(model=...)` stays
    (friendliest keyword, renaming buys nothing), and `_add_supported_param`'s
    `target_name=` argument is a different "target" — the destination parameter name —
    so a blind grep-and-replace will corrupt it; rename it to `param_name` or leave it.
  - Docs: `docs/llms.md` and `docs/models.md` (mkdocs sources; `site/` is generated)
    and a CHANGELOG entry. Vocabulary is already settled in `docs-agents/GLOSSARY.md`.
    One stale mention in this file to fix on the way through: "per-target capability
    resolution" under Shipped.
  - Deferred, to record as a decision rather than fix here: `provider_id` should
    probably never hold `"openai_compatible"` — that case wants
    `provider_id="vllm"` (or `llamacpp`) with the OpenAI-shaped wire format expressed
    purely as transport, which changes `openai_compatible()`'s signature.
- **Fix the two reasoning bugs blocking live coverage.** `thinking=True` hardcodes
  `reasoning_effort="low"`, which 400s on both Mistral reasoning served models; and
  `thinking=False` is a no-op on Ollama, where omitting the parameter leaves the model
  default — which is *on* for qwen3. Surfaced by probes from the now-deleted live test
  plan; recorded here because they gate any live suite.
- **Write a new provider test plan.** The old drafts (`llm_provider_test_plan.md`,
  `llm_provider_test_guide.md`, `llm_provider_requirements.md`, `llm_live_test_plan.md`)
  are deleted and not worth reviving — they predate the served-model vocabulary and the
  current capability layer. The replacement should define the test layers and their
  markers (contract, capability, adapter, reliability, live), how to mock LiteLLM and
  inject `_sleep`, and how to add a served model or a step. Write it after the rename so
  it uses the settled terms; it becomes the source for the Contributing guide below.
- **Capability-driven live test catalogue.** A curated served-model catalog plus one
  shared live suite parametrized over it, so adding a model is a single catalog entry.
  Replaces the ad-hoc per-provider `integration` tests and wires up the `live` marker.
  Depends on the two bug fixes and the new test plan.
- **Migrate anthropic to `claude-sonnet-5`.** Check support for `claude-sonnet-5` and
  add it in place of `claude-sonnet-4-6` (`_CATALOG`, examples, defaults); confirm
  capability parity (reasoning / batching / structured output) before removing the
  4.6 entry.

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

- Migrate existing per-provider `integration` tests onto the `live` marker and the shared catalogue once it lands; retire duplicated ad-hoc coverage.
- Unused markers (`multimodal`, `ollama`, `vllm`, `llamacpp`) are declared but not yet applied to tests.
