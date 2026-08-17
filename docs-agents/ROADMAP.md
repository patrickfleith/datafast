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

- Ollama capability probe and default: `probe_capabilities()` on the ollama served model
  asks the daemon's `/api/show` what the model can actually do, returning Ollama's own
  names (`completion`, `vision`, `audio`, `thinking`, `tools`). Which model is pulled is
  a property of the machine rather than of the id, so name heuristics can only be
  approximately right — the probe is the way out where being wrong matters, and it
  resolves the base URL exactly as LiteLLM does so it cannot reach a different daemon
  than the generate calls. The profiles stay static and the probe stays opt-in. The
  factory default moved from `gemma3:4b` to `gemma4:12b`, which also moves the default
  from `OLLAMA_CHAT` to `OLLAMA_REASONING_CHAT`; `docs/models.md` now records that this
  default is unlike the hosted ones and names lighter alternatives with their sizes.
  `docs/llms.md` gained a "Provider-Specific Methods" section covering the probe and
  Mistral's `upload_file`/`delete_file`, neither of which had been documented.

- `provider_id` always names a server, never a wire format (DEC-003):
  `openai_compatible()` takes a required `provider_id` keyword and rejects wire-format
  values with an actionable error; the `backend` parameter is gone, matching the
  glossary's "avoid: backend". Provider ids are normalized (`llama.cpp` → `llamacpp`)
  and stay free-form, so any self-hosted server works — known ids get their profile,
  unknown ones fall back to the conservative OpenAI-compatible one.

- Live provider test suites (`tests/live/`), one directory per provider, gated behind
  `--run-live` and self-skipping when the API key is absent. Anthropic, openai, mistral,
  ollama, gemini and openrouter have landed (generation, structured output, reasoning, multimodal, plus
  openai's Responses transport, fallback-concurrency batching and `OPENAI_CHAT`
  profile, mistral's reasoning allowlist and Files upload path, gemini's per-model
  reasoning floor across two models and its exact-item-count schema, and ollama's capability
  probe, `top_p` pass-through, batched message lists and nested-schema constrained
  decoding, openrouter's pinned-endpoint routing on `google/gemma-4-31b-it`, where one
  model id spans 19 endpoints that disagree about json_schema and image support).
  Every provider reachable without standing up a server is covered; vllm and llamacpp
  are tracked under Later, since they need one.
  Shared image/PDF assets live in `tests/live/assets/`. The legacy per-provider
  `integration` suites are gone — the last two files, `tests/test_openrouter.py` and the
  shared `tests/test_schemas.py`, were deleted once openrouter's live suite landed. Coverage those suites
  held and the live ones lacked was ported: batched message lists and nested schemas
  per provider, and the input-validation cases into the mocked
  `tests/test_served_model_unit.py`, where they never needed a network call.
  Ollama is the first local backend, and being local changes the setup rather than the
  tests: there is no key to guard on, so `require_ollama` checks the daemon answers and
  then that the model is pulled, and it resolves `OLLAMA_API_BASE` the way LiteLLM does
  so the guard and the calls cannot disagree about the host. The suite runs on
  `qwen3:0.6b` with `gemma4:12b` for vision only — small on purpose, because a small
  model turns a mapping bug into a visible failure instead of absorbing it.
  Four lessons worth carrying forward:
  - A declared modality can still be unreachable in practice. Mistral takes files only
    as an uploaded id (`files_require_file_id`); ollama declares `Modality.IMAGE` for
    every model while only some can see, which is why `probe_capabilities()` exists.
  - Provider docs are not authoritative about model ids — the listing endpoint is.
    Mistral's `/v1/models` confirmed every catalog id and caught `ministral-*` sitting
    on the self-hosted profile; ollama's `/api/show` reports per-model capabilities and
    caught the profiles over-declaring vision.
  - An ordering assertion must not depend on model knowledge. A wrong answer in the
    right slot is indistinguishable from a right answer in the wrong slot, so both
    concurrency tests now key on something the input dictates (an echoed token, or the
    country named in the prompt) rather than on a capital city the model must recall.
  - A schema constraint the test relies on belongs in the schema. The nested-schema test
    asserted a non-empty list while the schema permitted an empty one, so it passed only
    by the model's goodwill; `minItems: 2` makes the nested branch of the grammar
    unavoidable, and Ollama does enforce it.

- `claude-sonnet-5` support via a second Anthropic profile, `ANTHROPIC_ADAPTIVE_CHAT`.
  Adding the catalog key against `ANTHROPIC_CHAT` would have been wrong twice over, and
  both failures were silent. Sonnet 5 reasons unless told not to, so
  `reasoning_off_param=None` — which means "omitting the parameter already means off" —
  would have left `thinking=False` billing reasoning tokens, the same bug already fixed
  for gemini. And `reasoning_effort="none"` is not the fix: LiteLLM maps that value to
  *dropping* the parameter, landing back on the model's own default, so the off switch
  has to be Anthropic's native `thinking={"type": "disabled"}`. Measured rather than read
  off the docs: on a hard prompt the bare request and the explicitly-adaptive one both
  came back with a thinking block having spent the full 6000-token cap, while the
  disabled one answered in 1414 tokens. The second break is `temperature`, which this
  line rejects at any value but 1 whether or not reasoning is on, so it leaves
  `supported_params` rather than staying merely locked-while-thinking as on haiku. The
  accepted efforts are pinned to `low`/`medium`/`high`/`xhigh`/`max`: `none` would read
  as off while leaving the default in force, and `minimal` is silently mapped to `low`.
  Live coverage is `tests/live/anthropic/test_sonnet_5.py`, which needs both halves —
  reasoning on *and* reasoning off — since either alone would pass on a model that never
  reasons. Two facts came out of writing it. The trace is real but unreadable: thinking
  blocks arrive with empty text and `reasoning_content` empty, because Anthropic omits
  the written summary by default and datafast has no control to ask for one. And thinking
  shares the answer's token budget and sometimes eats all of it, so the reasoning-on test
  asserts the trace and not the answer — requiring both made it flaky at 10000 tokens.
  Not a default change: the `anthropic()` factory and `docs/models.md` stay on
  `claude-haiku-4-5`, and `claude-sonnet-4-6` keeps `ANTHROPIC_CHAT` and its entry.
  The anthropic example scripts stay on sonnet 4.6 deliberately — `06` and `10` print
  `reasoning_content`, which is 3162 characters there and empty on sonnet 5, so moving
  them would have turned two working demos into blank output.

## In progress

Nothing in flight — the next item is picked from Next up.

## Next up

Launch checklist, grouped by area. All pipeline-architecture items gating the
release have landed (see Shipped); what remains is provider hardening and
documentation.

### Provider hardening & tests

Nothing outstanding. The served-model rename and the `claude-sonnet-5` support both
landed (see Shipped); vocabulary is settled in `docs-agents/GLOSSARY.md`.

### Documentation (launch)

Bring the published docs (mkdocs, `docs/`) to release quality. The site today covers
Home, Concepts, a few Guides, three Cookbook recipes, Served models, Models, and API.
Gaps to close, roughly in priority order:

- **Check migration to Zensical, and migrate if confirmed.** Material for MkDocs
  reaches end of life on **November 5, 2026** — maintenance mode since Nov 2025, only
  critical bug and security fixes until then, no new features. The successor is
  Zensical, from the same maintainers, which reads `mkdocs.yml` natively. Our setup is
  the easy case: plain `theme: material`, no Insiders features, no theme overrides, no
  unusual plugins. Do a trial Zensical build against the current `mkdocs.yml`, confirm
  the feature set survives (navigation tabs/sections, search highlight, admonitions,
  pymdownx superfences/highlight/details, toc permalinks, mkdocstrings for the API
  page), then switch `pyproject.toml`'s `docs` extra and the build. Ref:
  https://github.com/squidfunk/mkdocs-material/issues/8523
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
  model or a step, and project layout. Written from the suites as they stand — the test
  layers are considered settled and get no separate plan document. Must document the
  real commands:
  `.venv/bin/pytest -m "not live"` for the default run, `--run-live` to opt in, and that
  live tests self-skip when their key or daemon is absent.
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
  All three are the same shape of work, and the same blocker: each needs a server
  running plus model ids that are local to whoever runs the suite, which is why they
  sit here rather than in TASKS. When one does land, `tests/live/ollama/` is the
  template — it is the only existing suite with no API key to guard on, so copy
  `require_ollama` and change the health endpoint it probes. Note that vllm and
  llamacpp are reached through `openai_compatible`, i.e. the OpenAI wire format, so
  unlike ollama they need no route-specific parameter translation.
- **Capability-driven live test catalogue.** A curated served-model catalog plus one
  shared live suite parametrized over it, so adding a model is a single catalog entry.
  Demoted from the launch checklist: the problem it was meant to solve — ad-hoc
  per-provider `integration` tests — is gone, and the six per-provider live suites
  replaced them instead. Weigh it against what the current shape buys, because each
  suite states what is peculiar to its provider (gemini's per-model reasoning floor,
  openrouter's endpoint pinning, ollama's capability probe) and a parametrized sweep
  would flatten exactly that. `tests/live/test_pipeline.py` is the one place the
  parametrized shape already earns its keep.
- Video input live coverage; `previous_response_id` continuation live scenario (E07); full-catalog live sweep (E08).

## Improvements & tech debt

Non-feature work: rework, refactor, performance, cleanup.

- ~~Migrate existing per-provider `integration` tests onto the `live` marker~~ — done. All six legacy files (`tests/test_{ollama,gemini,anthropic,openai,mistral,openrouter}.py`) are deleted, along with `tests/test_schemas.py`, their shared fixture module. Coverage the live suites lacked was ported first, per provider; the persona/QA/MCQ cases were dropped as model-quality tests, apart from one rewritten on gemini as a schema-constraint test.
- ~~Dead `supports_thinking` capability flag~~ — done. Declared on
  `ServedModelCapabilities` and set by `ANTHROPIC_CHAT`, but never read: whether
  `thinking=True` is accepted or refused is already decided by `supports_reasoning` plus
  the profile's `supported_params`. It was born unused in the commit that created the
  capability layer rather than left behind by a removed feature, and a flag that looks
  load-bearing while doing nothing is a trap for the next person writing a profile —
  the same reason `rate_limits` and runner-level `max_concurrent` went.
- ~~Unused markers~~ — done. `integration`, `slow`, `vllm` and `llamacpp` are no longer declared; `pytest.ini` now registers only markers that tests actually carry. The root conftest skips on `live` alone, and AGENTS.md's default test command is `-m "not live"`. A vllm or llamacpp live suite should re-add its marker when it lands.
