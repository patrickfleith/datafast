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
release have landed (see Shipped); what remains is documentation, plus the handful
of code and packaging fixes the doc audit turned up (see "Settle these first").

### Documentation (v1 launch)

Bring the published docs (mkdocs, `docs/`) to release quality for **v1**. Re-audited
against the code on 2026-08-17, after the served-model refactor; the notes below
replace the earlier list, which predated it and had gone stale in several places.

**Target shape.** Five things the site must deliver, in this order of importance:

1. **Quickstart** — install to first stored dataset, one page, no detours.
2. **Cookbooks** — two or three *deep* end-to-end recipes, plus an index mapping the
   45 scripts in `examples/scripts/` to what each demonstrates.
3. **Provider reference** — every factory, every supported model, every parameter.
4. **Component reference** — every step family and how they connect, source seed →
   stored dataset.
5. **Concepts & glossary** — the vocabulary, published rather than agent-only.

**Current state.** 13 published pages, ~1,220 lines, against ~10,100 lines of code.
The site is a tour, not a reference: it names the steps but documents almost none of
their parameters. Concretely, nothing published today covers the nine `Sample`
strategies, the 21 `Filter` operators, the eight `Rewrite` modes, the six `Extract`
presets, `Group`'s aggregation spec, `Pair`'s strategies, `Join`/`JoinBranches` modes,
or `Seed.expand`. `docs/models.md` (36 lines) lists seven factory defaults and nothing
else, while `capabilities.py` holds 17 catalogued models, 15 capability profiles and
four layers of fallback for everything not catalogued.

#### Settle these first — each one changes what gets written

- **Dead `RunConfig` fields.** `show_progress` and `log_level` are declared in
  `datafast/core/config.py` and read nowhere in the package. Documenting them would
  publish a lie; remove them or implement them before the execution guide is written.
  Same failure mode as `rate_limits` and runner-level `max_concurrent`, which is
  exactly what the old roadmap entry warned about.
- **Dependency surface.** `pyproject.toml` requires `instructor`,
  `google-generativeai`, `anthropic`, `openai`, `gradio` and `botocore`, and the
  package imports none of them — LiteLLM is the only LLM path. Meanwhile `pyarrow` and
  `huggingface_hub` are imported lazily with "install it with…" errors but are declared
  nowhere, and `datasets` is a hard dependency that only `HubSink` and
  `HuggingFaceSource` use, both behind lazy imports. An honest install page needs the
  real answer first: trim the unused six, and turn datasets/pyarrow/huggingface_hub
  into extras (`datafast[hub]`, `datafast[parquet]`, or one `datafast[data]`).
- **API page mechanism.** The old entry assumed mkdocstrings; it is not installed and
  not configured. `docs/api.md` is a hand-maintained bullet list, already missing
  `Seed.expand`, `LLMExecutionStrategy`, `PipelineChangedError`, `SeedDimension`,
  `HuggingFaceSource`, `configure_logger`, `get_version`, the concrete sink classes and
  everything in `datafast.llm` (`ContentPart`, `Modality`, `ServedModelCapabilities`, …)
  — 48 names in `__all__` against a list that covers roughly two-thirds. Decide:
  adopt mkdocstrings (add to the `docs` extra and `mkdocs.yml`), or keep it hand-written
  and pin it with a test. `tests/test_public_api.py` already exists and is the place.
- **Broken references to a deleted file.** `README.md` and `SOFTWARE_DESCRIPTION.md`
  both point at `datafast_new_design_document.md`, which no longer exists.
- **Release metadata.** Version is `0.0.35` with `Development Status :: 3 - Alpha`, and
  `[project.urls] Documentation` points at the GitHub repo rather than the docs site.
  All three need to change for a 1.0 tag.
- **`43_cookbook_persona_generation.py` chains two sinks**, which `compile()` rejects
  (tracked in TASKS). It is a cookbook script, so this is a docs blocker: the published
  recipe currently describes a pipeline that will not compile.

#### New pages to write

- **Quickstart.** Install, one API key, a ~15-line pipeline, the file it writes, and
  what the output rows look like. Today the only quickstart is a code block on
  `docs/index.md` wedged between "what changed" notes.
- **Installation & environment reference.** Extras (settled above), every env var —
  `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `MISTRAL_API_KEY`,
  `OPENROUTER_API_KEY`, `OLLAMA_API_BASE`, `HF_TOKEN`, `LANGFUSE_*`,
  `DATAFAST_LITELLM_SUPPRESS_DEBUG_INFO` — and the `.env` loading behaviour (loaded
  once, when a served model is constructed).
- **Step reference — the largest gap.** One page per family, every parameter:
  - *Sources & Seed* — `Source.list/file/jsonl/csv/tsv/txt/parquet/huggingface` (note
    `file` sniffs by extension, `txt` takes `text_column`, `huggingface` takes
    split/subset/columns/streaming/trust_remote_code); `Seed.values/range/expand/
    product/zip` — `expand` and `zip` are undocumented today.
  - *Sinks* — jsonl / csv / parquet / hub / list; sinks pass records through, so what a
    `run()` returns is the last step's output. `Sink.hub`: `HF_TOKEN` resolution,
    `private`, `train_size` + `seed` + `shuffle` for the split, `commit_message`, and
    the auto-injected `datafast-dataset` README tag.
  - *Data ops* — Map, FlatMap, AddUUID (`column`, `overwrite`); Filter with the full
    operator table (`$eq $ne $gt $gte $lt $lte $in $nin $contains $startswith
    $endswith $regex $len_gt $len_lt $len_eq $len_gte $len_lte $exists $type $all
    $any`, plus `$or`/`$and`, plus `fn=` and `keep=False`); Group (`by`, `collect`,
    `output_column`, `agg` as `"column:function"` or `"column:concat:separator"` over
    the nine functions count/sum/mean/min/max/first/last/collect/concat,
    `min_per_group`/`max_per_group`);
    Pair (`n`, strategy, `within`/`across`, `output_format`, `max_pairs`, `seed`);
    Concat; Join (`on`, `how`, `suffixes`).
  - *Sample* — all nine strategies (`uniform`, `first`, `last`, `top`, `bottom`,
    `weighted`, `stratified`, `systematic`, `gaussian`), the five that require `by`
    (top, bottom, weighted, stratified, gaussian), `systematic`'s required `step` and
    `gaussian`'s required `center`+`std`, `n` vs `frac`, `seed`, `replace`, and the
    step-vs-value duality (`Sample(...)` in a pipeline vs `.pick()` / `.sample()`
    inside an `LLMStep` argument).
  - *LLM steps* — LLMStep (expansion math prompt × model × language × num_outputs,
    `parse_mode` text/json/xml, `output_column` vs `output_columns`, prompt-from-`Path`,
    `forward_columns`/`exclude_columns`, `skip_if`, `system_prompt`,
    `{language}`/`{language_name}`, per-step `temperature`/`max_tokens` overriding the
    served model, and the `_model`/`_prompt_index`/`_language` metadata columns);
    Classify (`multi_label`, `labels_description`, `include_explanation`,
    `include_confidence`); Score (`score_range`, `criteria`, `rubric`); Compare
    (`output_mode`); the llm-vs-`fn` dual mode shared by all three; Rewrite (eight
    modes, three of which require a companion argument — `custom`→`custom_instruction`,
    `audience`→`target_audience`, `length`→`target_length`, plus `preserve` and
    `num_variations`); Extract (`fields` vs the six presets `entities`, `facts`,
    `keywords`, `metadata`, `summary_fields`, `topics`; `flatten`).
  - *Branch / JoinBranches* — path tagging (`_branch_id`, `_branch_name`,
    `_branch_input_keys`), cartesian join, `suffixes`, `how`, how the runner recurses
    into paths (nested batching, dotted-path checkpoint files, resume), the
    determinism requirement for non-LLM path steps, and that Branch-inside-Branch is
    rejected by `compile()`.
- **Pipeline & execution guide.** `>>` composition, `Pipeline.compile()` and what
  `PipelineValidationError` catches (source-first/sink-last, Branch↔JoinBranches
  pairing, column references, sub-pipeline rules), then everything `run()` /
  `RunConfig` / `run_pipeline()` expose: `checkpoint_dir`, `resume`, `resume_from`,
  `stop_after`, `limit`, `batch_size`, `llm_strategy` (`by_model` / `round_robin` /
  `by_record`), `checkpoint_every` — and that throughput and rate limiting live on the
  served model (`rpm_limit`, `max_concurrent`, `timeout`, retries), never on the runner.
  Absorbs the existing `guides/checkpointing.md`, which covers six of these in 33 lines.
- **Provider & served-model reference — one page per provider.** `docs/llms.md` (178
  lines) is a good start on the shared surface but says nothing per provider. Each page
  needs: the factory and its default model, the API-key env var, the **supported model
  table** (the 17 catalog entries plus the fallback rules — OpenAI's `gpt-5*`/`o1`/`o3`/
  `o4` prefix match, Mistral's `magistral`/`-reasoning` match, Ollama's eight reasoning
  families, and the five `_PROVIDER_DEFAULTS`), the transport (chat vs Responses), and
  the capability profile each model resolves to with what that implies — supported
  params, modalities, structured-output mode, batch mode, and the reasoning contract
  (`reasoning_effort_on`, `reasoning_off_param`, `reasoning_efforts`,
  `reasoning_always_on`, `reasoning_locks_temperature`). The 15 profiles' `notes`
  tuples in `capabilities.py` are already written prose and are the source material;
  the sonnet-5, gemini-3.7 and ollama findings in Shipped are the reason this page
  cannot be inferred by the reader.
  Cross-cutting on a shared page: every `ServedModelConfig` field, the
  `unsupported_params` policy (fail/warn/quiet) and what "dropped" means per parameter,
  `provider_params` as the unchecked escape hatch, reliability (`RetryPolicy`
  max_retries/base_delay/max_delay/jitter, `timeout`, `rpm_limit`), native batching vs
  fallback concurrency, and the provider-specific methods (`upload_file`/`delete_file`,
  `probe_capabilities`).
- **Calling a served model directly.** `generate` / `generate_batch` /
  `generate_response` / `generate_batch_response`, `NormalizedResponse` (`text`, `raw`,
  `reasoning_content`, `thinking_blocks`, `images`, `audio`, `output_items`), and when
  to reach for one instead of an `LLMStep`. Covered only by
  `examples/providers/*` today, which the site never links to.
- **Structured output guide.** Step-level `parse_mode` (text/json/xml) versus
  provider-level Pydantic `response_format`, and how the profile's
  `StructuredOutputMode` (`json_schema` / `json_object` / `prompted_json` / `none`)
  decides which of the two you actually get.
- **Multimodal input guide.** `ContentPart` (type, text, url, data, media_type,
  media_id, filename, provider_options), the image/video/file/document shapes,
  `Modality` gating per served model, and the two traps already paid for:
  `files_require_file_id` on Mistral, and OpenAI's Responses API rejecting inline file
  data without `filename`.
- **Error handling & troubleshooting.** `on_parse_error` (skip/raise) and that "skip"
  silently drops records, partial results, resuming after a crash,
  `PipelineChangedError` and when the pipeline hash invalidates a checkpoint, reading an
  `unsupported_params` warning, and common provider errors.
- **Glossary & concepts (published).** `docs/concepts.md` is 71 lines and does not
  define a single term from `docs-agents/GLOSSARY.md`. Publish the glossary — served
  model, provider, model, capabilities, capability profile, served-model catalog,
  transport, parse mode — and rewrite Concepts around the record → step → pipeline →
  runner model with the checkpoint/manifest vocabulary.
- **Contributing & development guide.** Project layout, the test layers
  (`tests/` mocked, `tests/live/<provider>/` gated), the real commands
  (`.venv/bin/pytest -m "not live"` by default, `--run-live` to opt in, live tests
  self-skip on a missing key or daemon), and how to add a served model or a step.
- **Release notes.** Populate `docs-agents/CHANGELOG.md` for 1.0 and publish a
  "What's in v1" page. **No migration guide** — everything before this is experimental
  and unsupported, so v1 is the starting point, not a transition.

#### Rewrites of existing pages

- **`docs/index.md` and `README.md` must stop being changelogs.** Both lead with "the
  old dataset-class API has been removed" and a "What Changed" section. For a v1 launch
  there is no old API to contrast against; both should open on what datafast *is*, what
  it produces, and where to start. README also needs the feature list, doc-site links
  and the fixed repo-layout section.
- **`docs/models.md` → the supported-model tables** described above, or fold it into
  the per-provider pages and delete it. As a standalone list of seven defaults it
  answers a question nobody asks twice.
- **`docs/guides/llm_steps.md` (49 lines)** is a teaser for what becomes the LLM step
  reference; keep it as a narrative guide only if the reference exists beside it.
- **Cookbooks.** Keep the three that exist, but deepen two or three into true
  end-to-end walkthroughs — seed design, model choice, prompt, execution, checkpoint,
  output schema, and the Hub push — rather than pointers at a script. Best candidates
  from `examples/scripts/`: `42_pipeline_preference_with_scoring.py` (Branch + Score,
  the most architecture per line), `38_pipeline_qa_generation.py`, and
  `40_pipeline_classification_dataset.py`. Add the 01–45 examples index as a table.
- **`docs/PUBLISHING.md`** sits in `docs/` but is absent from the nav — it is a
  maintainer runbook, not user documentation. Move it out of the published tree or add
  it under Contributing.

#### Build & infrastructure

- **Check migration to Zensical, and migrate if confirmed.** Material for MkDocs
  reaches end of life on **November 5, 2026** — maintenance mode since Nov 2025, only
  critical bug and security fixes until then, no new features. The successor is
  Zensical, from the same maintainers, which reads `mkdocs.yml` natively. Our setup is
  the easy case: plain `theme: material`, no Insiders features, no theme overrides,
  `search` as the only plugin. Trial-build against the current `mkdocs.yml`, confirm
  the feature set survives (navigation tabs/sections/indexes, search highlight,
  admonitions, pymdownx superfences/highlight/details/inlinehilite, toc permalinks,
  attr_list, md_in_html, def_list), then switch `pyproject.toml`'s `docs` extra and the
  build. Do this **before** writing the bulk of the pages, so nothing is written twice.
  If mkdocstrings is adopted above, its Zensical support is the one thing to verify
  first. Ref: https://github.com/squidfunk/mkdocs-material/issues/8523
- **Nav restructure.** The target IA is roughly: Home · Get started (install,
  quickstart, concepts, glossary) · Guides (pipelines, execution & checkpointing,
  structured output, multimodal, tracing, troubleshooting) · Reference (sources & seed,
  data ops, sample, LLM steps, branching, sinks, providers ×7, API) · Cookbook
  (recipes + examples index) · Contributing.
- **Ship `py.typed`.** The file does not exist; the package is fully annotated and
  advertises none of it. Add it and the `package-data` entry in `pyproject.toml`.
- **Retire `SOFTWARE_DESCRIPTION.md`.** Fold into the docs above, generate
  `docs-agents/SUM.md` with the `write-manual` skill, then delete it.


## Later / long term

- **vLLM support.** Delta live tests + example suite (needs a running server).
- **llama.cpp support.** Delta live tests + example suite (needs a running server).
- **Caching.** Full caching design from requirements: provider-native prompt caching, router/gateway caching, local prefix/KV reuse, optional client-side result cache; capability-aware cache keys/hints; cache tests (H01–H07). Only `cache_mode` metadata exists today.
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
