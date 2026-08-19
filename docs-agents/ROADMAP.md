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
- Dead `RunConfig` fields removed: `show_progress` and `log_level` were declared and read nowhere. `log_level` duplicated `configure_logger(level=...)`, which is exported from `datafast` and is the real mechanism — loguru's config is global, so a per-run field would have applied a process-wide side effect scoped to one `run()`. `show_progress` had no infrastructure behind it (no tqdm or rich, neither a dependency) and the runner already reports per-step progress via `logger.info`. `RunConfig`'s field set is now pinned by `tests/test_runner_execution.py`, so the execution guide can document all eight fields as live.
- Dependency surface trimmed to what the package imports. Runtime dependencies are now `litellm`, `loguru`, `pydantic`, `httpx` and `python-dotenv` — the five imported at module scope. The six unused declarations (`instructor`, `google-generativeai`, `anthropic`, `openai`, `gradio`, `botocore`) are gone: LiteLLM is the only LLM path and reaches Anthropic/Gemini over its own HTTP transport, and it declares `openai` itself. The three lazily-imported packages became feature extras — `datafast[parquet]` (pyarrow) and `datafast[hub]` (datasets, huggingface-hub), with `datafast[all]` for both — so `datasets` no longer forces pyarrow, pandas and the Hub stack into every install. A base install resolves to 50 packages against the old 107; `datafast[all]` takes 59. The five `ImportError` messages now name the extra. `tests/test_dependencies.py` walks the package AST and fails on drift in either direction: a declared dependency nothing imports, or a third-party import nothing declares.
- API page generated from docstrings. `docs/api.md` is now `:::` directives rendered by mkdocstrings (added to the `docs` extra and `mkdocs.yml`) instead of a hand-maintained bullet list that had drifted to 34 of 48 names. Chosen over pinning the hand-written list with a test because the content the audit wants — every step's parameters — already lives in the docstrings, so it publishes immediately rather than being transcribed and then maintained twice; a test can pin names but not parameter documentation. The page went from 81 lines of bare names to ~82k rendered characters and 85 parameter tables. Adopting it exposed the docstrings that rendered blank: the six provider factories had none and now document their API-key env var, transport and defaults, and `Filter` named 6 of its 23 operators and now carries the full table. `mkdocs build --strict` is clean, which required annotating eight public `**kwargs`. `tests/test_api_page.py` pins page coverage against `__all__`, and `tests/test_filter_operators.py` (36 cases) pins every operator the page publishes — they had no tests at all.
- Dangling design-document references removed. `README.md` and `SOFTWARE_DESCRIPTION.md` both pointed at `datafast_new_design_document.md`, deleted long ago; they now point at the published docs site and `docs/concepts.md`. Auditing the same class of claim turned up two more: `[project.urls] Documentation` pointed at the GitHub repo rather than the docs site, and `docs/PUBLISHING.md` asked for `PYPI_USERNAME`/`PYPI_PASSWORD` while the workflow uploads as `__token__` with `PYPI_API_TOKEN` — following that guide could not have worked. Both corrected.
- Release metadata set for v1: `1.0.0`, `Development Status :: 5 - Production/Stable`, documentation URL on the docs site (DEC-004). The roadmap's "version is `0.0.35`" was a stale read of this branch — `main` had already auto-bumped to `0.0.36`, so merging as-is would have made the publish workflow re-tag an existing `v0.0.35` and fail. Worth knowing for any future release: that workflow publishes whatever version is in `pyproject.toml` at merge time, straight to PyPI, and rejects non-`X.Y.Z` strings, so release candidates are not possible without changing it. The changelog's empty `[0.1.0]` placeholder, which contradicted the v1 framing and was never tagged, is gone; `[Unreleased]` became `[1.0.0] — 2026-08-18`.
- Docs CI aligned with the `docs` extra: `.github/workflows/deploy-docs.yml` installed a hand-listed set of packages and now installs `-e ".[docs]"` and builds with `--strict`, so CI and a local build cannot drift.
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

- A pipeline may end in several sinks (DEC-005). `compile()`'s rule went from "a sink
  must be the last step" to "nothing may follow the sinks", which is all that stood in
  the way: sinks already yield every record through, so the runner needed no change and
  chaining worked mechanically. This was the last documentation blocker —
  `43_cookbook_persona_generation.py` chained `Sink.jsonl >> Sink.hub` and could not
  compile, so the published recipe described a pipeline that would not run. Fixing it
  retired the script's out-of-pipeline `push_records_to_hub()` workaround, which had
  quietly made one run publish to two different Hub repos — the private one named by
  `HF_REPO_ID`, and a second, hardcoded, **public** one. A sweep of all 45 example
  scripts confirmed 43 was the only real chain; `44_cookbook_space_text_generation.py`
  keeps its helper deliberately, since its push is opt-in behind `DATAFAST_PUSH_TO_HUB`
  and a sink in the chain would run unconditionally.

- Docs site migrated to Zensical (DEC-006), before the v1 pages are written rather than
  after, so nothing gets written twice. Material for MkDocs goes end-of-life on
  2026-11-05. `mkdocs.yml` is unchanged — Zensical reads it natively — and the migration
  is two lines: the `docs` extra and the build command. The open question was
  mkdocstrings, adopted only just before this; Zensical supports it directly and names
  it in an error if it is configured but missing. Verified by building both and diffing:
  the same 16 pages, and an API page with identical 152 symbol anchors (zero difference
  in either direction), 46 parameter tables, 68 highlighted code blocks and 214
  permalinks. `zensical build --strict` is clean. Zensical is pre-1.0 at 0.0.55, which
  is the standing risk; `mkdocs.yml` staying the source of truth makes reverting a
  one-line change.

- Quickstart page (`docs/quickstart.md`), the first of the five target-shape pages:
  install, one API key, a 15-line pipeline, the file it writes and the row it writes.
  `tests/test_quickstart.py` extracts the page's own code block and executes it with a
  stub served model, so the documented pipeline is the one under test — it pins the
  six-row expansion, the exact output columns, and that the JSON sample row on the page
  names the same fields a run produces. Writing it caught two things a hand-written page
  would have shipped wrong: `Seed.values` returns a dimension, not a source (only
  `Seed.product`/`Seed.zip` return one), and `_model` is the single metadata column a
  plain single-model step adds. Note for anyone pinning docs this way: the block imports
  `openai` from `datafast`, so the factory has to be patched on the module — a stub
  injected into the exec namespace is silently overwritten by the block's own import,
  and the test calls the real API instead.

- `docs/index.md` and `README.md` stopped being changelogs. Both opened on "the old
  dataset-class API has been removed" and a "What Changed" section, contrasting v1
  against something no v1 reader has ever used. They now open on what datafast is, what
  it produces, and a "Why pipelines" section giving the four reasons the shape earns
  itself — declarative coverage, per-call resume, interchangeable providers, and
  `compile()` catching mistakes before the spend. The README gained the docs-site links
  it never had, `Sample` in a building-blocks list that had omitted it, and the
  `--run-live` note; its Langfuse section had a code block that imported `LLMStep` and
  `Seed`, used neither, and demonstrated nothing, now replaced by one that shows the
  auto-enable. All three documents open on the same pipeline as the quickstart, and
  `tests/test_quickstart.py` executes each of the three as written — a broken opening
  example being the worst kind to ship.

- Installation & environment reference (`docs/installation.md`): the base install and
  its five runtime dependencies, the four extras, all eleven environment variables, the
  `.env` rules (loaded once at first served-model construction, never overriding a real
  variable), Langfuse setup and the logging control. `tests/test_installation_page.py`
  scans the package for `os.getenv` and `env_key_name` and fails on any variable the
  page does not document — the direction that matters, since a variable the code reads
  and the docs omit cannot be discovered. It finds 10; `LANGFUSE_HOST` is written for
  langfuse to read and never read back, so it is pinned separately. The page also pins
  each API key to the factory that declares it, the extras against `optional-dependencies`,
  the dependency table against `dependencies`, and the Python floor against
  `requires-python`. Writing it corrected a claim that had already been drafted: building
  a served model does **not** validate the API key — construction stores `None` and the
  failure surfaces on the first call, which is worth knowing before a long run.

- Glossary published and Concepts rewritten (`docs/glossary.md`, `docs/concepts.md`) —
  the first of the ordered documentation groups, and first deliberately, because every
  reference page still to be written leans on this vocabulary. `docs-agents/GLOSSARY.md`
  grew from 9 served-model terms to 29 across two sections, adding the pipeline half it
  had never had: record, column, step, transform, pipeline, runner, checkpoint,
  manifest, compile, seed, dimension, branch path, prompt template, and the two
  strategy terms. The published page carries the definitions verbatim and renders the
  `_avoid:_` notes as **Avoid** lines, which earn their place on a public page rather
  than only an internal one: a reader searching for "row", "DAG" or "backend" lands on
  the term datafast actually uses. `tests/test_glossary_page.py` pins the pair in both
  directions — no canonical term unpublished, no published term invented — and asserts
  the definitions are byte-identical, since two definitions of one term is the exact
  failure a glossary exists to prevent. It also checks the code the glossary names:
  10 classes against `datafast.__all__`, 5 dotted attributes by `hasattr`, and the
  three execution strategies against the `LLMExecutionStrategy` enum.
  `docs/concepts.md` was 71 lines of headings restating the API; it is now 188 lines
  built on record → step → pipeline → runner, and says the things the old page left
  out — that a step's whole contract is `process(records) -> records`, that a pipeline
  is itself a step (which is how `Concat` takes pipelines), that the runner materializes
  each step in full and what that trade buys, that the manifest's pipeline fingerprint
  is what makes resume safe, and that checkpointing inside an LLM step is per call.
  `tests/test_concepts_page.py` executes every self-contained block and asserts the
  numbers the prose quotes. Writing the tests corrected the page twice: the `as_step`
  block imported `Source` without using it, and the source list named five factories
  when there are eight.

- Reference templates written, opening the parallel fan-out: `docs/reference/
  sources_and_seed.md` (271 lines) for the step reference, and
  `docs/reference/served_models.md` (199) plus `docs/reference/providers/openai.md`
  (95) for the provider reference. Three pages chosen deliberately over one: each
  family needs its own shape settled before five or six more are written against it,
  and the provider pages additionally need the shared configuration surface to link to
  rather than re-explain seven times.

  **The shape the remaining reference pages follow.** Intro saying what the family is
  for and where it sits · an at-a-glance table of every constructor · one section per
  constructor with a parameter table (`Parameter | Type | Default | Meaning`), a runnable
  example, and the behaviour worth knowing · a "Things worth knowing" list of the traps ·
  a "Where to go next" block. Links point only at pages that already exist.

  **The test shape**, which matters more, because it is what keeps 22 pages honest at
  once. Every page test asserts code → docs first: every parameter `inspect.signature`
  reports must appear on the page, since a parameter that exists and is undocumented
  cannot be discovered while the reverse is merely untidy. Then every self-contained
  example is executed, and every behavioural claim is asserted against a real call
  rather than paraphrased from the source. `tests/test_reference_sources_and_seed.py`
  (36 tests), `tests/test_reference_served_models.py` (24) and
  `tests/test_reference_provider_openai.py` (19) are the three to copy; the suite is
  375 passing, verified with sockets blocked.

  Writing them turned up things a hand-written page would have got wrong. `Source.file`
  reads `.json` as **JSONL**, so a conventional JSON array silently fails to load. A
  malformed JSONL line is skipped with a warning rather than raising, so a corrupt file
  loads partially. CSV and TSV values are always strings — no type inference. `Seed.range`
  is **inclusive** at both ends, unlike Python's `range`. Two seed dimensions filling the
  same column silently overwrite, last one winning. And the signature guard immediately
  caught its own page: `Seed.product` and `Seed.zip` had no parameter table when every
  other constructor did.

- Step and provider reference written in parallel: twelve pages by twelve agents against
  the two templates, in one pass. Step reference — `sinks.md` (169), `data_ops.md` (335),
  `sample.md` (288), `llm_step.md` (275), `llm_specialized.md` (311), `branching.md`
  (276). Provider reference — `anthropic.md` (108), `gemini.md` (98), `mistral.md` (120),
  `openrouter.md` (130), `ollama.md` (144), `openai_compatible.md` (158). The reference
  is now 14 pages and ~3000 lines. The suite went 375 → 848 passing, verified offline
  with sockets blocked; `zensical build --strict` clean with all 14 in the nav.

  The roadmap's single "LLM steps" page became two. `LLMStep` plus the five specialized
  steps is ~2200 lines of source, and one page over all six would have been the worst
  page on the site.

  **Parallelism held because the coupling was removed first, not managed.** A page
  outside the nav still builds clean, so twelve writers never touched `mkdocs.yml` — the
  one file they would all have collided on — and the nav was wired once at the end. Each
  agent created exactly two new files. Nothing was edited twice and no conflict occurred.

  **The test contract is what made it safe to write twelve pages without reading twelve
  pages.** Every page asserts code → docs first: each parameter `inspect.signature`
  reports must appear on the page. Four agents mutation-checked their own guards by
  breaking a claim and confirming the right test failed. That guard is why the pages
  describe the code rather than the docstrings — and the gap between the two turned out
  to be large. See CONCERNS.md for the ten defects and six risks this surfaced.

  One of them was mine. This roadmap listed "per-step `temperature`/`max_tokens`
  overriding the served model" as behaviour to document. Both are stored and never read,
  so the audit line was wrong: it was written from the constructor without checking the
  call site. Corrected above, documented truthfully on the page, and pinned by a test
  that fails the day someone wires them up.

## In progress

Nothing in flight — the next item is picked from Next up.

## Next up

Launch checklist, grouped by area. All pipeline-architecture items gating the
release have landed, and so have every code and packaging fix the doc audit turned
up, including the last blocker and the Zensical migration (see Shipped). What
remains is writing: one of the five target-shape deliverables is done, and the
other four are eleven pages, ordered under Documentation below.

### Documentation (v1 launch)

Bring the published docs (Zensical, `docs/`) to release quality for **v1**. Re-audited
against the code on 2026-08-17, after the served-model refactor; the notes below
replace the earlier list, which predated it and had gone stale in several places.

**Target shape.** Five things the site must deliver, in this order of importance:

1. ~~**Quickstart** — install to first stored dataset, one page, no detours.~~ — done,
   with `docs/installation.md` beside it and the landing pages rewritten to match.
2. **Cookbooks** — two or three *deep* end-to-end recipes, plus an index mapping the
   45 scripts in `examples/scripts/` to what each demonstrates.
3. **Provider reference** — every factory, every supported model, every parameter.
4. **Component reference** — every step family and how they connect, source seed →
   stored dataset.
5. **Concepts & glossary** — the vocabulary, published rather than agent-only.

**Current state.** Re-checked on 2026-08-18, after the reference templates landed.
The published site is 20 pages (`docs/` holds two more that the nav does not carry:
`PUBLISHING.md` and `cookbook/assets/index.md`). Nine of the twenty are v1 work —
`quickstart.md` (113 lines), `installation.md` (158), `concepts.md` (188),
`glossary.md` (70), the rewritten `index.md` (62), the generated `api.md` (85 lines of
directives, ~82k rendered characters) and the three reference templates
(`reference/sources_and_seed.md` 271, `reference/served_models.md` 199,
`reference/providers/openai.md` 95). The other eleven predate v1, and six of them
are under 60 lines: `guides/index.md` (8), `cookbook/index.md` (16),
`guides/checkpointing.md` (33), `models.md` (36), `guides/llm_steps.md` (49) and
`guides/building_pipelines.md` (59) are placeholders in all but name.

The parameter gap this section opened with is now largely closed by `docs/api.md`
becoming generated: the nine `Sample` strategies, the 23 `Filter` operators, the
`Rewrite` modes, the `Extract` presets, `Group`'s aggregation spec, `Pair`'s
strategies, `Join`/`JoinBranches` modes and `Seed.expand` all render from their
docstrings. What remains is not reference material but *narrative* — the prose pages
that say which step to reach for and why, in what order, and the worked examples. The
generated page is a lookup surface, not a guide, and it publishes only what the
docstrings say: `Pipeline`, `Step`, `Record`, `RunConfig`, `ServedModel` and the
concrete sinks still carry one-line docstrings and render thin. `docs/models.md`
(36 lines) lists seven factory defaults and nothing else, while `capabilities.py`
holds 17 catalogued models, 15 capability profiles and four layers of fallback for
everything not catalogued — the provider reference is still to write.

**What is left, in the order it should be written.** Eleven pages to write or
rewrite, and they are not equal — the first three groups are the launch, and the last is
bookkeeping that only makes sense once the pages exist.

1. ~~**Concepts & glossary** (2 pages)~~ — done. It went first, though it is the
   smallest group, because every reference page below leans on the vocabulary and
   writing them in the other order means writing the terms repeatedly and
   inconsistently.
2. ~~**Step reference**~~ — done, 7 pages (the single "LLM steps" page became two).
3. **Pipeline & execution guide** (1 page) — absorbs `guides/checkpointing.md`.
4. ~~**Provider & served-model reference**~~ — done, 8 pages. `docs/models.md` is now
   redundant and should be folded in or deleted during the nav restructure.
5. **Cookbooks** (3 deepened + 1 examples index) — the second target-shape deliverable,
   and the first thing that shows the library doing real work end to end.
6. **Specialist guides** (3 pages) — structured output, multimodal input, calling a
   served model directly.
7. **Error handling & troubleshooting** (1 page).
8. **Contributing** (1 page) and the **"What's in v1"** release-notes page (1 page).
9. **Build & infrastructure** — the nav restructure last, since it is a rearrangement of
   pages that must exist first; `py.typed` and retiring `SOFTWARE_DESCRIPTION.md` can
   happen at any point.

Each numbered group is a coherent unit of work — the larger ones split family by
family, one commit each — and each group is independently shippable, so the site stays
coherent if the launch date arrives partway down the list.

**Parallelism.** Groups 2, 4 and 5 are the wide ones: their pages are independent files
over independent code. With both templates written, 15 of the remaining 22 can be
written concurrently — 5 step-reference pages, 6 provider pages, and the 4 cookbook
items — plus Contributing and `py.typed`, which were never coupled to anything. Two things couple
them, and both are removable up front. `zensical build --strict` exits 1 on a link to a
page that does not exist yet (verified, not assumed), so either the whole file skeleton
and the final nav land in one commit first, or no page may link to an unwritten
sibling. And each family needs one page written first as the template — Sources & Seed
for the step reference, one provider page for the provider reference — or six pages
arrive in six different shapes. The parallelism is therefore 1-then-(N−1) per family,
not N. Nav restructure and the "What's in v1" page cannot be parallelised at all: the
first rearranges every page, the second summarises them.

#### New pages to write

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
    `{language}`/`{language_name}`, and the `_model`/`_prompt_index`/`_language`
    metadata columns) — note the earlier claim here that per-step `temperature` and
    `max_tokens` override the served model was **wrong**: both are stored and never
    read (see Shipped);
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
- **Contributing & development guide.** Project layout, the test layers
  (`tests/` mocked, `tests/live/<provider>/` gated), the real commands
  (`.venv/bin/pytest -m "not live"` by default, `--run-live` to opt in, live tests
  self-skip on a missing key or daemon), and how to add a served model or a step.
- **Release notes.** Populate `docs-agents/CHANGELOG.md` for 1.0 and publish a
  "What's in v1" page. **No migration guide** — everything before this is experimental
  and unsupported, so v1 is the starting point, not a transition.

#### Rewrites of existing pages

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
- ~~**`docs/PUBLISHING.md`** sits in `docs/` but is absent from the nav.~~ Done: moved
  to `docs-agents/PUBLISHING.md`, beside the other internal docs.

#### Build & infrastructure

- ~~**Nav restructure.**~~ Done: the nav is now Home · Get started · Guides ·
  Reference · Cookbook · Contributing, and carries every page on disk. Nine finished
  pages had been published but unreachable. `models.md` was kept rather than folded —
  all seven provider pages link to it for their default model id and none states its
  own. The three superseded guides (`building_pipelines`, `llm_steps`, `checkpointing`)
  were deleted and their 31 inbound links rewritten.
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
