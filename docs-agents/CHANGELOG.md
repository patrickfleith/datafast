# Changelog

## [Unreleased]

## [1.0.0] — 2026-08-18

First stable release.

### Added

- **The first reference pages.** `docs/reference/sources_and_seed.md` documents every
  way a pipeline can start — the eight `Source` constructors and the five `Seed` ones —
  with every parameter, and the behaviour that is easy to get wrong: `.json` files are
  read as JSONL, malformed JSONL lines are skipped rather than raised, CSV values are
  always strings, and `Seed.range` is inclusive at both ends.
  `docs/reference/served_models.md` documents every `ServedModelConfig` field, how
  capabilities are resolved through the catalog and its fallbacks, and the
  `unsupported_params` policy. `docs/reference/providers/openai.md` adds the model
  table, both capability profiles, and why reasoning has to be turned off explicitly on
  GPT-5.5. Each page is pinned by a test that fails when the code grows a parameter the
  page does not mention.

- **A published glossary and a rewritten Concepts page.** `docs/glossary.md` defines
  the 29 terms datafast uses precisely — the pipeline vocabulary (record, column, step,
  runner, checkpoint, manifest, compile, seed, dimension, branch path) alongside the
  served-model vocabulary — each with the terms it deliberately avoids, so searching
  for "row" or "backend" finds the word datafast uses instead. `docs/concepts.md` is
  rebuilt on record → step → pipeline → runner: the `process(records) -> records`
  contract every step satisfies, why the runner materializes each step in full, and how
  the manifest's pipeline fingerprint makes resume safe. Tests execute the page's code
  and pin the published glossary against the canonical one in both directions.

- **An installation & environment reference.** `docs/installation.md` documents the
  base install, the extras, every environment variable datafast reads, and the `.env`
  rules. A test scans the package for environment lookups and fails if the page misses
  one, so the list cannot fall behind the code.

- **A quickstart page.** `docs/quickstart.md` takes a reader from `pip install` to a
  stored dataset on one page: one API key, a 15-line pipeline, and the rows it writes.
  Its code block is executed by `tests/test_quickstart.py` against a stub served model,
  so the page cannot drift from the library without failing the suite.

- **A pipeline may end in several sinks.** `compile()` previously required the sink
  to be the single last step, so writing one dataset to both a file and the Hub took
  two runs. Sinks pass their records through, so a chain needs nothing from the
  runner — only the validation rule changed, from "a sink must be last" to "nothing
  may follow the sinks". `43_cookbook_persona_generation.py` is the recipe that
  motivated it: it chained `Sink.jsonl >> Sink.hub` and could not compile at all.

- **Docstrings for the six provider factories.** `openai`, `anthropic`, `gemini`,
  `mistral`, `openrouter` and `ollama` had none at all — they rendered as a bare
  signature on the generated API page. Each now documents its API-key environment
  variable, its transport, and the defaults worth knowing (OpenAI's Responses
  routing, Ollama's `OLLAMA_API_BASE` and `repeat_penalty`).
- **The full `Filter` operator reference.** The docstring named 6 of the 23
  operators; the other 17 were documented nowhere and tested nowhere. All are now
  described on `Filter` — comparison, membership, string, length, presence/type and
  the `$or` / `$and` logical forms — and pinned by `tests/test_filter_operators.py`.

- **Feature extras for optional I/O.** `datafast[parquet]` (pyarrow) enables
  `Source.parquet(...)` and `ParquetSink`; `datafast[hub]` (datasets,
  huggingface-hub) enables `HuggingFaceSource` and `HubSink`; `datafast[all]` takes
  both. `pyarrow` and `huggingface_hub` were previously imported with an "install it
  with…" hint while being declared nowhere — they are now installable by name. The
  `ImportError` messages name the extra rather than the raw package.

- **`top_p` and `frequency_penalty` as real config fields.** Both were declared by
  the capability profiles but had no field, so a caller's value slipped through
  `provider_params` unchecked — including to `OPENAI_RESPONSES`, whose reasoning
  models reject sampling controls. They are now gated like `temperature`: sent
  where the profile declares them, dropped under the `unsupported_params` policy
  where it does not.
- **`repeat_penalty` documented as Ollama's repetition control.** Ollama speaks its
  own API, where the knob is a multiplier neutral at `1.0` and values below it
  *reward* repetition; LiteLLM renames `frequency_penalty` onto it without
  rescaling, so an OpenAI-style `0.15` degenerated the output. The Ollama profiles
  no longer declare `frequency_penalty`; pass `ollama(repeat_penalty=1.2)` on its
  own scale instead. vLLM and llama.cpp are unaffected — they are reached over the
  OpenAI wire format, where `frequency_penalty` keeps its usual meaning.
- **`reasoning_summary` on served models.** OpenAI returns a reasoning summary only
  when asked, and the ask shares the Responses `reasoning` object with the effort.
  `openai(reasoning_summary="auto")` merges into that object, so `reasoning_content`
  is now reachable without hand-writing the provider's request shape through
  `provider_params` — which replaced the whole `reasoning` key rather than adding
  to it. Declared by `OPENAI_RESPONSES` only; elsewhere it goes through the
  `unsupported_params` policy like any other unsupported parameter.

### Changed

- **The README and the docs home page open on the library, not on its history.** Both
  led with the removal of the pre-1.0 dataset-class API and a "What Changed" section —
  a contrast against something no v1 reader has used. They now state what datafast is,
  what it produces, and why the pipeline shape earns itself, and both share the
  quickstart's pipeline, which the test suite executes as written.

- **The docs site builds with Zensical instead of MkDocs + Material.** Material for
  MkDocs goes end-of-life on 2026-11-05 and has been in maintenance since November
  2025. Zensical is its successor from the same maintainers, reads `mkdocs.yml`
  natively and renders the same theme, so no page and no configuration changed.
  Verified by building both and diffing the output: the generated API page carries
  the same 152 symbols and 46 parameter tables, with identical code highlighting,
  table-of-contents permalinks and navigation. Done before writing the v1 pages so
  nothing would be written twice.

- **`docs/api.md` is generated from docstrings.** The page was a hand-maintained
  bullet list that had drifted to 34 of the 48 exported names, and it published only
  names — never parameters. It is now `:::` directives rendered by mkdocstrings
  (added to the `docs` extra), so the reference cannot fall behind the code. Building
  the docs now requires `pip install "datafast[docs]"`; `zensical build --strict` is
  clean.

- **Breaking: renamed the provider layer to the served-model vocabulary.** A *provider*
  is now strictly the server, a *model* is the LLM it serves, and a **served model** is
  the two together plus its configuration — the object you construct and call.
  - `LLMProvider` → `ServedModel`, `TargetConfig` → `ServedModelConfig`,
    `TargetCapabilities` → `ServedModelCapabilities`.
  - The seven per-provider subclasses (`OpenAIProvider`, `AnthropicProvider`, …) are now
    private. The lowercase factories — `openai`, `anthropic`, `gemini`, `mistral`,
    `openrouter`, `ollama`, `openai_compatible` — are the only public entry points and
    are unchanged. `ServedModel` is exported for type annotations.
  - Fields: `provider` → `provider_id`, `litellm_provider` → `litellm_route`,
    `provider_name` → `provider_id`.
  - `datafast/llm/provider.py` → `datafast/llm/served_model.py`; the `datafast.llms`
    compatibility module is removed — import from `datafast` or `datafast.llm`.
  - Langfuse trace metadata: the `datafast_provider` key is now `datafast_provider_id`,
    matching its sibling `datafast_model_id`. The tracing component for served-model
    calls is now `served_model.generate`.
  - `LLMStep(model=...)` is unchanged.

### Removed

- **Breaking: six unused runtime dependencies.** `instructor`,
  `google-generativeai`, `anthropic`, `openai`, `gradio` and `botocore` were declared
  in `pyproject.toml` and imported nowhere in the package. LiteLLM is the only LLM
  path — it reaches Anthropic and Gemini over its own HTTP transport, and declares
  `openai` as its own dependency — so no provider SDK belongs here. Runtime
  dependencies are now exactly the five imported at module scope: `litellm`, `loguru`,
  `pydantic`, `httpx`, `python-dotenv`. Together with the `datasets` move below, a
  base install resolves to 50 packages instead of 107; `datafast[all]` takes 59.
- **Breaking: `datasets` is no longer a base dependency.** It backs only `HubSink`
  and `HuggingFaceSource`, both behind lazy imports, and it pulled pyarrow, pandas and
  the Hub stack into every install. It now lives in the `hub` extra.

- **Breaking: `RunConfig.show_progress` and `RunConfig.log_level`.** Both were
  declared and read nowhere, so they looked like working knobs while doing nothing.
  For logging, use `configure_logger(level=...)`, exported from `datafast` — loguru's
  configuration is global, so a per-run field was the wrong shape for it. Progress is
  reported per step through the logger at `INFO`. Callers passing either to `run()`,
  `run_pipeline()` or `RunConfig(...)` now get a `TypeError`; drop the argument.

### Fixed

- **Broken references to a deleted design document.** `README.md` and
  `SOFTWARE_DESCRIPTION.md` both pointed at `datafast_new_design_document.md`, which
  no longer exists. They now point at the published docs site and `docs/concepts.md`.
- **`[project.urls] Documentation` pointed at the GitHub repo**, not the
  documentation site it names. It is now
  <https://patrickfleith.github.io/datafast/>.
- **`docs/PUBLISHING.md` documented the wrong PyPI credentials.** It asked for
  `PYPI_USERNAME` and `PYPI_PASSWORD`; the workflow uploads as `__token__` with
  `PYPI_API_TOKEN`, so following the guide could not have worked.
- **The docs workflow hand-listed its dependencies**, so CI and a local build could
  drift. It now installs the `docs` extra and builds with `--strict`.
