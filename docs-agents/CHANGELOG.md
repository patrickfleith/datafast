# Changelog

## [Unreleased]

## [1.0.0] — 2026-08-18

First stable release.

### Added

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

- **`docs/api.md` is generated from docstrings.** The page was a hand-maintained
  bullet list that had drifted to 34 of the 48 exported names, and it published only
  names — never parameters. It is now `:::` directives rendered by mkdocstrings
  (added to the `docs` extra), so the reference cannot fall behind the code. Building
  the docs now requires `pip install "datafast[docs]"`; `mkdocs build --strict` is
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
