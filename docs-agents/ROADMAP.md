# Roadmap

## Shipped

### Served models

- Capability-aware served-model layer: capabilities resolved per provider, endpoint and model, with a `fail`/`warn`/`quiet` unsupported-parameter policy.
- Seven provider factories: `openai`, `anthropic`, `gemini`, `mistral`, `openrouter`, `ollama` and `openai_compatible`.
- Chat and Responses endpoint modes, Pydantic structured output, and first-class reasoning across anthropic, gemini, mistral and ollama.
- Multimodal input normalization for text, image, video and file/document content parts.
- Native batching with warned fallback concurrency, plus retries, backoff, jitter, timeout and client-side RPM throttling.
- Per-served-model reasoning controls: `thinking=True`/`False` resolve to each model's own values instead of a hardcoded effort and a silent no-op.
- `claude-sonnet-5` support via a second Anthropic profile, since it reasons unless disabled through Anthropic's native switch.
- Ollama capability probe reading the daemon's `/api/show`, plus the factory default moved from `gemma3:4b` to `gemma4:12b`.
- `provider_id` always names a server, never a wire format (DEC-003); `openai_compatible()` requires it and rejects wire-format values.
- `top_p`, `frequency_penalty` and `reasoning_summary` as real config fields, gated by capability rather than slipping through unchecked.
- Served-model vocabulary rename across code, docs and tests, leaving the lowercase factories as the only public entry points.

### Pipeline

- Runner reliability: a failed batch is retried per call, resume writes each record once, and `on_parse_error="raise"` stops the run.
- Pipeline pre-flight validation: `compile()` raises an actionable `PipelineValidationError` before execution, for structure and column references.
- `compile()` recurses into branch paths, `Concat` sources and `Join` right sides, naming the location in every error.
- Branch runner integration: the runner recurses into branch paths, so nested LLM steps get batching, ordering and per-call resume.
- A pipeline may end in several sinks (DEC-005). The rule became "nothing may follow the sinks" and needed no runner change.
- Execution controls `limit` and `resume_from`; runner-level throughput settings removed, because throughput belongs on the served model.
- Dead `RunConfig.show_progress` and `log_level` removed; the field set is now pinned by a test.

### Packaging and CI

- Dependency surface trimmed to the five packages actually imported. A base install resolves to 50 packages against the old 107.
- Release metadata set for v1: version `1.0.0`, Production/Stable classifier, and the documentation URL pointing at the docs site.
- `py.typed` shipped and verified inside a built wheel, so type checkers read the annotations the package already carried.
- CI runs the test suite on every pull request and gates the PyPI release on it.
- Docs CI installs the `docs` extra and builds with `--strict`, so CI and a local build cannot drift apart.

### Testing

- Mocked contract, capability, adapter and reliability tests covering retries, backoff growth, jitter, timeouts and batch-retry ordering.
- Live provider suites under `tests/live/`, one per provider, gated behind `--run-live` and self-skipping without a key or daemon.
- Example suites of 11 scripts each for all six hosted and local providers.

### Documentation (v1 launch)

- Docs site migrated to Zensical (DEC-006) before the v1 pages were written, so nothing had to be written twice.
- `docs/api.md` generated from docstrings, replacing a hand-maintained list that had drifted to 34 of 48 names.
- Quickstart page whose own code block is executed by the test suite against a stub served model.
- `docs/index.md` and `README.md` rewritten to open on what datafast is rather than on what pre-1.0 removed.
- Installation and environment reference, guarded by a test that scans the package for environment lookups.
- Glossary published and Concepts rewritten, pinned in both directions so no term is unpublished and none invented.
- Reference templates for the step and provider families, establishing the page shape and the code-to-docs test contract.
- Twelve reference pages written in parallel against those templates, taking the suite from 375 to 848 passing tests.
- Five guides: pipelines and execution, structured output, multimodal input, calling a served model directly, and troubleshooting.
- Three cookbooks deepened into end-to-end walkthroughs, a preference-with-scoring recipe added, and an index mapping all 45 example scripts.
- Contributing guide covering the layout, both test layers and the live gate, with every command it gives executed by its test.
- "What's in v1" release notes: what shipped, what is deliberately absent, and the rough edges, with no migration guide.
- Nav restructured around Home, Get started, Guides, Reference, Cookbook and Contributing; nine finished pages had been unreachable.
- Dark mode: a two-entry palette following the reader's OS setting, with a toggle that overrides it.
- Dangling references to a deleted design document removed, along with two wrong URLs the same audit turned up.
- `SOFTWARE_DESCRIPTION.md` retired after checking every section it carried is covered by the docs site.

## In progress

Nothing in flight — the next item is picked from Next up.

## Next up

- **Generate `docs-agents/SUM.md`** with the `write-manual` skill, which only Patrick can invoke.
- **Repository hygiene from `CONCERNS.md`.** Five small items, each costing a new contributor time.
  - `uv.lock` is stale and installs a different package than `pyproject.toml` describes.
  - The `[tool.pytest.ini_options]` block is dead; `pytest.ini` wins and pytest warns every run.
  - `ruff` is configured but unenforced, and the tree does not pass it.
  - The live gate matches parametrize ids, so an unmarked test named "live" is skipped.
  - `test_qa_pipeline.py` sits in the repository root and is not a test.

## Later / long term

- **Nested `Branch` inside a branch path.** Rejected by `compile()` because one `_branch_id` per record means an inner branch overwrites the outer tag.
  Supporting it needs a tag stack, touching tagging, the join, checkpoint names and the fingerprint at once.
  Deferred 2026-08-19: nothing in the docs, cookbook or examples needs it, and the rejection fails loudly.
- **vLLM support.** Delta live tests + example suite (needs a running server).
- **llama.cpp support.** Delta live tests + example suite (needs a running server).
- **openai-compatible generic backend.** Tests + example for the generic self-hosted path.
  All three share a blocker: each needs a server running plus model ids local to whoever
  runs the suite. `tests/live/ollama/` is the template — the only suite with no API key
  to guard on. vllm and llamacpp go through `openai_compatible`, i.e. the OpenAI wire
  format, so unlike ollama they need no route-specific parameter translation.
- **Caching.** Provider-native prompt caching, router/gateway caching, local prefix/KV reuse, optional client-side result cache; capability-aware cache keys. Only `cache_mode` metadata exists today.
- **Capability-driven live test catalogue.** One shared live suite parametrized over a curated catalog, so adding a model is a single entry.
  Demoted: the ad-hoc `integration` tests it would replace are already gone, and a parametrized sweep would flatten what each provider suite says is peculiar to it.
- Video input live coverage; `previous_response_id` continuation live scenario (E07); full-catalog live sweep (E08).

### Provider feature expansion

- **Audio input support.** Enable audio content parts end-to-end where `Modality.AUDIO` is declared; mocked gating test, one live test, and an example script.
- **File / document input support.** Implemented and covered for chat and Responses shapes, with live Anthropic and OpenAI tests. What remains:
  - Mocked gate-rejection tests for `OPENROUTER_CHAT` and `OLLAMA_CHAT`, which declare no `Modality.FILE`.
  - An example script for a document-capable provider.
  - A decision on `Modality.DOCUMENT`, which no served model declares.
- **Image output support.** Request-side selection and response normalization for image-generation models, exposed on `NormalizedResponse.images`; mocked test, live test, example.

## Improvements & tech debt

Non-feature work: rework, refactor, performance, cleanup.

- ~~Migrate per-provider `integration` tests onto the `live` marker~~ — done. All six legacy files and their shared fixture module are deleted, coverage ported first.
- ~~Dead `supports_thinking` capability flag~~ — done. It was set but never read; a flag that looks load-bearing while doing nothing is a trap.
- ~~Unused markers~~ — done. `integration`, `slow`, `vllm` and `llamacpp` are no longer declared; a future suite re-adds its own.
