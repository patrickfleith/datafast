# Tasks

<!-- Newest to do on top. Check off in place; move to Done when complete. -->

## To do

- [ ] <task> <!-- optional (context) -->

## Documentation improvements

- [ ] Split `docs/reference/sources_and_seed.md` into separate Source and Seed reference pages <!-- two nav entries; `tests/test_reference_sources_and_seed.py` splits with it -->
- [ ] Reorganize 'Key Concepts' pedagogically: Record → Step → Source → Seed (its own section) → served model → LLM step → Sample → Sink → Data operations <!-- Data operations reuses the `docs/reference/data_ops.md` 'At a glance' table, but each row gets a concrete, evocative example -->
- [ ] Retitle `docs/guides/index.md` from 'Guides' to 'How To Guides' — the page's `# ` heading only, leaving the nav item as 'Guides'
- [ ] Polish 'What's in v1'
- [ ] Move the Glossary out of 'Get started' — retitle it 'Full Glossary' and put it under Reference <!-- it is lookup material, not an onboarding step; reached via search and in-page links -->
- [ ] Polish the Quickstart
- [ ] Rename 'Concepts' to 'Key Concepts' and widen it to cover every key concept, absorbing what the glossary defines — pitched higher: not just what a term means, but how it works and where it fits in datafast <!-- glossary stays the precise lookup, incl. its Avoid lines -->
- [ ] Add an example datasets showcase to the docs — a grid graphic (e.g. 2x4 of labelled squares, one dataset type per square) linking each type to its cookbook or example script

## Conventions for documentation pages

Kept from the v1 launch, because every future page follows them.

- Copy `docs/reference/sources_and_seed.md` for a page and `tests/test_reference_sources_and_seed.py` for its test.
- Plain short sentences and simple words. Use the glossary's terms and never one it lists under *Avoid*.
- Assert code → docs first: every parameter `inspect.signature` reports must appear on the page in backticks.
- Then execute every self-contained example, prove each behavioural claim with a real call, and resolve every relative link.
- No test may pass vacuously — assert the introspection found something before asserting what it found.
- Never make a live LLM call. Stub the provider factory on the `datafast` module itself, not inside an `exec` namespace.
- Ground every claim in the source. Where the code disagrees with its docstring, document the code and log the mismatch in `CONCERNS.md`.

## Done

- [X] Retire `SOFTWARE_DESCRIPTION.md` — deleted after verifying every section is covered elsewhere
- [X] Ship `py.typed` — plus the package-data entry, verified inside a built wheel, with 74 tests guarding the annotations it exposes
- [X] Nav restructure — nine finished pages had been unreachable; three superseded guides deleted and their 31 inbound links rewritten
- [X] "What's in v1" release notes — 146 lines, 23 tests; the rough edges are user-facing, with no code detail
- [X] Contributing & development guide — 286 lines, 109 tests; every command on the page is executed, and six repository concerns were logged
- [X] Dark mode for the docs site — a two-entry palette with a toggle, 19 tests, and no CSS written
- [X] Examples index — 137 lines, 57 tests; every script has a row, and a test pins which scripts need no API key
- [X] New cookbook: preference data with scoring — 322 lines, 35 tests; documents two measured traps in the recipe that hits them
- [X] Deepen the space text generation cookbook — 378 lines, 40 tests; tables the opt-in Hub push trade-off rather than hiding it
- [X] Deepen the persona generation cookbook — 388 lines, 42 tests; the old page had drifted from its script in three places
- [X] Deepen the text classification cookbook — 349 lines, 39 tests, every number measured against a real stubbed run
- [X] Error handling & troubleshooting guide — 320 lines, 58 tests; measured the batch-group loss and resume duplication now in CONCERNS
- [X] Calling a served model directly — 211 lines, 45 tests; measured the endpoint asymmetry, the error contract and what retries cover
- [X] Multimodal input guide — 221 lines, 64 tests; found that an unknown content-part type is silently treated as text
- [X] Structured output guide — 226 lines, 41 tests; found that no step passes `response_format`, so provider-side schemas need a direct call
- [X] Pipeline & execution guide — 322 lines, 51 tests; measured that the checkpoint fingerprint covers only step names and classes
- [X] Fix the persona cookbook's chained sinks — resolved as DEC-005: a pipeline may end in several sinks
- [X] Review the documentation and identify missing blocks before v1 — became the roadmap's documentation section and found six non-doc blockers
- [X] Manually test all Gemini example scripts
- [X] Retire the unused pytest markers — dropped `integration`, `slow`, `vllm` and `llamacpp`; the rest are carried by real tests
- [X] Retire the last legacy `integration` suite — openrouter's live suite already covered everything except model-quality cases
- [X] Give OpenRouter a live suite — 13 tests pinning one endpoint per request, since one model id spans 19 that disagree
- [X] Retire the anthropic, openai and mistral legacy `integration` suites — 1200 lines deleted, 8 live tests ported first
- [X] Retire gemini's half of the legacy `integration` layer — ported three shapes the live suite lacked, dropped the model-quality cases
- [X] Give Gemini a live suite — 14 tests across two models, because the reasoning floor is per-model
- [X] Give `top_p` and `frequency_penalty` real config fields — both were declared by profiles but slipped through unchecked
- [X] Decide what `frequency_penalty` means on a local backend — only ollama is affected; it takes `repeat_penalty` on its own scale
- [X] Expose reasoning summaries — a real `reasoning_summary` config field rather than documenting a `provider_params` workaround
- [X] Add a live pipeline smoke test — source to sink over four providers, each self-skipping on its existing guard
- [X] Give Ollama a live suite, a capability probe and a current default — the probe reads the daemon, since the pulled model is machine-local
- [X] Retire ollama's half of the legacy `integration` layer — ported sampling params, batched message lists and nested schema
- [X] Make `Modality.FILE` state which carrier a served model accepts, and give Mistral an explicit file upload path
- [X] Narrow the roadmap's file/document input item to what is actually left
- [X] Make file content parts symmetric with image parts
- [X] Make sure the roadmap clearly outlines the release of the new version of datafast
- [X] Make a roadmap
- [X] Try out the OpenAI, Mistral, Anthropic and Ollama example scripts
