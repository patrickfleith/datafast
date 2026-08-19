# Changelog

## [Unreleased]

## [1.0.0] — 2026-08-19

First stable release.

### Added

- CI runs `pytest -m "not live"` on every pull request and before publishing, so a failing suite blocks the release.
- `py.typed` ships in the wheel, so type checkers read the annotations the package already carried.
- Reference pages for sources, seeds and served models, each pinned by a test that fails when the code grows an undocumented parameter.
- A published glossary of 29 terms, and a Concepts page rebuilt on record → step → pipeline → runner.
- An installation reference covering every extra and environment variable, guarded by a test that scans the package for lookups.
- A quickstart taking a reader from `pip install` to a stored dataset, its code block executed by the test suite.
- "What's in v1" release notes: what datafast does, what shipped, what is deliberately absent, and the rough edges.
- A pipeline may end in several sinks. The rule changed from "a sink must be last" to "nothing may follow the sinks".
- Docstrings for the six provider factories, which had none at all and rendered as bare signatures on the API page.
- The full `Filter` operator reference: the docstring named 6 of 23, and the other 17 were tested nowhere.
- Feature extras `datafast[parquet]`, `datafast[hub]` and `datafast[all]` for optional file formats and Hub I/O.
- `top_p` and `frequency_penalty` as real config fields, gated by capability like `temperature` rather than slipping through unchecked.
- `repeat_penalty` documented as Ollama's repetition control on its own scale; the Ollama profiles no longer declare `frequency_penalty`.
- `reasoning_summary` on served models, so OpenAI's summary is reachable without hand-writing the provider's request shape.

### Changed

- The docs nav carries every page, grouped as Home, Get started, Guides, Reference, Cookbook and Contributing.
- The README and docs home page open on what datafast is, not on the removal of a pre-1.0 API no v1 reader used.
- The docs site builds with Zensical instead of MkDocs + Material, which goes end-of-life on 2026-11-05.
- `docs/api.md` is generated from docstrings, replacing a hand-maintained list that had drifted to 34 of 48 names.
- **Breaking:** the provider layer renamed to served-model vocabulary; the seven subclasses are private, leaving the lowercase factories public.

### Removed

- `SOFTWARE_DESCRIPTION.md` and three superseded guides — every section they carried is covered by the docs site.
- **Breaking:** six unused runtime dependencies. A base install now resolves to 50 packages instead of 107.
- **Breaking:** `datasets` moved to the `hub` extra, so it no longer pulls pyarrow, pandas and the Hub stack into every install.
- **Breaking:** `RunConfig.show_progress` and `log_level`, both declared and read nowhere. Use `configure_logger(level=...)` instead.

### Fixed

- `Sample(n=0, strategy="last")` returned every record, because `items[-0:]` is the whole list.
- `ListSink.records` accumulated across runs; a run now replaces what the sink holds.
- `HubSink` re-prepended its README front-matter on every push, looking for a tag its own template never wrote.
- `stop_after` silently ignored a name matching no step. It now raises `ValueError` before the first step, as `resume_from` does.
- `PipelineValidationError` was missing from the top-level package, so catching it meant importing a private path.
- `SeedDimension.values` was annotated with the builtin `any` rather than `typing.Any`.
- `Concat`'s docstring claimed upstream records are yielded after its sources; they are discarded.
- Broken references to a deleted design document in `README.md` and `SOFTWARE_DESCRIPTION.md`.
- `[project.urls] Documentation` pointed at the GitHub repo rather than the documentation site it names.
- `docs/PUBLISHING.md` asked for PyPI credentials the workflow does not use, so following it could not have worked.
- The docs workflow hand-listed its dependencies; it now installs the `docs` extra and builds with `--strict`.
