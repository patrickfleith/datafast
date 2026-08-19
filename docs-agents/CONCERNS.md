# Concerns

Risks and defects to investigate. Found while writing the v1 reference pages
(2026-08-18): twelve pages were written by reading the source and proving each claim
with a test, which surfaced behaviour the docstrings describe wrongly or not at all.
The pages document what the code does today. What has since been fixed is listed at the
bottom rather than deleted.

## Defects

- **`LLMStep(temperature=..., max_tokens=...)` does nothing.** Both are stored at
  `datafast/transforms/llm_step.py:123-124` and never read; `ServedModel.generate()`
  takes no such arguments. The docstring says "Override model temperature for this
  step", which is false and renders on the generated API page. Either wire them up or
  remove them and the docstring. The ROADMAP repeated this claim and has been corrected.

- **`on_parse_error="raise"` is ignored under `Pipeline.run()`.** `LLMStep.process()`
  honours it, but the runner catches the exception in `_apply_llm_batch_results`, logs
  it and continues, so a run finishes with fewer records instead of raising. The default
  `"skip"` also swallows *every* exception, not only parse failures — a provider timeout
  silently drops a record.

- **A mistyped prompt file path becomes the prompt.** `_load_prompt_if_file` returns
  `str(prompt)` when the path is not a file, so `prompt=Path("prompts/typo.txt")` sends
  the literal string `prompts/typo.txt` to the model. No error, real spend.

- **`Filter`'s `$or` / `$and` ignore sibling keys.** They return immediately, so
  `{"$or": [...], "score": {"$gt": 100}}` silently drops the `score` condition and keeps
  a record scoring 0.

- **A branch path that rewrites an existing column loses the change.** `JoinBranches`
  copies pre-branch columns unsuffixed from the first path that produced a record and
  suffixes only *new* columns, so if two paths both rewrite `text`, the second is
  discarded silently.

- **`JoinBranches(how="outer")` does not fill with `None`** as its docstring promises —
  `_merge_group` skips a `None` record, so the missing path contributes no keys at all.

- **One failing LLM call abandons its whole batch group.** `_collect_llm_batch_results`
  wraps a whole per-model group in one `try`, and `_generate_llm_group` loops `generate`
  inside it, so the first exception aborts the rest of the group. Every record in it is
  logged as `LLM call failed` with that same error and dropped — including the calls
  never attempted. Measured on eight records with one failure on call 2:
  `batch_size=1` keeps 7, `batch_size=4` (the default) keeps 4, `batch_size=8` keeps 0.
  One transient 429 can cost a whole batch, and the log makes it look like every record
  failed. Catching per call inside the loop would confine the loss to the one that failed.

- **Resume duplicates the records completed since the last progress save.** Output records
  are appended one at a time by `append_record`, but the `completed_call_ids` list is only
  rewritten every `checkpoint_every` calls. Anything finished after the last save is in the
  step's JSONL without being marked done, so resume re-runs those calls and appends the
  records a second time. Reproduced: 8 source records, crash on call 6 with
  `checkpoint_every=2` → 9 output records, one input duplicated. The two files need to be
  written together, or the ids need appending as the records are.

## Risks

- **Resume can attach LLM results to the wrong records.** `_step_signature` renders a
  branch path by its step *class* names, so changing the lambda inside a `Map` inside a
  path leaves the checkpoint fingerprint unchanged and nothing warns. Call ids are
  positional, so a non-deterministic path step silently misaligns completed results on
  resume. This is the mechanism behind the determinism rule, and it currently depends
  entirely on the user obeying it.

- **Three providers have no model-name fallback.** OpenAI, Mistral and Ollama match
  uncatalogued models by name; Anthropic, Gemini and OpenRouter fall straight through to
  one provider default. For a reasoning model that default's "off" is either omission or
  nothing at all — exactly the silent reasoning-and-billing bug already fixed twice
  (gemini-3, sonnet-5). Any future reasoning model on those three reintroduces it until
  someone adds a catalog entry.

- **`Pair(strategy="random")` with no `max_pairs` yields 100,000 tuples per group.**
  Three input records produced exactly 100,000 outputs, duplicates allowed. The next LLM
  step pays for all of them.

- **Ragged records fail asymmetrically.** A later record with an extra column makes
  `CSVSink` raise and `ParquetSink` silently drop the column. Same input, one loud
  failure and one quiet data loss.

- **`Score` turns every bad answer into a confident number.** `_parse_llm_result` clamps
  to the range and falls back to its bottom, so on `score_range=(1, 10)`: `99` is stored as
  `10`, `"high"` as `1`, and a reply with no `score` key at all as `1`. Only a reply that is
  not JSON is treated as a failure. A model that ignored the instruction lands in the
  dataset as a perfect or a worst score, which then drives any downstream filter — cookbook
  42 filters on the margin between two of them. The clamp also mixes types in one column:
  an in-range answer is stored as a float (`7.0`) and a clamped one as the range bound
  itself (`10`, an int), because `max`/`min` return whichever operand won.

- **No step ever uses provider-side structured output.** `response_format` appears
  nowhere in `datafast/transforms/`: `LLMStep` appends JSON instructions to the prompt
  and parses the reply, and the specialized steps `json.loads` a reply they asked for in
  prose. Every shipped provider except an unprofiled `openai_compatible()` server
  declares `json_schema`, so a pipeline asks a model to please return JSON when the
  provider could have guaranteed it. Combined with `on_parse_error="skip"` being the
  default, a model that strays costs a dropped record and a wasted call. Wiring
  `output_columns` into a generated Pydantic model would close it — `JSONParser` already
  builds one in `_create_response_model`, and nothing calls that method.

- **A typo in a content part's `type` is silently forwarded as text.**
  `_normalize_content_part` returns an unrecognised part untouched and
  `_modality_for_part` reports `Modality.TEXT` for it, so `type="imgae"` passes the
  modality gate on a text-only served model and reaches the provider unnormalized. Every
  other malformed part raises locally with a useful message; this one comes back as an
  opaque provider error. A closed set of part types would be a one-line fix.

- **The default OpenAI served model warns on every batch.** `openai()` resolves to a
  reasoning model on the Responses endpoint, where `batch_mode` is
  `FALLBACK_CONCURRENCY`, so any multi-input call raises the "does not expose native
  batching" `UserWarning`. It is accurate, but it fires on the out-of-the-box
  configuration doing something ordinary, which trains users to ignore the warning that
  matters — the `prompted_json` one on the same channel. Worth demoting to a log line, or
  emitting once per served model.

- **A crash before the first progress save recovers nothing.** `checkpoint_every`
  defaults to 100, so an LLM step that dies at call 60 has no progress file at all and
  resume re-runs every call. Measured: crash on call 6 of 8 with the default → 8 calls
  paid again. The default is tuned for cheap steps; the expensive ones are exactly where
  it costs most. A time-based save, or a smaller default, would fit the failure it exists
  for.

- **`SeedDimension` is public, constructible and undocumented.** It is in
  `datafast.__all__`, and cookbook script 45 builds one directly to keep `label` and
  `label_description` in a single dimension. `sources_and_seed.md` names it only as a type
  in two parameter tables, so the reference page has no way to build a dimension of more
  than two columns — `Seed.expand` is parent/child only. Either document the constructor or
  give `Seed` a factory for the n-column case.

- **A partial JSON reply is a silent success.** `JSONParser.parse` fills any
  `output_columns` entry the reply omits with `""`, logs a warning and returns normally, so
  it is not a parse error and `on_parse_error` never sees it. A row whose whole value is
  the generated text (cookbook 44's `text`) lands in the dataset empty, having cost a call
  and looking like a row. Measured: a reply of `{"title": "T"}` against
  `output_columns=["title", "text"]` yields one record with `text == ""`, even with
  `on_parse_error="raise"`. Either count a missing column as a parse failure or let the
  step declare which columns are required.

- **`num_outputs > 1` leaves no marker on the extra records.** `_build_output_record`
  stamps `_model`, `_language` and `_prompt_index`, but nothing for the output index, so
  *n* records generated from one prompt/model/language combination are distinguishable
  only by their generated text. Nothing can group siblings, and a deduplication step
  cannot tell an intended repeat from an accidental one. `_prompt_index` shows the shape
  the fix would take.

- **`fn` mode is inconsistent.** In `Classify`, `Score` and `Compare` it ignores
  `forward_columns` / `exclude_columns` and adds no `_model`; in `Extract` those
  arguments work. Undiscoverable from the docstrings.

Found while writing `docs/contributing.md` (2026-08-19) — these are about the repository
rather than the library, and every one of them costs a new contributor time.

- **`uv.lock` is stale and installs a different package.** It pins `datafast 0.0.35`
  with `anthropic`, `openai`, `google-generativeai`, `instructor`, `gradio` and
  `botocore` — the six dependencies retired from `pyproject.toml` — and a `docs` extra
  of `mkdocs` + `mkdocs-material` rather than `zensical`. `uv sync` therefore
  contradicts `pyproject.toml` in both directions. Regenerate it or delete it; the
  contributing page currently has to warn people off it.

- **`[tool.pytest.ini_options]` in `pyproject.toml` is dead.** `pytest.ini` exists, and
  it wins, so pytest prints `configfile: pytest.ini (WARNING: ignoring pytest config in
  pyproject.toml!)` on every single run. The ignored block sets `addopts = "-ra -q"`,
  which nobody is getting. Delete the block or merge it into `pytest.ini`.

- **`ruff` is configured but unenforced, and the tree does not pass.** `ruff check .`
  reports 62 findings (35 `W293`, 10 `C901`, 8 `F401`), and `ruff format --check` would
  reformat 110 of 221 files. Either fix the tree and gate it in CI, or drop the tool
  from the `dev` extra — as it stands the config implies a standard nothing upholds.

- **The live gate matches on parametrize ids, not just markers.**
  `pytest_collection_modifyitems` tests `"live" in item.keywords`, and keywords include
  parametrize ids, so an unmarked mocked test parametrized with the string `"live"` is
  silently skipped. Measured: a two-case parametrization over `["live", "local"]`
  reports `1 passed, 1 skipped`. Checking `item.get_closest_marker("live")` instead
  would be exact.

- **`theme.palette.primary: black` has no effect.** Zensical ships a `classic` and a
  `modern` build of the Material stylesheets and the site loads `modern`, which defines
  `--md-primary-fg-color` for nineteen colours — `black` and `white` are not among them,
  though `classic` defines both. So the header renders in the theme's default indigo
  (`#4051b5`), not black, and has done since before dark mode. Pick a colour `modern`
  supports (`grey`, `blue-grey`) or drop the line. Links are unaffected: the dark scheme
  keys its `--md-typeset-a-color` override on the *attribute*, which is still `black`.
  Pinned by `test_the_configured_primary_is_ignored_by_this_stylesheet`.

- **`test_qa_pipeline.py` sits in the repository root.** It is tracked, named like a
  test, and is not one — it is a scratch OpenRouter pipeline script. `testpaths = tests`
  keeps it out of collection, so `pytest` never sees it, but `pytest test_qa_pipeline.py`
  would. Move it to `examples/scripts/` or delete it.

## Fixed

Cleared on 2026-08-18, each with a test that fails if the behaviour comes back.

- `Sample(n=0, strategy="last")` returned every record.
- `ListSink.records` accumulated across runs.
- `HubSink` re-prepended its README front-matter on every push.
- `stop_after` ignored a name that matched no step.
- `PipelineValidationError` was not exported from the top-level package.
- `SeedDimension.values` was annotated with the builtin `any`.
- `Concat`'s docstring contradicted its code.
- Cookbook script 43's `take_first_100` step took ten records; renamed `take_first_10`.
- No CI workflow ran the test suite, so a broken merge was released (2026-08-19).
