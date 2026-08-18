# Concerns

Risks and defects to investigate. Found while writing the v1 reference pages
(2026-08-18): twelve pages were written by reading the source and proving each claim
with a test, which surfaced behaviour the docstrings describe wrongly or not at all.
None of these were fixed — the pages document what the code does today.

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

- **`HubSink` stacks duplicate README front-matter on every push.** `_ensure_readme`
  skips only `if "datafast-dataset" in content`, but the template it writes tags the
  dataset `datafast`. The guard can never match what it wrote, so the block is
  re-prepended each push. The commit message says "Add datafast-dataset tag" while the
  tag is `datafast`.

- **`Filter`'s `$or` / `$and` ignore sibling keys.** They return immediately, so
  `{"$or": [...], "score": {"$gt": 100}}` silently drops the `score` condition and keeps
  a record scoring 0.

- **A branch path that rewrites an existing column loses the change.** `JoinBranches`
  copies pre-branch columns unsuffixed from the first path that produced a record and
  suffixes only *new* columns, so if two paths both rewrite `text`, the second is
  discarded silently.

- **`JoinBranches(how="outer")` does not fill with `None`** as its docstring promises —
  `_merge_group` skips a `None` record, so the missing path contributes no keys at all.

- **`Sample(n=0, strategy="last")` returns every record** (`items[-0:]`), while
  `strategy="first"` correctly returns none.

- **`ListSink.records` is never cleared**, so re-running one pipeline object accumulates
  both runs' records.

- **`Concat`'s docstring contradicts its code.** It claims upstream records are yielded
  first; `process` never reads its input and discards them.

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

- **`Score` clamps instead of rejecting.** A model answering `99` on a `(1, 5)` scale is
  stored as `5` — an out-of-range answer becomes a perfect score.

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

- **`stop_after` silently ignores an unknown step name.** The runner compares it to each
  index and name as it goes and just never matches, so `stop_after="typo"` runs the whole
  pipeline. `resume_from` validates its name and raises with the real list; these two
  arguments take the same kind of value and disagree about what a wrong one means.

- **`PipelineValidationError` is not exported from the top-level package.** `datafast`
  exports `PipelineChangedError` but not the exception `compile()` raises, so catching the
  more common of the two means importing from `datafast.core.validation` — a private-looking
  path for the error users will hit first.

- **`SeedDimension` is public, constructible and undocumented.** It is in
  `datafast.__all__`, and cookbook script 45 builds one directly to keep `label` and
  `label_description` in a single dimension. `sources_and_seed.md` names it only as a type
  in two parameter tables, so the reference page has no way to build a dimension of more
  than two columns — `Seed.expand` is parent/child only. Either document the constructor or
  give `Seed` a factory for the n-column case.

- **`SeedDimension.values` is annotated `list[dict[str, any]]`.** That is the builtin
  `any` function, not `typing.Any`. Harmless today because dataclasses do not evaluate
  annotations, but it is wrong and a type checker rejects it — which starts mattering the
  moment the pending `py.typed` task ships and users' checkers read this file.

- **Cookbook script 43's sampling step is named for a count it does not take.**
  `Sample(n=10, strategy="first").as_step("take_first_100")` — the name is left from a
  larger default. It is not cosmetic: step names become checkpoint file names and the
  `resume_from` argument, so the artefact on disk is `step_003_take_first_100.jsonl`
  holding ten records. The published page had copied the name's claim and documented 100
  rows; it now documents ten. Rename the step or restore `n=100`.

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
