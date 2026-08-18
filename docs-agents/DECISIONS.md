# Decisions

## TBD

- **Shortname — <pending decision>** — options/context in one line.

## Decided


### DEC-006 — the docs site moves to Zensical

Decided by: Author with Agent
Date: 2026-08-18

**Decision:** The `docs` extra drops `mkdocs` and `mkdocs-material` for `zensical`,
and both the deploy workflow and local builds run `zensical build --strict`.
`mkdocs.yml` is unchanged — Zensical reads it natively. `mkdocstrings` stays and is
supported directly by Zensical, which errors by name if it is configured but absent.

**Rationale:** Material for MkDocs goes end-of-life on 2026-11-05, and Zensical is
the successor from the same maintainers. Doing it before the v1 pages are written
means none of them get written twice. Confirmed by trial build rather than by
reputation: the two builds produce the same 16 pages, and the generated API page
carries the identical 152 symbol anchors — zero difference in either direction — with
the same 46 parameter tables, 68 highlighted code blocks and 214 permalinks. Zensical
is at 0.0.55 and pre-1.0, which is the standing risk; the mitigation is that
`mkdocs.yml` stays the source of truth, so reverting is a one-line change to the
`docs` extra.


### DEC-005 — a pipeline may end in several sinks

Decided by: Author with Agent
Date: 2026-08-18

**Decision:** `compile()` accepts a trailing chain of sinks. The rule changed from
"a sink must be the last step" to "no step may follow a sink", so
`... >> Sink.jsonl(path) >> Sink.hub(repo)` is one run writing two destinations.

**Rationale:** Sinks already yield every record through, so the runner needed no
change and chaining worked mechanically — only the validation rule forbade it. The
alternative, splitting into two runs, would have meant re-running or hand-carrying
records to reach a second destination, which is what
`43_cookbook_persona_generation.py` was doing with an out-of-pipeline
`push_records_to_hub()` helper. Sinks stay last as a group because a sink is a side
effect: a step after one would write before the pipeline had finished shaping the
records. An opt-in push (as in `44_cookbook_space_text_generation.py`, gated on
`DATAFAST_PUSH_TO_HUB`) still belongs outside the pipeline, since a sink in the chain
runs unconditionally.


### DEC-004 — v1 ships as 1.0.0 / Production/Stable, published on merge

Decided by: Author with Agent
Date: 2026-08-18

**Decision:** `pyproject.toml` carries `version = "1.0.0"` and
`Development Status :: 5 - Production/Stable`, and `[project.urls] Documentation`
points at <https://patrickfleith.github.io/datafast/> rather than the repo. Merging
this branch to `main` therefore tags `v1.0.0` and publishes it to PyPI, while parts
of the v1 documentation (quickstart, installation reference, provider pages, step
guides) are still unwritten. Taken knowingly: the release goes out first and the
remaining pages follow.

**Rationale:** The version had to change regardless — this branch sat at `0.0.35`
while `main` had already moved to `0.0.36`, so merging as-is would have made the
publish workflow re-tag an existing `v0.0.35` and fail. Given a forced choice, the
author elected to cut 1.0.0 now rather than reconcile to `0.0.36` and defer.

**Consequences:** The publish workflow (`.github/workflows/publish.yml`) treats a
version differing from the latest tag as a deliberate bump and preserves it, so the
number in `pyproject.toml` at merge time is the number that reaches PyPI —
irreversibly, since PyPI never permits reusing a version. It also validates against
`^[0-9]+\.[0-9]+\.[0-9]+$`, so pre-release strings like `1.0.0rc1` abort the
workflow; release candidates are not available without changing it first. Post-1.0,
the breaking changes in this release (dropped `RunConfig` fields, trimmed
dependencies, `datasets` moved to an extra) set the expectation that further
removals wait for 2.0.


### DEC-003 — provider_id always names a server, never a wire format

Decided by: Author with Agent
Date: 2026-08-03 11:15

**Decision:** `provider_id` may only name the server that serves the model (`vllm`,
`llamacpp`, `ollama`, `openai`, …), never `openai_compatible`. `openai_compatible()`
now requires an explicit `provider_id` keyword and rejects wire-format values; its
`backend` parameter is gone. Unknown provider ids stay legal and fall back to the
conservative capability profile.

**Rationale:** `GLOSSARY.md` already defines Provider as "the server that serves LLMs
… not the same thing as the wire protocol used to reach it", and Transport as the
OpenAI-shaped wire format that "does not imply OpenAI is the provider". A
`provider_id` of `openai_compatible` conflated the two and left self-hosted servers
indistinguishable from each other in traces and capability resolution.


### DEC-002 — More decisions

Decided by: Claude Code
Date: 2026-10-14 12:56

**Decision**: We have decided to mae more deicions.

**Rationale**: More is better.


### DEC-001 — title to be replaced

Decided by: Author | Author with Claude Code | Claude Code
Date: YYYY-MM-DD HH:MM

**Decision:** 1–2 sentences (what was decided).

**Rationale:** 1–2 sentences (why).
