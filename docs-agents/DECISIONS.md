# Decisions

## TBD

- **Shortname — <pending decision>** — options/context in one line.

## Decided


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
