# Decisions

## TBD

- **Shortname — <pending decision>** — options/context in one line.

## Decided


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
