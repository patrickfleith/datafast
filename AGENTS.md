# AGENTS.md

Instructions for AI coding agents (Claude Code, Codex, and any tool that reads
`AGENTS.md`) working in this repo. Claude loads it via `@AGENTS.md` in
`CLAUDE.md`; other tools read this file directly. It is loaded every session —
this is the whole mechanism, so keep it compact and current.

## How to work here

- **What this is:** `datafast` — a pipeline-first Python library for synthetic data generation with LLMs, for developers and ML engineers building datasets.
- Before acting, consult the relevant project doc below.
- Keep changes minimal and in the style of the surrounding code.
- **Tests:** `pytest` is not on PATH — run via `.venv/bin/pytest`. Integration/live tests hit real providers; deselect with `-m "not integration and not live"`.

## Project docs

Project docs live in `docs-agents/`. Consult the relevant one before acting. If a doc
doesn't exist yet, don't fabricate one — create it with the skill noted below.

| Doc          | Path                     | What it captures                        | Skill          |
|--------------|--------------------------|-----------------------------------------|----------------|
| PRD          | docs-agents/PRD.md       | Product intent, users, requirements.    | edit           |
| Roadmap      | docs-agents/ROADMAP.md   | Shipped / in progress / next / later.   | edit           |
| Changelog    | docs-agents/CHANGELOG.md | Released changes, per version.          | edit           |
| Manual (SUM) | docs-agents/SUM.md       | Install + usage guide for end users.    | `write-manual` |
| Decisions    | docs-agents/DECISIONS.md | Pending (TBD) and settled decisions.    | `decide`       |
| Glossary     | docs-agents/GLOSSARY.md  | Canonical term definitions.             | `glossary`     |
| Stack        | docs-agents/STACK.md     | Frameworks, libraries, tools.           | `log`          |
| Tasks        | docs-agents/TASKS.md     | Repo-level todo dump.                   | `log`          |
| Ideas        | docs-agents/IDEAS.md     | Captured ideas / possibilities.         | `log`          |
| Concerns     | docs-agents/CONCERNS.md  | Risks and concerns to investigate.      | `log`          |
| Questions    | docs-agents/QUESTIONS.md | Open questions + answers once resolved. | `log`          |

## Capabilities

Skills are auto-discovered from `.claude/skills/`. Currently available:

- `commit` — turn uncommitted work into atomic conventional commits.
- `log` — append an entry to STACK, TASKS, IDEAS, CONCERNS, or QUESTIONS.
- `decide` — record and manage decisions in DECISIONS.
- `glossary` — add and refine canonical terms in GLOSSARY.
- `write-manual` — write or update the SUM from the codebase.
