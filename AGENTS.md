# AGENTS.md

Instructions for AI coding agents working in this repo. Codex and Cursor read
this file directly, Codex cloud tasks included. Claude Code reads it through the
one-line `CLAUDE.md` adapter (`@AGENTS.md`). Keep it compact and current.

## How to work here

- **What this is:** `datafast` — a pipeline-first Python library for synthetic data generation with LLMs, for developers and ML engineers building datasets.
- Before acting, consult the relevant project doc below.
- Keep changes minimal and in the style of the surrounding code.
- **No live LLM calls unless asked:** never run code that hits a real provider API unless the user explicitly requests it. Default to `-m "not live"` and to examples/scripts that don't call out.
- **Tests:** `pytest` is not on PATH — run via `.venv/bin/pytest`. Tests under `tests/live/` hit real providers; they self-skip without `--run-live`, and deselect with `-m "not live"`.

## Writing rules

Everything you write follows `.agents/skills/unslop/SKILL.md`: answers in chat,
docs, commit messages, PR text. Read it before you write.

## Project docs

Project docs live in `docs-agents/`. Consult the relevant one before acting. If a doc
doesn't exist yet, don't fabricate one — create it with the skill noted below.

| Doc          | Path                     | What it captures                        | Skill          |
|--------------|--------------------------|-----------------------------------------|----------------|
| PRD          | docs-agents/PRD.md       | Product intent, users, requirements.    | edit           |
| Roadmap      | docs-agents/ROADMAP.md   | Shipped / in progress / next / later.   | edit           |
| Manual (SUM) | docs-agents/SUM.md       | Install + usage guide for end users.    | `write-manual` |
| Decisions    | docs-agents/DECISIONS.md | Pending (TBD) and settled decisions.    | `decide`       |
| Glossary     | docs-agents/GLOSSARY.md  | Canonical term definitions.             | `glossary`     |
| Stack        | docs-agents/STACK.md     | Frameworks, libraries, tools.           | `log`          |
| Tasks        | docs-agents/TASKS.md     | Repo-level todo dump.                   | `log`          |
| Ideas        | docs-agents/IDEAS.md     | Captured ideas / possibilities.         | `log`          |
| Concerns     | docs-agents/CONCERNS.md  | Risks and concerns to investigate.      | `log`          |
| Questions    | docs-agents/QUESTIONS.md | Open questions + answers once resolved. | `log`          |

## Capabilities

Skills live in `.agents/skills/`, committed with the repository so local and
cloud sessions use the same workflows. Codex and Cursor discover that directory
natively; `.claude/skills/` is a **generated** adapter for Claude Code, which
reads only its own path.

Never edit `.claude/skills/` by hand. Write the skill in `.agents/skills/` and
run `agents-control adapters`. Invoke a skill as `/name` in Claude Code, `$name`
in Codex, or by asking for it in Cursor. Currently available:

- `commit`: turn uncommitted work into atomic conventional commits.
- `log`: append an entry to STACK, TASKS, IDEAS, CONCERNS, or QUESTIONS.
- `decide`: record and manage decisions in DECISIONS.
- `glossary`: add and refine canonical terms in GLOSSARY.
- `write-manual`: write or update the SUM from the codebase.
- `unslop`: writing rules for anything an agent writes. Always applies.
