# Persona Cookbook Assets

This note records the supporting assets used by the persona-generation cookbook.

## Dataset Selection

- Dataset: `xsum`
- Split: `validation`
- Text field: `document`
- Summary field kept for inspection: `summary`
- Selection rule: keep documents whose whitespace-tokenized word counts are between `300` and `500`
- Cap: use the first `20` matching records

This keeps the cookbook deterministic and bounded while still using a well-known Hugging Face corpus with article lengths that fit the demonstration.

`GEM/xsum` was the original candidate, but the current `datasets` stack in this repo no longer supports dataset-script based loading for that asset. The script therefore uses the scriptless `xsum` dataset, which exposes the same `document` and summary-style fields needed for the cookbook.

## Prompt Assets

| Asset | Provenance | Purpose |
| --- | --- | --- |
| [text_to_persona.txt](text_to_persona.txt) | `paper-aligned` | Infer one specific persona from a source text |
| [persona_to_persona.txt](persona_to_persona.txt) | `paper-aligned` | Expand a persona through one close relationship |
| [persona_to_user_prompt.txt](persona_to_user_prompt.txt) | `repository-derived` | Generate a representative user prompt from a persona |

## Provenance Notes

- The Persona Hub paper describes `Text-to-Persona` and `Persona-to-Persona`, but it explicitly says the prompts shown in figures are simplified rather than the exact experiment strings.
- The `persona_to_user_prompt` asset is derived from the repository prompt family for instruction generation and adapted to return JSON fields that fit DataFast.
- The cookbook does not reuse Persona Hub code. It reimplements the workflow with DataFast primitives.
