# Persona Generation

This cookbook shows how to implement a Persona Hub-inspired workflow with DataFast without reusing Persona Hub code.

## Runnable Source

- Script: `examples/scripts/43_cookbook_persona_generation.py`
- Prompt assets: [asset index](assets/index.md)
- Output artifact: `examples/outputs/43_persona_cookbook.jsonl`

## What The Script Does

The pipeline is intentionally small:

1. Load `xsum` articles from the `validation` split.
2. Keep only the first `5` documents whose word counts fall between `300` and `500`.
3. Infer one likely persona from each article with a `Text-to-Persona` prompt.
4. Expand that persona into a closely related persona with a `Persona-to-Persona` prompt.
5. Generate one representative user prompt for the related persona.

```text
GEM/xsum article
    |
    v
Text-to-Persona
    |
    v
Persona-to-Persona
    |
    v
Representative user prompt
```

## Run

Prerequisites:

- `OPENROUTER_API_KEY` is set
- the project environment has the base dependencies from `pyproject.toml`
- the script uses OpenRouter model `nvidia/nemotron-3-super-120b-a12b`

Example:

```bash
.venv/bin/python examples/scripts/43_cookbook_persona_generation.py
```

## Prompt Summary

The cookbook keeps full prompts in asset files rather than embedding them here.

- [Text-to-Persona prompt](assets/text_to_persona.txt): a paper-aligned adaptation that infers one specific persona from a source text.
- [Persona-to-Persona prompt](assets/persona_to_persona.txt): a paper-aligned adaptation that expands a persona through one close relationship.
- [Persona-to-User-Prompt prompt](assets/persona_to_user_prompt.txt): a repository-derived prompt that asks for one realistic user request from the generated persona.

## Research Basis

The Persona Hub paper introduces `Text-to-Persona` and `Persona-to-Persona` as scalable persona-construction methods from web text. It also states that the prompts shown in the paper figures are simplified rather than the exact strings used in experiments, so this cookbook treats those persona-construction prompts as paper-aligned adaptations rather than verbatim reproductions.

For downstream prompt generation, the repository publishes a prompt family for persona-conditioned instruction generation. This cookbook adapts that idea to a DataFast JSON workflow and keeps the full asset path visible in the [asset index](assets/index.md).

## Output Shape

The JSONL output keeps the fields that matter for inspection:

- `summary`
- `document`
- `word_count`
- `persona`
- `persona_basis`
- `relationship_type`
- `related_persona`
- `user_prompt`
- `prompt_basis`
