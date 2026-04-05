# Persona Generation

This cookbook shows how to implement a Persona Hub-inspired workflow with DataFast without reusing Persona Hub code.

## Runnable Source

- Script: `examples/scripts/43_cookbook_persona_generation.py`
- Prompt assets: [asset index](assets/index.md)
- Output artifact: `examples/outputs/43_persona_cookbook.jsonl`
- Dataset publication target: the Hugging Face dataset repo in `PERSONA_COOKBOOK_HF_REPO_ID`

## What The Script Does

The pipeline is intentionally small:

1. Load `xsum` articles from the `validation` split.
2. Keep only the first `20` documents whose word counts fall between `300` and `500`.
3. Infer one likely persona from each article with a `Text-to-Persona` prompt.
4. Expand that persona into a closely related persona with a `Persona-to-Persona` prompt.
5. Generate one representative user prompt for the related persona.
6. Write the final records to local JSONL and publish the same rows to Hugging Face Hub.

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

- `MISTRAL_API_KEY` is set
- `PERSONA_COOKBOOK_HF_REPO_ID` points at the target Hugging Face dataset repo, for example `your-name/persona-cookbook-43`
- Hugging Face authentication is available through `HF_TOKEN` or a cached `huggingface_hub` login
- the project environment has the base dependencies from `pyproject.toml`
- the script uses the Mistral model `mistral-small-2603`

Example:

```bash
.venv/bin/python examples/scripts/43_cookbook_persona_generation.py
```

By default the dataset push is private. Set `PERSONA_COOKBOOK_HF_PRIVATE=false` if you want the published dataset to be public.

## Prompt Summary

The cookbook keeps full prompts in asset files rather than embedding them here.

- [Text-to-Persona prompt](assets/text_to_persona.txt): a paper-aligned adaptation that infers one specific persona from a source text.
- [Persona-to-Persona prompt](assets/persona_to_persona.txt): a paper-aligned adaptation that expands a persona through one close relationship.
- [Persona-to-User-Prompt prompt](assets/persona_to_user_prompt.txt): a repository-derived prompt that asks for one realistic user request from the generated persona.

## Research Basis

The Persona Hub paper introduces `Text-to-Persona` and `Persona-to-Persona` as scalable persona-construction methods from web text. It also states that the prompts shown in the paper figures are simplified rather than the exact strings used in experiments, so this cookbook treats those persona-construction prompts as paper-aligned adaptations rather than verbatim reproductions.

For downstream prompt generation, the repository publishes a prompt family for persona-conditioned instruction generation. This cookbook adapts that idea to a DataFast JSON workflow and keeps the full asset path visible in the [asset index](assets/index.md).

## Output Shape

The local JSONL output and the published Hugging Face dataset keep the same fields for inspection:

- `summary`
- `document`
- `word_count`
- `persona`
- `persona_basis`
- `relationship_type`
- `related_persona`
- `user_prompt`
- `prompt_basis`
