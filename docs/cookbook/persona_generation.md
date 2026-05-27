# Persona Generation

Build personas from real articles and expand them through relationships. Inspired by the Persona Hub paper, implemented entirely with DataFast.

## Source

- **Script:** `examples/scripts/43_cookbook_persona_generation.py`
- **Prompt assets:** [asset index](assets/index.md)
- **Local output:** `examples/outputs/43_persona_cookbook.jsonl`
- **Hub output:** pushed to the Hugging Face Hub repo IDs configured in the script

## Pipeline

1. Load `xsum` articles (`validation` split).
2. Filter to documents between 300 and 500 words. Keep the first 100 matches.
3. Assign a random life stage to the source persona.
4. **Text-to-Persona** — infer one persona from each article and life stage.
5. Assign a random life stage to the related persona.
6. **Persona-to-Persona** — expand that persona into a related individual.
7. Keep the final output fields, write JSONL, and push results to Hugging Face Hub.

Each LLM step randomly picks one prompt variant per record using `Sample(prompts, n=1)`. This adds diversity across generations.

```text
xsum article
    │
    ▼
life_stage  (random from configured stages)
    │
    ▼
Text-to-Persona  (random prompt from 3 variants)
    │
    ▼
related_life_stage  (random from configured stages)
    │
    ▼
Persona-to-Persona  (random prompt from 3 variants)
    │
    ▼
Hugging Face Hub
```

## Run

Prerequisites:

- `OPENROUTER_API_KEY` set in a `.env` file
- Hugging Face authentication via `HF_TOKEN` in `.env` or a cached `huggingface_hub` login
- Base dependencies from `pyproject.toml` installed

Before running, replace the example Hugging Face namespaces in the script with your own username or organization:

- `HF_REPO_ID = "<your-username-or-org>/new-persona-cookbook-dataset"` controls the private pipeline sink.
- `repo_id = "<your-username-or-org>/datafast-persona-cookbook"` inside `push_records_to_hub()` controls the public publish step.

```bash
python examples/scripts/43_cookbook_persona_generation.py
```

## Prompt Variants

Each step draws from multiple prompt files stored under `docs/cookbook/assets/`. See the [asset index](assets/index.md) for the full list.

- **Text-to-Persona:** 3 variants (`text_to_persona_v1.txt`, `v2`, `v3`)
- **Persona-to-Persona:** 3 variants (`persona_to_persona_v1.txt`, `v2`, `v3`)

Additional prompt variants for user-prompt generation are available (`persona_to_user_prompt_v1.txt`, `v2`, `v3`) but not used in the current pipeline.

## Research Basis

The Persona Hub paper introduces Text-to-Persona and Persona-to-Persona as scalable methods for building personas from web text. The paper states that its published prompts are simplified, not the exact experiment strings. This cookbook treats them as paper-aligned adaptations. It does not reuse any Persona Hub code.

## Output Fields

- `summary` — original article summary
- `document` — source article text
- `word_count` — whitespace token count
- `life_stage` — randomly selected life stage for the inferred persona
- `persona_description` — inferred persona
- `relationship_type` — link between the two personas
- `related_life_stage` — randomly selected life stage for the expanded persona
- `related_persona_description` — the expanded related persona
