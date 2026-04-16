# Persona Generation

Build personas from real articles and expand them through relationships. Inspired by the Persona Hub paper, implemented entirely with DataFast.

## Source

- **Script:** `examples/scripts/43_cookbook_persona_generation.py`
- **Prompt assets:** [asset index](assets/index.md)
- **Output:** pushed to a private Hugging Face Hub dataset

## Pipeline

1. Load `xsum` articles (`validation` split).
2. Filter to documents between 300 and 500 words. Keep the first 5.
3. **Text-to-Persona** — infer one persona from each article.
4. **Persona-to-Persona** — expand that persona into a related individual.
5. Push results to Hugging Face Hub.

Each LLM step randomly picks one prompt variant per record using `Sample(prompts, n=1)`. This adds diversity across generations.

```text
xsum article
    │
    ▼
Text-to-Persona  (random prompt from 3 variants)
    │
    ▼
Persona-to-Persona  (random prompt from 3 variants)
    │
    ▼
Hugging Face Hub
```

## Run

Prerequisites:

- `OPENROUTER_API_KEY` and `HF_TOKEN` set in a `.env` file
- Base dependencies from `pyproject.toml` installed

```bash
python examples/scripts/43_cookbook_persona_generation.py
```

## Prompt Variants

Each step draws from multiple prompt files stored under `docs/cookbook/assets/`. See the [asset index](assets/index.md) for the full list.

- **Text-to-Persona:** 3 variants (`text_to_persona_v1.txt`, `v2`, `v3`)
- **Persona-to-Persona:** 3 variants (`persona_to_persona_v1.txt`, `v2`, `v3`)

Additional prompt variants for user-prompt generation are available (`persona_to_user_prompt_v2.txt`, `v3`) but not used in the current pipeline.

## Research Basis

The Persona Hub paper introduces Text-to-Persona and Persona-to-Persona as scalable methods for building personas from web text. The paper states that its published prompts are simplified, not the exact experiment strings. This cookbook treats them as paper-aligned adaptations. It does not reuse any Persona Hub code.

## Output Fields

- `summary` — original article summary
- `document` — source article text
- `word_count` — whitespace token count
- `persona_description` — inferred persona
- `relationship_type` — link between the two personas
- `related_persona_description` — the expanded related persona
