# Space Engineering Text Generation

Build a raw technical text corpus across document types, topics, expertise levels,
languages, and model choices.

## Source

- **Script:** `examples/scripts/44_cookbook_space_text_generation.py`
- **Prompt assets:** [asset index](assets/index.md)
- **Local output:** `examples/outputs/44_space_text_generation_cookbook.jsonl`
- **Checkpoints:** `examples/checkpoints/44_space_text_generation_cookbook`
- **Hub output:** optional, controlled by `DATAFAST_PUSH_TO_HUB=1`

## Pipeline

1. Create a seed grid with `Seed.product`.
2. Cross document types, topics, and expertise levels explicitly.
3. Generate one section per seed and language with `LLMStep`.
4. Let the prompt variables drive the corpus variation.
5. Parse `title` and `text` from JSON mode.
6. Keep publication fields, add a row UUID, write JSONL, checkpoint progress,
   and optionally push to Hugging Face Hub.

The default model is `nvidia/nemotron-3-super-120b-a12b:nitro` through
OpenRouter.

```text
document_type x topic x expertise_level
    |
    v
LLMStep language expansion: English and French
    |
    v
JSON fields: title, text
    |
    v
examples/outputs/44_space_text_generation_cookbook.jsonl
```

## Row Count

The default script generates:

```text
3 document types x 8 topics x 3 expertise levels x 2 languages
x 1 generated output x 1 model = 144 rows
```

To use several models, add provider IDs to `MODEL_IDS`. `LLMStep` will run each
seed-language combination through every model and the row count will multiply by
the number of models.

## Run

Prerequisites:

- `OPENROUTER_API_KEY` set in a `.env` file
- Base dependencies from `pyproject.toml` installed
- Hugging Face authentication only if publishing

```bash
python examples/scripts/44_cookbook_space_text_generation.py
```

To publish, replace `HF_REPO_ID` in the script with a repository under your own
Hugging Face username or organization, then run:

```bash
DATAFAST_PUSH_TO_HUB=1 python examples/scripts/44_cookbook_space_text_generation.py
```

The run uses `checkpoint_dir` and `resume=True`. If generation is interrupted,
run the command again to continue from saved checkpoints.

## Prompt

The script uses one compact prompt file:

```text
Write one {document_type} excerpt about {topic} for {expertise_level} in {language_name}.
```

## Generation Controls

- `MODEL_IDS` controls which models generate each record.
- `LANGUAGES` controls language expansion and writes the emitted language code to
  the `language` field.
- `NUM_OUTPUTS` controls how many generated rows are created for each
  seed, language, and model combination.
- `PROMPT_PATH` controls the prompt file used for generation.
- `SEED` controls deterministic dataset splitting when publishing.
- `HF_REPO_ID` controls the optional Hugging Face Hub destination.

## Output Fields

- `id` - generated row UUID
- `document_type` - requested document style
- `topic` - space engineering topic
- `expertise_level` - intended reader level
- `language` - language code emitted by `LLMStep`
- `model` - model ID emitted by `LLMStep`
- `title` - generated section title
- `text` - generated corpus text
