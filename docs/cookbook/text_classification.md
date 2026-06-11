# Text Classification

Build a multilingual trail-conditions classification dataset with `datafast`.

## Source

- **Script:** `examples/scripts/45_cookbook_text_classification.py`
- **Prompt assets:** [asset index](assets/index.md)
- **Local output:** `examples/outputs/45_text_classification_cookbook.jsonl`
- **Checkpoints:** `examples/checkpoints/45_text_classification_cookbook`
- **Hub output:** optional, controlled by `DATAFAST_PUSH_TO_HUB=1`

## Use Case

This cookbook generates short hiker reports across four trail-condition labels
so teams can monitor trail quality and surface issues quickly.

The default setup is:

- multi-class: 4 trail-condition labels
- multi-lingual: English and French
- multi-model: two generation models by default
- publishable: optional push to Hugging Face Hub

## Pipeline

1. Create a seed grid from labels, trail types, and writing styles.
2. Generate one short hiker report for each seed across all configured models
   and languages.
3. Keep the label and prompt-variation provenance in flat output columns.
4. Add a UUID, write JSONL locally, and optionally push to Hugging Face Hub.

Variation is modeled explicitly through `Seed.product(...)`, which keeps the
generation axes inspectable and easy to count.

```text
label x trail_type x style
    |
    v
LLMStep language expansion: English and French
    |
    v
LLMStep model expansion
    |
    v
examples/outputs/45_text_classification_cookbook.jsonl
```

## Row Count

The default script generates:

```text
4 labels x 3 trail types x 2 styles x 2 languages
x 2 models = 96 rows
```

Each extra model in `MODEL_IDS` multiplies the total row count.

## Run

Prerequisites:

- `OPENROUTER_API_KEY` set in a `.env` file
- Base dependencies from `pyproject.toml` installed
- Hugging Face authentication only if publishing

```bash
python examples/scripts/45_cookbook_text_classification.py
```

To publish, replace `HF_REPO_ID` in the script with a repository under your own
Hugging Face username or organization, then run:

```bash
DATAFAST_PUSH_TO_HUB=1 python examples/scripts/45_cookbook_text_classification.py
```

The run uses `checkpoint_dir` and `resume=True`. If generation is interrupted,
run the command again to continue from saved checkpoints.

If you want to use provider-specific clients directly, replace `MODEL_IDS` or
the `model=MODELS` argument in `LLMStep` with providers such as `openai(...)`
or `anthropic(...)`. The default setup uses multiple OpenRouter-backed models
so it works with one API key.

## Prompt

The cookbook uses one prompt file and drives diversity through seed dimensions:

```text
Write one realistic hiker report in {language_name}.
```

See [text_classification_generation.txt](assets/text_classification_generation.txt)
for the full prompt.

## Generation Controls

- `LABELS` defines the target classes and their prompt descriptions.
- `TRAIL_TYPES` controls the trail settings used in generation.
- `STYLES` controls the voice and format of each report.
- `LANGUAGES` controls language expansion.
- `MODEL_IDS` controls which models generate records.
- `HF_REPO_ID` controls the optional Hugging Face Hub destination.

If you want an extra quality-control pass, add a downstream `Classify` and
`Filter` stage to verify that generated reports match their intended label.

## Output Fields

- `id` - generated row UUID
- `label` - target trail-condition label
- `trail_type` - prompt expansion axis for the trail setting
- `style` - prompt expansion axis for the report style
- `language` - language code emitted by `LLMStep`
- `model` - model ID emitted by `LLMStep`
- `text` - generated hiker report
