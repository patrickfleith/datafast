# Cookbook Assets

Prompt files and dataset details used by cookbook examples.

## Text Classification

### Dataset

- **Source:** seed dimensions created with `Seed.product`
- **Dimensions:** label, trail type, style, language, and model
- **Local output:** `examples/outputs/45_text_classification_cookbook.jsonl`
- **Checkpoints:** `examples/checkpoints/45_text_classification_cookbook`
- **Hub output:** optional, controlled by `DATAFAST_PUSH_TO_HUB=1`

This cookbook models variation directly as seed dimensions so the label, trail
type, style, language, and model are all explicit in the
pipeline.

### Prompt

| File | Style |
| --- | --- |
| [text_classification_generation.txt](text_classification_generation.txt) | One short trail report per call, with label, trail type, style, and language injected |

## Persona Generation

### Dataset

- **Source:** `xsum` (Hugging Face), `validation` split
- **Fields used:** `id`, `document`, `summary`
- **Filter:** 300–500 words, first 100 matches
- **Local output:** `examples/outputs/43_persona_cookbook.jsonl`
- **Checkpoints:** `examples/checkpoints/43_persona_cookbook`
- **Hub output:** set `HF_REPO_ID` and the `repo_id` in `push_records_to_hub()` to repos under your own Hugging Face username or organization

The example keeps first-match sampling for reproducibility. For local JSONL corpora with metadata such as `document_filename`, stratified sampling is usually a better fit.

### Prompt Variants

Each LLM step picks one prompt at random per record. The script also assigns random `life_stage` and `related_life_stage` values before the corresponding LLM steps. Multiple variants add diversity.

#### Text-to-Persona

| File | Style |
| --- | --- |
| [text_to_persona_v1.txt](text_to_persona_v1.txt) | Direct inference of a reader persona |
| [text_to_persona_v2.txt](text_to_persona_v2.txt) | XML-tagged source text, writer/reader framing |
| [text_to_persona_v3.txt](text_to_persona_v3.txt) | System-role preamble, search-interest angle |

#### Persona-to-Persona

| File | Style |
| --- | --- |
| [persona_to_persona_v1.txt](persona_to_persona_v1.txt) | Close relationship, standalone description |
| [persona_to_persona_v2.txt](persona_to_persona_v2.txt) | Rule-list format, explicit separation of description and relationship |
| [persona_to_persona_v3.txt](persona_to_persona_v3.txt) | XML-tagged input, concise vivid output |

### Provenance

- Text-to-Persona and Persona-to-Persona prompts are paper-aligned adaptations. The Persona Hub paper states its published prompts are simplified, not exact.
- No Persona Hub code is reused. The workflow is built with datafast primitives.

## Space Engineering Text Generation

### Dataset

- **Source:** seed dimensions created with `Seed.product`
- **Dimensions:** document type, topic, expertise level, and language
- **Local output:** `examples/outputs/44_space_text_generation_cookbook.jsonl`
- **Checkpoints:** `examples/checkpoints/44_space_text_generation_cookbook`
- **Hub output:** optional, controlled by `DATAFAST_PUSH_TO_HUB=1`

### Prompt

The text-generation cookbook uses one compact prompt and relies on seed
dimensions for variation.

| File | Style |
| --- | --- |
| [space_text_generation.txt](space_text_generation.txt) | Minimal variable-driven request |
