# Persona Cookbook Assets

Prompt files and dataset details used by the persona-generation cookbook.

## Dataset

- **Source:** `xsum` (Hugging Face), `validation` split
- **Fields used:** `document`, `summary`
- **Filter:** 300–500 words, first 5 matches

## Prompt Variants

Each LLM step picks one prompt at random per record. Multiple variants add diversity.

### Text-to-Persona

| File | Style |
| --- | --- |
| [text_to_persona_v1.txt](text_to_persona_v1.txt) | Direct inference of a reader persona |
| [text_to_persona_v2.txt](text_to_persona_v2.txt) | XML-tagged source text, writer/reader framing |
| [text_to_persona_v3.txt](text_to_persona_v3.txt) | System-role preamble, search-interest angle |

### Persona-to-Persona

| File | Style |
| --- | --- |
| [persona_to_persona_v1.txt](persona_to_persona_v1.txt) | Close relationship, standalone description |
| [persona_to_persona_v2.txt](persona_to_persona_v2.txt) | Rule-list format, explicit separation of description and relationship |
| [persona_to_persona_v3.txt](persona_to_persona_v3.txt) | XML-tagged input, concise vivid output |

### Persona-to-User-Prompt (not in current pipeline)

| File | Style |
| --- | --- |
| [persona_to_user_prompt_v2.txt](persona_to_user_prompt_v2.txt) | XML-tagged person, AI assistant framing |
| [persona_to_user_prompt_v3.txt](persona_to_user_prompt_v3.txt) | Requirements-first ordering |

## Provenance

- Text-to-Persona and Persona-to-Persona prompts are paper-aligned adaptations. The Persona Hub paper states its published prompts are simplified, not exact.
- User-prompt variants are derived from the repository's instruction-generation prompt family.
- No Persona Hub code is reused. The workflow is built with DataFast primitives.
