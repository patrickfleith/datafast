## MODIFIED Requirements

### Requirement: Cookbook pages identify runnable source
Each cookbook page SHALL identify the authoritative executable source file, the runtime prerequisites, the expected local output location, and any Hugging Face dataset publication target for the example it documents.

#### Scenario: Reader opens a cookbook page
- **WHEN** a reader opens the persona-generation cookbook page
- **THEN** the page shows the script path, required Mistral and Hugging Face configuration, the local output artifact path, and the dataset publication target needed to reproduce the example
