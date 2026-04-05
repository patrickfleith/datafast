## Purpose
The persona-generation cookbook documents a bounded, runnable DataFast workflow inspired by Persona Hub research and shows how generated personas can drive later synthetic-data creation.
## Requirements
### Requirement: Persona Hub-inspired persona workflow
The persona-generation cookbook SHALL implement a runnable DataFast workflow that explores Persona Hub-inspired `Text-to-Persona` and `Persona-to-Persona` methods from bounded sample inputs drawn from Hugging Face `xsum`, using Mistral AI as the documented LLM provider path.

#### Scenario: Script generates personas from source texts
- **WHEN** a user runs the cookbook script with a configured Mistral model
- **THEN** the script selects up to the first twenty documents from the `validation` split whose lengths are between 300 and 500 words and produces output records that include personas inferred from source text and personas expanded from prior personas

### Requirement: Prompt provenance is explicit
The cookbook SHALL distinguish between paper-aligned persona-generation prompts, repository-derived downstream prompt templates, and DataFast-specific prompt adaptations, and it SHALL NOT claim verbatim reproduction where the source material does not publish exact prompt strings.

#### Scenario: Reader inspects prompt usage
- **WHEN** a reader reviews the cookbook code or documentation
- **THEN** each prompt used in the workflow is labeled by provenance and any paper-derived persona prompt is described as an adaptation rather than an exact reproduction

### Requirement: Cookbook demonstrates downstream persona usage
The cookbook SHALL include at least one downstream persona-conditioned generation step implemented with DataFast to show how generated personas can drive later synthetic-data creation.

#### Scenario: Script reaches downstream synthesis
- **WHEN** the cookbook script completes its final stage
- **THEN** the outputs include at least one artifact generated from a persona-conditioned prompt, such as a representative user request

### Requirement: Standalone execution is documented and bounded
The cookbook SHALL be runnable as a standalone Python script with a bounded execution path suitable for manual verification and dataset publication.

#### Scenario: User performs a smoke run
- **WHEN** a user executes the documented smoke-run command with configured Mistral and Hugging Face credentials
- **THEN** the script uses Mistral model id `mistral-small-2603`, processes only the documented bounded sample size, writes inspectable output artifacts, and publishes the resulting dataset without requiring repo code changes

### Requirement: Cookbook publishes the final synthetic dataset
The persona-generation cookbook SHALL push the final synthetic records to a configured Hugging Face Hub dataset after local generation completes successfully.

#### Scenario: Script publishes generated records
- **WHEN** the cookbook script reaches its final sink stage with a configured Hugging Face dataset repo and token
- **THEN** it pushes the same final record fields that are written locally to the configured dataset repository

