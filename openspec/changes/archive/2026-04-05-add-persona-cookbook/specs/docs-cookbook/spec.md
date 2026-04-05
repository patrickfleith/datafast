## ADDED Requirements

### Requirement: Cookbook navigation
The documentation site SHALL expose a Cookbook section in the MkDocs navigation, and each cookbook entry SHALL resolve to a Markdown page under `docs/`.

#### Scenario: Cookbook section appears in navigation
- **WHEN** the documentation site configuration is loaded
- **THEN** the navigation includes a Cookbook section with an entry for the persona-generation cookbook

### Requirement: Cookbook pages identify runnable source
Each cookbook page SHALL identify the authoritative executable source file, the runtime prerequisites, and the expected output location for the example it documents.

#### Scenario: Reader opens a cookbook page
- **WHEN** a reader opens the persona-generation cookbook page
- **THEN** the page shows the script path, required provider configuration, and the output artifact path needed to reproduce the example
