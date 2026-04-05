## Why

DataFast has runnable examples and narrative docs, but it does not yet have a cookbook area that connects a real executable script to a documentation-page walkthrough. A persona-generation cookbook is a strong first entry because it showcases a realistic synthetic-data workflow and lets DataFast demonstrate a paper-aligned reimplementation of Persona Hub ideas without copying their code.

## What Changes

- Add a cookbook section under `docs/` and expose it in the MkDocs navigation.
- Establish a cookbook pattern where the executable Python script is authored first and the documentation page is built from that runnable example.
- Add the first cookbook for persona generation, implemented with DataFast primitives and explicitly inspired by Persona Hub’s `Text-to-Persona`, `Persona-to-Persona`, and persona-conditioned prompting ideas.
- Document the research basis of the cookbook, including which prompt patterns come from the paper or repository and which parts are DataFast-specific adaptations.
- Keep the implementation independent from Persona Hub code: reuse methodology and prompt logic where appropriate, but do not vendor or call their code.

## Capabilities

### New Capabilities
- `docs-cookbook`: Provide a cookbook area in the documentation for executable, real-world DataFast examples that can later render cleanly on the docs site.
- `persona-generation-cookbook`: Provide a first cookbook that explores persona generation with DataFast using Persona Hub-inspired methods, prompts, and workflow notes.

### Modified Capabilities

- None.

## Impact

- Affected docs and site navigation in `docs/` and `mkdocs.yml`.
- New runnable cookbook source files and supporting prompt/reference material.
- No public DataFast API changes are required.
- Requires explicit handling of provider prerequisites and provenance notes because the Persona Hub paper states that figure prompts are simplified rather than exact experiment strings.
