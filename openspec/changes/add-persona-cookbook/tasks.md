## 1. Cookbook scaffolding

- [x] 1.1 Add the Cookbook section to `mkdocs.yml` and create the base `docs/cookbook/` pages needed for navigation.
- [x] 1.2 Add prompt/reference assets for the persona cookbook and label each asset with its provenance (`paper-aligned`, `repository-derived`, or `DataFast adaptation`).
- [x] 1.3 Add a dataset-selection note for `xsum` `validation`, including the 300 to 500 word filter and first-five cap used by the example.

## 2. Runnable persona example

- [x] 2.1 Implement the standalone persona-generation script in `examples/scripts/` with `xsum` `validation` inputs, a 300 to 500 word filter, a first-five cap, and JSONL outputs.
- [x] 2.2 Add the `Text-to-Persona` and `Persona-to-Persona` stages using DataFast primitives and structured outputs where practical.
- [x] 2.3 Add a downstream persona-conditioned user-prompt generation stage that demonstrates how the generated personas drive a later synthetic-data step.
- [x] 2.4 Perform a bounded smoke run with OpenRouter and confirm the documented command produces inspectable output artifacts.

## 3. Cookbook documentation

- [x] 3.1 Write the persona-generation cookbook page under `docs/cookbook/` with prerequisites, script path, run command, output path, and explanation of the workflow.
- [x] 3.2 Document the research basis and prompt provenance, including the limitation that Persona Hub’s paper shows simplified persona-creation prompts rather than exact experiment strings.
- [x] 3.3 Summarize the key prompts in the cookbook page and link to the prompt asset files instead of embedding full prompt text inline.
- [x] 3.4 Review the cookbook navigation and page content to ensure the docs site can expose the new section cleanly.
