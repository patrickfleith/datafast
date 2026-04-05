## Why

Cookbook 43 currently demonstrates persona-conditioned synthetic data generation with OpenRouter and a five-record sample cap, which is too narrow for the larger synthetic-data workflow the repo now wants to showcase. The next revision should reflect a first-party Mistral AI execution path, produce a more representative 20-sample output set, and document publishing the resulting dataset to Hugging Face Hub.

## What Changes

- Update `examples/scripts/43_cookbook_persona_generation.py` to use the repo's Mistral provider path instead of OpenRouter.
- Expand the bounded sample size from the first `5` eligible `xsum` validation records to the first `20`.
- Extend the cookbook pipeline so it can publish the generated synthetic dataset to Hugging Face Hub after local generation succeeds.
- Revise the cookbook documentation to describe the Mistral runtime prerequisites, larger sample count, local output artifact, and Hugging Face dataset destination.
- Preserve the existing Persona Hub-inspired prompt provenance model while adapting the execution and publication flow.

## Capabilities

### New Capabilities
- None.

### Modified Capabilities
- `docs-cookbook`: Change cookbook-page requirements so a published cookbook can document both its local output artifact and its Hugging Face dataset publication target.
- `persona-generation-cookbook`: Change the cookbook workflow requirements to use Mistral AI, generate 20 bounded samples, and publish the resulting synthetic dataset to Hugging Face Hub.

## Impact

- Affected runnable example in `examples/scripts/43_cookbook_persona_generation.py`.
- Affected cookbook documentation in `docs/cookbook/persona_generation.md` and supporting asset references.
- Requires Mistral runtime configuration via `MISTRAL_API_KEY` and dataset publication credentials via `HF_TOKEN`.
- Uses existing DataFast Hub sink support for dataset publishing; no new public API surface is required.
