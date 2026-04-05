## Context

DataFast already separates runnable examples (`examples/scripts/`) from rendered documentation (`docs/`), but it does not yet have a cookbook layer that ties a real script to a narrative, site-facing walkthrough. This change introduces that pattern and uses a Persona Hub-inspired persona-generation example as the first cookbook because the repo already exposes the core primitives needed for it: `Source`/`Seed`, `LLMStep`, and `Sink`.

The research constraint is important. Persona Hub publishes the high-level methods for persona construction in the paper, but the paper also states that figure prompts are simplified rather than the exact experiment strings. The repository publishes exact prompt templates for persona-conditioned synthesis tasks such as instruction generation, but not a full canonical `Text-to-Persona` implementation prompt. The design therefore needs an explicit provenance model so DataFast can be faithful to the method without overstating prompt fidelity.

## Goals / Non-Goals

**Goals:**
- Add a reusable cookbook pattern that connects docs navigation, a runnable script, and an explanatory Markdown page.
- Implement a first cookbook that demonstrates Persona Hub-inspired persona creation with DataFast, centered on `Text-to-Persona` and `Persona-to-Persona`.
- Preserve provenance by separating paper-aligned prompts, repository-derived templates, and DataFast-specific prompt adaptations.
- Keep the example practically runnable with a bounded sample size and explicit provider prerequisites.

**Non-Goals:**
- Reproducing Persona Hub’s full billion-persona data pipeline or dataset scale.
- Reusing or vendoring Persona Hub code.
- Adding new public DataFast APIs solely for this cookbook.
- Building an automated docs-code sync system beyond what is needed for the first cookbook.

## Decisions

### 1. Keep executable source in `examples/scripts/` and renderable narrative in `docs/cookbook/`

The runnable example will follow the repo’s existing convention and live under `examples/scripts/`. The new cookbook section will live under `docs/cookbook/`, with `mkdocs.yml` updated to expose it in navigation.

Why:
- The repo already teaches runnable examples from `examples/scripts/`.
- Keeping executable code there avoids turning `docs/` into a mixed content area full of Python files and generated artifacts.
- The cookbook page can still act as the canonical reader-facing entry point while pointing to the authoritative script.

Alternative considered:
- Store the script directly under `docs/cookbook/`. Rejected because it breaks current example organization and makes docs content noisier to maintain.

### 2. Use prompt assets with explicit provenance labels

Prompt text used by the cookbook will be stored as dedicated assets instead of being embedded only inside Python strings. Each prompt asset will be labeled as one of:
- paper-aligned adaptation
- repository-derived template
- DataFast-specific adaptation

Why:
- The paper gives simplified prompt forms for persona creation, not exact experiment strings.
- The repository does publish exact downstream templates such as instruction synthesis.
- External prompt assets make it easier to show provenance in both the script and the cookbook page without duplicating large prompt blocks.

Alternative considered:
- Inline all prompts in the script. Rejected because provenance becomes harder to audit and the doc page is more likely to drift from the executable source.

### 3. Model the cookbook as a small multi-stage DataFast pipeline

The first cookbook script will be structured as a bounded pipeline with three stages:
- source text to inferred persona (`Text-to-Persona`)
- persona to related persona (`Persona-to-Persona`)
- downstream persona-conditioned generation via a representative user-prompt generation step inspired by the Persona Hub repository template

Outputs should be written to JSONL so the script leaves inspectable artifacts after execution.

Why:
- This demonstrates both persona construction methods from the paper and one concrete persona-driven synthesis pattern from the repository.
- JSONL outputs fit existing repo conventions and are easy to inspect in docs.
- A multi-stage pipeline shows DataFast’s composition model better than a single prompt call.

Alternative considered:
- Stop after the first persona-generation stage. Rejected because it under-explains the value of generated personas and misses the strongest link to the repository prompt templates.

### 4. Use `xsum` validation articles as the seed corpus

The cookbook will draw source texts from Hugging Face `xsum`, using the `validation` split and selecting up to the first five documents whose lengths fall between 300 and 500 words.

Why:
- `xsum` is a well-known Hugging Face dataset with concise full articles that fit the desired demonstration size well.
- The `validation` split gives a stable, reproducible sample source for the cookbook.
- Limiting the selection to at most five articles keeps the run small enough for a cookbook while still showing multiple persona inferences.
- It works with the current `datasets>=3.0` stack in this repo, whereas `GEM/xsum` now depends on a dataset-script loading path that is no longer supported here.

Alternative considered:
- Use `GEM/xsum`. Rejected during implementation because the current `datasets` dependency no longer supports the required dataset script.

### 5. Favor structured outputs where practical

The script should ask the model for structured fields where that improves inspectability, for example:
- inferred persona description
- relationship type used for expansion
- related persona description
- downstream user prompt or artifact

Why:
- Structured output is easier to verify manually.
- It produces cleaner JSONL for later cookbook rendering.
- It reduces ambiguity when comparing prompt variants.

Alternative considered:
- Use only free-form text outputs. Rejected because the resulting artifacts are harder to reuse in documentation.

### 6. Use OpenRouter as the documented execution path

The cookbook will document and smoke-test an OpenRouter-based execution path using `nvidia/nemotron-3-super-120b-a12b` as the default model id.

Why:
- The user selected OpenRouter as the desired provider.
- It matches existing repo examples and keeps the cookbook aligned with current usage patterns.
- A single concrete provider path reduces ambiguity in the first cookbook.

Alternative considered:
- Keep the smoke-run path provider-agnostic. Rejected because it weakens reproducibility for the initial cookbook.

### 7. Require explicit runtime prerequisites instead of CI-level execution guarantees

The cookbook will be documented as a real runnable script that requires a configured LLM provider. The implementation should keep sample sizes small and output paths explicit, but it will not require new test infrastructure or a fake public provider abstraction.

Why:
- The repo’s current examples already assume a configured provider.
- No public fake-provider path exists for full end-to-end example execution.
- This keeps the first cookbook lightweight and consistent with existing example ergonomics.

Alternative considered:
- Introduce a fake provider or snapshot-based offline harness only for cookbook validation. Rejected for this change because it expands scope beyond the user request.

## Risks / Trade-offs

- [Prompt fidelity ambiguity] → Label prompts by provenance and explicitly state that the paper’s displayed persona-creation prompts are simplified, not exact experiment strings.
- [Docs/example drift] → Make the Python script the implementation source of truth and have the cookbook page reference concrete script and prompt asset paths.
- [Networked example friction] → Keep runs bounded, document provider prerequisites clearly, and use a small default sample size for manual execution.
- [Cookbook scope creep] → Limit v1 to a single persona-generation cookbook and a minimal cookbook section in navigation.

## Migration Plan

1. Add the cookbook docs section and navigation entry.
2. Add prompt/reference assets with provenance notes.
3. Implement the runnable persona-generation script using `xsum` validation articles, capped at the first five 300 to 500 word documents, and write outputs to a stable example output path.
4. Author the cookbook Markdown page that explains the workflow and references the script outputs.
5. Run a bounded manual smoke test with OpenRouter and capture the expected invocation in the docs.

Rollback is straightforward because the change is additive: remove the cookbook nav entry and the newly added docs/example files.

## Open Questions

- Whether the word-count filter should use a simple whitespace tokenization or a slightly stricter normalization rule before selecting the five `xsum` documents.
