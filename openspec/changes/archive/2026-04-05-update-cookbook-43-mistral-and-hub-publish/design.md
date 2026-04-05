## Context

Cookbook 43 already provides a runnable persona-generation example and a matching docs page, but the current implementation is tuned for a small OpenRouter smoke run and stops at a local JSONL artifact. The requested update changes three user-visible behaviors at once: the documented LLM path moves to DataFast's native Mistral provider, the bounded run expands from five records to twenty, and the final synthetic records must be publishable to Hugging Face Hub.

This change stays within the existing cookbook pattern. The repo already exposes `mistral(...)` for provider construction and `Sink.hub(...)` for dataset publication, so the main design work is deciding how Cookbook 43 will compose those existing primitives without losing the bounded, inspectable nature of the example.

## Goals / Non-Goals

**Goals:**
- Update Cookbook 43 so its documented execution path uses Mistral AI instead of OpenRouter.
- Expand the bounded synthetic-data run from 5 to 20 filtered `xsum` validation records.
- Preserve a local inspectable output artifact while also publishing the final records to Hugging Face Hub.
- Document the exact runtime prerequisites needed for both generation and publishing.

**Non-Goals:**
- Changing the Persona Hub-inspired prompt assets or provenance labels beyond what is needed for provider and publication updates.
- Generalizing every cookbook to support automatic dataset publication.
- Adding new public DataFast sink APIs or a new provider abstraction.
- Turning the cookbook into an unbounded production pipeline.

## Decisions

### 1. Use DataFast's native Mistral provider helper as the cookbook default

Cookbook 43 will switch from `openrouter(...)` to `mistral(...)` and document `MISTRAL_API_KEY` as the required LLM credential. The documented model path will be pinned to `mistral-small-2603`.

Why:
- The repo already ships and tests a first-party Mistral provider.
- This makes the cookbook align with a provider that DataFast owns directly instead of routing through OpenRouter.
- Reusing the helper keeps the example consistent with existing provider ergonomics.

Alternative considered:
- Keep the script provider-agnostic. Rejected because the user asked for a concrete switch to Mistral AI and the docs need a reproducible path.

### 2. Keep the `xsum` validation workflow and raise only the bounded sample cap

The cookbook will continue to source records from `xsum` `validation`, apply the existing 300 to 500 word filter, and use `Sample(..., strategy=\"first\")` for deterministic selection. Only the cap changes, from 5 to 20.

Why:
- This preserves the original example's reproducibility and keeps the change narrowly focused on requested scale.
- Twenty records is still small enough for cookbook verification while being materially more representative than five.
- Holding the dataset and filter constant isolates the effect of the provider and publication changes.

Alternative considered:
- Switch to random sampling or a different dataset. Rejected because it adds unnecessary variability and moves the change away from Cookbook 43's current behavior.

### 3. Publish the final synthetic records to Hugging Face Hub while retaining local JSONL output

The final transformed records should remain inspectable in `examples/outputs/43_persona_cookbook.jsonl`, and the same final schema should also be pushed to Hugging Face Hub through `Sink.hub(...)`. The cookbook should treat local JSONL as the audit artifact and the Hub dataset as the publication artifact.

The Hub destination should be runtime-configurable instead of hard-coded, using the standard `HF_TOKEN` plus a dataset repo identifier supplied through configuration that the script can read without requiring source edits.

Why:
- The user asked to push the resulting dataset, but cookbook readers still need a simple local file they can inspect immediately.
- `Sink.hub(...)` already implements dataset creation and push semantics, so the example can stay within existing APIs.
- A configurable repo id avoids baking a personal or temporary dataset namespace into the repository.

Alternative considered:
- Replace the JSONL sink entirely with a Hub-only push. Rejected because it weakens local inspectability and makes failure analysis harder.

### 4. Expand the cookbook docs to describe both generation and publication prerequisites

`docs/cookbook/persona_generation.md` should document the authoritative script, prompt assets, local output path, Mistral credential requirements, Hugging Face publication requirements, and the configured dataset target. The page should also update its workflow summary from five samples to twenty and explain that the final records are published after generation.

Why:
- The current docs spec already treats the cookbook page as the reader-facing source of truth.
- Provider and publication configuration are both runtime prerequisites, not hidden implementation details.
- Cookbook readers need to know where artifacts end up locally and remotely.

Alternative considered:
- Document only the script change and leave Hub publication discoverable from code. Rejected because it would violate the cookbook pattern of making execution requirements explicit.

## Risks / Trade-offs

- [Higher run cost and latency from 20 samples] → Keep deterministic first-20 sampling and preserve the cookbook's bounded scope.
- [Publishing requires user-specific dataset configuration] → Use standard Hugging Face credentials and a runtime-configurable repo id rather than a hard-coded namespace.
- [Local and remote outputs can drift] → Push the same final record shape that is written to JSONL and keep the field-selection step immediately before both sinks.
- [Hub push failures can obscure generation failures] → Preserve local JSONL output so generation artifacts remain inspectable even if publication fails.

## Migration Plan

1. Update the Cookbook 43 script to construct a Mistral provider, keep the existing `xsum` filter, and raise the bounded sample cap to 20.
2. Add publication configuration and wire the final record stream to both local JSONL output and a Hugging Face Hub sink.
3. Revise the cookbook docs to describe Mistral setup, Hugging Face publication requirements, the 20-sample behavior, and the local plus remote outputs.
4. Run a bounded manual verification path that confirms the documented configuration and output locations are coherent.

Rollback is straightforward: revert the script and docs to the previous OpenRouter plus JSONL-only flow and remove the publication configuration path.

## Open Questions

- Which repo-level configuration mechanism should hold the Hugging Face dataset repo id for the example: a dedicated environment variable, a top-of-script constant, or a small CLI argument surface?
- Whether the published dataset should remain private by default or be documented as user-selectable at runtime.
