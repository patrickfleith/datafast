## 1. Cookbook Script

- [x] 1.1 Replace the OpenRouter provider setup in `examples/scripts/43_cookbook_persona_generation.py` with the DataFast Mistral provider path and the required runtime configuration.
- [x] 1.2 Raise the bounded sample selection from the first 5 eligible `xsum` validation records to the first 20 while preserving the existing word-count filter and output schema.
- [x] 1.3 Add a final Hugging Face Hub publication step that pushes the generated synthetic records after local output generation succeeds.

## 2. Cookbook Documentation

- [x] 2.1 Update `docs/cookbook/persona_generation.md` to document the Mistral prerequisites, 20-sample behavior, local output artifact, and Hugging Face dataset publication target.
- [x] 2.2 Revise supporting cookbook notes or asset references that still describe the OpenRouter path or five-sample limit.

## 3. Verification

- [x] 3.1 Perform a bounded verification of the updated script path and confirm the documented Mistral plus Hugging Face configuration is coherent.
- [x] 3.2 Verify the cookbook page and script describe the same local record fields and remote publication behavior.
