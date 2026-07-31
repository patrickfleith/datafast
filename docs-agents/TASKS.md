# Tasks

<!-- Newest to do on top. Check off in place; move to Done when complete. -->

## To do

- [ ] Fix `43_cookbook_persona_generation.py`: it chains two sinks (`Sink.jsonl >> Sink.hub`), which `compile()` rejects — decide whether to split into two runs or let a pipeline end in several sinks
- [ ] Consider supporting nested `Branch` inside a branch path — needs a metadata stack; `compile()` rejects the shape today because the inner branch overwrites the outer `_branch_id`
- [X] Manually test all Gemini example scripts <!-- examples/providers/gemini -->
- [ ] Review the documentation and identify missing blocks before publishing the new version of datafast
- [ ] <task> <!-- optional (context) -->

## Done

- [X] Make sure the roadmap clearly outlines the release of the new version of datafast
- [X] Make a roadmap
- [X] Try out the OpenAI example scripts <!-- examples/providers/openai -->
- [X] Try out the Mistral example scripts <!-- examples/providers/mistral -->
- [X] Try out the Anthropic example scripts <!-- examples/providers/anthropic -->
- [X] Try out the Ollama example script <!-- examples/providers/ollama -->

