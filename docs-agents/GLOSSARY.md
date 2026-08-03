# Glossary

**Capabilities** — What a served model can actually do: the intersection of the model's own traits and the provider's features, declared by Datafast rather than probed at runtime. Resolved once, when the served model is constructed. _avoid:_ features, support matrix

**Capability profile** — A named, reusable Capabilities record shared by served models that behave alike. _avoid:_ preset, template

**Model** — The LLM itself, the weights being served. Not the object you call — that's a served model. _avoid:_ engine

**Parse mode** — The step-level choice of how one raw LLM response is split into named dataset columns (text, json, xml). Distinct from provider-level validation of a whole response against a Pydantic schema. _avoid:_ output format, parser mode

**Provider** — The server that serves LLMs, whether cloud (OpenAI, Anthropic, Mistral, OpenRouter) or local (Ollama, vLLM, llama.cpp). It serves one or many models, and is not the same thing as the wire protocol used to reach it. _avoid:_ backend, vendor, host

**Served model** — A provider and a model together, with its configuration: the thing you construct and call. _avoid:_ target, configured target, deployment, model instance

**Served-model catalog** — The table mapping known provider-and-model pairs to their capabilities, with per-provider and per-model-family fallbacks for pairs it doesn't list. _avoid:_ registry, model list

**Target** — Reserved for the pipeline sense: what a transformation aims at, such as a target audience, target length, or target column. Never the provider-plus-model pair, which is a served model. _avoid:_ (as a synonym for served model)

**Transport** — The wire protocol and route used to reach a provider: chat completions versus the Responses API, plus the routing prefix handed to LiteLLM. Independent of the provider, so an OpenAI-shaped transport does not imply OpenAI is the provider. _avoid:_ provider, backend, endpoint
