# LLM Provider Test Plan (Draft)

## Goal

Test the provider redesign without exploding the matrix.

Main idea:

- Test shared behavior once at the common layer.
- Test only provider/model deltas at the capability layer.
- Run a meaningful live suite against selected real models.
- Keep the live suite maintainable through a small curated model catalog.
- Defer multimodal live coverage until after the first stable text-first provider test suite is in place.
- Defer caching coverage until after the first stable text-first provider test suite is in place.

## Test Layers

| Layer | Purpose | Typical tools |
|---|---|---|
| Unit / contract | Validate request normalization, capability resolution, retry logic, batching decisions, parsing, caching decisions | mocked LiteLLM / fake adapters |
| Adapter tests | Verify mapping from Datafast request to LiteLLM request per endpoint mode | mocked `completion()`, `batch_completion()`, `responses()` |
| Live acceptance | Verify selected real models are safe for Datafast users | live API / local server |

## Marker Strategy

Recommended markers:

- `live`: any test hitting a real provider endpoint
- `multimodal`: reserved for later image / audio / document / video coverage
- `ollama`: real Ollama backend
- `vllm`: real vLLM backend
- `llamacpp`: real `llama.cpp` backend

Suggested usage:

- default CI: mocked tests only
- provider CI / pre-release: `-m live`
- targeted local runs: `-m "live and ollama"` / `-m "live and vllm"` / `-m "live and llamacpp"`

## Matrix Reduction Strategy

- Do not test every feature against every provider.
- Run a compact acceptance suite against a curated list of selected models.
- Choose one representative provider/model endpoint per endpoint mode for mocked tests.
- Choose one representative provider/model endpoint per modality for deeper live tests.
- For each provider, test only what is different from the shared contract.
- Keep local-backend tests separate from hosted-provider smoke tests.

## Selected Model Catalog

Maintain one curated list of current supported / recommended test targets per provider.

This catalog should not aim to include every available model. It should be a curated test surface for capability coverage and user confidence, not a registry of all provider inventory.

Each catalog entry should record at least:

- provider
- model_id
- endpoint mode
- hosted vs local
- expected modalities
- expected structured-output support
- expected reasoning / thinking support
- expected batching behavior
- expected cache mechanism type
- test markers to apply, such as `live`, `multimodal`, `ollama`, `vllm`, `llamacpp`

Design goal:

- adding a new model should usually mean adding one catalog entry
- most live tests should parametrize over that catalog
- provider/model-specific regressions should be captured as capability expectations in the catalog

Models are good candidates for the catalog when they are:

- recommended to Datafast users
- used in docs or examples
- representative of a distinct capability shape
- newly added and worth validating before being treated as supported
- known to be tricky or historically unstable

Models are usually not good candidates when they:

- do not add meaningful new capability coverage
- are deprecated or not intended for ongoing support
- are only one of many near-identical variants from the same provider

### Current Catalog Decisions

Current agreed shortlist as of June 2026:

- OpenAI: `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`
- Anthropic: `claude-sonnet-4-6`, `claude-haiku-4-5`
- Gemini: `gemini-2.5-pro`, `gemini-3.5-flash`, `gemini-3.1-flash-lite`
- Mistral hosted: `mistral-medium-3-5`, `mistral-large-2512`, `mistral-small-2603`
- Mistral local / self-hosted: `ministral-14b-2512`, `ministral-8b-2512`, `ministral-3b-2512`

Current exclusions / constraints:

- Exclude Anthropic `claude-fable-5` and `claude-opus-4-8` due to cost.
- Exclude Gemini `gemini-2.5-flash`.
- Keep the catalog curated for capability coverage, not exhaustive by provider inventory.
- Keep hosted Mistral and local Mistral entries separate in the catalog.
- Treat local-server capability expectations as backend-specific, especially for `vLLM`, `llama.cpp`, and other OpenAI-compatible servers.
- If a compact local Mistral subset is needed later, start with `ministral-8b-2512` and `ministral-3b-2512`.

## Live Acceptance Suite

These should run against the curated selected-model catalog.

| ID | Test |
|---|---|
| L01 | Basic text generation works for every selected live model |
| L02 | Structured output works for every selected live model that claims support |
| L03 | Batch request works for every selected live model using the expected execution path, and emits a warning if fallback batching is used |
| L04 | Common params such as `timeout` and `temperature` are accepted or handled according to capability expectations |
| L05 | Declared unsupported params follow `unsupported_params` policy as expected for that model |
| L06 | Endpoint mode matches expectation: chat completions vs Responses API |
| L07 | Provider-specific factory entry point works for that model |
| L08 | Metadata / tracing path does not break live requests |

For local backends, include:

| ID | Test |
|---|---|
| L09 | `api_base_url` path works |
| L10 | no-API-key path works where expected |

## Core Contract Tests

These should run with mocks only.

| ID | Test |
|---|---|
| C01 | Factory functions such as `openai(...)`, `openrouter(...)`, `ollama(...)` create the expected internal target/config shape |
| C02 | Single prompt returns a single result |
| C03 | Batch prompts return ordered list results |
| C04 | `messages` input works for single request |
| C05 | Batched `messages` input works and preserves order |
| C06 | Reject `prompt=None` and `messages=None` |
| C07 | Reject providing both `prompt` and `messages` |
| C08 | Structured output with Pydantic parses successfully |
| C09 | Structured output surfaces a clear validation error on invalid JSON / schema mismatch |
| C10 | Text responses are normalized consistently |
| C11 | Metadata / tracing payload is attached to requests |

## Capability Layer Tests

These should validate the resolved target rules.

| ID | Test |
|---|---|
| K01 | Supported params are forwarded for a target that allows them |
| K02 | Unsupported params are omitted by default when capability is unknown |
| K03 | `unsupported_params="warn"` omits unsupported params and emits a warning |
| K04 | `unsupported_params="fail"` raises a clear error before request dispatch |
| K05 | `unsupported_params="quiet"` omits unsupported params without warning |
| K06 | Provider-specific aliases map correctly to the internal common config |
| K07 | `thinking=False` suppresses `reasoning_effort` |
| K08 | `thinking=True` with no explicit `reasoning_effort` uses target default |
| K09 | Endpoint mode resolves correctly: chat completions vs Responses API |
| K10 | Capability notes such as "accepted but ignored" or "translated internally" are represented correctly |
| K11 | OpenAI-compatible target is not assumed to support all OpenAI features |
| K12 | Local target requiring a chat template is flagged correctly |

## Adapter Tests

These verify the LiteLLM call shape.

| ID | Test |
|---|---|
| A01 | Chat-completions target calls `litellm.completion()` for single input |
| A02 | Native same-target batch calls `litellm.batch_completion()` when supported |
| A03 | If native batching is unavailable, batch input is executed via bounded parallel single requests, preserves ordered batch outputs, and emits a user warning |
| A04 | Responses target calls `litellm.responses()` |
| A05 | Responses target forwards `previous_response_id` when present |
| A06 | Structured output maps to the correct LiteLLM field per endpoint mode |
| A07 | Provider-specific extra params pass only through the escape hatch |
| A08 | `api_base_url` and optional `api_key` are passed correctly for local / self-hosted targets |

## Reliability Tests

| ID | Test |
|---|---|
| R01 | Retryable error triggers bounded retries |
| R02 | Non-retryable error fails immediately |
| R03 | Backoff grows across retries |
| R04 | Jitter is applied within the expected range |
| R05 | Timeout is forwarded and timeout failure is surfaced clearly |
| R06 | Client-side `rpm_limit` throttles before provider error |
| R07 | Batch retry behavior preserves output ordering |

## Multimodal Tests

Multimodal coverage should come later.

For the first rollout:

- keep multimodal tests out of the required live acceptance suite
- allow a small number of mocked multimodal contract tests if useful
- add real multimodal coverage only after the text-first live suite is stable

| ID | Test |
|---|---|
| M01 | Text-only message content remains supported |
| M02 | Image content part is accepted for a target with image input support |
| M03 | Audio content part is accepted for a target with audio input support |
| M04 | Video content part is accepted for a target with video input support |
| M05 | File / document content part is accepted for a target with document support |
| M06 | Unsupported modality is rejected clearly for a text-only target |
| M07 | Mixed text + image multimodal message preserves part order |
| M08 | Stable media ID / UUID is forwarded when provided |
| M09 | Non-text output path is selected correctly for image-generation-capable chat target |

## Caching Tests

Caching coverage should come later.

For the first rollout:

- keep caching tests out of the required live acceptance suite
- allow mocked cache-resolution tests if useful
- add real cache-behavior coverage only after the text-first live suite is stable

| ID | Test |
|---|---|
| H01 | Cache mode resolves correctly for provider-native prompt caching |
| H02 | Cache mode resolves correctly for local prefix / KV caching |
| H03 | Cache key / cache hint changes when model changes |
| H04 | Cache key / cache hint changes when relevant generation params change |
| H05 | Cache key / cache hint changes when multimodal input identity changes |
| H06 | Stable media identity enables multimodal reuse hint when supported |
| H07 | Public API does not claim cache hit semantics that the target cannot guarantee |

## Provider / Model Delta Live Tests

Add only when a selected model has behavior that differs meaningfully from the common suite.

| ID | Example |
|---|---|
| D01 | Responses-only reasoning model |
| D02 | OpenRouter model with provider-specific capability caveat |
| D03 | vLLM deployment with structured-output expectations |
| D04 | `llama.cpp` target with chat-template requirement |
| D05 | model with unusual unsupported-param behavior expectations |
| D06 | multimodal model with image input support |
| D07 | cache-relevant local backend behavior |

## Extended Live Scenarios

These are later-phase tests, not required for the initial rollout.

| ID | Target | Test |
|---|---|---|
| E01 | Multimodal hosted model | text + image input |
| E02 | Audio or document-capable model | real multimodal request |
| E03 | Structured-output target | real Pydantic schema validation |
| E04 | Provider with prompt caching | repeated request with cache-relevant setup |
| E05 | vLLM | prefix-cache-friendly repeated prompt |
| E06 | local multimodal target | document or image input if supported |
| E07 | Responses target | `previous_response_id` continuation |
| E08 | selected-model sweep | run the full acceptance suite across the full catalog |

## New Model Onboarding

When a new model comes out:

1. Add it to the selected-model catalog with expected capabilities.
2. Run the shared live acceptance suite against it.
3. Add a provider/model delta test only if it differs from the standard expectations.
4. Add an extended live scenario only if it adds meaningful new capability coverage.

## Suggested Priorities

- Phase 1: `C*`, `K*`, `A*`, `R*`
- Phase 2: selected-model `L*` live suite
- Phase 3: `M*`, `H*`, `D*`
- Phase 4: `E*`

## Success Criteria

- Shared behavior is covered mostly by fast mocked tests.
- The curated live suite gives confidence against real provider endpoints.
- Provider/model-specific logic is tested as deltas, not full re-runs of the whole matrix.
- Adding a new model is mostly a catalog update plus, if needed, one delta test.
