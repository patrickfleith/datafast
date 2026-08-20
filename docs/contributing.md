# Contributing

How to set datafast up for development, run the tests, build the documentation, and add
the two things people most often want to add: a served model and a step.

## Set up

```bash
git clone https://github.com/patrickfleith/datafast.git
```

Make a virtual environment, then install the package in editable mode with the `dev` and
`docs` extras:

```bash
pip install -e ".[dev,docs]"
```

`dev` brings `pytest`, `ruff`, and the `parquet` and `hub` extras, so the whole test
suite can run. `docs` brings the site generator. Datafast needs Python 3.10 or newer.

### Keys

Provider API keys are read from the environment, and datafast loads a `.env` file in the
working directory automatically. See [Installation](installation.md) for the variable
names. You need no key at all to run the default test command, and no key to run
examples 01 to 14 — see the [examples index](cookbook/examples.md).

## Layout

| Path | What is in it |
|---|---|
| `datafast/core/` | `Step`, `Pipeline`, the runner, `RunConfig`, checkpointing, validation |
| `datafast/sources/` | `Source` and `Seed` |
| `datafast/transforms/` | every step between the source and the sink |
| `datafast/sinks/` | `Sink` and the five sink classes |
| `datafast/llm/` | `ServedModel`, the provider factories, capability profiles, parsing |
| `tests/` | the mocked suite — no network |
| `tests/live/` | the live suite — one directory per provider |
| `docs/` | the published site |
| `examples/scripts/` | 45 numbered runnable scripts |
| `examples/providers/` | per-provider scripts for calling a served model directly |
| `docs-agents/` | internal project notes, not published |

## Tests

There are two layers, and the difference matters because one of them spends money.

**`tests/`** is mocked. It never opens a socket, and it covers the library itself.

**`tests/live/`** hits real provider endpoints. Every module there is marked `live`, and
the root `tests/conftest.py` skips every `live` test unless you pass `--run-live`.

### The commands

`pytest` is usually not on your `PATH` after an editable install into a virtual
environment, so the commands below call it from the environment directly.

The default command, and the one to run before opening a pull request:

```bash
.venv/bin/pytest -m "not live"
```

One file, which is what you want while writing a test:

```bash
.venv/bin/pytest tests/test_reference_sinks.py
```

The live suite, which you must ask for:

```bash
.venv/bin/pytest --run-live -m "anthropic"
```

### What each form does

Running with no arguments is already safe — the gate skips the live tests rather than
letting them through. `-m "not live"` is still the better default because it removes
them from the run entirely instead of reporting a wall of skips.

| Command | Live tests |
|---|---|
| `.venv/bin/pytest` | collected, then **skipped** |
| `.venv/bin/pytest -m "not live"` | **deselected** — never collected |
| `.venv/bin/pytest --run-live` | **run**, against real endpoints |

### Markers

`pytest.ini` registers these, and every one of them is carried by real tests:

| Marker | Selects |
|---|---|
| `live` | tests that hit a real provider endpoint |
| `multimodal` | tests that exercise multimodal provider behavior |
| `anthropic` | tests that require a real Anthropic API key |
| `openai` | tests that require a real OpenAI API key |
| `mistral` | tests that require a real Mistral API key |
| `gemini` | tests that require a real Gemini API key |
| `openrouter` | tests that require a real OpenRouter API key |
| `ollama` | tests that require a real Ollama backend |

A live module carries two: `pytestmark = [pytest.mark.live, pytest.mark.anthropic]`. The
first one arms the gate, the second lets you run one provider at a time.

The gate matches on pytest's keywords, and keywords include parametrize ids. A test
parametrized with the string `"live"` is skipped by the gate as well, even in the mocked
suite. Name that parameter something else.

### Live tests skip themselves

`--run-live` says you are willing to make real calls. It does not promise you can. Each
live test also guards on what it needs, so an incomplete `.env` produces skips rather
than failures:

- `require_api_key("ANTHROPIC_API_KEY")` skips when the variable is unset.
- `require_ollama("qwen3:0.6b")` skips when no daemon answers, and skips again when the
  daemon is up but that model is not pulled.

Both fixtures live in `tests/live/conftest.py`. `require_ollama` is there rather than in
`tests/live/ollama/` because the root-level pipeline test needs the same guard, and two
copies could disagree about which host to check.

### Where pytest is configured

`pytest.ini`, and nowhere else. `pyproject.toml` carries no `[tool.pytest.ini_options]`
block: `pytest.ini` would win over it anyway, and pytest warns on every run when both
exist.

## Documentation

The site is built by [Zensical](https://zensical.org), which reads `mkdocs.yml`.

Build it the way CI does, which fails on a link to a page that does not exist:

```bash
.venv/bin/zensical build --strict
```

Preview it while writing:

```bash
.venv/bin/zensical serve
```

### Every page is pinned by a test

Documentation here is not written and then trusted. Most pages under `docs/` carry a
test file in `tests/` that fails when the page and the code disagree.
`docs/reference/sinks.md` is checked by `tests/test_reference_sinks.py`, and so on.
A page test asserts four things, in this order:

1. **Code → docs.** For every callable the page documents, every parameter
   `inspect.signature` reports appears on the page in backticks. A parameter that exists
   and is undocumented cannot be discovered.
2. **Every self-contained example runs.**
3. **Every behavioural claim is proved by a real call**, not paraphrased from the source.
4. **Every relative link resolves.**

No test may pass vacuously, so each one asserts that the introspection found something
before checking it. `docs/reference/sources_and_seed.md` and
`tests/test_reference_sources_and_seed.py` are the template to copy.

If you change a signature, the page test tells you which page to update.

## Add a served model

A provider is more than a name — it is a wire route, a key, and a set of capabilities
that decide which parameters are allowed through. All of it is declared up front rather
than discovered at call time.

1. **Declare a capability profile** in `datafast/llm/capabilities.py`. A profile is a
   `ServedModelCapabilities` record: endpoint modes, supported parameters, modalities,
   structured output mode, batch mode. Reuse an existing one if the behaviour matches —
   `MISTRAL_CHAT` is `HOSTED_CHAT` with one field replaced.
2. **Make it reachable.** Add the pair to `_SERVED_MODEL_CATALOG` for one specific
   model, or to `_PROVIDER_DEFAULTS` for every model on that provider. A provider that
   needs per-model logic gets a `_resolve_<provider>_capabilities` function instead, as
   OpenAI, Mistral and Ollama do.
3. **Subclass `ServedModel`** in `datafast/llm/served_model.py`, passing the provider
   id, the `litellm_route` and the `env_key_name`. A local backend passes
   `env_key_name=None`.
4. **Add the provider factory** — a module-level function taking `model_id` and
   `**kwargs`, with a default model id. This is the public way to build a served model.
5. **Export it** from `datafast/__init__.py`, in the imports and in `__all__`.
6. **Add a live suite** at `tests/live/<provider>/`, with a `conftest.py` holding a
   `served_model` fixture that calls `require_api_key`, and modules marked
   `[pytest.mark.live, pytest.mark.<provider>]`. Register the new marker in `pytest.ini`.
7. **Write the reference page** at `docs/reference/providers/<provider>.md` and its test.

Keep the live suite cheap. Pick the smallest model that exercises everything the profile
declares, and make the assertion depend on datafast rather than on the model being
clever.

## Add a step

A step is any object that takes records and yields records. The base class asks for one
method:

```python
from collections.abc import Iterable

from datafast import Step
from datafast.core.types import Record


class Shout(Step):
    def __init__(self, column: str) -> None:
        super().__init__()
        self._column = column
        self._input_columns = [column]

    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        for record in records:
            yield {**record, self._column: record[self._column].upper()}
```

That is enough to use it:

```python
from datafast import Source

pipeline = Source.list([{"text": "hi"}]) >> Shout("text")
records = pipeline.run()
```

Three things happen for free. `>>` works, because `Step` defines it. The step is
checkpointed under its class name — `step_001_Shout.jsonl` — and `as_step("shout")`
renames both the step and its checkpoint file. And `process` is a generator, so records
stream through it.

### Make it visible to `compile()`

Setting `_input_columns` is what lets [`compile()`](guides/pipelines_and_execution.md)
catch a missing column before the run starts:

```
Step 'Shout' references column(s) ['text'] that are not available.
Available columns: ['a'].
```

Leave it off and the step is silent at compile time. There is a cost either way: a step
the validator does not recognise makes the schema **unknown from that point on**, so
column checks stop for every step after it. That is deliberate — guessing would report
missing columns that are really there — but it means a custom step in the middle of a
pipeline turns off validation downstream of itself.

If your step is close to an existing one, prefer configuring that one. `Map` and
`FlatMap` already cover most per-record reshaping.

## Style and checks

Match the code around you. That rule outranks any linter here.

`ruff` ships in the `dev` extra and `pyproject.toml` configures it with an 88-character
line length. It is not a gate: the tree does not pass `ruff check` today, and no
workflow runs it. Do not reformat files you are not otherwise changing — a formatting
sweep buries the change you actually made.

`.github/workflows/tests.yml` runs `pytest -m "not live"` on every pull request, on
Python 3.10 and 3.13. `publish.yml` calls the same job and releases nothing unless it
passes. Run the suite locally anyway — a red pull request is a slower way to learn the
same thing.

## Open a pull request

- Run `.venv/bin/pytest -m "not live"`.
- Run `.venv/bin/zensical build --strict` if you touched `docs/`.
- Add or update the page test for any documentation you changed.
- Say in the description whether you ran the live suite, and against which provider.

Bugs and feature requests go to the
[issue tracker](https://github.com/patrickfleith/datafast/issues); questions go to
[discussions](https://github.com/patrickfleith/datafast/discussions).
