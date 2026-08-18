"""The troubleshooting guide, pinned against the code it documents.

1. Every exception the package defines, every `on_parse_error` value, every
   `unsupported_params` policy and every warning the library emits is named on the page
   (code → docs). An error a reader cannot look up is an error they cannot fix.
2. Every self-contained example executes, against a stubbed provider factory.
3. Every error the page tables is triggered for real, and the page's description of its
   cause is checked against the message the code actually produces.

No test here makes a live LLM call: the only LLM steps are driven by stub served models,
and the one example that builds a real one has its provider factory replaced.
"""

import re
import warnings
from pathlib import Path

import pytest

import datafast
from datafast import (
    Branch,
    Concat,
    Group,
    JoinBranches,
    LLMStep,
    ListSink,
    Map,
    Pipeline,
    Sink,
    Source,
)
from datafast.core.checkpoint import PipelineChangedError
from datafast.core.config import RunConfig
from datafast.core.validation import PipelineValidationError
from datafast.llm.types import UnsupportedParamsPolicy

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "guides" / "troubleshooting.md"


def _page() -> str:
    return PAGE.read_text()


def _code_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", _page(), re.DOTALL)


class StubModel:
    """A served model that answers without a network call."""

    model_id = "stub"

    def __init__(self, answer: str = "answer", boom: Exception | None = None) -> None:
        self.answer = answer
        self.boom = boom
        self.calls = 0

    def generate(self, messages=None, metadata=None, **kwargs) -> str:
        self.calls += 1
        if self.boom is not None:
            raise self.boom
        return self.answer


class CrashingModel(StubModel):
    """Stops the run for real: the runner catches Exception, not BaseException."""

    def __init__(self, crash_at: int) -> None:
        super().__init__()
        self.crash_at = crash_at

    def generate(self, messages=None, metadata=None, **kwargs) -> str:
        self.calls += 1
        if self.calls == self.crash_at:
            raise KeyboardInterrupt("simulated crash")
        return self.answer


def _step(model, **kwargs) -> LLMStep:
    return LLMStep(
        prompt="{text}", input_columns=["text"], output_column="out", model=model, **kwargs
    )


# --- code → docs -----------------------------------------------------------------


def test_every_exception_the_package_defines_is_documented():
    """Two, and a reader hitting either must be able to look it up."""
    defined = set()
    for path in (ROOT / "datafast").rglob("*.py"):
        defined.update(re.findall(r"^class (\w+)\((?:Exception|\w*Error)\)", path.read_text(), re.M))
    assert defined == {"PipelineValidationError", "PipelineChangedError"}, (
        f"the package's exception set changed: {sorted(defined)} — update the page"
    )
    missing = [name for name in defined if f"`{name}`" not in _page()]
    assert not missing, f"exceptions undocumented on the page: {missing}"


def test_the_page_shows_where_each_exception_is_imported_from():
    """They live in different places and only one is on the top-level package."""
    assert not hasattr(datafast, "PipelineValidationError")
    assert hasattr(datafast, "PipelineChangedError")
    assert "from datafast import PipelineChangedError" in _page()
    assert "from datafast.core.validation import PipelineValidationError" in _page()


def test_every_on_parse_error_value_is_documented():
    values = ["skip", "raise"]
    for value in values:
        _step(StubModel(), on_parse_error=value)  # the code accepts it
        assert f'`"{value}"`' in _page(), f"on_parse_error={value!r} undocumented"
    with pytest.raises(ValueError, match="on_parse_error must be"):
        _step(StubModel(), on_parse_error="warn")


def test_every_unsupported_params_policy_is_documented():
    members = [p.value for p in UnsupportedParamsPolicy]
    assert members, "UnsupportedParamsPolicy is empty — check the test, not the page"
    missing = [m for m in members if f'`"{m}"`' not in _page()]
    assert not missing, f"unsupported_params policies undocumented: {missing}"


def test_every_warning_the_library_emits_is_documented():
    """Five warn() sites; the page tables what each one means."""
    sources = [
        (ROOT / "datafast" / "llm" / "served_model.py").read_text(),
        (ROOT / "datafast" / "tracing.py").read_text(),
    ]
    total = sum(s.count("warnings.warn(") for s in sources)
    assert total == 5, f"the library now emits {total} warnings — update the page"

    for fragment in (
        "is not supported by",
        "drop_params=True",
        "does not expose native batching",
        "no declared native schema support",
        "Langfuse",
    ):
        assert fragment in _page(), f"warning {fragment!r} undocumented on the page"


def test_the_checkpoint_every_default_the_page_names_is_the_real_one():
    assert RunConfig().checkpoint_every == 100
    assert "`100`" in _page(), "the page must name the real checkpoint_every default"


# --- the examples ----------------------------------------------------------------


@pytest.fixture
def stub_factory(monkeypatch):
    """Replace the provider factory on the datafast module, which is what the page's
    own `from datafast import openai` resolves to. Nothing reaches a network."""
    monkeypatch.setattr(datafast, "openai", lambda *a, **k: StubModel())
    monkeypatch.setattr(datafast, "anthropic", lambda *a, **k: StubModel())


@pytest.mark.parametrize("block", _code_blocks(), ids=lambda b: b.split("\n")[0][:40])
def test_every_example_executes(block, tmp_path, monkeypatch, stub_factory):
    monkeypatch.chdir(tmp_path)
    exec(compile(block, str(PAGE), "exec"), {})


def test_there_are_examples_to_execute():
    assert len(_code_blocks()) >= 4, "the page lost its examples"


# --- compile(): every message the page tables, triggered for real ----------------


COMPILE_CASES = [
    ("Pipeline is empty.", lambda: Pipeline([])),
    (
        "must start with a source",
        lambda: Map(lambda r: r) >> Sink.list(),
    ),
    (
        "discards upstream records",
        lambda: Source.list([{"a": 1}]) >> Source.list([{"a": 2}]),
    ),
    (
        "comes after the sink at position",
        lambda: Source.list([{"a": 1}]) >> Sink.list() >> Map(lambda r: r),
    ),
    (
        "that are not available",
        lambda: Source.list([{"a": 1}]) >> Group(by="topic") >> Sink.list(),
    ),
    (
        "is never closed by a JoinBranches",
        lambda: Source.list([{"a": 1}])
        >> Branch(x=Map(lambda r: r), y=Map(lambda r: r)),
    ),
    (
        "has no matching Branch",
        lambda: Source.list([{"a": 1}]) >> JoinBranches(),
    ),
    (
        "nesting a Branch inside a branch path is not supported",
        lambda: Source.list([{"a": 1}])
        >> Branch(p=Branch(m=Map(lambda r: r), n=Map(lambda r: r)), q=Map(lambda r: r))
        >> JoinBranches(),
    ),
    (
        "is not allowed inside Branch path",
        lambda: Source.list([{"a": 1}])
        >> Branch(p=Sink.list(), q=Map(lambda r: r))
        >> JoinBranches(),
    ),
    (
        "must start with a source",
        lambda: Concat(Map(lambda r: r), Source.list([{"a": 1}])) >> Sink.list(),
    ),
]


@pytest.mark.parametrize("fragment,build", COMPILE_CASES, ids=[c[0][:34] for c in COMPILE_CASES])
def test_every_documented_compile_error_is_real_and_worded_as_the_page_says(fragment, build):
    with pytest.raises(PipelineValidationError) as caught:
        build().compile()
    assert fragment in str(caught.value)
    assert fragment in _page(), f"the page does not carry the real message {fragment!r}"


def test_a_valid_pipeline_compiles_and_compile_returns_itself():
    """The page's compile() example; chaining is what makes it usable."""
    pipeline = Source.list([{"text": "hello"}]) >> Map(lambda r: r) >> Sink.list()
    assert pipeline.compile() is pipeline


def test_several_sinks_may_be_chained_as_the_page_says():
    (Source.list([{"a": 1}]) >> Sink.list() >> Sink.list()).compile()


def test_compile_stops_checking_columns_after_an_opaque_step():
    """The page says a clean compile() is not proof every column reference is right."""
    pipeline = (
        Source.list([{"a": 1}])
        >> Map(lambda r: r)
        >> Group(by="never-produced")
        >> Sink.list()
    )
    pipeline.compile()  # no error, because Map made the schema unknown


# --- what stops a run and what does not ------------------------------------------


def test_a_failing_llm_call_drops_the_record_and_the_run_finishes():
    """The page's central claim: the quiet failure."""
    model = StubModel(boom=RuntimeError("provider exploded"))
    results = (Source.list([{"text": "a"}, {"text": "b"}]) >> _step(model) >> ListSink()).run()
    assert results == [], "a failed call must not stop the run"


class FailOnce(StubModel):
    """Fails exactly one call, so the blast radius is measurable."""

    def __init__(self, fail_on: int) -> None:
        super().__init__()
        self.fail_on = fail_on

    def generate(self, messages=None, metadata=None, **kwargs) -> str:
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("one transient failure")
        return self.answer


@pytest.mark.parametrize(
    "batch_size,attempted,kept", [(1, 8, 7), (4, 6, 4), (8, 2, 0)]
)
def test_one_failing_call_takes_its_whole_batch_with_it(batch_size, attempted, kept):
    """The page's batch_size table, measured rather than reasoned about."""
    model = FailOnce(fail_on=2)
    results = (
        Source.list([{"text": str(i)} for i in range(8)]) >> _step(model) >> ListSink()
    ).run(batch_size=batch_size)
    assert model.calls == attempted, "records in the group were never attempted"
    assert len(results) == kept
    row = re.search(rf"^\| `{batch_size}`.*$", _page(), re.M)
    assert row, f"the page's table has no row for batch_size={batch_size}"
    cells = [c.strip() for c in row.group(0).strip("|").split("|")]
    assert cells[1:] == [str(attempted), str(kept)], (
        f"the page says {cells[1:]}, the code does {[attempted, kept]}"
    )


def test_on_parse_error_raise_is_ignored_under_run():
    """Documented plainly because it contradicts what the argument's name promises."""
    model = StubModel(boom=RuntimeError("provider exploded"))
    step = _step(model, on_parse_error="raise")
    assert (Source.list([{"text": "a"}]) >> step >> ListSink()).run() == []


def test_on_parse_error_raise_does_raise_under_process():
    step = _step(StubModel(boom=RuntimeError("provider exploded")), on_parse_error="raise")
    with pytest.raises(RuntimeError, match="provider exploded"):
        list(step.process(iter([{"text": "a"}])))


def test_skip_swallows_every_exception_not_only_parse_failures():
    """A timeout and a malformed reply are treated identically."""
    for error in (TimeoutError("timed out"), ConnectionError("no route")):
        results = (
            Source.list([{"text": "a"}]) >> _step(StubModel(boom=error)) >> ListSink()
        ).run()
        assert results == []


def test_an_unparseable_reply_is_dropped_the_same_way():
    step = LLMStep(
        prompt="{text}",
        input_columns=["text"],
        output_columns=["x", "y"],
        model=StubModel(answer="not json"),
        parse_mode="json",
    )
    assert (Source.list([{"text": "a"}]) >> step >> ListSink()).run() == []


def test_a_normal_step_raising_does_stop_the_run():
    def boom(record):
        raise KeyError("topic")

    with pytest.raises(KeyError):
        (Source.list([{"a": 1}]) >> Map(boom) >> ListSink()).run()


def test_a_missing_prompt_placeholder_stops_the_run_before_any_call():
    """The page calls this a loud failure that looks like a quiet one."""
    model = StubModel()
    step = LLMStep(
        prompt="Write about {topic}",
        input_columns=["text"],
        output_column="out",
        model=model,
    )
    with pytest.raises(KeyError, match="topic"):
        (Source.list([{"text": "a"}]) >> step >> ListSink()).run()
    assert model.calls == 0, "the page says nothing has been spent"


# --- checkpoints and resume ------------------------------------------------------


def _pipeline(model=None, extra=False):
    steps = Source.list([{"text": "a"}]) >> Map(lambda r: r)
    if extra:
        steps = steps >> Map(lambda r: r)
    return steps >> ListSink()


def test_a_changed_pipeline_raises_pipeline_changed_error_on_resume(tmp_path):
    directory = str(tmp_path / "ckpt")
    _pipeline().run(checkpoint_dir=directory)
    with pytest.raises(PipelineChangedError) as caught:
        _pipeline(extra=True).run(checkpoint_dir=directory, resume=True)
    assert "Use resume=False to start fresh." in str(caught.value)
    assert "Use resume=False to start fresh." in _page()


def test_without_resume_a_changed_pipeline_clears_the_checkpoint_instead(tmp_path):
    directory = str(tmp_path / "ckpt")
    _pipeline().run(checkpoint_dir=directory)
    assert _pipeline(extra=True).run(checkpoint_dir=directory) == [{"text": "a"}]


def test_the_fingerprint_does_not_see_a_changed_prompt(tmp_path):
    """Why the page says to point a changed pipeline at a fresh directory."""
    directory = str(tmp_path / "ckpt")

    def build(prompt):
        model = StubModel()
        step = LLMStep(
            prompt=prompt, input_columns=["text"], output_column="out", model=model
        )
        return (Source.list([{"text": "a"}]) >> step >> ListSink()), model

    first, _ = build("one {text}")
    first.run(checkpoint_dir=directory)

    second, second_model = build("completely different {text}")
    second.run(checkpoint_dir=directory, resume=True)  # no PipelineChangedError
    assert second_model.calls == 0, "resume reused the old step's records"


RESUME_FROM_CASES = [
    ("resume_from requires checkpoint_dir to be set.", {"resume_from": "Map"}),
    ("requires an existing checkpoint in", {"resume_from": "Map", "empty_dir": True}),
    ("not found. Steps:", {"resume_from": "Nope", "seeded": True}),
]


@pytest.mark.parametrize(
    "fragment,options", RESUME_FROM_CASES, ids=[c[0][:30] for c in RESUME_FROM_CASES]
)
def test_every_documented_resume_from_error_is_real(fragment, options, tmp_path):
    options = dict(options)
    empty = options.pop("empty_dir", False)
    seeded = options.pop("seeded", False)
    if empty:
        options["checkpoint_dir"] = str(tmp_path / "nothing-here")
    if seeded:
        options["checkpoint_dir"] = str(tmp_path / "ckpt")
        _pipeline().run(checkpoint_dir=options["checkpoint_dir"])
        options["resume"] = True

    with pytest.raises(ValueError) as caught:
        _pipeline().run(**options)
    assert fragment in str(caught.value)
    assert fragment in _page(), f"the page does not carry the real message {fragment!r}"


def test_the_step_names_resume_from_wants_are_class_names(tmp_path):
    directory = str(tmp_path / "ckpt")
    _pipeline().run(checkpoint_dir=directory)
    with pytest.raises(ValueError) as caught:
        _pipeline().run(checkpoint_dir=directory, resume=True, resume_from="Nope")
    assert "ListSource, Map, ListSink" in str(caught.value)
    for name in ("ListSource", "LLMStep", "JSONLSink"):
        assert f"`{name}`" in _page()


def _crashing_run(directory, crash_at, checkpoint_every, total=8):
    model = CrashingModel(crash_at)
    pipeline = (
        Source.list([{"text": str(i)} for i in range(total)])
        >> _step(model)
        >> ListSink()
    )
    try:
        pipeline.run(
            checkpoint_dir=directory, batch_size=1, checkpoint_every=checkpoint_every
        )
    except KeyboardInterrupt:
        pass
    return model


def _resumed_run(directory, checkpoint_every, total=8):
    model = StubModel()
    pipeline = (
        Source.list([{"text": str(i)} for i in range(total)])
        >> _step(model)
        >> ListSink()
    )
    return pipeline.run(
        checkpoint_dir=directory,
        resume=True,
        batch_size=1,
        checkpoint_every=checkpoint_every,
    ), model


def test_a_crash_before_the_first_progress_save_loses_every_completed_call(tmp_path):
    """Why the page tells you to lower checkpoint_every on expensive steps."""
    directory = str(tmp_path / "ckpt")
    crashed = _crashing_run(directory, crash_at=6, checkpoint_every=100)
    assert crashed.calls == 6
    assert not list(Path(directory).glob("*.progress.json")), "nothing was recorded"

    results, resumed = _resumed_run(directory, checkpoint_every=100)
    assert resumed.calls == 8, "all eight calls were paid for a second time"
    assert len(results) == 8


def test_resume_duplicates_the_records_completed_since_the_last_progress_save(tmp_path):
    """The page's warning about exact counts, proved rather than paraphrased."""
    directory = str(tmp_path / "ckpt")
    _crashing_run(directory, crash_at=6, checkpoint_every=2)

    results, resumed = _resumed_run(directory, checkpoint_every=2)
    assert resumed.calls == 4, "the four recorded calls were skipped"
    assert len(results) == 9, "eight records in, nine out — one is duplicated"

    duplicated = [r["text"] for r in results if [x["text"] for x in results].count(r["text"]) > 1]
    assert duplicated, "the page says a record appears twice"


# --- provider errors -------------------------------------------------------------


def _served_model(**kwargs):
    from datafast.llm.served_model import ServedModel
    from datafast.llm.types import (
        BatchMode,
        EndpointMode,
        ServedModelCapabilities,
        StructuredOutputMode,
    )

    return ServedModel(
        provider_id="openai",
        model_id="fake",
        litellm_route="openai",
        env_key_name="OPENAI_API_KEY",
        capabilities=ServedModelCapabilities(
            endpoint_modes=frozenset({EndpointMode.CHAT}),
            default_endpoint_mode=EndpointMode.CHAT,
            batch_mode=BatchMode.FALLBACK_CONCURRENCY,
            structured_output=StructuredOutputMode.JSON_SCHEMA,
        ),
        **{"api_key": "not-a-real-key", **kwargs},
    )


@pytest.mark.parametrize(
    "fragment,call",
    [
        ("Either prompt or messages must be provided", lambda m: m.generate()),
        (
            "Provide either prompt or messages, not both",
            lambda m: m.generate(prompt="a", messages=[{"role": "user", "content": "b"}]),
        ),
    ],
    ids=["neither", "both"],
)
def test_the_documented_input_value_errors_are_real(fragment, call):
    with pytest.raises(ValueError) as caught:
        call(_served_model())
    assert fragment in str(caught.value)
    assert fragment in _page()


def test_a_missing_api_key_raises_the_value_error_the_page_quotes(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError) as caught:
        _served_model(api_key=None).generate(prompt="hi")
    message = str(caught.value)
    assert "OPENAI_API_KEY environment variable not set." in message
    assert "OPENAI_API_KEY environment variable not set." in _page()


def test_a_provider_failure_is_wrapped_in_runtime_error_naming_the_provider(monkeypatch):
    import datafast.llm.served_model as served_model_module

    def explode(**params):
        raise Exception("401 incorrect api key")

    monkeypatch.setattr(served_model_module.litellm, "completion", explode)
    with pytest.raises(RuntimeError) as caught:
        _served_model().generate(prompt="hi")
    message = str(caught.value)
    assert message.startswith("Error generating response with openai:")
    assert "401 incorrect api key" in message, "the page says to read from the end"
    assert "Error generating response with <provider>" in _page()


@pytest.mark.parametrize("policy", ["warn", "fail", "quiet"])
def test_each_unsupported_params_policy_behaves_as_the_page_says(policy, monkeypatch):
    import types

    import datafast.llm.served_model as served_model_module

    def fake_completion(**params):
        message = types.SimpleNamespace(
            content="ok",
            reasoning_content=None,
            thinking_blocks=None,
            images=None,
            audio=None,
        )
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message)], model="fake", usage=None
        )

    monkeypatch.setattr(served_model_module.litellm, "completion", fake_completion)

    if policy == "fail":
        with pytest.raises(ValueError, match="is not supported by"):
            _served_model(unsupported_params=policy, thinking=True).generate(prompt="hi")
        return

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _served_model(unsupported_params=policy, thinking=True).generate(prompt="hi")
    raised = [str(w.message) for w in caught if "is not supported by" in str(w.message)]
    assert bool(raised) is (policy == "warn")


def test_the_unsupported_parameter_warning_names_the_wire_parameter(monkeypatch):
    """The page's note: you set `thinking`, the warning says `reasoning_effort`."""
    import types

    import datafast.llm.served_model as served_model_module

    monkeypatch.setattr(
        served_model_module.litellm,
        "completion",
        lambda **params: types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content="ok",
                        reasoning_content=None,
                        thinking_blocks=None,
                        images=None,
                        audio=None,
                    )
                )
            ],
            model="fake",
            usage=None,
        ),
    )
    with pytest.warns(UserWarning, match="reasoning_effort"):
        _served_model(thinking=True).generate(prompt="hi")
    assert "`reasoning_effort`" in _page() and "`thinking=True`" in _page()


def test_only_the_five_documented_error_types_are_retried():
    from datafast.llm.served_model import _is_retryable_error

    import litellm.exceptions as litellm_exceptions

    retryable = (
        litellm_exceptions.RateLimitError,
        litellm_exceptions.APIConnectionError,
        litellm_exceptions.Timeout,
        litellm_exceptions.InternalServerError,
        litellm_exceptions.ServiceUnavailableError,
    )
    for kind in retryable:
        assert _is_retryable_error(kind.__new__(kind))
    for kind in (ValueError, KeyError, PermissionError):
        assert not _is_retryable_error(kind("nope"))
    assert "fails on the first attempt" in _page()


# --- the silent failures ---------------------------------------------------------


def test_a_mistyped_prompt_file_path_becomes_the_prompt(tmp_path, monkeypatch):
    """One of the page's silent failures, with real spend behind it."""
    monkeypatch.chdir(tmp_path)
    seen: list[str] = []

    class Recorder(StubModel):
        def generate(self, messages=None, metadata=None, **kwargs):
            seen.append(messages[-1]["content"])
            return "answer"

    step = LLMStep(
        prompt=Path("prompts/typo.txt"),
        input_columns=["text"],
        output_column="out",
        model=Recorder(),
    )
    (Source.list([{"text": "a"}]) >> step >> ListSink()).run()
    assert seen == ["prompts/typo.txt"], "the path itself was sent to the model"


def test_stop_after_with_an_unknown_name_runs_the_whole_pipeline():
    results = (Source.list([{"a": 1}]) >> Map(lambda r: {**r, "b": 2}) >> ListSink()).run(
        stop_after="does-not-exist"
    )
    assert results == [{"a": 1, "b": 2}], "the page says the name is not checked"


def test_temperature_and_max_tokens_on_a_step_never_reach_the_model():
    seen: list[dict] = []

    class Recorder(StubModel):
        def generate(self, messages=None, metadata=None, **kwargs):
            seen.append(kwargs)
            return "answer"

    step = _step(Recorder(), temperature=0.1, max_tokens=7)
    (Source.list([{"text": "a"}]) >> step >> ListSink()).run()
    assert seen == [{}], "the page says both are ignored"


# --- links -----------------------------------------------------------------------


def test_every_page_linked_to_exists():
    links = re.findall(r"\]\((?!https?:)([^)#]+\.md)", _page())
    assert links, "the page has no links — check the test, not the page"
    missing = sorted(link for link in links if not (PAGE.parent / link).resolve().exists())
    assert not missing, f"links to pages that do not exist: {missing}"
