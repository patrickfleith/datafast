"""The pipelines & execution guide, pinned against the code it documents.

Checked in order of how much it matters:

1. Every run control the code accepts is named on the page (code → docs). A control
   that exists and is undocumented cannot be discovered.
2. Every self-contained example executes; every snippet at least parses.
3. The behavioural claims — the ones a reader would be burned by if wrong — are
   asserted against real runs, not paraphrased from the source.

No test here makes a live LLM call: the only LLM step used is driven by a stub served
model, and the one example that builds a real served model only constructs it.
"""

import inspect
import re
from dataclasses import fields
from pathlib import Path

import pytest

from datafast import Map, Pipeline, Source
from datafast.core.checkpoint import PipelineChangedError, compute_pipeline_hash
from datafast.core.config import LLMExecutionStrategy, RunConfig
from datafast.core.runner import Runner, _step_signature, run_pipeline
from datafast.core.validation import PipelineValidationError
from datafast.transforms.llm_step import LLMStep

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "guides" / "pipelines_and_execution.md"


def _page() -> str:
    return PAGE.read_text()


def _code_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", _page(), re.DOTALL)


class StubModel:
    """A served model that answers without a network call."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def generate(self, messages=None, metadata=None) -> str:
        return f"{self.model_id}:{messages[-1]['content']}"


def _hash_of(pipeline: Pipeline) -> str:
    return compute_pipeline_hash(
        [s.name for s in pipeline.steps],
        [_step_signature(s) for s in pipeline.steps],
    )


# --- code → docs -----------------------------------------------------------------


def test_every_run_config_field_is_documented():
    """RunConfig is the whole surface of run(); a field it omits is undiscoverable."""
    names = [f.name for f in fields(RunConfig)]
    assert names, "RunConfig has no fields — check the test, not the page"
    missing = [n for n in names if f"`{n}`" not in _page()]
    assert not missing, f"RunConfig fields undocumented on the page: {missing}"


@pytest.mark.parametrize("func", [Pipeline.run, run_pipeline], ids=lambda f: f.__name__)
def test_every_run_parameter_is_documented(func):
    parameters = [
        p.name
        for p in inspect.signature(func).parameters.values()
        if p.kind is not p.VAR_KEYWORD and p.name not in ("self", "pipeline")
    ]
    assert parameters, f"{func.__name__} takes nothing — check the test, not the page"
    missing = [p for p in parameters if f"`{p}`" not in _page()]
    assert not missing, f"{func.__name__} takes {missing}, undocumented on the page"


def test_run_and_run_pipeline_take_the_same_controls():
    """The page says run() is a thin wrapper over run_pipeline()."""
    run_params = set(inspect.signature(Pipeline.run).parameters) - {"self"}
    fn_params = set(inspect.signature(run_pipeline).parameters) - {"pipeline"}
    assert run_params == fn_params


def test_documented_defaults_match_run_config():
    """The run-controls table prints a default for each field; none may drift."""
    documented = {
        "checkpoint_dir": "None", "resume": "False", "resume_from": "None",
        "stop_after": "None", "limit": "None", "batch_size": "4",
        "llm_strategy": '"by_model"', "checkpoint_every": "100",
    }
    assert documented.keys() == {f.name for f in fields(RunConfig)}
    for field in fields(RunConfig):
        assert repr(field.default) == documented[field.name].replace('"', "'"), (
            f"{field.name} defaults to {field.default!r}, page says "
            f"{documented[field.name]}"
        )
        # `str \| None` in the type column escapes its pipe; drop those first.
        table = _page().replace(r"\|", "")
        row = re.search(rf"\|\s*`{field.name}`\s*\|[^|]*\|\s*`([^`]+)`\s*\|", table)
        assert row, f"{field.name} has no row in the run-controls table"
        assert row.group(1) == documented[field.name]


def test_every_execution_strategy_is_documented():
    values = [s.value for s in LLMExecutionStrategy]
    assert values
    missing = [v for v in values if f"`{v}`" not in _page()]
    assert not missing, f"strategies undocumented: {missing}"


def test_the_documented_exception_names_are_the_real_ones():
    for name in ("PipelineValidationError", "PipelineChangedError"):
        assert f"`{name}`" in _page()


# --- examples --------------------------------------------------------------------


@pytest.mark.parametrize("block", _code_blocks(), ids=lambda b: b.split("\n")[0][:40])
def test_every_example_is_valid_python(block):
    """Snippets that assume a pipeline in scope still have to parse."""
    compile(block, str(PAGE), "exec")


@pytest.mark.parametrize(
    "block",
    [b for b in _code_blocks() if "from datafast import" in b],
    ids=lambda b: b.split("\n")[0][:40],
)
def test_every_self_contained_example_executes(block, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exec(compile(block, str(PAGE), "exec"), {})


def test_the_page_has_examples_of_both_kinds():
    """Guards the two tests above against passing on an empty list."""
    blocks = _code_blocks()
    assert sum("from datafast import" in b for b in blocks) >= 4
    assert len(blocks) >= 8


def test_every_relative_link_resolves():
    links = re.findall(r"\]\((\.\./[^)#]+\.md|[a-z_]+\.md)\)", _page())
    assert links, "the page should link somewhere"
    for link in links:
        assert (PAGE.parent / link).resolve().exists(), f"broken link: {link}"


# --- behaviour -------------------------------------------------------------------


@pytest.mark.parametrize(
    "build,fragment",
    [
        (lambda: Map(lambda r: r) >> Map(lambda r: r), "must start with a source"),
        (
            lambda: Source.list([{"a": 1}]) >> Source.list([{"b": 2}]),
            "a source may only be the first step",
        ),
    ],
    ids=["no source", "second source"],
)
def test_compile_rejects_what_the_page_says_it_rejects(build, fragment):
    with pytest.raises(PipelineValidationError, match=re.escape(fragment)):
        build().compile()
    assert fragment in _page(), "the page must quote the message it promises"


def test_compile_reports_a_missing_column_with_the_columns_it_has():
    pipeline = Source.list([{"a": 1}]) >> LLMStep(
        prompt="{b}", input_columns=["b"], output_column="out", model=StubModel("stub")
    )
    with pytest.raises(PipelineValidationError) as excinfo:
        pipeline.compile()
    assert "references column(s) ['b'] that are not available" in str(excinfo.value)
    assert "Available columns: ['a']" in str(excinfo.value)


def test_compile_returns_the_pipeline_so_it_can_be_chained():
    pipeline = Source.list([{"a": 1}]) >> Map(lambda r: r)
    assert pipeline.compile() is pipeline


def test_chained_pipelines_are_flattened_not_nested():
    """The page prints 3 for this pipeline."""
    prepare = Source.list([{"text": "hello"}]) >> Map(lambda r: {**r, "n": 1})
    finish = Map(lambda r: {**r, "n": r["n"] + 1})
    assert len((prepare >> finish).steps) == 3


def test_an_unnamed_step_is_named_after_its_class():
    assert Map(lambda r: r).name == "Map"
    assert Map(lambda r: r).as_step("clean_text").name == "clean_text"


def test_limit_truncates_after_the_source_has_read_everything():
    read = []
    source = Source.list([{"n": i} for i in range(10)])
    pipeline = source >> Map(lambda r: (read.append(r), r)[1])
    assert len(pipeline.run(limit=3)) == 3
    assert len(read) == 3, "limit truncates before the next step sees the records"


def test_stop_after_accepts_an_index_and_a_name():
    pipeline = (
        Source.list([{"n": 1}])
        >> Map(lambda r: {**r, "b": 2}).as_step("clean_text")
        >> Map(lambda r: {**r, "c": 3})
    )
    assert pipeline.run(stop_after="clean_text") == [{"n": 1, "b": 2}]
    assert pipeline.run(stop_after=0) == [{"n": 1}]


def test_an_unknown_run_keyword_raises_rather_than_being_ignored():
    pipeline = Source.list([{"n": 1}]) >> Map(lambda r: r)
    with pytest.raises(TypeError, match="checkpoint_evry|unexpected keyword"):
        pipeline.run(checkpoint_evry=1)


def test_checkpoint_every_reaches_run_config_through_kwargs(tmp_path):
    pipeline = Source.list([{"n": 1}]) >> Map(lambda r: r)
    assert pipeline.run(checkpoint_dir=str(tmp_path), checkpoint_every=1) == [{"n": 1}]


@pytest.mark.parametrize(
    "strategy,expected",
    [
        ("by_model", ["a", "a", "a", "b", "b", "b"]),
        ("round_robin", ["a", "b", "a", "b", "a", "b"]),
        ("by_record", ["a", "b", "a", "b", "a", "b"]),
    ],
)
def test_the_strategy_decides_the_order_records_come_out_in(strategy, expected):
    pipeline = Source.list([{"t": str(i)} for i in range(3)]) >> LLMStep(
        prompt="{t}",
        input_columns=["t"],
        output_column="out",
        model=[StubModel("a"), StubModel("b")],
    )
    records = pipeline.run(llm_strategy=strategy, batch_size=10)
    assert [r["_model"] for r in records] == expected


def test_checkpoint_files_are_named_by_index_and_step_name(tmp_path):
    pipeline = (
        Source.list([{"n": 1}])
        >> Map(lambda r: r)
        >> Map(lambda r: r).as_step("clean_text")
    )
    pipeline.run(checkpoint_dir=str(tmp_path))
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "manifest.json",
        "step_000_ListSource.jsonl",
        "step_001_Map.jsonl",
        "step_002_clean_text.jsonl",
    ]


def test_the_manifest_records_status_and_counts_per_step(tmp_path):
    import json

    pipeline = Source.list([{"n": 1}]) >> Map(lambda r: r)
    pipeline.run(checkpoint_dir=str(tmp_path))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert set(manifest) >= {"pipeline_hash", "steps", "current_step", "config"}
    assert [s["status"] for s in manifest["steps"]] == ["complete", "complete"]
    assert manifest["steps"][1] == {
        "index": 1, "name": "Map", "status": "complete",
        "records_in": 1, "records_out": 1,
    }


def test_resume_reuses_completed_steps_instead_of_rerunning_them(tmp_path):
    ran = []
    pipeline = Source.list([{"n": 1}]) >> Map(lambda r: (ran.append(1), r)[1])
    pipeline.run(checkpoint_dir=str(tmp_path))
    assert pipeline.run(checkpoint_dir=str(tmp_path), resume=True) == [{"n": 1}]
    assert len(ran) == 1, "a complete checkpoint should be returned, not recomputed"


def test_resume_from_reruns_that_step_and_reuses_the_ones_before(tmp_path):
    ran = []
    pipeline = (
        Source.list([{"n": 1}])
        >> Map(lambda r: (ran.append("first"), r)[1]).as_step("first")
        >> Map(lambda r: (ran.append("second"), r)[1]).as_step("second")
    )
    pipeline.run(checkpoint_dir=str(tmp_path))
    ran.clear()
    pipeline.run(checkpoint_dir=str(tmp_path), resume_from="second")
    assert ran == ["second"]


def test_resume_from_needs_a_checkpoint_dir_and_a_known_step_name(tmp_path):
    pipeline = Source.list([{"n": 1}]) >> Map(lambda r: r).as_step("only")
    with pytest.raises(ValueError, match="requires checkpoint_dir"):
        pipeline.run(resume_from="only")
    pipeline.run(checkpoint_dir=str(tmp_path))
    with pytest.raises(ValueError, match="not found. Steps: ListSource, only"):
        pipeline.run(checkpoint_dir=str(tmp_path), resume_from="nope")


def test_a_changed_pipeline_raises_on_resume_and_starts_fresh_without_it(tmp_path):
    original = Source.list([{"n": 1}]) >> Map(lambda r: r)
    original.run(checkpoint_dir=str(tmp_path))

    changed = Source.list([{"n": 1}]) >> Map(lambda r: r) >> Map(lambda r: r)
    with pytest.raises(PipelineChangedError):
        changed.run(checkpoint_dir=str(tmp_path), resume=True)

    changed.run(checkpoint_dir=str(tmp_path))
    assert (tmp_path / "step_002_Map.jsonl").exists()


def test_the_fingerprint_sees_structure_not_behaviour():
    """The page's determinism warning rests on exactly this."""
    same_shape = [
        Source.list([{"n": 1}]) >> Map(lambda r: {**r, "b": 2}),
        Source.list([{"n": 999}]) >> Map(lambda r: {**r, "b": "totally different"}),
    ]
    assert _hash_of(same_shape[0]) == _hash_of(same_shape[1])

    renamed = Source.list([{"n": 1}]) >> Map(lambda r: r).as_step("renamed")
    longer = Source.list([{"n": 1}]) >> Map(lambda r: r) >> Map(lambda r: r)
    assert _hash_of(same_shape[0]) != _hash_of(renamed)
    assert _hash_of(same_shape[0]) != _hash_of(longer)


def test_a_pipeline_may_end_in_several_sinks_but_nothing_may_follow_them():
    from datafast import ListSink

    (Source.list([{"a": 1}]) >> ListSink() >> ListSink()).compile()
    with pytest.raises(PipelineValidationError, match="sinks must be the last steps"):
        (Source.list([{"a": 1}]) >> ListSink() >> Map(lambda r: r)).compile()


def test_running_through_a_sink_still_returns_the_records():
    from datafast import ListSink

    assert (Source.list([{"a": 1}]) >> ListSink()).run() == [{"a": 1}]


def test_the_documented_throughput_settings_belong_to_the_served_model(tmp_path):
    """The page's central claim: these live on the model, never on the runner."""
    from datafast.llm.served_model import ServedModel

    on_model = set(inspect.signature(ServedModel.__init__).parameters)
    on_runner = {f.name for f in fields(RunConfig)}
    for name in ("rpm_limit", "max_concurrent", "timeout", "retry_limit"):
        assert name in on_model, f"{name} is not a served-model setting"
        assert name not in on_runner, f"{name} is also a run control — page is wrong"
        assert f"`{name}`" in _page()


def test_the_runner_does_not_compile_but_run_does():
    """The page tells readers to expect exactly this difference."""
    invalid = Pipeline([Map(lambda r: r)])
    with pytest.raises(PipelineValidationError):
        invalid.run()
    assert Runner(invalid, RunConfig()).execute() == []
