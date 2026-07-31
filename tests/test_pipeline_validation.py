import pytest

from datafast import (
    AddUUID,
    Branch,
    Concat,
    Filter,
    Join,
    JoinBranches,
    LLMStep,
    ListSink,
    Map,
    Source,
)
from datafast.core.validation import PipelineValidationError


class _Model:
    model_id = "fake-model"
    provider_name = "fake"

    def generate(self, **kwargs):
        return "ok"


def _llm(input_columns, **kwargs):
    return LLMStep(
        prompt="{topic}",
        input_columns=input_columns,
        output_column="result",
        model=_Model(),
        **kwargs,
    )


def test_valid_pipeline_compiles():
    pipeline = (
        Source.list([{"topic": "x"}]) >> _llm(["topic"]) >> AddUUID() >> ListSink()
    )
    assert pipeline.compile() is pipeline


def test_missing_source_rejected():
    pipeline = Map(lambda r: r) >> ListSink()
    with pytest.raises(PipelineValidationError, match="must start with a source"):
        pipeline.compile()


def test_source_after_first_rejected():
    pipeline = Source.list([{"a": 1}]) >> Source.list([{"b": 2}])
    with pytest.raises(PipelineValidationError, match="discards upstream records"):
        pipeline.compile()


def test_sink_must_be_last():
    pipeline = Source.list([{"a": 1}]) >> ListSink() >> Map(lambda r: r)
    with pytest.raises(PipelineValidationError, match="must be the last step"):
        pipeline.compile()


def test_unclosed_branch_rejected():
    pipeline = Source.list([{"a": 1}]) >> Branch(x=Map(lambda r: r), y=Map(lambda r: r))
    with pytest.raises(PipelineValidationError, match="never closed"):
        pipeline.compile()


def test_join_without_branch_rejected():
    pipeline = Source.list([{"a": 1}]) >> JoinBranches()
    with pytest.raises(PipelineValidationError, match="no matching Branch"):
        pipeline.compile()


def test_unknown_input_column_rejected():
    pipeline = Source.list([{"topic": "x"}]) >> _llm(["nope"])
    with pytest.raises(PipelineValidationError, match=r"references column\(s\) \['nope'\]"):
        pipeline.compile()


def test_forward_column_reference_validated():
    pipeline = Source.list([{"topic": "x"}]) >> _llm(["topic"], forward_columns=["ghost"])
    with pytest.raises(PipelineValidationError, match="ghost"):
        pipeline.compile()


def test_column_check_skipped_after_opaque_step():
    """A Map makes the schema unknown, so downstream refs are not flagged."""
    pipeline = Source.list([{"topic": "x"}]) >> Map(lambda r: r) >> _llm(["nope"])
    assert pipeline.compile() is pipeline


def test_columns_added_upstream_are_available():
    """A column produced by an LLM step satisfies a later step's reference."""
    pipeline = (
        Source.list([{"topic": "x"}])
        >> _llm(["topic"])  # adds "result"
        >> Filter(where={"result": "ok"})
    )
    assert pipeline.compile() is pipeline


# ---------------------------------------------------------------------------
# Branch paths (inherited sub-pipelines)
# ---------------------------------------------------------------------------


def _branch(**paths):
    return Source.list([{"topic": "x"}]) >> Branch(**paths) >> JoinBranches()


def test_valid_branch_paths_compile():
    pipeline = _branch(a=_llm(["topic"]), b=Map(lambda r: r))
    assert pipeline.compile() is pipeline


def test_unknown_column_inside_branch_path_rejected():
    """A path's column references are checked against the branch's input schema."""
    with pytest.raises(PipelineValidationError, match=r"references column\(s\) \['nope'\]"):
        _branch(a=_llm(["nope"]), b=Map(lambda r: r)).compile()


def test_branch_path_error_names_the_path():
    with pytest.raises(PipelineValidationError, match="inside Branch path 'a'"):
        _branch(a=_llm(["nope"]), b=Map(lambda r: r)).compile()


def test_columns_added_before_branch_are_available_in_paths():
    pipeline = (
        Source.list([{"topic": "x"}])
        >> _llm(["topic"])  # adds "result"
        >> Branch(a=Filter(where={"result": "ok"}), b=Map(lambda r: r))
        >> JoinBranches()
    )
    assert pipeline.compile() is pipeline


def test_source_inside_branch_path_rejected():
    with pytest.raises(PipelineValidationError, match="discards upstream records"):
        _branch(a=Source.list([{"topic": "y"}]), b=Map(lambda r: r)).compile()


def test_sink_inside_branch_path_rejected():
    with pytest.raises(PipelineValidationError, match="not allowed inside Branch path"):
        _branch(a=Map(lambda r: r) >> ListSink(), b=Map(lambda r: r)).compile()


def test_multi_step_branch_path_is_validated():
    """Recursion reaches every step of a multi-step path, not just the first."""
    path = Map(lambda r: r) >> AddUUID() >> ListSink()
    with pytest.raises(PipelineValidationError, match="not allowed inside Branch path"):
        _branch(a=path, b=Map(lambda r: r)).compile()


def test_join_branches_inside_branch_path_rejected():
    with pytest.raises(PipelineValidationError, match="no matching Branch"):
        _branch(
            a=Map(lambda r: r) >> JoinBranches(),
            b=Map(lambda r: r),
        ).compile()


def test_nested_branch_rejected():
    """A Branch inside a branch path clobbers the outer branch metadata."""
    inner = Branch(x=Map(lambda r: r), y=Map(lambda r: r)) >> JoinBranches()
    with pytest.raises(PipelineValidationError, match="nesting a Branch"):
        _branch(a=inner, b=Map(lambda r: r)).compile()


# ---------------------------------------------------------------------------
# Concat sources and Join right sides (sourced sub-pipelines)
# ---------------------------------------------------------------------------


def test_valid_concat_compiles():
    pipeline = (
        Concat(
            Source.list([{"topic": "x"}]) >> _llm(["topic"]),
            Source.list([{"topic": "y"}]),
        )
        >> ListSink()
    )
    assert pipeline.compile() is pipeline


def test_concat_source_without_source_rejected():
    pipeline = Concat(Map(lambda r: r), Source.list([{"topic": "x"}])) >> ListSink()
    with pytest.raises(PipelineValidationError, match="Concat source 0 must start with a source"):
        pipeline.compile()


def test_unknown_column_inside_concat_source_rejected():
    pipeline = Concat(Source.list([{"topic": "x"}]) >> _llm(["nope"])) >> ListSink()
    with pytest.raises(PipelineValidationError, match="inside Concat source 0"):
        pipeline.compile()


def test_sink_inside_concat_source_rejected():
    pipeline = Concat(Source.list([{"topic": "x"}]) >> ListSink()) >> ListSink()
    with pytest.raises(PipelineValidationError, match="not allowed inside Concat source 0"):
        pipeline.compile()


def test_valid_join_compiles():
    pipeline = (
        Source.list([{"topic": "x", "key": 1}])
        >> Join(Source.list([{"key": 1, "extra": "e"}]), on="key")
        >> ListSink()
    )
    assert pipeline.compile() is pipeline


def test_join_key_missing_on_left_rejected():
    pipeline = Source.list([{"topic": "x"}]) >> Join(
        Source.list([{"key": 1}]), on="key"
    )
    with pytest.raises(PipelineValidationError, match=r"references column\(s\) \['key'\]"):
        pipeline.compile()


def test_join_right_side_without_source_rejected():
    pipeline = Source.list([{"key": 1}]) >> Join(Map(lambda r: r), on="key")
    with pytest.raises(PipelineValidationError, match="Join right side .* must start with a source"):
        pipeline.compile()


def test_unknown_column_inside_join_right_side_rejected():
    right = Source.list([{"other": 1}]) >> _llm(["nope"])
    pipeline = Source.list([{"key": 1}]) >> Join(right, on="key")
    with pytest.raises(PipelineValidationError, match="inside Join right side"):
        pipeline.compile()
