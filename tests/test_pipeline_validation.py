import pytest

from datafast import (
    AddUUID,
    Branch,
    Filter,
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
