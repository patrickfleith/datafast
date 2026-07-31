"""Tests for Branch execution through the runner (batching, resume)."""

import pytest

from datafast import Branch, JoinBranches, LLMStep, ListSink, Map, Source
from datafast.core.checkpoint import PipelineChangedError


class RecordingModel:
    """Model that records the batches the runner hands it."""

    provider_name = "fake"

    def __init__(self, model_id: str = "fake-model") -> None:
        self.model_id = model_id
        self.batches: list[int] = []
        self.calls = 0

    def generate(self, prompt=None, messages=None, metadata=None, response_format=None):
        self.calls += 1
        return f"{self.model_id}-out{self.calls}"

    def generate_batch(self, messages_list, metadata=None, response_format=None):
        self.batches.append(len(messages_list))
        return [
            self.generate(messages=messages) for messages in messages_list
        ]


class _Crash(BaseException):
    """Non-Exception so the runner's per-call `except Exception` doesn't swallow it."""


class CrashOnNthModel:
    provider_name = "fake"
    model_id = "crash-model"

    def __init__(self, crash_on: int) -> None:
        self.crash_on = crash_on
        self.calls = 0

    def generate(self, prompt=None, messages=None, metadata=None, response_format=None):
        self.calls += 1
        if self.calls == self.crash_on:
            raise _Crash("simulated crash")
        return f"ok{self.calls}"


def _branch_pipeline(model, sink=None):
    return (
        Source.list([{"question": f"q{i}"} for i in range(4)])
        >> Branch(
            chosen=LLMStep(
                prompt="Expert answer: {question}",
                input_columns=["question"],
                output_column="response",
                model=model,
            ),
            rejected=LLMStep(
                prompt="Brief answer: {question}",
                input_columns=["question"],
                output_column="response",
                model=model,
            ),
        ).as_step("branch")
        >> JoinBranches()
        >> (sink or ListSink())
    )


def test_branch_llm_steps_are_batched_by_the_runner():
    """Nested LLM steps go through the runner's batching, not step.process."""
    model = RecordingModel()

    output = _branch_pipeline(model).run(batch_size=4)

    assert model.calls == 8  # 4 records x 2 paths
    assert model.batches == [4, 4]  # one runner batch per path, not per record
    assert len(output) == 4
    assert set(output[0]) == {"question", "response_chosen", "response_rejected",
                              "_model_chosen", "_model_rejected"}


def test_branch_output_matches_direct_process():
    """Runner-driven Branch produces the same records as Branch.process."""
    via_runner = _branch_pipeline(RecordingModel()).run()

    pipeline = _branch_pipeline(RecordingModel())
    # Bypass the runner: run every step through plain process().
    records = list(pipeline.process(iter([])))

    assert via_runner == records


def test_branch_llm_strategy_groups_calls_across_models():
    """A multi-model nested LLM step is reordered by the runner's strategy."""
    m1, m2 = RecordingModel("m1"), RecordingModel("m2")

    pipeline = (
        Source.list([{"question": f"q{i}"} for i in range(2)])
        >> Branch(
            a=LLMStep(
                prompt="A: {question}",
                input_columns=["question"],
                output_column="response",
                model=[m1, m2],
            ),
            b=Map(lambda r: {**r, "response": "static"}),
        )
        >> JoinBranches()
        >> ListSink()
    )
    pipeline.run(batch_size=2, llm_strategy="by_model")

    # by_model puts both m1 calls in the first batch, both m2 calls in the second.
    assert m1.batches == [2]
    assert m2.batches == [2]


def test_branch_resume_skips_completed_calls(tmp_path):
    """After a crash inside a Branch path, resume re-runs only unfinished calls."""
    model = CrashOnNthModel(crash_on=6)
    ckpt = str(tmp_path / "ckpt")

    with pytest.raises(_Crash):
        _branch_pipeline(model).run(
            checkpoint_dir=ckpt, batch_size=1, checkpoint_every=1
        )
    # Path 'chosen' fully done (4 calls), 'rejected' did 1 then crashed on the 2nd.
    assert model.calls == 6

    output = _branch_pipeline(model).run(
        checkpoint_dir=ckpt, resume=True, batch_size=1, checkpoint_every=1
    )

    assert model.calls == 9  # only the 3 remaining 'rejected' calls ran
    assert len(output) == 4


def test_branch_resume_reuses_a_completed_path(tmp_path):
    """A path that finished before the crash is not re-run on resume."""
    model = CrashOnNthModel(crash_on=5)
    ckpt = str(tmp_path / "ckpt")

    with pytest.raises(_Crash):
        _branch_pipeline(model).run(
            checkpoint_dir=ckpt, batch_size=1, checkpoint_every=1
        )
    assert model.calls == 5  # 'chosen' complete, crash on the first 'rejected' call

    _branch_pipeline(model).run(
        checkpoint_dir=ckpt, resume=True, batch_size=1, checkpoint_every=1
    )

    assert model.calls == 9  # the 4 'chosen' calls were reused, 4 'rejected' ran


def test_branch_nested_pipeline_path_is_batched():
    """An LLM step inside a multi-step branch path still batches."""
    model = RecordingModel()

    pipeline = (
        Source.list([{"question": f"q{i}"} for i in range(3)])
        >> Branch(
            a=(
                Map(lambda r: {**r, "question": r["question"].upper()})
                >> LLMStep(
                    prompt="A: {question}",
                    input_columns=["question"],
                    output_column="response",
                    model=model,
                )
            ),
            b=Map(lambda r: {**r, "response": "static"}),
        )
        >> JoinBranches()
        >> ListSink()
    )
    output = pipeline.run(batch_size=3)

    assert model.batches == [3]
    assert len(output) == 3


def test_branch_path_change_invalidates_checkpoint(tmp_path):
    """Editing a step inside a branch path is detected as a pipeline change."""
    ckpt = str(tmp_path / "ckpt")
    _branch_pipeline(RecordingModel()).run(checkpoint_dir=ckpt)

    changed = (
        Source.list([{"question": f"q{i}"} for i in range(4)])
        >> Branch(
            chosen=Map(lambda r: {**r, "response": "x"}),
            rejected=Map(lambda r: {**r, "response": "y"}),
        ).as_step("branch")
        >> JoinBranches()
        >> ListSink()
    )
    with pytest.raises(PipelineChangedError, match="Pipeline structure has changed"):
        changed.run(checkpoint_dir=ckpt, resume=True)


def test_branch_with_no_input_records():
    pipeline = (
        Source.list([])
        >> Branch(
            a=Map(lambda r: r),
            b=Map(lambda r: r),
        )
        >> JoinBranches()
        >> ListSink()
    )
    assert pipeline.run() == []
