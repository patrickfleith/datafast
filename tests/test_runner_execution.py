import pytest

from datafast import LLMStep, ListSink, Map, Source
from datafast.core.config import LLMCall, RunConfig
from datafast.core.runner import Runner


class CountingModel:
    provider_name = "fake"
    model_id = "fake-model"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt=None, messages=None, metadata=None, response_format=None):
        self.calls += 1
        return "ok"


def test_limit_truncates_source_records():
    """limit=N processes only the first N source records downstream."""
    model = CountingModel()
    records = [{"topic": f"t{i}"} for i in range(10)]

    pipeline = (
        Source.list(records)
        >> LLMStep(
            prompt="Say something about {topic}.",
            input_columns=["topic"],
            output_column="result",
            model=model,
        )
        >> ListSink()
    )

    output = pipeline.run(limit=3)

    assert len(output) == 3
    assert model.calls == 3
    assert [r["topic"] for r in output] == ["t0", "t1", "t2"]


def test_resume_from_reruns_step_and_reuses_upstream(tmp_path):
    """resume_from re-runs the named step onward, reusing cached upstream LLM output."""
    model = CountingModel()
    transform_runs = []

    def transform(record):
        transform_runs.append(record["topic"])
        return {**record, "seen": True}

    def build():
        return (
            Source.list([{"topic": f"t{i}"} for i in range(3)])
            >> LLMStep(
                prompt="About {topic}.",
                input_columns=["topic"],
                output_column="result",
                model=model,
            ).as_step("generate")
            >> Map(transform).as_step("transform")
            >> ListSink()
        )

    ckpt = str(tmp_path / "ckpt")
    build().run(checkpoint_dir=ckpt)
    assert model.calls == 3
    assert len(transform_runs) == 3

    output = build().run(checkpoint_dir=ckpt, resume_from="transform")

    assert model.calls == 3  # generate reused from checkpoint, not re-called
    assert len(transform_runs) == 6  # transform re-ran on the 3 cached records
    assert len(output) == 3


def test_resume_from_requires_checkpoint_dir():
    pipeline = Source.list([{"topic": "x"}]) >> ListSink()
    with pytest.raises(ValueError, match="resume_from requires checkpoint_dir"):
        pipeline.run(resume_from="anything")


def test_resume_from_unknown_step_raises(tmp_path):
    ckpt = str(tmp_path / "ckpt")
    pipeline = (
        Source.list([{"topic": "x"}]) >> Map(lambda r: r).as_step("m") >> ListSink()
    )
    pipeline.run(checkpoint_dir=ckpt)
    with pytest.raises(ValueError, match="not found"):
        pipeline.run(checkpoint_dir=ckpt, resume_from="nope")


def test_stop_after_halts_before_later_steps():
    """stop_after returns the named step's output and never runs later steps."""
    a_runs, b_runs = [], []

    def mark_a(record):
        a_runs.append(record["n"])
        return record

    def mark_b(record):
        b_runs.append(record["n"])
        return record

    pipeline = (
        Source.list([{"n": i} for i in range(3)])
        >> Map(mark_a).as_step("a")
        >> Map(mark_b).as_step("b")
        >> ListSink()
    )

    output = pipeline.run(stop_after="a")

    assert a_runs == [0, 1, 2]
    assert b_runs == []
    assert len(output) == 3


def _order(strategy: str, model_ids: list[str]) -> list[str]:
    """Run the given call model_ids through Runner._order_calls, return call_ids."""
    runner = Runner(Source.list([]) >> ListSink(), RunConfig(llm_strategy=strategy))
    calls = [
        LLMCall(
            call_id=f"c{i}",
            record={},
            record_index=i,
            prompt_template="",
            prompt_index=0,
            model_id=model_id,
            language_code="",
            language_name="",
            messages=[],
            output_index=0,
        )
        for i, model_id in enumerate(model_ids)
    ]
    return [call.call_id for call in runner._order_calls(calls)]


def test_llm_strategy_orders_calls():
    # by_record preserves input order
    assert _order("by_record", ["m1", "m2", "m1"]) == ["c0", "c1", "c2"]
    # by_model groups per model in first-seen order
    assert _order("by_model", ["m1", "m2", "m1"]) == ["c0", "c2", "c1"]
    # round_robin interleaves across models
    assert _order("round_robin", ["m1", "m1", "m2"]) == ["c0", "c2", "c1"]


def test_resume_returns_cached_results(tmp_path):
    """A fully-completed run resumes to cached output without re-calling the model."""
    model = CountingModel()

    def build():
        return (
            Source.list([{"topic": f"t{i}"} for i in range(3)])
            >> LLMStep(
                prompt="About {topic}.",
                input_columns=["topic"],
                output_column="result",
                model=model,
            ).as_step("generate")
            >> ListSink()
        )

    ckpt = str(tmp_path / "ckpt")
    first = build().run(checkpoint_dir=ckpt)
    assert model.calls == 3

    second = build().run(checkpoint_dir=ckpt, resume=True)

    assert model.calls == 3  # nothing re-run
    assert second == first


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


def test_resume_mid_llm_step_skips_completed_calls(tmp_path):
    """After a crash mid-LLM-step, resume runs only the calls that hadn't completed."""
    model = CrashOnNthModel(crash_on=3)

    def build():
        return (
            Source.list([{"topic": f"t{i}"} for i in range(4)])
            >> LLMStep(
                prompt="About {topic}.",
                input_columns=["topic"],
                output_column="result",
                model=model,
            ).as_step("generate")
            >> ListSink()
        )

    ckpt = str(tmp_path / "ckpt")
    with pytest.raises(_Crash):
        build().run(checkpoint_dir=ckpt, batch_size=1, checkpoint_every=1)
    assert model.calls == 3  # 2 completed and checkpointed, 3rd crashed

    output = build().run(checkpoint_dir=ckpt, resume=True, batch_size=1, checkpoint_every=1)

    assert model.calls == 5  # only the 2 remaining calls ran
    assert len(output) == 4
