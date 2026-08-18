"""The branching reference, pinned against the code it documents.

Same shape as ``test_reference_sources_and_seed.py``:

1. Every parameter the code takes, and every metadata column it adds, is named on
   the page (code → docs). Something that exists and is undocumented cannot be
   discovered; the reverse is merely untidy.
2. Every self-contained example executes.
3. The behavioural claims — the ones a reader would be burned by if wrong — are
   asserted against real runs, not paraphrased from the source.

No test here reaches a provider: the one model object is a local stub.
"""

import inspect
import re
from pathlib import Path

import pytest

from datafast import (
    Branch,
    FlatMap,
    Filter,
    JoinBranches,
    LLMStep,
    ListSink,
    Map,
    Source,
)
from datafast.core.validation import PipelineValidationError
from datafast.transforms import branch as branch_module

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "reference" / "branching.md"

DOCUMENTED = [Branch, JoinBranches]


def _page() -> str:
    return PAGE.read_text()


def _code_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", _page(), re.DOTALL)


def _two_paths(**paths):
    """A branch pipeline over two records, ending in a list sink."""
    return (
        Source.list([{"text": "hello"}, {"text": "goodbye"}])
        >> Branch(**paths)
        >> JoinBranches()
        >> ListSink()
    )


class StubModel:
    """A local stand-in for a served model. Reaches nothing."""

    provider_id = "stub"
    model_id = "stub-model"

    def generate(self, prompt=None, messages=None, metadata=None, response_format=None):
        return "out"

    def generate_batch(self, messages_list, metadata=None, response_format=None):
        return ["out"] * len(messages_list)


# --------------------------------------------------------------------------
# code → docs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("step", DOCUMENTED, ids=lambda s: s.__name__)
def test_every_parameter_is_named_on_the_page(step):
    """A parameter the page forgets is one a reader never finds.

    ``**paths`` counts: it is the whole of Branch's API.
    """
    parameters = [
        p.name
        for p in inspect.signature(step).parameters.values()
        if p.name != "self"
    ]
    assert parameters, f"{step.__name__} takes nothing — check the test, not the page"
    missing = [p for p in parameters if f"`{p}`" not in _page()]
    assert not missing, f"{step.__name__} takes {missing}, undocumented on the page"


def test_every_branch_metadata_column_is_named_on_the_page():
    """The columns Branch adds are read by user code, so all of them are documented."""
    columns = sorted(branch_module._BRANCH_META_KEYS)
    assert len(columns) >= 3, "expected the branch metadata keys — check the source"
    missing = [c for c in columns if f"`{c}`" not in _page()]
    assert not missing, f"Branch adds {missing}, undocumented on the page"


def test_the_documented_join_modes_are_the_ones_the_code_accepts():
    for how in ("inner", "outer"):
        JoinBranches(how=how)
        assert f'`"{how}"`' in _page()
    with pytest.raises(ValueError, match="inner"):
        JoinBranches(how="left")


@pytest.mark.parametrize("block", _code_blocks(), ids=lambda b: b.split("\n")[0][:40])
def test_every_example_executes(block):
    exec(compile(block, str(PAGE), "exec"), {})


def test_every_page_linked_to_exists():
    links = re.findall(r"\]\((?!https?:)([^)#]+\.md)", _page())
    assert links, "the page links nowhere — check the test, not the page"
    missing = sorted(link for link in links if not (PAGE.parent / link).resolve().exists())
    assert not missing, f"links to pages that do not exist: {missing}"


# --------------------------------------------------------------------------
# tagging and merging
# --------------------------------------------------------------------------


def test_branch_tags_every_output_record_with_the_documented_columns():
    tagged = list(
        Branch(
            upper=Map(lambda r: {**r, "shout": r["text"].upper()}),
            length=Map(lambda r: {**r, "shout": str(len(r["text"]))}),
        ).process(iter([{"text": "hello"}, {"text": "goodbye"}]))
    )

    assert len(tagged) == 4  # 2 records x 2 paths
    for record in tagged:
        assert branch_module._BRANCH_META_KEYS <= set(record)
        assert record["_branch_input_keys"] == ["text"]
    assert [r["_branch_name"] for r in tagged] == ["upper", "upper", "length", "length"]
    # _branch_id is the input position, shared by every path.
    assert [r["_branch_id"] for r in tagged] == [0, 1, 0, 1]


def test_join_suffixes_new_columns_and_strips_the_metadata():
    records = _two_paths(
        upper=Map(lambda r: {**r, "shout": r["text"].upper()}),
        length=Map(lambda r: {**r, "shout": str(len(r["text"]))}),
    ).run()

    assert records[0] == {"text": "hello", "shout_upper": "HELLO", "shout_length": "5"}
    assert not branch_module._BRANCH_META_KEYS & set(records[0])


def test_a_custom_suffix_applies_only_to_the_path_it_names():
    pipeline = (
        Source.list([{"q": "hi"}])
        >> Branch(
            a=Map(lambda r: {**r, "answer": "short"}),
            b=Map(lambda r: {**r, "answer": "long"}),
        )
        >> JoinBranches(suffixes={"a": "_first"})
        >> ListSink()
    )
    assert pipeline.run() == [
        {"q": "hi", "answer_first": "short", "answer_b": "long"}
    ]


def test_a_path_that_rewrites_an_existing_column_loses_to_the_first_path():
    """The page's sharpest trap: shared columns are copied, never suffixed."""
    pipeline = (
        Source.list([{"text": "hello"}])
        >> Branch(
            first=Map(lambda r: {**r, "text": "FIRST"}),
            second=Map(lambda r: {**r, "text": "SECOND"}),
        )
        >> JoinBranches()
        >> ListSink()
    )
    assert pipeline.run() == [{"text": "FIRST"}], "the second path's rewrite is dropped"


def test_paths_cannot_see_each_others_records():
    """Each path gets a deep copy, so in-place edits do not leak."""
    def mutate(record):
        record["text"] = "mutated"
        return record

    records = _two_paths(
        a=Map(mutate),
        b=Map(lambda r: {**r, "seen": r["text"]}),
    ).run()

    assert records[0]["seen_b"] == "hello", "path b saw path a's mutation"
    assert records[0]["text"] == "mutated"  # path a's own copy did change


def test_multiple_records_per_path_merge_as_a_cartesian_product():
    pipeline = (
        Source.list([{"n": 1}])
        >> Branch(
            a=FlatMap(lambda r: [{**r, "a": 1}, {**r, "a": 2}]),
            b=FlatMap(lambda r: [{**r, "b": 10}, {**r, "b": 20}]),
        )
        >> JoinBranches()
        >> ListSink()
    )
    records = pipeline.run()

    assert len(records) == 4
    assert {(r["a_a"], r["b_b"]) for r in records} == {(1, 10), (1, 20), (2, 10), (2, 20)}


def _missing_path_pipeline(how: str):
    return (
        Source.list([{"n": 1}, {"n": 2}])
        >> Branch(
            keep=Map(lambda r: {**r, "a": r["n"]}),
            picky=Filter(lambda r: r["n"] == 1) >> Map(lambda r: {**r, "b": r["n"]}),
        )
        >> JoinBranches(how=how)
        >> ListSink()
    )


def test_inner_drops_the_input_record_when_a_path_produced_nothing():
    assert _missing_path_pipeline("inner").run() == [{"n": 1, "a_keep": 1, "b_picky": 1}]


def test_outer_keeps_the_record_and_the_missing_column_is_absent_not_none():
    records = _missing_path_pipeline("outer").run()

    assert records == [{"n": 1, "a_keep": 1, "b_picky": 1}, {"n": 2, "a_keep": 2}]
    assert "b_picky" not in records[1], "the page says absent, not None"


# --------------------------------------------------------------------------
# what compile() rejects
# --------------------------------------------------------------------------


def test_compile_rejects_a_branch_inside_a_branch_path():
    inner = Branch(c=Map(lambda r: r), d=Map(lambda r: r))
    pipeline = (
        Source.list([{"n": 1}])
        >> Branch(a=inner, b=Map(lambda r: r))
        >> JoinBranches()
        >> ListSink()
    )
    with pytest.raises(PipelineValidationError, match="nesting a Branch"):
        pipeline.compile()


def test_compile_rejects_a_branch_with_no_joinbranches():
    pipeline = (
        Source.list([{"n": 1}])
        >> Branch(a=Map(lambda r: r), b=Map(lambda r: r))
        >> ListSink()
    )
    with pytest.raises(PipelineValidationError, match="never closed by"):
        pipeline.compile()


def test_compile_rejects_a_joinbranches_with_no_branch():
    pipeline = Source.list([{"n": 1}]) >> JoinBranches() >> ListSink()
    with pytest.raises(PipelineValidationError, match="no matching Branch"):
        pipeline.compile()


def test_compile_rejects_a_second_branch_opened_before_the_first_is_closed():
    pipeline = (
        Source.list([{"n": 1}])
        >> Branch(a=Map(lambda r: r), b=Map(lambda r: r))
        >> Branch(c=Map(lambda r: r), d=Map(lambda r: r))
        >> JoinBranches()
        >> ListSink()
    )
    with pytest.raises(PipelineValidationError, match="opens before"):
        pipeline.compile()


def test_compile_rejects_a_source_or_a_sink_inside_a_path():
    with pytest.raises(PipelineValidationError, match="discards upstream records"):
        (
            Source.list([{"n": 1}])
            >> Branch(a=Source.list([{"n": 9}]), b=Map(lambda r: r))
            >> JoinBranches()
            >> ListSink()
        ).compile()

    with pytest.raises(PipelineValidationError, match="is not allowed"):
        (
            Source.list([{"n": 1}])
            >> Branch(a=(Map(lambda r: r) >> ListSink()), b=Map(lambda r: r))
            >> JoinBranches()
            >> ListSink()
        ).compile()


def test_branch_requires_at_least_two_paths():
    with pytest.raises(ValueError, match="at least 2 named paths"):
        Branch(only=Map(lambda r: r))


# --------------------------------------------------------------------------
# the runner
# --------------------------------------------------------------------------


def _llm_branch_pipeline():
    return (
        Source.list([{"q": "a"}, {"q": "b"}])
        >> Branch(
            a=LLMStep(prompt="{q}", input_columns=["q"], output_column="r", model=StubModel()),
            b=(
                Map(lambda r: r)
                >> LLMStep(
                    prompt="{q}", input_columns=["q"], output_column="r", model=StubModel()
                )
            ),
        )
        >> JoinBranches()
        >> ListSink()
    )


def test_checkpoint_files_are_keyed_by_the_dotted_path_name(tmp_path):
    """The page shows these three names; the checkpoint directory must contain them."""
    _llm_branch_pipeline().run(checkpoint_dir=str(tmp_path), checkpoint_every=1)

    written = {p.name for p in tmp_path.iterdir()}
    for name in (
        "step_001_Branch.jsonl",
        "step_001_Branch.a.jsonl",
        "step_001_Branch.b.1_LLMStep.jsonl",
    ):
        assert name in written, f"{name} not written; the page shows it"
        assert name in _page(), f"{name} written but not shown on the page"


def test_renaming_the_branch_step_renames_its_path_checkpoint_files(tmp_path):
    """The page says as_step renames every one of those files."""
    pipeline = (
        Source.list([{"q": "a"}])
        >> Branch(
            a=LLMStep(prompt="{q}", input_columns=["q"], output_column="r", model=StubModel()),
            b=Map(lambda r: {**r, "r": "static"}),
        ).as_step("compare")
        >> JoinBranches()
        >> ListSink()
    )
    pipeline.run(checkpoint_dir=str(tmp_path), checkpoint_every=1)

    assert "step_001_compare.a.jsonl" in {p.name for p in tmp_path.iterdir()}


def test_a_nested_llm_step_is_batched_by_the_runner():
    """The page claims four records against one path go out as one batch of four."""
    batches: list[int] = []

    class RecordingModel(StubModel):
        def generate_batch(self, messages_list, metadata=None, response_format=None):
            batches.append(len(messages_list))
            return ["out"] * len(messages_list)

    pipeline = (
        Source.list([{"q": f"q{i}"} for i in range(4)])
        >> Branch(
            a=LLMStep(
                prompt="{q}", input_columns=["q"], output_column="r", model=RecordingModel()
            ),
            b=Map(lambda r: {**r, "r": "static"}),
        )
        >> JoinBranches()
        >> ListSink()
    )
    pipeline.run(batch_size=4)

    assert batches == [4]


def test_changing_a_map_function_inside_a_path_does_not_change_the_fingerprint():
    """The page warns about this: the fingerprint records step classes, not functions."""
    from datafast.core.runner import _step_signature

    one = Branch(a=Map(lambda r: {**r, "x": 1}), b=Map(lambda r: r))
    two = Branch(a=Map(lambda r: {**r, "x": 2}), b=Map(lambda r: r))

    assert _step_signature(one) == _step_signature(two)
    # Changing which steps a path contains does change it.
    three = Branch(a=(Map(lambda r: r) >> Filter(lambda r: True)), b=Map(lambda r: r))
    assert _step_signature(three) != _step_signature(one)
