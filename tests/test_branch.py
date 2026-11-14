"""Tests for Branch and JoinBranches steps."""

from unittest.mock import MagicMock

import pytest

from datafast import Classify, LLMStep, ListSink, Map, Pipeline, Source
from datafast.transforms.branch import (
    Branch,
    JoinBranches,
    _BRANCH_ID,
    _BRANCH_INPUT_KEYS,
    _BRANCH_META_KEYS,
    _BRANCH_NAME,
)
from datafast.transforms.data_ops import Filter, FlatMap


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RECORDS = [
    {"id": 1, "question": "What is Python?"},
    {"id": 2, "question": "What is Rust?"},
    {"id": 3, "question": "What is Go?"},
]


def _upper_map(records):
    """Map that uppercases the question and adds a 'response' column."""
    for r in records:
        yield {**r, "response": r["question"].upper()}


def _lower_map(records):
    """Map that lowercases the question and adds a 'response' column."""
    for r in records:
        yield {**r, "response": r["question"].lower()}


class _UpperStep:
    """Minimal step-like object for testing."""

    def process(self, records):
        return _upper_map(records)


class _LowerStep:
    """Minimal step-like object for testing."""

    def process(self, records):
        return _lower_map(records)


# ---------------------------------------------------------------------------
# Branch — init validation
# ---------------------------------------------------------------------------


class TestBranchInit:
    def test_requires_at_least_two_paths(self):
        with pytest.raises(ValueError, match="at least 2"):
            Branch(only_one=Map(lambda r: r))

    def test_accepts_two_paths(self):
        step = Branch(
            a=Map(lambda r: {**r, "x": 1}),
            b=Map(lambda r: {**r, "x": 2}),
        )
        assert step.path_names == ["a", "b"]

    def test_path_names_preserved(self):
        step = Branch(
            chosen=Map(lambda r: r),
            rejected=Map(lambda r: r),
            neutral=Map(lambda r: r),
        )
        assert step.path_names == ["chosen", "rejected", "neutral"]


# ---------------------------------------------------------------------------
# Branch — tagging
# ---------------------------------------------------------------------------


class TestBranchTagging:
    def test_output_records_have_branch_metadata(self):
        step = Branch(
            a=Map(lambda r: {**r, "resp": "A"}),
            b=Map(lambda r: {**r, "resp": "B"}),
        )
        results = list(step.process(iter(RECORDS)))

        # 3 input × 2 branches = 6 output records
        assert len(results) == 6

        for rec in results:
            assert _BRANCH_ID in rec
            assert _BRANCH_NAME in rec
            assert _BRANCH_INPUT_KEYS in rec

    def test_branch_ids_are_contiguous_integers(self):
        step = Branch(
            a=Map(lambda r: {**r, "resp": "A"}),
            b=Map(lambda r: {**r, "resp": "B"}),
        )
        results = list(step.process(iter(RECORDS)))
        ids = sorted({r[_BRANCH_ID] for r in results})
        assert ids == [0, 1, 2]

    def test_branch_names_match_path_keys(self):
        step = Branch(
            chosen=Map(lambda r: {**r, "resp": "C"}),
            rejected=Map(lambda r: {**r, "resp": "R"}),
        )
        results = list(step.process(iter(RECORDS)))
        names = {r[_BRANCH_NAME] for r in results}
        assert names == {"chosen", "rejected"}

    def test_input_keys_exclude_branch_metadata(self):
        step = Branch(
            a=Map(lambda r: {**r, "x": 1}),
            b=Map(lambda r: {**r, "x": 2}),
        )
        results = list(step.process(iter(RECORDS)))
        for rec in results:
            input_keys = set(rec[_BRANCH_INPUT_KEYS])
            assert input_keys & _BRANCH_META_KEYS == set()
            assert "id" in input_keys
            assert "question" in input_keys

    def test_empty_input(self):
        step = Branch(
            a=Map(lambda r: {**r, "x": 1}),
            b=Map(lambda r: {**r, "x": 2}),
        )
        results = list(step.process(iter([])))
        assert results == []


# ---------------------------------------------------------------------------
# Branch — isolation between paths
# ---------------------------------------------------------------------------


class TestBranchIsolation:
    def test_paths_get_independent_copies(self):
        mutations: list[str] = []

        def mutating_fn(r):
            r["mutated"] = True
            mutations.append(r["id"])
            return r

        step = Branch(
            a=Map(mutating_fn),
            b=Map(lambda r: {**r, "safe": True}),
        )
        results = list(step.process(iter(RECORDS)))

        # Branch b should NOT see 'mutated' key from branch a.
        b_records = [r for r in results if r[_BRANCH_NAME] == "b"]
        for rec in b_records:
            assert "mutated" not in rec


# ---------------------------------------------------------------------------
# JoinBranches — init validation
# ---------------------------------------------------------------------------


class TestJoinBranchesInit:
    def test_rejects_invalid_how(self):
        with pytest.raises(ValueError, match="inner.*outer"):
            JoinBranches(how="left")

    def test_accepts_inner(self):
        JoinBranches(how="inner")

    def test_accepts_outer(self):
        JoinBranches(how="outer")

    def test_default_is_inner(self):
        jb = JoinBranches()
        assert jb._how == "inner"


# ---------------------------------------------------------------------------
# JoinBranches — basic merging
# ---------------------------------------------------------------------------


class TestJoinBranchesMerge:
    def test_basic_two_branch_merge(self):
        branch = Branch(
            chosen=Map(lambda r: {**r, "response": "expert"}),
            rejected=Map(lambda r: {**r, "response": "brief"}),
        )
        join = JoinBranches()

        tagged = list(branch.process(iter(RECORDS)))
        merged = list(join.process(iter(tagged)))

        assert len(merged) == 3
        for rec in merged:
            assert "id" in rec
            assert "question" in rec
            assert "response_chosen" in rec
            assert "response_rejected" in rec
            assert rec["response_chosen"] == "expert"
            assert rec["response_rejected"] == "brief"
            # No branch metadata in output.
            assert _BRANCH_ID not in rec
            assert _BRANCH_NAME not in rec
            assert _BRANCH_INPUT_KEYS not in rec

    def test_original_columns_preserved(self):
        branch = Branch(
            a=Map(lambda r: {**r, "out": "A"}),
            b=Map(lambda r: {**r, "out": "B"}),
        )
        join = JoinBranches()

        tagged = list(branch.process(iter(RECORDS)))
        merged = list(join.process(iter(tagged)))

        for i, rec in enumerate(merged):
            assert rec["id"] == RECORDS[i]["id"]
            assert rec["question"] == RECORDS[i]["question"]

    def test_custom_suffixes(self):
        branch = Branch(
            formal=Map(lambda r: {**r, "text": "formal version"}),
            casual=Map(lambda r: {**r, "text": "casual version"}),
        )
        join = JoinBranches(
            suffixes={"formal": "_formal", "casual": "_casual"}
        )

        tagged = list(branch.process(iter(RECORDS)))
        merged = list(join.process(iter(tagged)))

        assert len(merged) == 3
        for rec in merged:
            assert "text_formal" in rec
            assert "text_casual" in rec
            assert rec["text_formal"] == "formal version"
            assert rec["text_casual"] == "casual version"

    def test_three_branches(self):
        branch = Branch(
            a=Map(lambda r: {**r, "val": 1}),
            b=Map(lambda r: {**r, "val": 2}),
            c=Map(lambda r: {**r, "val": 3}),
        )
        join = JoinBranches()

        tagged = list(branch.process(iter(RECORDS)))
        merged = list(join.process(iter(tagged)))

        assert len(merged) == 3
        for rec in merged:
            assert rec["val_a"] == 1
            assert rec["val_b"] == 2
            assert rec["val_c"] == 3


# ---------------------------------------------------------------------------
# JoinBranches — inner vs outer join
# ---------------------------------------------------------------------------


class TestJoinBranchesJoinMode:
    def test_inner_skips_incomplete_groups(self):
        # Branch where one path filters out record id=2.
        branch = Branch(
            a=Map(lambda r: {**r, "x": "A"}),
            b=Filter(fn=lambda r: r.get("id") != 2),
        )
        join = JoinBranches(how="inner")

        tagged = list(branch.process(iter(RECORDS)))
        merged = list(join.process(iter(tagged)))

        # Record id=2 is missing from branch b → skipped.
        assert len(merged) == 2
        merged_ids = {r["id"] for r in merged}
        assert 2 not in merged_ids

    def test_outer_includes_incomplete_groups(self):
        branch = Branch(
            a=Map(lambda r: {**r, "x": "A"}),
            b=Filter(fn=lambda r: r.get("id") != 2),
        )
        join = JoinBranches(how="outer")

        tagged = list(branch.process(iter(RECORDS)))
        merged = list(join.process(iter(tagged)))

        # All 3 records present; record id=2 has no branch-b columns.
        assert len(merged) == 3
        rec_2 = next(r for r in merged if r["id"] == 2)
        assert rec_2["x_a"] == "A"
        # Branch b produced no new columns for this record, so
        # there should be no x_b key.
        assert "x_b" not in rec_2


# ---------------------------------------------------------------------------
# JoinBranches — multiple records per branch (cartesian product)
# ---------------------------------------------------------------------------


class TestJoinBranchesCartesian:
    def test_cartesian_product_when_branch_produces_multiple(self):
        branch = Branch(
            a=Map(lambda r: {**r, "resp": "single"}),
            b=FlatMap(
                lambda r: [
                    {**r, "resp": "v1"},
                    {**r, "resp": "v2"},
                ]
            ),
        )
        join = JoinBranches()

        single_record = [{"id": 1, "question": "Q"}]
        tagged = list(branch.process(iter(single_record)))
        # branch a: 1 record, branch b: 2 records → 3 tagged total
        assert len(tagged) == 3

        merged = list(join.process(iter(tagged)))
        # Cartesian: 1 × 2 = 2
        assert len(merged) == 2
        assert merged[0]["resp_a"] == "single"
        assert {merged[0]["resp_b"], merged[1]["resp_b"]} == {"v1", "v2"}


# ---------------------------------------------------------------------------
# JoinBranches — empty / no metadata
# ---------------------------------------------------------------------------


class TestJoinBranchesEdgeCases:
    def test_empty_input(self):
        join = JoinBranches()
        results = list(join.process(iter([])))
        assert results == []

    def test_no_branch_metadata_passes_through(self):
        join = JoinBranches()
        plain = [{"id": 1, "text": "hello"}]
        results = list(join.process(iter(plain)))
        assert len(results) == 1
        assert results[0] == {"id": 1, "text": "hello"}


# ---------------------------------------------------------------------------
# End-to-end: pipeline chaining with >> operator
# ---------------------------------------------------------------------------


class TestBranchPipelineIntegration:
    def test_pipeline_chaining(self):
        pipe = (
            Map(lambda r: {**r, "prepared": True})
            >> Branch(
                upper=Map(lambda r: {**r, "response": r["question"].upper()}),
                lower=Map(lambda r: {**r, "response": r["question"].lower()}),
            )
            >> JoinBranches()
        )

        assert isinstance(pipe, Pipeline)
        results = list(pipe.process(iter(RECORDS)))

        assert len(results) == 3
        for rec in results:
            assert rec["prepared"] is True
            assert rec["response_upper"] == rec["question"].upper()
            assert rec["response_lower"] == rec["question"].lower()
            # No branch metadata leaked.
            assert _BRANCH_ID not in rec

    def test_source_to_sink_pipeline(self):
        sink = ListSink()
        pipeline = (
            Source.list(RECORDS)
            >> Branch(
                a=Map(lambda r: {**r, "out": "A"}),
                b=Map(lambda r: {**r, "out": "B"}),
            )
            >> JoinBranches()
            >> sink
        )

        results = list(pipeline.process(iter([])))
        assert len(results) == 3
        assert len(sink.records) == 3
        for rec in sink.records:
            assert rec["out_a"] == "A"
            assert rec["out_b"] == "B"

    def test_pipeline_run(self):
        pipeline = (
            Source.list(RECORDS)
            >> Branch(
                a=Map(lambda r: {**r, "v": "A"}),
                b=Map(lambda r: {**r, "v": "B"}),
            )
            >> JoinBranches()
        )
        results = pipeline.run()
        assert len(results) == 3
        assert all("v_a" in r and "v_b" in r for r in results)


# ---------------------------------------------------------------------------
# LLM steps inside branches (mocked)
# ---------------------------------------------------------------------------


class TestBranchWithLLMSteps:
    def test_mocked_llm_step_inside_branch(self):
        mock_model = MagicMock()
        mock_model.model_id = "mock-model"
        mock_model.generate.return_value = '{"answer": "42"}'

        step_a = LLMStep(
            prompt="Expert: {question}",
            input_columns=["question"],
            output_columns=["answer"],
            parse_mode="json",
            model=mock_model,
        )
        step_b = LLMStep(
            prompt="Brief: {question}",
            input_columns=["question"],
            output_columns=["answer"],
            parse_mode="json",
            model=mock_model,
        )

        branch = Branch(chosen=step_a, rejected=step_b)
        join = JoinBranches()

        tagged = list(branch.process(iter(RECORDS)))
        merged = list(join.process(iter(tagged)))

        assert len(merged) == 3
        for rec in merged:
            assert "answer_chosen" in rec
            assert "answer_rejected" in rec

    def test_mocked_classify_inside_branch(self):
        mock_model = MagicMock()
        mock_model.model_id = "mock-model"
        mock_model.generate.return_value = '{"label": "positive"}'

        branch = Branch(
            model_a=Classify(
                labels=["positive", "negative"],
                input_columns=["question"],
                llm=mock_model,
            ),
            model_b=Classify(
                labels=["positive", "negative"],
                input_columns=["question"],
                llm=mock_model,
            ),
        )
        join = JoinBranches()

        tagged = list(branch.process(iter(RECORDS)))
        merged = list(join.process(iter(tagged)))

        assert len(merged) == 3
        for rec in merged:
            # Classify defaults to output_column="label"
            assert "label_model_a" in rec
            assert "label_model_b" in rec


# ---------------------------------------------------------------------------
# Checkpointing compatibility (records are JSON-serializable)
# ---------------------------------------------------------------------------


class TestBranchCheckpointCompat:
    def test_tagged_records_are_json_safe(self):
        import json

        branch = Branch(
            a=Map(lambda r: {**r, "resp": "hello"}),
            b=Map(lambda r: {**r, "resp": "world"}),
        )
        tagged = list(branch.process(iter(RECORDS)))

        for rec in tagged:
            serialized = json.dumps(rec, default=str)
            deserialized = json.loads(serialized)
            assert deserialized[_BRANCH_ID] == rec[_BRANCH_ID]
            assert deserialized[_BRANCH_NAME] == rec[_BRANCH_NAME]
