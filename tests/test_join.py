"""Tests for the Join step."""

import pytest

from datafast import Join, Map, Source


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

USERS = [
    {"user_id": 1, "name": "Alice"},
    {"user_id": 2, "name": "Bob"},
    {"user_id": 3, "name": "Charlie"},
]

ACTIONS = [
    {"user_id": 1, "action": "login"},
    {"user_id": 2, "action": "purchase"},
    {"user_id": 4, "action": "signup"},
]

SCORES_LEFT = [
    {"qid": "q1", "score": 8},
    {"qid": "q2", "score": 5},
]

SCORES_RIGHT = [
    {"qid": "q1", "score": 3},
    {"qid": "q2", "score": 9},
]


# ---------------------------------------------------------------------------
# Init validation
# ---------------------------------------------------------------------------


class TestJoinInit:
    def test_rejects_invalid_how(self):
        right = Source.list(ACTIONS)
        with pytest.raises(ValueError, match="how must be one of"):
            Join(right, on="user_id", how="cross")

    def test_accepts_all_valid_how(self):
        right = Source.list(ACTIONS)
        for how in ("inner", "left", "right", "outer"):
            Join(right, on="user_id", how=how)

    def test_on_string_normalized_to_list(self):
        right = Source.list(ACTIONS)
        j = Join(right, on="user_id")
        assert j._on == ["user_id"]

    def test_on_list_preserved(self):
        right = Source.list(ACTIONS)
        j = Join(right, on=["a", "b"])
        assert j._on == ["a", "b"]


# ---------------------------------------------------------------------------
# Inner join
# ---------------------------------------------------------------------------


class TestJoinInner:
    def test_inner_join_basic(self):
        left = Source.list(USERS)
        right = Source.list(ACTIONS)
        step = Join(right, on="user_id", how="inner")

        results = list(step.process(left.process(iter([]))))

        # Users 1 & 2 match; user 3 has no action, user 4 has no name.
        assert len(results) == 2
        ids = {r["user_id"] for r in results}
        assert ids == {1, 2}

        for r in results:
            assert "name" in r
            assert "action" in r

    def test_inner_join_no_overlap_columns(self):
        results = list(
            Join(Source.list(ACTIONS), on="user_id").process(
                iter(USERS)
            )
        )
        # No overlapping non-key columns → no suffixes needed.
        for r in results:
            assert "name" in r
            assert "action" in r
            # No suffixed columns.
            assert not any(k.endswith("_left") or k.endswith("_right") for k in r)


# ---------------------------------------------------------------------------
# Left join
# ---------------------------------------------------------------------------


class TestJoinLeft:
    def test_left_join_keeps_unmatched_left(self):
        step = Join(Source.list(ACTIONS), on="user_id", how="left")
        results = list(step.process(iter(USERS)))

        assert len(results) == 3
        ids = {r["user_id"] for r in results}
        assert ids == {1, 2, 3}

        charlie = next(r for r in results if r["user_id"] == 3)
        assert "action" not in charlie or charlie.get("action") is None


# ---------------------------------------------------------------------------
# Right join
# ---------------------------------------------------------------------------


class TestJoinRight:
    def test_right_join_keeps_unmatched_right(self):
        step = Join(Source.list(ACTIONS), on="user_id", how="right")
        results = list(step.process(iter(USERS)))

        assert len(results) == 3
        ids = {r["user_id"] for r in results}
        assert ids == {1, 2, 4}

        signup = next(r for r in results if r["user_id"] == 4)
        assert signup["action"] == "signup"
        assert "name" not in signup or signup.get("name") is None


# ---------------------------------------------------------------------------
# Outer join
# ---------------------------------------------------------------------------


class TestJoinOuter:
    def test_outer_join_keeps_all(self):
        step = Join(Source.list(ACTIONS), on="user_id", how="outer")
        results = list(step.process(iter(USERS)))

        assert len(results) == 4
        ids = {r["user_id"] for r in results}
        assert ids == {1, 2, 3, 4}


# ---------------------------------------------------------------------------
# Overlapping columns & suffixes
# ---------------------------------------------------------------------------


class TestJoinSuffixes:
    def test_overlapping_columns_get_suffixed(self):
        step = Join(
            Source.list(SCORES_RIGHT),
            on="qid",
            how="inner",
        )
        results = list(step.process(iter(SCORES_LEFT)))

        assert len(results) == 2
        for r in results:
            assert "qid" in r
            assert "score_left" in r
            assert "score_right" in r
            # Plain "score" should NOT be present.
            assert "score" not in r

    def test_custom_suffixes(self):
        step = Join(
            Source.list(SCORES_RIGHT),
            on="qid",
            how="inner",
            suffixes=("_chosen", "_rejected"),
        )
        results = list(step.process(iter(SCORES_LEFT)))

        for r in results:
            assert "score_chosen" in r
            assert "score_rejected" in r

    def test_suffix_values_correct(self):
        step = Join(
            Source.list(SCORES_RIGHT),
            on="qid",
            how="inner",
        )
        results = list(step.process(iter(SCORES_LEFT)))
        q1 = next(r for r in results if r["qid"] == "q1")
        assert q1["score_left"] == 8
        assert q1["score_right"] == 3


# ---------------------------------------------------------------------------
# Composite key
# ---------------------------------------------------------------------------


class TestJoinCompositeKey:
    def test_composite_key(self):
        left = [
            {"a": 1, "b": "x", "val": "L1"},
            {"a": 1, "b": "y", "val": "L2"},
        ]
        right_data = [
            {"a": 1, "b": "x", "val": "R1"},
            {"a": 1, "b": "z", "val": "R3"},
        ]
        step = Join(Source.list(right_data), on=["a", "b"])
        results = list(step.process(iter(left)))

        # Only (1, "x") matches.
        assert len(results) == 1
        assert results[0]["a"] == 1
        assert results[0]["b"] == "x"
        assert results[0]["val_left"] == "L1"
        assert results[0]["val_right"] == "R1"


# ---------------------------------------------------------------------------
# One-to-many
# ---------------------------------------------------------------------------


class TestJoinOneToMany:
    def test_one_to_many(self):
        left = [{"user_id": 1, "name": "Alice"}]
        right_data = [
            {"user_id": 1, "action": "login"},
            {"user_id": 1, "action": "purchase"},
            {"user_id": 1, "action": "logout"},
        ]
        step = Join(Source.list(right_data), on="user_id")
        results = list(step.process(iter(left)))

        assert len(results) == 3
        assert all(r["name"] == "Alice" for r in results)
        actions = {r["action"] for r in results}
        assert actions == {"login", "purchase", "logout"}


# ---------------------------------------------------------------------------
# Empty inputs
# ---------------------------------------------------------------------------


class TestJoinEmpty:
    def test_empty_left(self):
        step = Join(Source.list(ACTIONS), on="user_id")
        results = list(step.process(iter([])))
        assert results == []

    def test_empty_right(self):
        step = Join(Source.list([]), on="user_id")
        results = list(step.process(iter(USERS)))
        assert results == []

    def test_empty_right_left_join(self):
        step = Join(Source.list([]), on="user_id", how="left")
        results = list(step.process(iter(USERS)))
        assert len(results) == 3

    def test_empty_left_right_join(self):
        step = Join(Source.list(ACTIONS), on="user_id", how="right")
        results = list(step.process(iter([])))
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


class TestJoinPipeline:
    def test_pipeline_chaining(self):
        pipeline = (
            Source.list(USERS)
            >> Join(Source.list(ACTIONS), on="user_id")
        )
        results = list(pipeline.process(iter([])))
        assert len(results) == 2

    def test_pipeline_run(self):
        pipeline = (
            Source.list(USERS)
            >> Join(Source.list(ACTIONS), on="user_id", how="left")
            >> Map(lambda r: {**r, "processed": True})
        )
        results = pipeline.run()
        assert len(results) == 3
        assert all(r["processed"] for r in results)
