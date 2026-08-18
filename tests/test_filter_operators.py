"""Pin the `Filter` operators that `docs/api.md` publishes.

The API page renders `Filter`'s docstring, so the operator table there is only as
true as this file. Every row of that table has a case below.
"""

import pytest

from datafast import Filter


def keeps(where: dict, record: dict) -> bool:
    """Whether `where` keeps `record`."""
    return list(Filter(where=where).process([record])) == [record]


@pytest.mark.parametrize(
    "where,record",
    [
        ({"a": {"$eq": 1}}, {"a": 1}),
        ({"a": 1}, {"a": 1}),  # a bare value is $eq
        ({"a": {"$ne": 1}}, {"a": 2}),
        ({"a": {"$gt": 1}}, {"a": 2}),
        ({"a": {"$gte": 2}}, {"a": 2}),
        ({"a": {"$lt": 3}}, {"a": 2}),
        ({"a": {"$lte": 2}}, {"a": 2}),
    ],
)
def test_comparison_operators(where, record):
    assert keeps(where, record)


@pytest.mark.parametrize(
    "where,record",
    [
        ({"a": {"$in": [1, 2]}}, {"a": 1}),
        ({"a": {"$nin": [1, 2]}}, {"a": 3}),
        ({"a": {"$contains": "ell"}}, {"a": "hello"}),  # substring
        ({"a": {"$contains": 2}}, {"a": [1, 2]}),  # list member
        ({"a": {"$all": [1, 2]}}, {"a": [1, 2, 3]}),
        ({"a": {"$any": [9, 2]}}, {"a": [1, 2]}),
    ],
)
def test_membership_operators(where, record):
    assert keeps(where, record)


@pytest.mark.parametrize(
    "where,record",
    [
        ({"a": {"$startswith": "he"}}, {"a": "hello"}),
        ({"a": {"$endswith": "lo"}}, {"a": "hello"}),
        ({"a": {"$regex": "ell"}}, {"a": "hello"}),  # search, not match
    ],
)
def test_string_operators(where, record):
    assert keeps(where, record)


def test_string_operators_reject_non_strings():
    assert not keeps({"a": {"$startswith": "h"}}, {"a": 5})


@pytest.mark.parametrize(
    "where,record",
    [
        ({"a": {"$len_eq": 5}}, {"a": "hello"}),
        ({"a": {"$len_gt": 4}}, {"a": "hello"}),
        ({"a": {"$len_gte": 5}}, {"a": "hello"}),
        ({"a": {"$len_lt": 6}}, {"a": "hello"}),
        ({"a": {"$len_lte": 5}}, {"a": "hello"}),
        ({"a": {"$len_eq": 2}}, {"a": {"x": 1, "y": 2}}),  # anything with a length
    ],
)
def test_length_operators(where, record):
    assert keeps(where, record)


@pytest.mark.parametrize(
    "where,record",
    [
        ({"a": {"$exists": True}}, {"a": 1}),
        ({"a": {"$exists": False}}, {"a": None}),  # null counts as absent
        ({"a": {"$type": "str"}}, {"a": "x"}),
        ({"a": {"$type": "none"}}, {"a": None}),
    ],
)
def test_presence_and_type_operators(where, record):
    assert keeps(where, record)


def test_unknown_type_is_rejected():
    with pytest.raises(ValueError):
        keeps({"a": {"$type": "complex"}}, {"a": 1})


def test_unknown_operator_is_rejected():
    with pytest.raises(ValueError):
        keeps({"a": {"$nope": 1}}, {"a": 1})


@pytest.mark.parametrize(
    "where,record,expected",
    [
        ({"$or": [{"a": 1}, {"a": 2}]}, {"a": 2}, True),
        ({"$or": [{"a": 1}, {"a": 2}]}, {"a": 3}, False),
        ({"$and": [{"a": 1}, {"b": 2}]}, {"a": 1, "b": 2}, True),
        ({"$and": [{"a": 1}, {"b": 2}]}, {"a": 1, "b": 9}, False),
    ],
)
def test_logical_operators(where, record, expected):
    assert keeps(where, record) is expected


def test_conditions_combine_conjunctively():
    """Several operators on a column, and several columns, must all hold."""
    assert keeps({"a": {"$len_gt": 1, "$len_lt": 9}}, {"a": "hello"})
    assert not keeps({"a": {"$len_gt": 1, "$len_lt": 3}}, {"a": "hello"})
    assert keeps({"a": 1, "b": 2}, {"a": 1, "b": 2})
    assert not keeps({"a": 1, "b": 2}, {"a": 1, "b": 9})


def test_missing_column_fails_its_condition():
    """A missing column is not an error — except under `$exists: False`."""
    assert not keeps({"a": {"$gt": 1}}, {"b": 1})
    assert keeps({"a": {"$exists": False}}, {"b": 1})


def test_keep_false_inverts_the_match():
    assert list(Filter(where={"a": 1}, keep=False).process([{"a": 1}])) == []
    assert list(Filter(where={"a": 1}, keep=False).process([{"a": 2}])) == [{"a": 2}]
