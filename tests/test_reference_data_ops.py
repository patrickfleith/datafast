"""The data operations reference, pinned against the code it documents.

Same contract as ``test_reference_sources_and_seed.py``:

1. Every parameter the code takes, and every ``Filter`` operator it implements, is named
   on the page (code → docs). Something that exists and is undocumented cannot be found.
2. Every example on the page executes.
3. The behavioural claims — the ones a reader would be burned by if wrong — are asserted
   against real calls on real records.
"""

import inspect
import re
from pathlib import Path

import pytest

from datafast import AddUUID, Concat, Filter, FlatMap, Group, Join, Map, Pair, Source
from datafast.core.step import Step
from datafast.core.validation import PipelineValidationError
from datafast.transforms import data_ops

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "reference" / "data_ops.md"
SOURCE = ROOT / "datafast" / "transforms" / "data_ops.py"

DOCUMENTED = [Map, FlatMap, AddUUID, Filter, Group, Pair, Concat, Join]


def _page() -> str:
    return PAGE.read_text()


def _code_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", _page(), re.DOTALL)


def _public_steps() -> list[type]:
    return [
        obj for obj in vars(data_ops).values()
        if inspect.isclass(obj)
        and issubclass(obj, Step)
        and obj.__module__ == data_ops.__name__
    ]


def _implemented_operators() -> set[str]:
    return set(re.findall(r"\$[a-z_]+", SOURCE.read_text()))


# --------------------------------------------------------------------------- code → docs


def test_the_page_documents_every_public_step():
    """A step the page forgets is one a reader never finds."""
    steps = _public_steps()
    assert len(steps) >= 8, "introspection found no steps — check the test, not the page"
    missing = [s.__name__ for s in steps if f"`{s.__name__}(" not in _page()]
    assert not missing, f"steps defined in data_ops.py but undocumented: {missing}"
    assert {s.__name__ for s in steps} == {s.__name__ for s in DOCUMENTED}


@pytest.mark.parametrize("step", DOCUMENTED, ids=lambda s: s.__name__)
def test_every_parameter_is_named_on_the_page(step):
    parameters = [
        p.name for p in inspect.signature(step).parameters.values()
        if p.kind is not p.VAR_KEYWORD
    ]
    assert parameters, f"{step.__name__} takes nothing — check the test, not the page"
    missing = [p for p in parameters if f"`{p}`" not in _page()]
    assert not missing, f"{step.__name__} takes {missing}, undocumented on the page"


def test_every_filter_operator_the_code_implements_is_on_the_page():
    operators = _implemented_operators()
    assert len(operators) >= 23, "found no operators in the source — check the test"
    missing = sorted(op for op in operators if f"`{op}`" not in _page())
    assert not missing, f"operators the code implements but the page omits: {missing}"


def test_every_aggregation_function_and_pair_option_is_on_the_page():
    assert Group.AGG_FUNCTIONS, "no aggregation functions found — check the test"
    for func in Group.AGG_FUNCTIONS:
        assert f"`{func}`" in _page(), f"aggregation '{func}' undocumented"
    for strategy in Pair.VALID_STRATEGIES:
        assert f'`"{strategy}"`' in _page(), f"strategy '{strategy}' undocumented"
    for fmt in Pair.VALID_OUTPUT_FORMATS:
        assert f'`"{fmt}"`' in _page(), f"output_format '{fmt}' undocumented"
    for how in Join.VALID_HOW:
        assert f'`"{how}"`' in _page(), f"join type '{how}' undocumented"
    for name in Filter.TYPE_MAP:
        assert f'`"{name}"`' in _page(), f"$type value '{name}' undocumented"


def test_every_example_executes():
    blocks = _code_blocks()
    assert len(blocks) >= 8, "the page lost its examples"
    for block in blocks:
        exec(compile(block, str(PAGE), "exec"), {})


def test_every_page_linked_to_exists():
    links = re.findall(r"\]\((?!https?:)([^)#]+\.md)", _page())
    assert links, "the page links nowhere"
    missing = sorted(link for link in links if not (PAGE.parent / link).resolve().exists())
    assert not missing, f"links to pages that do not exist: {missing}"


# --------------------------------------------------------------------------- Map, FlatMap


def test_map_replaces_the_record_rather_than_merging_it():
    """The page warns that nothing is merged for you."""
    step = Map(lambda r: {"length": len(r["text"])})
    assert list(step.process(iter([{"text": "hello", "keep": 1}]))) == [{"length": 5}]


def test_flatmap_returning_an_empty_list_drops_the_record():
    step = FlatMap(lambda r: [] if r["drop"] else [r])
    assert list(step.process(iter([{"drop": True}, {"drop": False}]))) == [{"drop": False}]


# --------------------------------------------------------------------------- AddUUID


def test_adduuid_defaults_to_id_and_writes_a_uuid_string():
    import uuid

    (record,) = AddUUID().process(iter([{"text": "hi"}]))
    assert record["text"] == "hi"
    assert uuid.UUID(record["id"]).version == 4


def test_adduuid_keeps_an_existing_value_even_when_it_is_none():
    """The page says the check is presence, not emptiness."""
    assert list(AddUUID().process(iter([{"id": None}]))) == [{"id": None}]
    (record,) = AddUUID(overwrite=True).process(iter([{"id": None}]))
    assert record["id"] is not None


# --------------------------------------------------------------------------- Filter

OPERATORS = [
    ("$eq", 5, 5, 6),
    ("$ne", 5, 6, 5),
    ("$gt", 5, 6, 5),
    ("$gte", 5, 5, 4),
    ("$lt", 5, 4, 5),
    ("$lte", 5, 5, 6),
    ("$in", [1, 2], 1, 3),
    ("$nin", [1, 2], 3, 1),
    ("$contains", "ab", "xaby", "xy"),
    ("$all", [1, 2], [1, 2, 3], [1, 3]),
    ("$any", [1, 2], [2, 9], [9]),
    ("$startswith", "ab", "abc", "xabc"),
    ("$endswith", "bc", "abc", "abcd"),
    ("$regex", r"a\d", "xa1", "ab"),
    ("$len_eq", 3, "abc", "ab"),
    ("$len_gt", 2, "abc", "ab"),
    ("$len_gte", 3, "abc", "ab"),
    ("$len_lt", 3, "ab", "abc"),
    ("$len_lte", 2, "ab", "abc"),
    ("$exists", True, "x", None),
    ("$type", "int", 5, "5"),
]


def test_the_operator_table_covers_every_operator_the_code_implements():
    """Without this, an operator added to the code is silently untested."""
    logical = {"$or", "$and"}
    assert _implemented_operators() - logical == {op for op, *_ in OPERATORS}


@pytest.mark.parametrize("op,argument,matching,failing", OPERATORS, ids=[o[0] for o in OPERATORS])
def test_each_operator_keeps_what_the_page_says_it_keeps(op, argument, matching, failing):
    step = Filter(where={"c": {op: argument}})
    kept = list(step.process(iter([{"c": matching}, {"c": failing}])))
    assert kept == [{"c": matching}]


def test_a_bare_value_is_equality_and_several_conditions_must_all_hold():
    step = Filter(where={"score": {"$gte": 7}, "category": "science"})
    records = [
        {"score": 9, "category": "science"},
        {"score": 9, "category": "history"},
        {"score": 3, "category": "science"},
    ]
    assert list(step.process(iter(records))) == [records[0]]


def test_string_operators_reject_non_strings_instead_of_failing():
    step = Filter(where={"c": {"$startswith": "a"}})
    assert list(step.process(iter([{"c": ["a", "b"]}, {"c": 1}]))) == []


def test_length_operators_apply_to_lists_and_dicts_too():
    step = Filter(where={"c": {"$len_gt": 1}})
    kept = list(step.process(iter([{"c": [1, 2]}, {"c": {"a": 1, "b": 2}}, {"c": [1]}])))
    assert kept == [{"c": [1, 2]}, {"c": {"a": 1, "b": 2}}]


def test_a_missing_column_fails_most_conditions_but_satisfies_three():
    record = {"other": 1}
    for where in ({"c": {"$gt": 0}}, {"c": {"$in": [1]}}, {"c": {"$exists": True}}, {"c": 1}):
        assert list(Filter(where=where).process(iter([record]))) == []
    for where in ({"c": {"$ne": 1}}, {"c": {"$nin": [1]}}, {"c": {"$exists": False}}):
        assert list(Filter(where=where).process(iter([record]))) == [record]


def test_or_keeps_a_record_matching_any_condition():
    step = Filter(where={"$or": [{"category": "science"}, {"score": {"$gte": 9}}]})
    records = [
        {"category": "history", "score": 9},
        {"category": "science", "score": 2},
        {"category": "history", "score": 2},
    ]
    assert list(step.process(iter(records))) == records[:2]


def test_and_keeps_a_record_matching_all_conditions():
    step = Filter(where={"$and": [{"a": 1}, {"b": 2}]})
    assert list(step.process(iter([{"a": 1, "b": 2}, {"a": 1, "b": 3}]))) == [{"a": 1, "b": 2}]


def test_a_logical_operator_silently_ignores_its_siblings():
    """The page's sharpest Filter trap: the score condition below does nothing."""
    step = Filter(where={"$or": [{"a": 1}], "score": {"$gt": 100}})
    assert list(step.process(iter([{"a": 1, "score": 0}]))) == [{"a": 1, "score": 0}]


def test_keep_false_inverts_the_condition():
    step = Filter(where={"quality": {"$lt": 3}}, keep=False)
    assert list(step.process(iter([{"quality": 1}, {"quality": 5}]))) == [{"quality": 5}]


def test_fn_filters_and_obeys_keep():
    records = [{"text": "long enough"}, {"text": "no"}]
    assert list(Filter(fn=lambda r: len(r["text"]) > 5).process(iter(records))) == [records[0]]
    assert list(Filter(fn=lambda r: len(r["text"]) > 5, keep=False).process(iter(records))) == [records[1]]


def test_filter_needs_exactly_one_of_fn_and_where():
    with pytest.raises(ValueError, match="either"):
        Filter()
    with pytest.raises(ValueError, match="both"):
        Filter(fn=lambda r: True, where={"a": 1})


def test_an_unknown_operator_is_an_error():
    with pytest.raises(ValueError, match="Unknown operator"):
        list(Filter(where={"a": {"$nope": 1}}).process(iter([{"a": 1}])))


# --------------------------------------------------------------------------- Group

CHUNKS = [
    {"doc_id": 1, "chunk": "a", "score": 2, "extra": "dropped"},
    {"doc_id": 1, "chunk": "b", "score": 4, "extra": "dropped"},
    {"doc_id": 2, "chunk": "c", "score": None, "extra": "dropped"},
]


def test_group_output_holds_only_by_collect_and_agg_columns():
    """The page's central Group claim: every other column is dropped."""
    step = Group(by="doc_id", collect="chunk", agg={"n": "chunk:count"})
    first, _ = step.process(iter(CHUNKS))
    assert first == {"doc_id": 1, "chunk_list": ["a", "b"], "n": 2}


def test_collect_names_the_column_col_list_unless_output_column_renames_it():
    (record,) = Group(by="doc_id", collect="chunk").process(iter(CHUNKS[:2]))
    assert "chunk_list" in record
    (renamed,) = Group(by="doc_id", collect="chunk", output_column="chunks").process(iter(CHUNKS[:2]))
    assert "chunks" in renamed and "chunk_list" not in renamed


def test_output_column_is_ignored_when_several_columns_are_collected():
    """The page says output_column only applies to a single collected column."""
    (record,) = Group(
        by="doc_id", collect=["chunk", "score"], output_column="chunks"
    ).process(iter(CHUNKS[:2]))
    assert "chunks" not in record
    assert record["chunk_list"] == ["a", "b"] and record["score_list"] == [2, 4]


def test_every_documented_aggregation_function_computes_what_the_page_says():
    step = Group(by="doc_id", agg={
        "count": "score:count", "sum": "score:sum", "mean": "score:mean",
        "min": "score:min", "max": "score:max", "first": "chunk:first",
        "last": "chunk:last", "collect": "chunk:collect", "concat": "chunk:concat",
    })
    first, second = step.process(iter(CHUNKS))
    assert first == {
        "doc_id": 1, "count": 2, "sum": 6, "mean": 3.0, "min": 2, "max": 4,
        "first": "a", "last": "b", "collect": ["a", "b"], "concat": "a\nb",
    }
    assert second["count"] == 1, "count counts records, including null values"
    assert second["mean"] is None, "mean of nothing but nulls is None"


def test_concat_takes_a_separator_as_a_third_part_and_defaults_to_a_newline():
    (record,) = Group(by="doc_id", agg={"j": "chunk:concat: | "}).process(iter(CHUNKS[:2]))
    assert record["j"] == "a | b"
    (default,) = Group(by="doc_id", agg={"j": "chunk:concat"}).process(iter(CHUNKS[:2]))
    assert default["j"] == "a\nb"


def test_min_per_group_drops_small_groups_and_max_per_group_truncates_them():
    small = Group(by="doc_id", collect="chunk", min_per_group=2)
    assert [r["doc_id"] for r in small.process(iter(CHUNKS))] == [1]
    capped = Group(by="doc_id", collect="chunk", max_per_group=1)
    assert [r["chunk_list"] for r in capped.process(iter(CHUNKS))] == [["a"], ["c"]]


def test_a_bad_aggregation_spec_fails_when_the_step_is_built():
    with pytest.raises(ValueError, match="Invalid aggregation spec"):
        Group(by="a", agg={"n": "score"})
    with pytest.raises(ValueError, match="Unknown aggregation function"):
        Group(by="a", agg={"n": "score:median"})


def test_group_accepts_several_key_columns():
    records = [{"a": 1, "b": 1, "c": "x"}, {"a": 1, "b": 2, "c": "y"}]
    assert len(list(Group(by=["a", "b"], collect="c").process(iter(records)))) == 2


# --------------------------------------------------------------------------- Pair

PAIRABLE = [{"doc_id": 1, "chunk": c} for c in "abc"]


def test_columns_output_prefixes_every_column_with_chunk_n():
    (record,) = Pair(n=2, strategy="sequential").process(iter(PAIRABLE[:2]))
    assert record == {
        "chunk_1_doc_id": 1, "chunk_1_chunk": "a",
        "chunk_2_doc_id": 1, "chunk_2_chunk": "b",
    }


def test_list_output_holds_the_whole_records_plus_one_list_per_column():
    (record,) = Pair(n=2, strategy="sequential", output_format="list").process(iter(PAIRABLE[:2]))
    assert record["chunks"] == PAIRABLE[:2]
    assert record["chunk_list"] == ["a", "b"] and record["doc_id_list"] == [1, 1]


def test_sequential_sliding_and_all_produce_the_documented_tuples():
    def chunks(step):
        return [(r["chunk_1_chunk"], r["chunk_2_chunk"]) for r in step.process(iter(PAIRABLE))]

    assert chunks(Pair(strategy="sequential")) == [("a", "b")]
    assert chunks(Pair(strategy="sliding")) == [("a", "b"), ("b", "c")]
    assert chunks(Pair(strategy="all")) == [("a", "b"), ("a", "c"), ("b", "c")]


def test_n_sets_the_tuple_size():
    (record,) = Pair(n=3, strategy="sliding").process(iter(PAIRABLE))
    assert record["chunk_3_chunk"] == "c"


def test_within_pairs_only_records_sharing_the_column():
    records = [{"doc": 1, "c": "a"}, {"doc": 2, "c": "b"}, {"doc": 1, "c": "z"}]
    step = Pair(strategy="all", within="doc")
    pairs = [(r["chunk_1_c"], r["chunk_2_c"]) for r in step.process(iter(records))]
    assert pairs == [("a", "z")]


def test_across_requires_the_records_to_differ():
    records = [{"author": "x", "c": 1}, {"author": "x", "c": 2}, {"author": "y", "c": 3}]
    step = Pair(strategy="all", across="author")
    pairs = [(r["chunk_1_c"], r["chunk_2_c"]) for r in step.process(iter(records))]
    assert pairs == [(1, 3), (2, 3)]


def test_a_group_smaller_than_n_produces_nothing():
    assert list(Pair(n=3, strategy="all").process(iter(PAIRABLE[:2]))) == []


def test_max_pairs_caps_the_total_and_seed_makes_random_repeatable():
    assert len(list(Pair(strategy="all", max_pairs=2).process(iter(PAIRABLE)))) == 2
    run = lambda: list(Pair(seed=7, max_pairs=5).process(iter(PAIRABLE)))
    assert run() == run()
    assert run() != list(Pair(seed=8, max_pairs=5).process(iter(PAIRABLE)))


def test_random_without_max_pairs_makes_a_hundred_thousand_tuples():
    """The number the page warns about, measured rather than guessed."""
    assert len(list(Pair(seed=0).process(iter(PAIRABLE)))) == 100_000


def test_pair_rejects_bad_arguments_when_the_step_is_built():
    with pytest.raises(ValueError, match="at least 2"):
        Pair(n=1)
    with pytest.raises(ValueError, match="Invalid strategy"):
        Pair(strategy="pairwise")
    with pytest.raises(ValueError, match="Invalid output_format"):
        Pair(output_format="dict")


# --------------------------------------------------------------------------- Concat


def test_concat_yields_each_source_in_order_and_discards_upstream_records():
    step = Concat(Source.list([{"text": "a"}]), Source.list([{"text": "b"}]))
    assert list(step.process(iter([{"text": "upstream"}]))) == [{"text": "a"}, {"text": "b"}]


def test_concat_needs_at_least_one_source():
    with pytest.raises(ValueError, match="at least one source"):
        Concat()


def test_concat_must_be_the_first_step():
    pipeline = Source.list([{"a": 1}]) >> Concat(Source.list([{"a": 2}]))
    with pytest.raises(PipelineValidationError, match="discards upstream records"):
        pipeline.compile()


# --------------------------------------------------------------------------- Join

LEFT = [{"user_id": 1, "name": "ada"}, {"user_id": 3, "name": "bob"}]
RIGHT = Source.list([{"user_id": 1, "action": "click"}, {"user_id": 2, "action": "scroll"}])


def test_inner_join_keeps_only_keys_found_on_both_sides():
    step = Join(RIGHT, on="user_id")
    assert list(step.process(iter(LEFT))) == [{"user_id": 1, "name": "ada", "action": "click"}]


def test_how_selects_which_unmatched_records_survive():
    def keys(how):
        return sorted(r["user_id"] for r in Join(RIGHT, on="user_id", how=how).process(iter(LEFT)))

    assert keys("inner") == [1]
    assert keys("left") == [1, 3]
    assert keys("right") == [1, 2]
    assert keys("outer") == [1, 2, 3]


def test_an_unmatched_record_leaves_the_other_side_absent_not_null():
    """The page promises no null-filling."""
    (_, unmatched) = Join(RIGHT, on="user_id", how="left").process(iter(LEFT))
    assert unmatched == {"user_id": 3, "name": "bob"}, "no 'action' column at all"


def test_overlapping_columns_get_the_suffixes_and_the_key_never_does():
    right = Source.list([{"qid": 1, "text": "rejected"}])
    step = Join(right, on="qid", suffixes=("_chosen", "_rejected"))
    (record,) = step.process(iter([{"qid": 1, "text": "chosen"}]))
    assert record == {"qid": 1, "text_chosen": "chosen", "text_rejected": "rejected"}


def test_a_repeated_key_yields_one_record_per_combination():
    right = Source.list([{"k": 1, "r": "x"}, {"k": 1, "r": "y"}])
    step = Join(right, on="k")
    assert len(list(step.process(iter([{"k": 1, "l": 1}, {"k": 1, "l": 2}])))) == 4


def test_records_missing_the_key_column_join_on_none():
    right = Source.list([{"other": "r"}])
    (record,) = Join(right, on="k").process(iter([{"other": "l"}]))
    assert record["k"] is None


def test_join_accepts_several_key_columns_and_rejects_an_unknown_how():
    right = Source.list([{"a": 1, "b": 2, "r": "x"}])
    assert len(list(Join(right, on=["a", "b"]).process(iter([{"a": 1, "b": 2}])))) == 1
    with pytest.raises(ValueError, match="how must be one of"):
        Join(right, on="a", how="cross")


# --------------------------------------------------------------------------- compile()


def test_compile_checks_the_columns_these_steps_name():
    for pipeline in (
        Source.list([{"a": 1}]) >> Group(by="missing"),
        Source.list([{"a": 1}]) >> Pair(within="missing"),
        Source.list([{"a": 1}]) >> Join(Source.list([{"missing": 1}]), on="missing"),
    ):
        with pytest.raises(PipelineValidationError, match="missing"):
            pipeline.compile()


def test_compile_stops_checking_columns_after_an_opaque_step():
    """The page says these steps make the later schema unknowable."""
    pipeline = Source.list([{"a": 1}]) >> Map(lambda r: r) >> Group(by="missing")
    assert pipeline.compile() is pipeline
