"""The sources & seed reference, pinned against the code it documents.

This is the template for the rest of the step reference. Three things are checked, in
order of how much they matter:

1. Every parameter the code takes is named on the page (code → docs). A parameter that
   exists and is undocumented cannot be discovered; the reverse is merely untidy.
2. Every self-contained example executes.
3. The behavioural claims — the ones a reader would be burned by if wrong — are
   asserted against real calls, not paraphrased from the source.
"""

import inspect
import re
from pathlib import Path

import pytest

from datafast import Seed, Source

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "reference" / "sources_and_seed.md"

DOCUMENTED = [
    Source.list, Source.file, Source.jsonl, Source.csv, Source.tsv, Source.txt,
    Source.parquet, Source.huggingface,
    Seed.values, Seed.range, Seed.expand, Seed.product, Seed.zip,
]


def _page() -> str:
    return PAGE.read_text()


def _code_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", _page(), re.DOTALL)


def test_the_page_documents_every_public_constructor():
    """A constructor the page forgets is one a reader never finds."""
    for factory in (Source, Seed):
        for name in [n for n in dir(factory) if not n.startswith("_")]:
            assert f"{factory.__name__}.{name}" in _page(), f"{factory.__name__}.{name} undocumented"


@pytest.mark.parametrize("func", DOCUMENTED, ids=lambda f: f.__qualname__)
def test_every_parameter_is_named_on_the_page(func):
    parameters = [
        p.name for p in inspect.signature(func).parameters.values()
        if p.kind is not p.VAR_KEYWORD
    ]
    assert parameters, f"{func.__qualname__} takes nothing — check the test, not the page"
    missing = [p for p in parameters if f"`{p}`" not in _page()]
    assert not missing, f"{func.__qualname__} takes {missing}, undocumented on the page"


@pytest.mark.parametrize("block", _code_blocks(), ids=lambda b: b.split("\n")[0][:40])
def test_every_example_executes(block, tmp_path, monkeypatch):
    """Examples that read a file are constructed, not run — construction is the claim."""
    monkeypatch.chdir(tmp_path)
    exec(compile(block, str(PAGE), "exec"), {})


def test_the_documented_extension_table_matches_the_detector():
    """The page tables the extension → format mapping; the detector owns the truth."""
    documented = {
        ".jsonl": "jsonl", ".json": "jsonl", ".csv": "csv", ".tsv": "tsv",
        ".txt": "txt", ".parquet": "parquet", ".pq": "parquet",
    }
    for extension, expected in documented.items():
        source = Source.file(f"x{extension}")
        assert source._format == expected, f"{extension} is not {expected}"
        assert f"`{extension}`" in _page()


def test_an_unknown_extension_is_rejected_at_construction():
    """The page promises this happens before the pipeline runs."""
    with pytest.raises(ValueError, match="Cannot auto-detect format"):
        Source.file("notes.md")


def test_range_end_is_inclusive_as_documented():
    values = [r["grade"] for r in Seed.product(Seed.range("grade", 1, 12)).process(iter([]))]
    assert values == list(range(1, 13)), "the page says twelve values, 1 through 12"


def test_expand_is_one_dimension_not_two():
    """The page's central claim about expand: Physics is never paired with Genetics."""
    dimension = Seed.expand("topic", "subtopic", {
        "Physics": ["Quantum", "Relativity"],
        "Biology": ["Genetics", "Evolution"],
    })
    assert len(dimension) == 4
    pairs = {(v["topic"], v["subtopic"]) for v in dimension.values}
    assert ("Physics", "Genetics") not in pairs


def test_product_multiplies_and_len_reports_it_before_running():
    seed = Seed.product(Seed.values("a", [1, 2]), Seed.values("b", [1, 2, 3]))
    assert len(seed) == 6


def test_zip_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        Seed.zip(Seed.values("a", [1, 2]), Seed.values("b", [1]))


def test_combining_nothing_gives_an_empty_source():
    assert len(Seed.product()) == 0 and len(Seed.zip()) == 0


def test_overlapping_columns_let_the_last_dimension_win():
    """Documented as a silent overwrite, which is why the page warns about it."""
    records = list(Seed.product(Seed.values("a", [1]), Seed.values("a", [9])).process(iter([])))
    assert records == [{"a": 9}]


def test_a_dimension_is_not_a_step():
    """The page calls this the most common mistake with seeds."""
    from datafast import Map

    with pytest.raises(TypeError):
        Seed.values("a", [1]) >> Map(lambda r: r)


def test_csv_values_arrive_as_strings(tmp_path):
    path = tmp_path / "scores.csv"
    path.write_text("name,score\nada,1\n")
    (record,) = list(Source.csv(path).process(iter([])))
    assert record == {"name": "ada", "score": "1"}, "the page says no type inference"


def test_a_malformed_jsonl_line_is_skipped_not_raised(tmp_path):
    path = tmp_path / "partial.jsonl"
    path.write_text('{"a": 1}\nnot json at all\n{"a": 2}\n')
    assert list(Source.jsonl(path).process(iter([]))) == [{"a": 1}, {"a": 2}]


def test_txt_skips_blank_lines_and_uses_the_documented_default_column(tmp_path):
    path = tmp_path / "lines.txt"
    path.write_text("one\n\ntwo\n")
    assert list(Source.txt(path).process(iter([]))) == [{"text": "one"}, {"text": "two"}]


def test_the_extras_the_page_names_are_the_ones_the_code_names():
    """Both steps raise ImportError naming an extra; the page must name the same one."""
    source = (ROOT / "datafast" / "sources" / "source.py").read_text()
    for extra in ("datafast[parquet]", "datafast[hub]"):
        assert extra in source and extra in _page()


def test_every_page_linked_to_exists():
    links = re.findall(r"\]\((?!https?:)([^)#]+\.md)", _page())
    missing = sorted(link for link in links if not (PAGE.parent / link).resolve().exists())
    assert not missing, f"links to pages that do not exist: {missing}"
