"""The source reference, pinned against the code it documents.

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

from datafast import Map, Source
from datafast.core.validation import PipelineValidationError

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "reference" / "source.md"

DOCUMENTED = [
    Source.list, Source.file, Source.jsonl, Source.csv, Source.tsv, Source.txt,
    Source.parquet, Source.huggingface,
]


def _page() -> str:
    return PAGE.read_text()


def _code_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", _page(), re.DOTALL)


def test_the_page_documents_every_public_constructor():
    """A constructor the page forgets is one a reader never finds."""
    names = [n for n in dir(Source) if not n.startswith("_")]
    assert names, "Source exposes nothing — check the test, not the page"
    for name in names:
        assert f"Source.{name}" in _page(), f"Source.{name} undocumented"


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


def test_a_pipeline_must_start_with_a_source():
    """The page's positional claim, in both directions."""
    with pytest.raises(PipelineValidationError, match="must start with a source"):
        (Map(lambda r: r) >> Map(lambda r: r)).compile()
    with pytest.raises(PipelineValidationError, match="may only be the first step"):
        (Source.list([{"a": 1}]) >> Source.list([{"a": 2}])).compile()


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
    assert links, "the page links nowhere — check the test, not the page"
    missing = sorted(link for link in links if not (PAGE.parent / link).resolve().exists())
    assert not missing, f"links to pages that do not exist: {missing}"
