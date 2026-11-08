"""The seed reference, pinned against the code it documents.

Same contract as ``test_reference_source.py``:

1. Every parameter the code takes is named on the page (code → docs).
2. Every self-contained example executes.
3. The behavioural claims — the ones a reader would be burned by if wrong — are
   asserted against real calls, not paraphrased from the source.
"""

import inspect
import re
from pathlib import Path

import pytest

from datafast import Map, Seed
from datafast.core.validation import PipelineValidationError

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "reference" / "seed.md"

DOCUMENTED = [Seed.values, Seed.range, Seed.expand, Seed.product, Seed.zip]


def _page() -> str:
    return PAGE.read_text()


def _code_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", _page(), re.DOTALL)


def test_the_page_documents_every_public_constructor():
    """A constructor the page forgets is one a reader never finds."""
    names = [n for n in dir(Seed) if not n.startswith("_")]
    assert names, "Seed exposes nothing — check the test, not the page"
    for name in names:
        assert f"Seed.{name}" in _page(), f"Seed.{name} undocumented"


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
    monkeypatch.chdir(tmp_path)
    exec(compile(block, str(PAGE), "exec"), {})


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
    with pytest.raises(TypeError):
        Seed.values("a", [1]) >> Map(lambda r: r)


def test_a_combined_seed_only_goes_first():
    """The page says a seed comes first, like any other source."""
    seed = Seed.product(Seed.values("a", [1]))
    with pytest.raises(PipelineValidationError, match="may only be the first step"):
        (seed >> Seed.product(Seed.values("b", [2]))).compile()


def test_every_page_linked_to_exists():
    links = re.findall(r"\]\((?!https?:)([^)#]+\.md)", _page())
    assert links, "the page links nowhere — check the test, not the page"
    missing = sorted(link for link in links if not (PAGE.parent / link).resolve().exists())
    assert not missing, f"links to pages that do not exist: {missing}"
