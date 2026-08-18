"""The concepts page teaches the execution model, so its code must actually run.

Every self-contained block on the page is executed here, and the numbers the prose
quotes are asserted against what the library returns. None of it touches a provider:
the page deliberately explains the model with non-LLM steps.
"""

import re
from pathlib import Path

import pytest

from datafast.core.validation import PipelineValidationError

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "concepts.md"


def _code_blocks(language: str, doc: Path) -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", doc.read_text(), re.DOTALL)


def _runnable_blocks() -> list[str]:
    """Blocks that stand alone — the ones that import what they use.

    The page also shows a bare `process` signature and a line lifted from the runner;
    neither is meant to run, and neither imports anything.
    """
    return [b for b in _code_blocks("python", PAGE) if "from datafast import" in b]


def test_the_page_has_runnable_blocks():
    assert len(_runnable_blocks()) >= 4


@pytest.mark.parametrize("block", _runnable_blocks(), ids=lambda b: b.split("\n")[0][:40])
def test_every_self_contained_block_executes(block, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exec(compile(block, str(PAGE), "exec"), {})


def _block_containing(needle: str) -> str:
    (block,) = [b for b in _runnable_blocks() if needle in b]
    return block


def test_the_pipeline_block_returns_the_documented_records():
    namespace: dict = {}
    exec(compile(_block_containing("AddUUID()"), str(PAGE), "exec"), namespace)
    records = namespace["records"]
    assert len(records) == 2, "the page's source holds two records"
    # The prose says a step that adds a column returns a dict with one more key.
    assert set(records[0]) == {"text", "length", "id"}


def test_naming_a_step_gives_it_that_name():
    namespace: dict = {}
    exec(compile(_block_containing("as_step"), str(PAGE), "exec"), namespace)
    assert namespace["step"].name == "normalize"


def test_the_seed_block_expands_to_the_documented_four_records():
    namespace: dict = {}
    exec(compile(_block_containing("Seed.product"), str(PAGE), "exec"), namespace)
    # "two topics and two audiences produce four records"
    assert len(list(namespace["seed"].process(iter([])))) == 4


def test_compile_raises_what_the_page_says_it_raises():
    """The page names PipelineValidationError and the rule it enforces."""
    from datafast import Map, Sink, Source

    pipeline = Source.list([{"a": 1}]) >> Sink.list() >> Map(lambda r: r)
    with pytest.raises(PipelineValidationError, match="comes after the sink"):
        pipeline.compile()


def test_every_page_the_concepts_page_links_to_exists():
    """A link to an unwritten page fails `zensical build --strict`, so catch it here."""
    links = re.findall(r"\]\((?!https?:)([^)#]+\.md)", PAGE.read_text())
    assert links, "the page should link onward"
    missing = sorted(link for link in links if not (PAGE.parent / link).exists())
    assert not missing, f"concepts.md links to pages that do not exist: {missing}"
