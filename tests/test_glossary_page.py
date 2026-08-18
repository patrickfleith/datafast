"""The published glossary must not drift from the canonical one, or from the code.

`docs-agents/GLOSSARY.md` is the source of truth agents read; `docs/glossary.md` is what
readers get. Two definitions of one term is exactly the failure a glossary exists to
prevent, so the pair is pinned here — and so are the code identifiers the terms name,
since a glossary that describes a class the package no longer exports is worse than none.
"""

import re
from pathlib import Path

import pytest

import datafast

ROOT = Path(__file__).parent.parent
SOURCE = ROOT / "docs-agents" / "GLOSSARY.md"
PAGE = ROOT / "docs" / "glossary.md"

TERM = re.compile(r"^\*\*(?P<term>[^*]+)\*\* — (?P<definition>.+?)\s*(?:_avoid:_|\*\*Avoid:\*\*)\s*(?P<avoid>.+)$")


def _terms(path: Path) -> dict[str, str]:
    """Map every term on a glossary to its definition, ignoring the avoid list."""
    entries = {}
    for line in path.read_text().splitlines():
        match = TERM.match(line)
        if match:
            entries[match["term"]] = match["definition"]
    return entries


def test_every_canonical_term_is_published():
    missing = sorted(set(_terms(SOURCE)) - set(_terms(PAGE)))
    assert not missing, f"defined in docs-agents/GLOSSARY.md but not published: {missing}"


def test_the_page_invents_no_terms_of_its_own():
    extra = sorted(set(_terms(PAGE)) - set(_terms(SOURCE)))
    assert not extra, f"published but not canonical: {extra}"


def test_published_definitions_are_verbatim():
    source, page = _terms(SOURCE), _terms(PAGE)
    drifted = sorted(t for t in source if source[t] != page.get(t))
    assert not drifted, f"definition differs from docs-agents/GLOSSARY.md: {drifted}"


def test_both_sections_survive_publication():
    for heading in ("## Pipelines", "## Models and providers"):
        assert heading in PAGE.read_text()


def test_the_glossary_is_not_empty():
    """Guards every assertion above, all of which pass vacuously on an empty parse."""
    assert len(_terms(SOURCE)) >= 25


def _quoted_names() -> set[str]:
    return set(re.findall(r"`([A-Za-z_][\w.]*(?:\(\))?)`", PAGE.read_text()))


def test_every_class_the_glossary_names_is_exported():
    classes = {n for n in _quoted_names() if re.fullmatch(r"[A-Z][A-Za-z]+", n)}
    assert classes, "the glossary should name some classes"
    unknown = sorted(c for c in classes if c not in datafast.__all__)
    assert not unknown, f"named in the glossary but not exported by datafast: {unknown}"


def test_every_attribute_the_glossary_names_exists():
    dotted = {n for n in _quoted_names() if re.fullmatch(r"[A-Z][A-Za-z]+\.\w+(\(\))?", n)}
    assert dotted, "the glossary should name some methods"
    for name in sorted(dotted):
        owner, attribute = name.rstrip("()").split(".")
        assert hasattr(getattr(datafast, owner), attribute), f"{name} does not exist"


@pytest.mark.parametrize("strategy", ["by_model", "round_robin", "by_record"])
def test_documented_execution_strategies_are_the_real_ones(strategy):
    """The 'Execution strategy' entry lists all three; the enum must agree."""
    values = {s.value for s in datafast.LLMExecutionStrategy}
    assert strategy in values and f"`{strategy}`" in PAGE.read_text()
    assert len(values) == 3, f"the glossary lists three strategies, code has {values}"
