"""Pin `docs/api.md` against the public surface.

The page renders docstrings through mkdocstrings, so its *content* cannot drift
from the code. Its *coverage* still can: a new export is only documented once a
`:::` directive names it. The hand-maintained page this replaced had drifted to
34 of 48 names.
"""

import re
from pathlib import Path

import datafast

API_PAGE = Path(__file__).resolve().parent.parent / "docs" / "api.md"
MKDOCS_YML = Path(__file__).resolve().parent.parent / "mkdocs.yml"


def documented_names() -> set[str]:
    """Names pulled in by a `::: datafast.Name` directive."""
    return set(re.findall(r"^::: datafast\.(\w+)$", API_PAGE.read_text(), re.MULTILINE))


def test_every_public_name_is_on_the_api_page():
    assert documented_names() == set(datafast.__all__)


def test_api_page_documents_nothing_it_should_not():
    """A directive naming a non-export would render, but off the public surface."""
    for name in documented_names():
        assert hasattr(datafast, name), name


def test_mkdocstrings_is_configured():
    """The page is directives only — without the plugin it renders as raw text."""
    assert "mkdocstrings" in MKDOCS_YML.read_text()
