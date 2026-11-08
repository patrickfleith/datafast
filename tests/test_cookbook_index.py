"""The example datasets grid on the Cookbook index, pinned against what it links to.

Each square names a kind of dataset and points at the cookbook page or the example
script that builds it. A square whose link breaks, or whose script was renumbered, sends
a reader to the wrong place, so every square is resolved here, anchors included.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "cookbook" / "index.md"
EXAMPLES_PAGE = ROOT / "docs" / "cookbook" / "examples.md"
SCRIPTS = ROOT / "examples" / "scripts"


def _grid() -> str:
    match = re.search(
        r'<div class="grid cards showcase" markdown>\n(.*?)\n</div>', PAGE.read_text(), re.DOTALL
    )
    assert match, "no showcase grid on the Cookbook index"
    return match.group(1)


def _squares() -> list[dict]:
    """One dict per square: its title, link target, and the line naming what it links to."""
    squares = []
    for block in re.split(r"^-   ", _grid(), flags=re.MULTILINE)[1:]:
        lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
        title, target = re.fullmatch(r"\*\*\[(.+)\]\((.+)\)\*\*", lines[0]).groups()
        squares.append({"title": title, "target": target, "body": lines[1:-1],
                        "label": lines[-1]})
    return squares


def _slug(heading: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")


def test_the_grid_has_eight_squares_with_distinct_titles():
    """2 x 4 is the layout the stylesheet is built for."""
    titles = [s["title"] for s in _squares()]
    assert len(titles) == 8
    assert len(set(titles)) == 8


@pytest.mark.parametrize("square", _squares(), ids=lambda s: s["title"])
def test_every_square_has_a_description_and_says_what_it_links_to(square):
    assert square["body"], f"{square['title']} has no description"
    assert re.fullmatch(r"Cookbook|Script `\d\d`", square["label"]), square["label"]


@pytest.mark.parametrize("square", _squares(), ids=lambda s: s["title"])
def test_every_square_links_to_a_page_and_heading_that_exist(square):
    path, _, anchor = square["target"].partition("#")
    page = (PAGE.parent / path).resolve()
    assert page.exists(), f"{square['title']} links to {path}, which does not exist"
    if anchor:
        headings = re.findall(r"^#+ (.+)$", page.read_text(), re.MULTILINE)
        assert anchor in {_slug(h) for h in headings}, f"no #{anchor} on {path}"


@pytest.mark.parametrize(
    "square", [s for s in _squares() if s["label"] != "Cookbook"], ids=lambda s: s["title"]
)
def test_every_script_square_points_where_that_script_is_listed(square):
    """The square links to an Examples-page section; the script must be in it."""
    number = re.search(r"`(\d\d)`", square["label"]).group(1)
    (script,) = SCRIPTS.glob(f"{number}_*.py")
    anchor = square["target"].partition("#")[2]
    body = EXAMPLES_PAGE.read_text()
    sections = re.split(r"^## ", body, flags=re.MULTILINE)
    (section,) = [s for s in sections if _slug(s.splitlines()[0]) == anchor]
    assert f"`{script.name}`" in section, f"{script.name} is not under #{anchor}"


def test_every_cookbook_walkthrough_has_a_square():
    """A new cookbook page should earn a place in the grid."""
    walkthroughs = {
        p.name for p in PAGE.parent.glob("*.md") if p.name not in {"index.md", "examples.md"}
    }
    assert walkthroughs, "no cookbook pages found — check the test, not the page"
    linked = {s["target"] for s in _squares()}
    assert walkthroughs <= linked, f"no square for {sorted(walkthroughs - linked)}"


def test_the_grid_stylesheet_is_loaded_by_the_site():
    css = ROOT / "docs" / "stylesheets" / "showcase.css"
    assert css.exists()
    assert ".grid.cards.showcase" in css.read_text()
    assert "- stylesheets/showcase.css" in (ROOT / "mkdocs.yml").read_text()


def test_the_extensions_the_grid_needs_are_enabled():
    """Without md_in_html the grid renders as a raw list inside an empty div."""
    config = (ROOT / "mkdocs.yml").read_text()
    for extension in ("md_in_html", "attr_list"):
        assert f"  - {extension}" in config
