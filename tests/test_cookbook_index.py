"""The cookbook grid and example dataset list on the Cookbook index, pinned against what they link to.

Each square names a cookbook and the script behind it. Each list item names an example
script and the Examples-page section that lists it. A link that breaks, or a script that
was renumbered, sends a reader to the wrong place, so every link is resolved here.
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


def _example_datasets() -> list[dict]:
    """One dict per item in the 'Example datasets' list: its title, link target, and script."""
    section = PAGE.read_text().partition("## Example datasets")[2]
    items = re.findall(r"^- \[(.+?)\]\((.+?)\):(.*?Script `(\d\d)`)", section, re.M | re.S)
    assert items, "no example datasets listed on the Cookbook index"
    return [{"title": t, "target": target, "number": n} for t, target, _, n in items]


def test_the_grid_has_four_squares_with_distinct_titles():
    """2 x 2 is the layout the stylesheet is built for."""
    titles = [s["title"] for s in _squares()]
    assert len(titles) == 4
    assert len(set(titles)) == 4


@pytest.mark.parametrize("square", _squares(), ids=lambda s: s["title"])
def test_every_square_has_a_description_and_says_what_it_links_to(square):
    assert square["body"], f"{square['title']} has no description"
    assert re.fullmatch(r"Script `\d\d`", square["label"]), square["label"]


@pytest.mark.parametrize("square", _squares(), ids=lambda s: s["title"])
def test_every_square_links_to_a_page_and_heading_that_exist(square):
    path, _, anchor = square["target"].partition("#")
    page = (PAGE.parent / path).resolve()
    assert page.exists(), f"{square['title']} links to {path}, which does not exist"
    if anchor:
        headings = re.findall(r"^#+ (.+)$", page.read_text(), re.MULTILINE)
        assert anchor in {_slug(h) for h in headings}, f"no #{anchor} on {path}"


@pytest.mark.parametrize("square", _squares(), ids=lambda s: s["title"])
def test_every_square_names_the_script_its_cookbook_walks_through(square):
    number = re.search(r"`(\d\d)`", square["label"]).group(1)
    (script,) = SCRIPTS.glob(f"{number}_*.py")
    page = (PAGE.parent / square["target"]).read_text()
    assert f"`examples/scripts/{script.name}`" in page, (
        f"{square['target']} does not walk through {script.name}"
    )


@pytest.mark.parametrize("item", _example_datasets(), ids=lambda i: i["title"])
def test_every_example_dataset_points_where_that_script_is_listed(item):
    """The item links to an Examples-page section; the script must be in it."""
    (script,) = SCRIPTS.glob(f"{item['number']}_*.py")
    path, _, anchor = item["target"].partition("#")
    assert path == "examples.md", item["target"]
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
