"""`docs/whats_in_v1.md` is the release page. It summarises the other pages, so its
version, its links and its example all have to track the code rather than a snapshot."""

import ast
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "whats_in_v1.md"


def _page() -> str:
    return PAGE.read_text()


def _flat() -> str:
    """Claims are checked against the page with its line wrapping collapsed."""
    return " ".join(_page().lower().split())


def _links() -> list[str]:
    links = sorted(set(re.findall(r"\]\((?!https?:)([^)#]+\.md)", _page())))
    assert links, "no relative links found — the test would pass vacuously"
    return links


def test_the_version_it_names_is_the_one_being_released():
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert f"datafast {version}" in _page()


@pytest.mark.parametrize("link", _links())
def test_every_page_it_links_to_exists(link):
    assert (PAGE.parent / link).resolve().exists(), f"{link} is linked but not on disk"


@pytest.mark.parametrize("block", re.findall(r"```python\n(.*?)```", _page(), re.DOTALL))
def test_every_example_is_valid_python(block):
    ast.parse(block)


def test_the_example_uses_parameters_the_step_really_takes():
    """A release page showing a signature that does not exist is worse than none."""
    import inspect

    from datafast import LLMStep

    block = re.search(r"```python\n(.*?)```", _page(), re.DOTALL).group(1)
    call = next(
        node
        for node in ast.walk(ast.parse(block))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "LLMStep"
    )
    real = set(inspect.signature(LLMStep.__init__).parameters)
    used = {kw.arg for kw in call.keywords}
    assert used, "the example passes no keyword arguments — check the test"
    assert used <= real, f"the page passes {used - real}, which LLMStep does not take"


def test_it_says_there_is_no_migration_guide():
    """The task that specified this page ruled one out; readers must not wait for it."""
    assert "no migration guide" in _flat()


@pytest.mark.parametrize(
    "claim",
    ["no async api", "no progress bar", "no nested branching"],
)
def test_every_deliberate_omission_is_still_true(claim):
    assert claim in _flat()


def test_the_package_really_has_no_async_api():
    """The page promises `run()` blocks. If an async path lands, this page is stale."""
    sources = (ROOT / "datafast").rglob("*.py")
    offenders = [p.name for p in sources if "async def" in p.read_text()]
    assert not offenders, f"async API added in {offenders} — update the page"


def test_the_page_is_reachable_from_the_nav():
    assert "whats_in_v1.md" in (ROOT / "mkdocs.yml").read_text()
