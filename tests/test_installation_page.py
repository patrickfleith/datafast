"""The installation page makes checkable claims, so they are checked.

The valuable direction is code → page: a variable the package reads but the page never
mentions is a variable a user cannot discover.
"""

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "installation.md"
PYPROJECT = ROOT / "pyproject.toml"

# `LANGFUSE_HOST` is written by configure_langfuse_tracing for langfuse itself to read,
# never read back by datafast, so it cannot be discovered by scanning for reads.
SET_NOT_READ = {"LANGFUSE_HOST"}


def _page() -> str:
    return PAGE.read_text()


def _project() -> dict:
    return tomllib.loads(PYPROJECT.read_text())["project"]


def _env_vars_read_by_package() -> set[str]:
    """Every environment variable the package resolves at runtime."""
    found: set[str] = set()
    for path in (ROOT / "datafast").rglob("*.py"):
        source = path.read_text()
        found.update(re.findall(r'os\.getenv\(\s*"([A-Z][A-Z0-9_]+)"', source))
        found.update(re.findall(r'env_key_name\s*=\s*"([A-Z][A-Z0-9_]+)"', source))
        # The suppression flag is referenced through a module constant.
        found.update(re.findall(r'^[A-Z_]+ENV\s*=\s*"([A-Z][A-Z0-9_]+)"', source, re.M))
    return found


def test_every_environment_variable_the_package_reads_is_documented():
    documented = _page()
    undocumented = sorted(v for v in _env_vars_read_by_package() if v not in documented)
    assert not undocumented, (
        f"read by datafast but absent from installation.md: {undocumented}"
    )


def test_documented_langfuse_host_is_still_written_by_the_tracing_module():
    """Guards the one variable the scan above cannot see."""
    tracing = (ROOT / "datafast" / "tracing.py").read_text()
    for name in SET_NOT_READ:
        assert name in tracing and name in _page()


def test_every_extra_the_page_offers_exists():
    declared = set(_project()["optional-dependencies"])
    offered = set(re.findall(r"datafast\[([a-z,]+)\]", _page()))
    # `all` is a real extra; comma forms like `parquet,hub` are not offered on this page.
    assert offered, "the page should name its extras"
    assert offered <= declared, f"page offers unknown extras: {sorted(offered - declared)}"


def test_runtime_dependency_table_matches_pyproject():
    """The page tables the five runtime dependencies; it must not drift from the list."""
    declared = {re.split(r"[><=\[]", d)[0] for d in _project()["dependencies"]}
    for name in declared:
        assert f"`{name}`" in _page(), f"{name} is a runtime dependency but untabled"
    assert len(declared) == 5, "the page says five runtime dependencies"


def test_documented_python_floor_matches_pyproject():
    assert _project()["requires-python"] == ">=3.10"
    assert "3.10" in _page()


@pytest.mark.parametrize(
    "factory,variable",
    [
        ("openai", "OPENAI_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("gemini", "GEMINI_API_KEY"),
        ("mistral", "MISTRAL_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
    ],
)
def test_documented_key_belongs_to_the_factory_the_page_names(factory, variable):
    """Each row of the API-key table must match the factory's own declaration."""
    served_model = (ROOT / "datafast" / "llm" / "served_model.py").read_text()
    assert f'env_key_name="{variable}"' in served_model
    assert re.search(rf"\|\s*`{variable}`\s*\|\s*`{factory}\(\)`", _page()), (
        f"installation.md should pair {variable} with {factory}()"
    )
