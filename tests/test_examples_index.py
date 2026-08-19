"""The examples index, pinned against `examples/scripts/`.

The directory is the source of truth. The whole point of the page is that it stays
complete, so the two guards that matter are:

1. Every script in `examples/scripts/` appears on the page.
2. Every script the page names exists.

Everything after that checks the claims the page makes about the scripts: which ones need
no API key, which ones publish, and which ones have a walkthrough.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "cookbook" / "examples.md"
SCRIPTS = ROOT / "examples" / "scripts"

PROVIDER_FACTORIES = (
    "ollama",
    "openai",
    "anthropic",
    "gemini",
    "mistral",
    "openrouter",
    "openai_compatible",
)


def _page() -> str:
    return PAGE.read_text()


def _scripts() -> list[Path]:
    return sorted(p for p in SCRIPTS.glob("*.py") if not p.name.startswith("_"))


def _named_on_page() -> set[str]:
    return set(re.findall(r"`(\d{2}_[\w]+\.py)`", _page()))


def _calls_a_provider(source: str) -> bool:
    """A provider factory called in live code, not in a comment."""
    for factory in PROVIDER_FACTORIES:
        if re.search(rf"^\s*[^#\n]*\b{factory}\(", source, re.M):
            return True
    return False


# --- the two guards the page exists for ------------------------------------------


def test_every_script_appears_on_the_page():
    scripts = _scripts()
    assert len(scripts) == 45, f"the directory holds {len(scripts)} scripts, not 45"
    missing = sorted(p.name for p in scripts if p.name not in _named_on_page())
    assert not missing, f"scripts missing from the page: {missing}"


def test_every_script_the_page_names_exists():
    named = _named_on_page()
    assert named, "the page names no scripts — check the test, not the directory"
    missing = sorted(name for name in named if not (SCRIPTS / name).is_file())
    assert not missing, f"the page names scripts that do not exist: {missing}"


def test_the_page_says_how_many_scripts_there_are():
    assert "Forty-five runnable scripts" in _page()
    assert len(_scripts()) == 45


# --- what the page says about them -----------------------------------------------


def test_the_scripts_the_page_calls_key_free_really_are():
    """01–14 are listed under 'no API key'; nothing in them may reach a provider."""
    for script in _scripts():
        number = int(script.name[:2])
        calls = _calls_a_provider(script.read_text())
        if number <= 14:
            assert not calls, f"{script.name} calls a provider but the page says it does not"
        else:
            assert calls, f"{script.name} calls no provider but the page implies it does"
    assert "**Scripts 01–14 need no API key.**" in _page()
    assert _page().count("— no API key") == 2, "the two key-free sections are labelled"


def test_every_llm_script_reaches_openrouter():
    """The page says everything from 15 on goes through OpenRouter."""
    for script in _scripts():
        if int(script.name[:2]) <= 14:
            continue
        source = script.read_text()
        assert re.search(r"^\s*[^#\n]*\bopenrouter\(", source, re.M), (
            f"{script.name} uses another provider; the page names only OpenRouter"
        )


def test_the_local_model_alternative_the_page_promises_is_there():
    """'Most of the LLM scripts carry a commented-out ollama line.'"""
    with_comment = [
        script.name
        for script in _scripts()
        if re.search(r"^\s*#\s*model\s*=\s*ollama\(", script.read_text(), re.M)
    ]
    llm_scripts = [s for s in _scripts() if int(s.name[:2]) > 14]
    assert len(with_comment) > len(llm_scripts) / 2, (
        f"only {len(with_comment)} of {len(llm_scripts)} carry the line; 'most' is wrong"
    )
    assert "commented-out `ollama(...)` line" in _page()


def test_the_scripts_the_page_calls_publishers_are_the_ones_that_publish():
    publishing = {
        script.name
        for script in _scripts()
        if "Sink.hub" in script.read_text() or "Source.huggingface" in script.read_text()
    }
    assert publishing == {
        "43_cookbook_persona_generation.py",
        "44_cookbook_space_text_generation.py",
        "45_cookbook_text_classification.py",
    }, f"the set of Hub scripts changed: {sorted(publishing)}"
    assert 'pip install "datafast[hub]"' in _page()
    assert "HF_REPO_ID" in _page()


def test_every_cookbook_script_has_the_walkthrough_the_page_links_to():
    rows = re.findall(
        r"\| `(\d{2}_[\w]+\.py)` \| ([^|]+) \| \[[^\]]+\]\(([\w_]+\.md)\) \|", _page()
    )
    assert len(rows) == 4, f"the cookbook table has {len(rows)} rows, expected 4"
    for script, _, page in rows:
        assert (SCRIPTS / script).is_file(), f"{script} does not exist"
        assert (PAGE.parent / page).is_file(), f"{page} does not exist"


def test_every_script_is_described_once_and_only_the_capstone_twice():
    """A script listed twice reads as two examples; a script listed nowhere is lost."""
    listed = re.findall(r"^\| `(\d{2}_[\w]+\.py)`", _page(), re.M)
    assert set(listed) == {p.name for p in _scripts()}, "a script has no table row"
    twice = sorted(name for name in set(listed) if listed.count(name) > 1)
    assert twice == ["42_pipeline_preference_with_scoring.py"], (
        f"listed more than once: {twice} — only the capstone is, as pipeline and cookbook"
    )


# --- the descriptions ------------------------------------------------------------


@pytest.mark.parametrize("script", _scripts(), ids=lambda p: p.name)
def test_every_script_has_a_header_the_description_can_come_from(script):
    docstring = ast.get_docstring(ast.parse(script.read_text()))
    assert docstring, f"{script.name} has no module docstring"
    assert "Demonstrates" in docstring, f"{script.name} does not say what it demonstrates"


def test_every_row_carries_a_description():
    rows = re.findall(r"^\| `(\d{2}_[\w]+\.py)` \| (.+?) \|", _page(), re.M)
    assert len(rows) == 46, "45 scripts, with the capstone listed twice"
    for name, description in rows:
        assert len(description.strip()) > 15, f"{name} has no real description"


def test_every_relative_link_resolves():
    links = re.findall(r"\]\((?!https?://)([^)#]+)", _page())
    assert links, "the page lost its links"
    for link in links:
        assert (PAGE.parent / link).exists(), f"broken link: {link}"


def test_the_run_command_the_page_gives_names_a_real_script():
    command = re.search(r"python (examples/scripts/[\w.]+)", _page())
    assert command, "the page no longer shows how to run a script"
    assert (ROOT / command.group(1)).is_file()
