"""The landing code is the first thing a user runs, so it is pinned rather than trusted.

The quickstart, the docs home page and the README all open on the same pipeline. Each
block is extracted and executed with a stub in place of the served model, so the
documented code is the code under test and no provider is called.
"""

import json
import re
from pathlib import Path

import pytest

import datafast

ROOT = Path(__file__).parent.parent
QUICKSTART = ROOT / "docs" / "quickstart.md"
# Every document whose opening example must stay runnable.
LANDING_DOCS = [QUICKSTART, ROOT / "docs" / "index.md", ROOT / "README.md"]

STUB_REPLY = '{"question": "Why is the sky blue?", "answer": "Rayleigh scattering."}'


class StubModel:
    """Stands in for `openai()` so the documented code runs without a network call."""

    provider_id = "openai"
    model_id = "gpt-5.5"

    def generate(self, prompt=None, messages=None, metadata=None, response_format=None):
        return STUB_REPLY


def _code_blocks(language: str, doc: Path = QUICKSTART) -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", doc.read_text(), re.DOTALL)


def _run_documented_pipeline(tmp_path, monkeypatch, doc: Path = QUICKSTART):
    """Execute the quickstart's python block, swapping in the stub model.

    The block imports `openai` from `datafast` itself, so the factory is patched on
    the module rather than injected into the exec namespace — an injected name would
    be overwritten by the block's own import, and the test would call the real API.
    """
    monkeypatch.setattr(datafast, "openai", lambda *args, **kwargs: StubModel())
    monkeypatch.chdir(tmp_path)

    # The README carries several python blocks; the pipeline is the one that counts.
    (code,) = [b for b in _code_blocks("python", doc) if "pipeline = (" in b]
    namespace: dict = {}
    exec(compile(code, str(doc), "exec"), namespace)
    return namespace["pipeline"]


def test_quickstart_pipeline_compiles_and_runs(tmp_path, monkeypatch):
    pipeline = _run_documented_pipeline(tmp_path, monkeypatch)
    pipeline.compile()

    written = tmp_path / "quickstart.jsonl"
    assert written.exists(), "the page tells the reader to expect quickstart.jsonl"

    rows = [json.loads(line) for line in written.read_text().splitlines()]
    assert len(rows) == 6, "3 topics x 2 levels; the page says six rows"


def test_quickstart_output_columns_match_the_documented_row(tmp_path, monkeypatch):
    """The sample row on the page must name exactly the columns a run produces."""
    _run_documented_pipeline(tmp_path, monkeypatch)
    rows = [
        json.loads(line)
        for line in (tmp_path / "quickstart.jsonl").read_text().splitlines()
    ]

    (documented,) = _code_blocks("json")
    assert set(json.loads(documented)) == set(rows[0])
    assert set(rows[0]) == {"topic", "level", "question", "answer", "_model"}


def test_quickstart_install_command_matches_the_distribution():
    assert "pip install datafast" in QUICKSTART.read_text()


@pytest.mark.parametrize(
    "extra", ["datafast[parquet]", "datafast[hub]", "datafast[all]"]
)
def test_quickstart_names_only_real_extras(extra):
    """Every extra the page offers must exist in pyproject."""
    import tomllib

    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text())["project"]["optional-dependencies"]

    assert extra in QUICKSTART.read_text()
    assert extra[len("datafast[") : -1] in declared


@pytest.mark.parametrize("doc", LANDING_DOCS, ids=lambda p: p.name)
def test_landing_example_runs_and_produces_six_rows(doc, tmp_path, monkeypatch):
    """The quickstart, the home page and the README open on the same pipeline.

    Each is executed as written; a broken opening example is the worst one to ship.
    """
    pipeline = _run_documented_pipeline(tmp_path, monkeypatch, doc)
    pipeline.compile()

    (written,) = tmp_path.glob("*.jsonl")
    rows = [json.loads(line) for line in written.read_text().splitlines()]

    assert len(rows) == 6, f"{doc.name} claims three topics x two levels"
    assert set(rows[0]) == {"topic", "level", "question", "answer", "_model"}
