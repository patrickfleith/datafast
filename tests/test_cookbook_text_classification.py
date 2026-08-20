"""The text-classification cookbook, pinned against the script it walks through.

The script is the source of truth. This test asserts, in order of how much it matters:

1. Every step, constant and column the script uses is named on the page (code → docs).
   The page describes a run a reader will repeat; a drifted script makes it a lie.
2. Every self-contained example on the page executes.
3. The behavioural claims — the row counts, the language placeholders, the column set of
   a published row, the checkpoint file names — come from running the script's own
   pipeline with stub served models, not from reading it.

No test here makes a live LLM call: `datafast.openrouter` is replaced before the script
is imported, so the served models it builds at import time are stubs.
"""

import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

import datafast
from datafast import Seed, SeedDimension
from datafast.sinks.sink import HubSink, JSONLSink
from datafast.sources.seed import SeedSource
from datafast.transforms.data_ops import AddUUID
from datafast.transforms.llm_step import LLMStep

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "cookbook" / "text_classification.md"
SCRIPT = ROOT / "examples" / "scripts" / "45_cookbook_text_classification.py"
PROMPT = ROOT / "docs" / "cookbook" / "assets" / "text_classification_generation.txt"


def _page() -> str:
    return PAGE.read_text()


def _code_blocks(language: str = "python") -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", _page(), re.DOTALL)


class StubModel:
    """Stands in for an OpenRouter served model. Answers without a network call."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.prompts: list[str] = []

    def generate(self, messages=None, metadata=None, **kwargs) -> str:
        self.prompts.append(messages[-1]["content"])
        return "Bridge over the creek is out, had to wade across. Cold."


@pytest.fixture(scope="module")
def script():
    """Import the cookbook script with its provider factory stubbed.

    The factory must be replaced on the `datafast` module itself: the script does its own
    `from datafast import openrouter`, which resolves through that module at import time.
    """
    original = datafast.openrouter
    stubs: list[StubModel] = []

    def fake_openrouter(model_id, **kwargs):
        stub = StubModel(model_id)
        stubs.append(stub)
        return stub

    datafast.openrouter = fake_openrouter
    try:
        spec = importlib.util.spec_from_file_location("cookbook_45", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        sys.modules["cookbook_45"] = module
        spec.loader.exec_module(module)
    finally:
        datafast.openrouter = original

    module._stubs = stubs
    return module


@pytest.fixture
def completed_run(script, tmp_path, monkeypatch):
    """Run the script's own pipeline to completion, offline, in a temp directory."""
    monkeypatch.chdir(tmp_path)
    # The publishing example is guarded by this variable. Never let it fire in a test.
    monkeypatch.delenv("DATAFAST_PUSH_TO_HUB", raising=False)
    destination = tmp_path / script.PROMPT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(PROMPT, destination)

    records = script.pipeline.run(
        batch_size=4, checkpoint_dir=script.CHECKPOINT_DIR, resume=True
    )
    return records, tmp_path


# --- code → docs -----------------------------------------------------------------


def test_every_step_the_script_uses_is_named_on_the_page(script):
    """A step in the pipeline that the page never mentions is a gap in the walkthrough."""
    classes = {type(step).__name__ for step in script.pipeline.steps}
    assert classes, "the pipeline has no steps — check the test, not the page"
    documented = {
        "SeedSource": "Seed.product",
        "LLMStep": "LLMStep",
        "Map": "Map",
        "AddUUID": "AddUUID",
        "JSONLSink": "Sink.jsonl",
    }
    assert classes == set(documented), f"the script's steps changed: {sorted(classes)}"
    missing = [name for cls, name in documented.items() if f"`{name}" not in _page()]
    assert not missing, f"steps undocumented on the page: {missing}"


def test_every_step_name_the_script_sets_is_on_the_page(script):
    names = [step.name for step in script.pipeline.steps]
    assert names == [
        "seed_trail_report_grid",
        "generate_trail_reports",
        "keep_output_fields",
        "add_uuid",
        "JSONLSink",
    ], f"the script renamed its steps: {names}"
    missing = [name for name in names if name not in _page()]
    assert not missing, f"step names undocumented: {missing}"


def test_every_tunable_constant_is_documented(script):
    """The page's "making it yours" table is only useful if the names are real."""
    for constant in ("LABELS", "TRAIL_TYPES", "STYLES", "LANGUAGES", "MODEL_IDS", "HF_REPO_ID"):
        assert hasattr(script, constant), f"the script no longer defines {constant}"
        assert f"`{constant}`" in _page(), f"{constant} undocumented on the page"


def test_every_label_the_script_defines_is_named_on_the_page(script):
    labels = [entry["label"] for entry in script.LABELS]
    assert len(labels) == 4
    missing = [label for label in labels if f"`{label}`" not in _page()]
    assert not missing, f"labels undocumented on the page: {missing}"


def test_the_paths_the_page_quotes_are_the_script_s_paths(script):
    for path in (script.OUTPUT_PATH, script.CHECKPOINT_DIR, script.PROMPT_PATH):
        assert path in _page(), f"the page does not name {path}"
    assert SCRIPT.relative_to(ROOT).as_posix() in _page()


def test_the_prompt_lines_the_page_quotes_are_in_the_prompt_file():
    """The page argues from two specific constraints; they must still be there."""
    prompt = PROMPT.read_text()
    for quoted in _code_blocks("text"):
        for line in quoted.strip().splitlines():
            if line.startswith("- Do not"):
                assert line in prompt, f"the prompt no longer says: {line}"


def test_the_prompt_uses_the_placeholder_the_page_says_it_does(script):
    prompt = PROMPT.read_text()
    assert "{language_name}" in prompt
    assert "{language}" not in prompt.replace("{language_name}", "")
    for column in script.pipeline.steps[1]._input_columns:
        assert "{" + column + "}" in prompt, f"{column} is declared but never used"


# --- the examples ----------------------------------------------------------------


@pytest.mark.parametrize("block", _code_blocks(), ids=lambda b: b.split("\n")[0][:40])
def test_every_example_executes(block, tmp_path, monkeypatch, script):
    """The pipeline block references the script's own constants, so run it against them."""
    monkeypatch.chdir(tmp_path)
    destination = tmp_path / script.PROMPT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(PROMPT, destination)
    if "resume_from=" in block:
        # The page's resume example needs the checkpoint an earlier run left behind.
        script.pipeline.run(checkpoint_dir=script.CHECKPOINT_DIR)
    namespace = {
        name: getattr(script, name)
        for name in (
            "LABELS", "TRAIL_TYPES", "STYLES", "LANGUAGES", "MODELS", "MODEL_IDS",
            "PROMPT_PATH", "OUTPUT_PATH", "CHECKPOINT_DIR", "HF_REPO_ID", "SEED",
            "keep_output_fields", "pipeline",
        )
    }
    namespace["os"] = __import__("os")
    namespace["records"] = []
    exec(compile(block, str(PAGE), "exec"), namespace)


def test_there_are_examples_to_execute():
    assert len(_code_blocks()) >= 5, "the page lost its examples"


# --- the run, measured -----------------------------------------------------------


def test_the_seed_produces_the_number_of_records_the_page_tables(script):
    seed = script.pipeline.steps[0]
    assert isinstance(seed, SeedSource)
    assert len(seed) == 24, "4 labels x 3 trail types x 2 styles"
    assert "| `seed_trail_report_grid` | 0 | 24 |" in _page()


def test_the_expected_row_count_the_page_names_is_the_script_s(script):
    assert script.EXPECTED_ROWS == 96
    assert "96" in _page()


def test_the_run_produces_the_rows_the_page_promises(completed_run):
    records, _ = completed_run
    assert len(records) == 96


def test_the_step_by_step_counts_on_the_page_are_the_real_ones(completed_run, script):
    """Every row of the record-count table, checked against the checkpoint files."""
    _, tmp_path = completed_run
    directory = tmp_path / script.CHECKPOINT_DIR
    manifest = json.loads((directory / "manifest.json").read_text())
    measured = {s["name"]: (s["records_in"], s["records_out"]) for s in manifest["steps"]}

    for name, (records_in, records_out) in measured.items():
        row = re.search(rf"^\| `{re.escape(name)}` \| (\d+) \| (\d+) \|", _page(), re.M)
        assert row, f"the page's table has no row for {name}"
        assert (int(row.group(1)), int(row.group(2))) == (records_in, records_out), (
            f"the page says {row.group(1)} → {row.group(2)} for {name}, "
            f"the run did {records_in} → {records_out}"
        )


def test_the_checkpoint_file_names_the_page_lists_are_the_real_ones(completed_run, script):
    _, tmp_path = completed_run
    written = sorted(p.name for p in (tmp_path / script.CHECKPOINT_DIR).iterdir())
    assert written, "the run wrote no checkpoint"
    missing = [name for name in written if name not in _page()]
    assert not missing, f"checkpoint files the page does not list: {missing}"


def test_a_published_row_has_exactly_the_columns_the_page_shows(completed_run):
    """The page prints one row as JSON; its keys and order are the claim."""
    records, _ = completed_run
    shown = json.loads(re.search(r"```json\n(.*?)```", _page(), re.DOTALL).group(1))
    assert list(records[0]) == list(shown), "the page's row has drifted from the output"


def test_both_languages_and_both_models_appear_in_the_output(completed_run, script):
    records, _ = completed_run
    assert {r["language"] for r in records} == set(script.LANGUAGES)
    assert {r["model"] for r in records} == set(script.MODEL_IDS)


def test_every_label_gets_the_same_number_of_rows(completed_run, script):
    """The page's claim that the product balances the classes by construction."""
    records, _ = completed_run
    counts = dict.fromkeys((e["label"] for e in script.LABELS), 0)
    for record in records:
        counts[record["label"]] += 1
    assert len(set(counts.values())) == 1, f"the classes are not balanced: {counts}"


def test_the_map_drops_the_label_description(completed_run):
    """The page says the description is written for the model, not for the dataset."""
    records, _ = completed_run
    assert "label_description" not in records[0]
    assert "label_description" in _page(), "the page must say where it went"


def test_the_output_file_holds_the_same_records(completed_run, script):
    _, tmp_path = completed_run
    lines = (tmp_path / script.OUTPUT_PATH).read_text().splitlines()
    assert len(lines) == 96
    assert json.loads(lines[0])["label"] in {e["label"] for e in script.LABELS}


# --- the claims that would burn a reader -----------------------------------------


def test_language_name_is_the_name_and_language_is_the_code(script, completed_run):
    """The page's placeholder table, proved from the prompts that were actually sent."""
    _, _ = completed_run
    sent = [prompt for stub in script._stubs for prompt in stub.prompts]
    assert sent, "no prompt was recorded — check the test, not the page"
    for name in script.LANGUAGES.values():
        assert any(f"comment in {name}" in prompt for prompt in sent), (
            f"the prompt was never filled with the language name {name!r}"
        )
    for code in script.LANGUAGES:
        assert not any(f"comment in {code}\n" in prompt for prompt in sent)


def test_the_label_and_its_description_never_come_apart(script):
    """The page's central seeding claim: 4 pairs, not 16."""
    seed = script.pipeline.steps[0]
    correct = {(entry["label"], entry["label_description"]) for entry in script.LABELS}
    seen = {(r["label"], r["label_description"]) for r in seed.process(iter([]))}
    assert seen == correct, "a label was crossed with another label's description"


def test_a_multi_column_dimension_counts_as_one_dimension():
    """The page's `SeedDimension` example, and the mistake it prevents."""
    labels = [
        {"label": "hazards", "label_description": "risk"},
        {"label": "positive_conditions", "label_description": "clear"},
    ]
    one = SeedDimension(columns=["label", "label_description"], values=labels)
    assert len(one) == 2
    assert len(Seed.product(one)) == 2

    crossed = Seed.product(
        Seed.values("label", [e["label"] for e in labels]),
        Seed.values("label_description", [e["label_description"] for e in labels]),
    )
    assert len(crossed) == 4, "the page says crossing them multiplies the mistake"


def test_input_columns_is_a_whitelist_for_the_prompt(script):
    """The page's warning, triggered for real: a real column, left out, still fails."""
    from datafast import ListSink, Source

    step = LLMStep(
        prompt="Category: {label}. Definition: {label_description}",
        input_columns=["label"],
        output_column="text",
        model=StubModel("stub"),
    )
    record = {"label": "hazards", "label_description": "risk"}
    with pytest.raises(KeyError, match="label_description"):
        (Source.list([record]) >> step >> ListSink()).run()


def test_add_uuid_runs_after_the_map_so_the_id_belongs_to_the_published_row(script):
    steps = script.pipeline.steps
    assert [type(s).__name__ for s in steps].index("Map") < [
        type(s).__name__ for s in steps
    ].index("AddUUID")
    uuid_step = next(s for s in steps if isinstance(s, AddUUID))
    assert uuid_step._column == "id" and uuid_step._overwrite is True


def test_the_models_carry_the_temperature_the_page_names(script):
    source = SCRIPT.read_text()
    assert "temperature=0.8" in source
    assert "`temperature=0.8`" in _page()


def test_the_hub_push_is_outside_the_pipeline_and_opt_in(script):
    """The page explains why; the script must still be built that way."""
    assert not any(isinstance(step, HubSink) for step in script.pipeline.steps)
    assert isinstance(script.pipeline.steps[-1], JSONLSink)
    source = SCRIPT.read_text()
    assert 'os.getenv("DATAFAST_PUSH_TO_HUB") == "1"' in source
    assert "DATAFAST_PUSH_TO_HUB=1" in _page()


def test_the_hub_arguments_the_page_shows_are_the_script_s():
    """A published split is hard to undo; the page must not invent its settings."""
    source = SCRIPT.read_text()
    for argument in ("train_size=0.8", "shuffle=True", "private=False", "seed=SEED"):
        assert argument in source, f"the script no longer passes {argument}"
        assert argument in _page(), f"the page does not show {argument}"


def test_the_hub_sink_is_a_generator_so_list_is_required():
    """The page's note about `list(...)`; without it the push never runs."""
    import inspect

    assert inspect.isgeneratorfunction(HubSink.process)
    assert "list(" in _page()


# --- links -----------------------------------------------------------------------


def test_every_page_linked_to_exists():
    links = re.findall(r"\]\((?!https?:)([^)#]+\.md)", _page())
    assert links, "the page has no links — check the test, not the page"
    missing = sorted(link for link in links if not (PAGE.parent / link).resolve().exists())
    assert not missing, f"links to pages that do not exist: {missing}"


def test_the_linked_assets_exist():
    links = re.findall(r"\]\((?!https?:)(assets/[^)#]+)\)", _page())
    assert links, "the page no longer links its prompt asset"
    missing = [link for link in links if not (PAGE.parent / link).exists()]
    assert not missing, f"linked assets that do not exist: {missing}"
