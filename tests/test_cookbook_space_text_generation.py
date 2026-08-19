"""The space text-generation cookbook, pinned against the script it walks through.

The script is the source of truth. Checked in order of how much it matters:

1. Every step, step name, constant, path and output column the script uses is named on
   the page (code → docs).
2. Every self-contained example on the page executes.
3. The behavioural claims — the record counts, the appended JSON instruction, what a
   partial reply does, what a second run costs, what `num_outputs` leaves behind — are
   measured by running the script's own pipeline against a stub served model.

Nothing here reaches a network: `datafast.openrouter` is replaced before the script is
imported, and no test touches the Hub.
"""

import importlib.util
import inspect
import json
import re
import sys
from pathlib import Path

import pytest

import datafast
from datafast import ListSink, LLMStep, Source
from datafast.sinks.sink import JSONLSink
from datafast.sources.seed import SeedSource
from datafast.transforms.data_ops import AddUUID, Map

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "cookbook" / "space_text_generation.md"
SCRIPT = ROOT / "examples" / "scripts" / "44_cookbook_space_text_generation.py"


def _page() -> str:
    return PAGE.read_text()


def _code_blocks(language: str = "python") -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", _page(), re.DOTALL)


class StubModel:
    """Answers with the JSON the step's output_columns ask for."""

    model_id = "stub-model"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, messages=None, metadata=None, **kwargs) -> str:
        self.prompts.append(messages[-1]["content"])
        return json.dumps({"title": "Operating Without Weight", "text": "In orbit..."})


@pytest.fixture(scope="module")
def script():
    """Import the cookbook script with its provider factory stubbed."""
    original = datafast.openrouter
    stub = StubModel()
    datafast.openrouter = lambda model_id, **kwargs: stub
    try:
        spec = importlib.util.spec_from_file_location("cookbook_44", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        sys.modules["cookbook_44"] = module
        spec.loader.exec_module(module)
    finally:
        datafast.openrouter = original
    module._stub = stub
    return module


def _with_prompt(tmp_path: Path, script) -> None:
    """Copy the script's prompt file into a temporary working tree."""
    destination = tmp_path / script.PROMPT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text((ROOT / script.PROMPT_PATH).read_text())


@pytest.fixture
def completed_run(script, tmp_path, monkeypatch):
    """Run the script's real pipeline offline, in a temporary directory."""
    monkeypatch.chdir(tmp_path)
    _with_prompt(tmp_path, script)
    script._stub.prompts.clear()
    records = script.build_pipeline().run(
        batch_size=4, checkpoint_dir=script.CHECKPOINT_DIR, resume=False
    )
    return records, tmp_path


# --- code → docs -----------------------------------------------------------------


def test_every_step_the_script_uses_is_named_on_the_page(script):
    classes = {type(step).__name__ for step in script.build_pipeline().steps}
    assert classes, "the pipeline has no steps — check the test, not the page"
    documented = {
        "SeedSource": "Seed.product",
        "LLMStep": "LLMStep",
        "Map": "Map",
        "AddUUID": "AddUUID",
        "JSONLSink": "Sink.jsonl",
    }
    assert classes == set(documented), f"the script's steps changed: {sorted(classes)}"
    missing = [name for name in documented.values() if f"`{name}" not in _page()]
    assert not missing, f"steps undocumented on the page: {missing}"


def test_every_step_name_the_script_sets_is_on_the_page(script):
    names = [step.name for step in script.build_pipeline().steps]
    assert names == [
        "seed_space_text_grid",
        "generate_space_text",
        "finalize_record",
        "add_uuid",
        "JSONLSink",
    ], f"the script renamed or reordered its steps: {names}"
    missing = [name for name in names if name not in _page()]
    assert not missing, f"step names undocumented: {missing}"


def test_every_llm_step_argument_the_script_passes_is_documented(script):
    """The step is the page's subject; an argument it sets and the page omits is a gap."""
    source = inspect.getsource(script.build_pipeline)
    passed = set(re.findall(r"(\w+)=", source.split("LLMStep(")[1].split(").as_step")[0]))
    assert passed == {
        "prompt",
        "input_columns",
        "output_columns",
        "parse_mode",
        "model",
        "language",
        "num_outputs",
        "on_parse_error",
    }, f"the script's LLMStep call changed: {sorted(passed)}"
    missing = [name for name in passed if f"`{name}" not in _page()]
    assert not missing, f"LLMStep arguments undocumented: {missing}"


def test_every_tunable_constant_is_documented(script):
    constants = (
        "DOCUMENT_TYPES",
        "TOPICS",
        "EXPERTISE_LEVELS",
        "LANGUAGES",
        "MODEL_IDS",
        "NUM_OUTPUTS",
        "PROMPT_PATH",
        "HF_REPO_ID",
        "SEED",
    )
    for constant in constants:
        assert hasattr(script, constant), f"the script no longer defines {constant}"
        assert f"`{constant}`" in _page(), f"{constant} undocumented on the page"


def test_the_seed_values_the_script_uses_are_the_ones_the_page_describes(script):
    assert len(script.DOCUMENT_TYPES) == 3
    assert len(script.TOPICS) == 8
    assert len(script.EXPERTISE_LEVELS) == 3
    assert script.LANGUAGES == {"en": "English", "fr": "French"}
    for topic in script.TOPICS:
        assert topic.lower() in _page().lower(), f"topic not on the page: {topic}"
    for level in script.EXPERTISE_LEVELS:
        assert level in _page(), f"expertise level not on the page: {level}"


def test_the_paths_the_page_quotes_are_the_script_s_paths(script):
    for path in (script.OUTPUT_PATH, script.CHECKPOINT_DIR, script.PROMPT_PATH):
        assert path in _page(), f"the page does not name {path}"
    assert SCRIPT.relative_to(ROOT).as_posix() in _page()


def test_the_prompt_file_the_page_quotes_is_the_real_one(script):
    prompt = (ROOT / script.PROMPT_PATH).read_text().strip()
    assert prompt in _page(), "the page's prompt is not the file's prompt"
    assert "JSON" not in prompt.upper(), "the page says the file never mentions JSON"


def test_every_output_column_is_explained_on_the_page(script):
    source = inspect.getsource(script.finalize_record)
    columns = re.findall(r'"(\w+)":', source) + ["id"]
    assert len(columns) == 8, f"the output schema changed: {columns}"
    missing = [c for c in columns if f'"{c}"' not in _page() and f"`{c}`" not in _page()]
    assert not missing, f"output columns undocumented: {missing}"


def test_every_relative_link_resolves():
    links = re.findall(r"\]\((?!https?://)([^)#]+)", _page())
    assert links, "the page lost its links"
    for link in links:
        assert (PAGE.parent / link).exists(), f"broken link: {link}"


# --- the examples ----------------------------------------------------------------


@pytest.mark.parametrize("block", _code_blocks(), ids=lambda b: b.split("\n")[0][:40])
def test_every_example_executes(block, tmp_path, monkeypatch, script):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DATAFAST_PUSH_TO_HUB", raising=False)
    _with_prompt(tmp_path, script)
    namespace = {
        name: getattr(script, name)
        for name in (
            "DOCUMENT_TYPES", "TOPICS", "EXPERTISE_LEVELS", "LANGUAGES", "MODEL_IDS",
            "NUM_OUTPUTS", "PROMPT_PATH", "OUTPUT_PATH", "CHECKPOINT_DIR", "HF_REPO_ID",
            "SEED", "make_models", "finalize_record", "expected_row_count",
        )
    }
    namespace["pipeline"] = script.build_pipeline()
    if "Sink.hub" in block:
        # Executing this one would publish. Prove it parses.
        compile(block, str(PAGE), "exec")
        return
    exec(compile(block, str(PAGE), "exec"), namespace)


def test_there_are_examples_to_execute():
    assert len(_code_blocks()) >= 5, "the page lost its examples"


# --- the run, measured -----------------------------------------------------------


def test_the_run_produces_the_row_count_the_page_promises(completed_run, script):
    records, _ = completed_run
    assert len(records) == 144
    assert script.expected_row_count() == 144
    assert "**144 rows** by default" in _page()


def test_the_step_by_step_counts_on_the_page_are_the_real_ones(completed_run, script):
    _, tmp_path = completed_run
    manifest = json.loads((tmp_path / script.CHECKPOINT_DIR / "manifest.json").read_text())
    measured = {s["name"]: (s["records_in"], s["records_out"]) for s in manifest["steps"]}
    assert len(measured) == 5, "the manifest lost a step"
    for name, (records_in, records_out) in measured.items():
        row = re.search(rf"^\| `{re.escape(name)}` \| (\d+) \| (\d+) \|", _page(), re.M)
        assert row, f"the page's table has no row for {name}"
        assert (int(row.group(1)), int(row.group(2))) == (records_in, records_out)


def test_the_seed_grid_is_the_size_the_page_says(script):
    seed = script.build_pipeline().steps[0]
    assert isinstance(seed, SeedSource)
    assert len(list(seed.process(iter([])))) == 72


def test_one_llm_call_per_output_row(completed_run, script):
    records, _ = completed_run
    assert len(script._stub.prompts) == len(records) == 144
    assert "one LLM call each" in _page()


def test_the_output_columns_are_exactly_the_ones_the_page_shows(completed_run):
    records, _ = completed_run
    shown = json.loads(_code_blocks("json")[0])
    assert set(records[0]) == set(shown), (
        f"the run produced {sorted(records[0])}, the page shows {sorted(shown)}"
    )
    for column in ("document_type", "topic", "expertise_level", "language"):
        assert records[0][column] == shown[column], f"{column} drifted from the page"


def test_the_metadata_columns_lose_their_underscore(completed_run):
    """The page's explanation of what the Map is for."""
    records, _ = completed_run
    assert "_language" not in records[0] and "_model" not in records[0]
    assert records[0]["language"] in {"en", "fr"}
    assert records[0]["model"] == "stub-model"


def test_add_uuid_runs_after_the_map_so_every_row_keeps_its_id(completed_run):
    records, _ = completed_run
    ids = [r["id"] for r in records]
    assert len(set(ids)) == len(ids)
    assert len(ids[0]) == 36, "AddUUID writes a uuid4"


def test_the_corpus_covers_every_seed_value(completed_run, script):
    records, _ = completed_run
    assert {r["document_type"] for r in records} == set(script.DOCUMENT_TYPES)
    assert {r["topic"] for r in records} == set(script.TOPICS)
    assert {r["expertise_level"] for r in records} == set(script.EXPERTISE_LEVELS)
    assert {r["language"] for r in records} == set(script.LANGUAGES)


def test_the_checkpoint_file_names_the_page_lists_are_the_real_ones(completed_run, script):
    _, tmp_path = completed_run
    written = {p.name for p in (tmp_path / script.CHECKPOINT_DIR).iterdir()}
    quoted = set(re.findall(r"step_\d{3}_\w+\.jsonl", _page()))
    assert quoted, "the page quotes no checkpoint file"
    assert not quoted - written, f"the page names files the run never wrote: {quoted - written}"


def test_a_second_run_costs_nothing_and_returns_the_same_records(completed_run, script):
    """The page's claim, and the reason publishing can be a separate command."""
    records, tmp_path = completed_run
    before = len(script._stub.prompts)
    again = script.build_pipeline().run(
        batch_size=4, checkpoint_dir=script.CHECKPOINT_DIR, resume=True
    )
    assert len(script._stub.prompts) == before, "the resumed run made LLM calls"
    assert [r["id"] for r in again] == [r["id"] for r in records]
    written = (tmp_path / script.OUTPUT_PATH).read_text().strip().splitlines()
    assert len(written) == 144, "the sink appended instead of rewriting"


# --- the claims that would burn a reader -----------------------------------------


def _step(model, **kwargs) -> LLMStep:
    return LLMStep(
        prompt="write about {topic}",
        input_columns=["topic"],
        output_columns=["title", "text"],
        parse_mode="json",
        model=model,
        **kwargs,
    )


class _Reply:
    model_id = "stub"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.index = 0

    def generate(self, messages=None, metadata=None, **kwargs) -> str:
        reply = self.replies[self.index % len(self.replies)]
        self.index += 1
        return reply


def _run(model, **kwargs) -> list[dict]:
    return (Source.list([{"topic": "vacuum"}]) >> _step(model, **kwargs) >> ListSink()).run()


def test_the_json_instruction_the_page_quotes_is_appended_verbatim(script, completed_run):
    """The prompt file never mentions JSON; the page quotes what the step adds."""
    _, _ = completed_run
    quoted = [b for b in _code_blocks("text") if "Respond with valid JSON" in b]
    assert len(quoted) == 1, "the page no longer quotes the appended instruction"
    for line in quoted[0].strip().splitlines():
        assert any(line in prompt for prompt in script._stub.prompts), (
            f"no prompt sent contained: {line}"
        )


def test_the_prompt_carries_the_language_name_not_the_code(completed_run, script):
    """The page's table: {language_name} is what the prompt uses."""
    _, _ = completed_run
    assert any(p.startswith("Write one") and " in English." in p for p in script._stub.prompts)
    assert any(" in French." in p for p in script._stub.prompts)
    assert not any(" in en." in p for p in script._stub.prompts)


def test_a_fenced_reply_still_parses():
    fenced = "```json\n" + json.dumps({"title": "T", "text": "body"}) + "\n```"
    assert _run(_Reply(fenced))[0]["text"] == "body"


def test_a_missing_field_becomes_an_empty_string_and_keeps_the_row():
    """The page's warning: this is not a parse error, so on_parse_error never sees it."""
    records = _run(_Reply(json.dumps({"title": "T"})), on_parse_error="raise")
    assert len(records) == 1, "the row was kept"
    assert records[0]["text"] == "", "the missing column is filled with an empty string"


def test_every_parsed_column_is_a_string():
    records = _run(_Reply(json.dumps({"title": "T", "text": ["a", "b"]})))
    assert records[0]["text"] == "['a', 'b']"


def test_on_parse_error_raise_stops_the_run(script):
    """Set by the script, and now honoured under `run()` as well as `process()`."""
    llm_step = script.build_pipeline().steps[1]
    assert llm_step._on_parse_error == "raise"
    with pytest.raises(Exception, match="(?i)json"):
        _run(_Reply("not json at all"), on_parse_error="raise")


def test_a_placeholder_outside_input_columns_raises_before_any_call():
    """input_columns is a whitelist, as the page says."""
    model = _Reply(json.dumps({"title": "T", "text": "x"}))
    step = LLMStep(
        prompt="{topic} for {expertise_level}",
        input_columns=["topic"],
        output_columns=["title", "text"],
        parse_mode="json",
        model=model,
    )
    with pytest.raises(KeyError, match="expertise_level"):
        (
            Source.list([{"topic": "vacuum", "expertise_level": "executives"}])
            >> step
            >> ListSink()
        ).run()
    assert model.index == 0, "a call was made before the failure"


def test_num_outputs_multiplies_rows_and_leaves_no_marker_column():
    """The page's note: the extra rows carry nothing saying which output they were."""
    records = _run(
        _Reply(
            json.dumps({"title": "A", "text": "first"}),
            json.dumps({"title": "B", "text": "second"}),
        ),
        num_outputs=2,
    )
    assert len(records) == 2
    assert records[0].keys() == records[1].keys()
    assert {k: v for k, v in records[0].items() if k not in ("title", "text")} == {
        k: v for k, v in records[1].items() if k not in ("title", "text")
    }


def test_output_columns_is_required_for_json_mode():
    with pytest.raises(ValueError, match="output_columns required"):
        LLMStep(
            prompt="{topic}",
            input_columns=["topic"],
            parse_mode="json",
            model=_Reply("{}"),
        )


def test_expected_row_count_prices_a_model_change(script):
    """The page's second assertion in the counting example."""
    assert script.expected_row_count(3) == 432
    assert "432" in _page()


# --- publishing, which no test may actually do -----------------------------------


def test_the_push_is_a_function_behind_an_environment_variable(script):
    """The trade-off the page tables: the push is not a step in the pipeline."""
    assert callable(script.push_records_to_hub)
    classes = [type(step).__name__ for step in script.build_pipeline().steps]
    assert "HubSink" not in classes, "the push moved into the pipeline"
    assert isinstance(script.build_pipeline().steps[-1], JSONLSink)
    main = inspect.getsource(script.main)
    assert 'os.getenv("DATAFAST_PUSH_TO_HUB") == "1"' in main
    assert "DATAFAST_PUSH_TO_HUB=1" in _page()


def test_the_push_wraps_process_in_list_because_it_is_a_generator(script):
    source = inspect.getsource(script.push_records_to_hub)
    assert "list(" in source and ".process(records)" in source
    assert "`process` is a generator" in _page()


def test_the_push_settings_the_page_names_are_the_script_s(script):
    source = inspect.getsource(script.push_records_to_hub)
    for setting in ("private=True", "train_size=0.8", "seed=SEED", "shuffle=True"):
        assert setting in source, f"the script no longer passes {setting}"
        assert setting in _page(), f"the page does not name {setting}"


def test_a_map_returning_a_new_dict_drops_what_it_does_not_name(script):
    """Why finalize_record is enough to strip the internal columns."""
    step = Map(script.finalize_record)
    record = {
        "document_type": "d", "topic": "t", "expertise_level": "e",
        "_language": "en", "_model": "m", "title": "T", "text": "x",
        "_prompt_index": 0, "left_over": "gone",
    }
    out = list(step.process(iter([record])))[0]
    assert "left_over" not in out and "_prompt_index" not in out
    assert list(out) == [
        "document_type", "topic", "expertise_level", "language", "model", "title", "text"
    ]
    assert isinstance(AddUUID(column="id", overwrite=True), AddUUID)
