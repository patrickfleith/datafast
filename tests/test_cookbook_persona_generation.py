"""The persona-generation cookbook, pinned against the script it walks through.

The script is the source of truth. Checked in order of how much it matters:

1. Every step, constant, prompt file and output column the script uses is named on the
   page (code → docs).
2. Every self-contained example on the page executes.
3. The behavioural claims — the step order, the column set, the appended JSON
   instruction, the two chained sinks, what a random prompt picker does — are proved by
   running the script's own pipeline with a stub served model.

Nothing here reaches a network. `datafast.openrouter` is replaced before the script is
imported, and the Hugging Face source is swapped for a fake corpus before the run: the
real one would download a dataset.
"""

import importlib.util
import inspect
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

import datafast
from datafast import Sample, Source
from datafast.core.step import Pipeline
from datafast.sinks.sink import HubSink, JSONLSink
from datafast.sources.source import HuggingFaceSource
from datafast.transforms.data_ops import Filter
from datafast.transforms.llm_step import LLMStep

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "cookbook" / "persona_generation.md"
SCRIPT = ROOT / "examples" / "scripts" / "43_cookbook_persona_generation.py"
ASSETS = ROOT / "docs" / "cookbook" / "assets"


def _page() -> str:
    return PAGE.read_text()


def _code_blocks(language: str = "python") -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", _page(), re.DOTALL)


class StubModel:
    """Answers both steps with the JSON their output_columns ask for."""

    model_id = "stub"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, messages=None, metadata=None, **kwargs) -> str:
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        if "relationship_type" in prompt:
            return json.dumps(
                {
                    "relationship_type": "daughter",
                    "related_persona_description": "A student who visits at weekends.",
                }
            )
        return json.dumps(
            {"persona_description": "A council officer who tracks flood risk."}
        )


@pytest.fixture(scope="module")
def script():
    """Import the cookbook script with its provider factory stubbed."""
    original = datafast.openrouter
    stub = StubModel()
    datafast.openrouter = lambda model_id, **kwargs: stub
    try:
        spec = importlib.util.spec_from_file_location("cookbook_43", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        sys.modules["cookbook_43"] = module
        spec.loader.exec_module(module)
    finally:
        datafast.openrouter = original
    module._stub = stub
    return module


def _fake_corpus(count: int = 30) -> list[dict]:
    """Articles inside the script's own word-count window."""
    return [
        {"id": f"xsum-{i}", "document": " ".join(["word"] * 350), "summary": f"s{i}"}
        for i in range(count)
    ]


@pytest.fixture
def completed_run(script, tmp_path, monkeypatch):
    """Run the script's pipeline offline: fake corpus in, Hub sink removed."""
    monkeypatch.chdir(tmp_path)
    for prompt in script.TEXT_TO_PERSONA_PROMPTS + script.PERSONA_TO_PERSONA_PROMPTS:
        destination = tmp_path / prompt
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / prompt, destination)

    script._stub.prompts.clear()
    steps = list(script.build_pipeline().steps)
    steps[0] = Source.list(_fake_corpus())
    offline = Pipeline([s for s in steps if not isinstance(s, HubSink)])

    records = offline.run(
        batch_size=1, checkpoint_dir=script.CHECKPOINT_DIR, resume=False
    )
    return records, tmp_path


# --- code → docs -----------------------------------------------------------------


def test_every_step_the_script_uses_is_named_on_the_page(script):
    classes = {type(step).__name__ for step in script.build_pipeline().steps}
    assert classes, "the pipeline has no steps — check the test, not the page"
    documented = {
        "HuggingFaceSource": "Source.huggingface",
        "Map": "Map",
        "Filter": "Filter",
        "Sample": "Sample",
        "LLMStep": "LLMStep",
        "AddUUID": "AddUUID",
        "JSONLSink": "Sink.jsonl",
        "HubSink": "Sink.hub",
    }
    assert classes == set(documented), f"the script's steps changed: {sorted(classes)}"
    missing = [name for name in documented.values() if f"`{name}" not in _page()]
    assert not missing, f"steps undocumented on the page: {missing}"


def test_every_step_name_the_script_sets_is_on_the_page(script):
    names = [step.name for step in script.build_pipeline().steps]
    assert names == [
        "HuggingFaceSource",
        "add_word_count",
        "filter_word_count",
        "take_first_10",
        "assign_life_stage",
        "text_to_persona",
        "assign_related_life_stage",
        "persona_to_persona",
        "keep_output_fields",
        "add_uuid",
        "JSONLSink",
        "HubSink",
    ], f"the script renamed or reordered its steps: {names}"
    named = [name for name in names if name.islower()]
    assert len(named) == 9, "the script stopped naming its steps"
    missing = [name for name in named if name not in _page()]
    assert not missing, f"step names undocumented: {missing}"


def test_every_tunable_constant_is_documented(script):
    for constant in ("MODEL_ID", "HF_REPO_ID", "LIFE_STAGES"):
        assert hasattr(script, constant), f"the script no longer defines {constant}"
        assert f"`{constant}`" in _page(), f"{constant} undocumented on the page"


def test_every_prompt_file_the_script_names_exists_and_is_reachable(script):
    prompts = script.TEXT_TO_PERSONA_PROMPTS + script.PERSONA_TO_PERSONA_PROMPTS
    assert len(prompts) == 6, "three variants per step"
    for prompt in prompts:
        assert (ROOT / prompt).is_file(), f"{prompt} does not exist"
    assert "three variants per step" in _page()


def test_the_paths_the_page_quotes_are_the_script_s_paths(script):
    for path in (script.OUTPUT_PATH, script.CHECKPOINT_DIR):
        assert path in _page(), f"the page does not name {path}"
    assert SCRIPT.relative_to(ROOT).as_posix() in _page()


def test_every_output_column_is_explained_on_the_page(script):
    """The page's column table is the reader's schema; a gap in it is a gap in the data."""
    source = inspect.getsource(script.keep_output_fields)
    columns = re.findall(r'"(\w+)":', source) + ["id"]
    assert len(columns) == 10, f"the output schema changed: {columns}"
    missing = [c for c in columns if f"`{c}`" not in _page()]
    assert not missing, f"output columns undocumented: {missing}"


# --- the examples ----------------------------------------------------------------


@pytest.mark.parametrize("block", _code_blocks(), ids=lambda b: b.split("\n")[0][:40])
def test_every_example_executes(block, tmp_path, monkeypatch, script):
    monkeypatch.chdir(tmp_path)
    namespace = {
        name: getattr(script, name)
        for name in (
            "MODEL_ID", "OUTPUT_PATH", "CHECKPOINT_DIR", "HF_REPO_ID", "LIFE_STAGES",
            "TEXT_TO_PERSONA_PROMPTS", "PERSONA_TO_PERSONA_PROMPTS",
            "add_word_count", "assign_life_stage", "assign_related_life_stage",
            "keep_output_fields", "build_pipeline",
        )
    }
    namespace["model"] = script._stub
    namespace["openrouter"] = lambda *a, **k: script._stub
    monkeypatch.delenv("DATAFAST_PUSH_TO_HUB", raising=False)
    if "build_pipeline().run(" in block:
        # The run example would download XSum and push to the Hub. Prove it parses.
        compile(block, str(PAGE), "exec")
        return
    exec(compile(block, str(PAGE), "exec"), namespace)


def test_there_are_examples_to_execute():
    assert len(_code_blocks()) >= 6, "the page lost its examples"


# --- the run, measured -----------------------------------------------------------


def test_the_run_produces_the_row_count_the_page_promises(completed_run):
    records, _ = completed_run
    assert len(records) == 10, "the page says 10 rows by default"
    assert "**10 rows** by default" in _page()


def test_each_row_costs_the_two_llm_calls_the_page_names(completed_run, script):
    records, _ = completed_run
    assert len(script._stub.prompts) == 2 * len(records)
    assert "two LLM calls per row" in _page()


def test_the_output_columns_are_exactly_the_ones_the_page_tables(completed_run):
    records, _ = completed_run
    table = re.search(r"\| Column \| Where it comes from \|\n\|---\|---\|\n((?:\|.*\n)+)", _page())
    assert table, "the page lost its column table"
    documented = set(re.findall(r"^\| `(\w+)`", table.group(1), re.M))
    documented |= set(re.findall(r"`(\w+)`, `(\w+)`", table.group(1))[0]) if re.findall(r"`(\w+)`, `(\w+)`", table.group(1)) else set()
    assert set(records[0]) == documented, (
        f"the run produced {sorted(set(records[0]))}, the table lists {sorted(documented)}"
    )


def test_the_source_id_is_the_dataset_id_and_id_is_a_new_uuid(completed_run):
    """The page's explanation of the rename; both columns must really be there."""
    records, _ = completed_run
    assert records[0]["source_id"].startswith("xsum-")
    assert records[0]["id"] != records[0]["source_id"]
    assert len(records[0]["id"]) == 36, "AddUUID writes a uuid4"


def test_the_checkpoint_file_names_the_page_lists_are_the_real_ones(completed_run, script):
    _, tmp_path = completed_run
    written = {p.name for p in (tmp_path / script.CHECKPOINT_DIR).iterdir()}
    quoted = set(re.findall(r"step_\d{3}_\w+\.jsonl", _page()))
    assert quoted, "the page quotes no checkpoint file"
    missing = sorted(quoted - written)
    assert not missing, f"the page names checkpoint files the run does not write: {missing}"


def test_the_word_count_filter_keeps_only_the_documented_window(script):
    """300 to 500 words, as the page says — measured on both edges."""
    steps = {s.name: s for s in script.build_pipeline().steps}
    filter_step = steps["filter_word_count"]
    assert isinstance(filter_step, Filter)

    def sized(words):
        return script.add_word_count({"document": " ".join(["w"] * words)})

    kept = [r["word_count"] for r in filter_step.process(iter([sized(n) for n in (299, 300, 500, 501)]))]
    assert kept == [300, 500]
    assert "300" in _page() and "500" in _page()


def test_the_sample_step_takes_the_number_its_name_claims(script):
    """The step name is the checkpoint file name, so it has to tell the truth."""
    steps = {s.name: s for s in script.build_pipeline().steps}
    sample = steps["take_first_10"]
    assert isinstance(sample, Sample)
    kept = list(sample.process(iter([{"i": i} for i in range(30)])))
    assert kept == [{"i": i} for i in range(10)], "first ten, in order"
    assert "step_003_take_first_10.jsonl" in _page()


# --- the claims that would burn a reader -----------------------------------------


def test_a_sample_with_items_is_a_picker_and_without_items_is_a_step():
    """The page's central Sample claim: one class, two jobs."""
    picker = Sample(["a.txt", "b.txt", "c.txt"], n=1)
    assert len(picker.pick()) == 1
    assert picker.pick()[0] in {"a.txt", "b.txt", "c.txt"}

    step = Sample(n=2, strategy="first")
    assert list(step.process(iter([{"i": 0}, {"i": 1}, {"i": 2}]))) == [{"i": 0}, {"i": 1}]


def test_the_prompt_picker_is_drawn_once_per_record(script, completed_run):
    """Three variants across ten records; a single draw for the whole step would show."""
    _, _ = completed_run
    first_stage = script._stub.prompts[:10]
    openings = {prompt.split("\n")[0] for prompt in first_stage}
    assert len(openings) > 1, "every record got the same prompt variant"


def test_the_json_instruction_the_page_quotes_is_appended_verbatim(script, completed_run):
    """Neither prompt file mentions JSON; the page quotes what the step adds."""
    _, _ = completed_run
    quoted = [b for b in _code_blocks("text") if "Respond with valid JSON" in b]
    assert len(quoted) == 1, "the page no longer quotes the appended instruction"
    for line in quoted[0].strip().splitlines():
        assert any(line in prompt for prompt in script._stub.prompts), (
            f"no prompt sent contained: {line}"
        )


def test_no_prompt_file_mentions_json(script):
    for prompt in script.TEXT_TO_PERSONA_PROMPTS + script.PERSONA_TO_PERSONA_PROMPTS:
        assert "JSON" not in (ROOT / prompt).read_text().upper()


def test_the_second_step_never_sees_the_article(script):
    """The page's claim that the related persona is derived from the persona alone."""
    steps = {s.name: s for s in script.build_pipeline().steps}
    second = steps["persona_to_persona"]
    assert isinstance(second, LLMStep)
    assert "document" not in second._input_columns
    assert second._input_columns == ["persona_description", "related_life_stage"]


def test_both_llm_steps_set_on_parse_error_raise(script):
    """The page warns this does not raise under run(); the script must still set it."""
    steps = script.build_pipeline().steps
    llm_steps = [s for s in steps if isinstance(s, LLMStep)]
    assert len(llm_steps) == 2
    assert all(s._on_parse_error == "raise" for s in llm_steps)
    assert "on_parse_error=" in _page()


def test_on_parse_error_raise_stops_the_run(script):
    """The script sets `raise` on both steps; under `run()` that used to be ignored."""
    from datafast import ListSink

    step = LLMStep(
        prompt="{document}",
        input_columns=["document"],
        output_columns=["persona_description"],
        model=_Unparseable(),
        parse_mode="json",
        on_parse_error="raise",
    )
    with pytest.raises(Exception, match="(?i)json"):
        (Source.list([{"document": "a"}]) >> step >> ListSink()).run()


class _Unparseable:
    model_id = "stub"

    def generate(self, messages=None, metadata=None, **kwargs) -> str:
        return "not json at all"


def test_the_life_stages_are_drawn_twice_and_independently(script):
    """Two draws, so the related persona is not forced into the same stage."""
    source = SCRIPT.read_text()
    assert source.count("random.choice(LIFE_STAGES)") == 2
    steps = [s.name for s in script.build_pipeline().steps]
    assert steps.index("assign_life_stage") < steps.index("text_to_persona")
    assert steps.index("assign_related_life_stage") < steps.index("persona_to_persona")


# --- the two sinks ---------------------------------------------------------------


def test_the_pipeline_really_ends_in_two_sinks(script):
    steps = script.build_pipeline().steps
    assert isinstance(steps[-2], JSONLSink)
    assert isinstance(steps[-1], HubSink)


def test_two_chained_sinks_compile_and_a_step_after_them_does_not(script):
    """DEC-005, and the rule that replaced it — both stated on the page."""
    from datafast import ListSink, Map, Sink

    (Source.list([{"a": 1}]) >> Sink.jsonl("out.jsonl") >> ListSink()).compile()

    from datafast.core.validation import PipelineValidationError

    with pytest.raises(PipelineValidationError, match="sinks must be the last steps"):
        (
            Source.list([{"a": 1}]) >> ListSink() >> Map(lambda r: r) >> ListSink()
        ).compile()


def test_a_sink_yields_its_records_through_unchanged(tmp_path, monkeypatch):
    """Why two sinks in a row works at all."""
    from datafast import ListSink, Sink

    monkeypatch.chdir(tmp_path)
    collector = ListSink()
    records = [{"a": 1}, {"a": 2}]
    result = (Source.list(records) >> Sink.jsonl("out.jsonl") >> collector).run()
    assert result == records
    assert collector.records == records
    assert (tmp_path / "out.jsonl").read_text().count("\n") == 2


def test_the_hub_push_is_private_and_unguarded(script):
    """Unlike the other cookbooks, this one publishes on every run. The page says so."""
    source = SCRIPT.read_text()
    assert "private=True" in source
    assert "DATAFAST_PUSH_TO_HUB" not in source
    assert "`private=True`" in _page()
    assert "it runs every time" in _page()


def test_the_run_settings_the_page_quotes_are_the_script_s(script):
    source = SCRIPT.read_text()
    for setting in ("batch_size=1", "resume=False"):
        assert setting in source, f"the script no longer passes {setting}"
        assert f"`{setting}`" in _page(), f"the page does not explain {setting}"


def test_the_source_the_page_describes_is_the_script_s(script):
    source_step = script.build_pipeline().steps[0]
    assert isinstance(source_step, HuggingFaceSource)
    assert source_step._columns == ["id", "document", "summary"]
    assert 'split="validation"' in SCRIPT.read_text()


# --- links -----------------------------------------------------------------------


def test_every_page_linked_to_exists():
    links = re.findall(r"\]\((?!https?:)([^)#]+\.md)", _page())
    assert links, "the page has no links — check the test, not the page"
    missing = sorted(link for link in links if not (PAGE.parent / link).resolve().exists())
    assert not missing, f"links to pages that do not exist: {missing}"
