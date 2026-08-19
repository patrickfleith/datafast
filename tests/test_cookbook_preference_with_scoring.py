"""The preference-with-scoring cookbook, pinned against the script it walks through.

The script is the source of truth. Checked in order of how much it matters:

1. Every step, step name, branch path and Score argument the script uses is named on the
   page (code → docs).
2. Every self-contained example on the page executes.
3. The behavioural claims — the record counts, the join's column names, what a Score does
   with a bad reply, which pairs the margin keeps, what a second run costs — are measured
   by running the script itself against a stub served model.

The script runs its pipeline at import, so it is only ever executed with
`datafast.openrouter` replaced and inside a temporary directory.
"""

import json
import re
from pathlib import Path

import pytest

import datafast
from datafast import (
    Branch,
    Filter,
    JoinBranches,
    LLMStep,
    ListSink,
    Score,
    Seed,
    Sink,
    Source,
)
from datafast.sinks.sink import JSONLSink
from datafast.sources.seed import SeedSource

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "cookbook" / "preference_with_scoring.md"
SCRIPT = ROOT / "examples" / "scripts" / "42_pipeline_preference_with_scoring.py"

CHECKPOINT_DIR = "examples/checkpoints/42_preference"
OUTPUT_PATH = "examples/outputs/42_preference_dataset.jsonl"


def _page() -> str:
    return PAGE.read_text()


def _code_blocks(language: str = "python") -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", _page(), re.DOTALL)


class StubModel:
    """Answers every step in the script: a question, an answer, or a score."""

    model_id = "stub-model"

    def __init__(self) -> None:
        self.messages: list[list[dict]] = []

    def generate(self, messages=None, metadata=None, **kwargs) -> str:
        self.messages.append(messages)
        prompt = messages[-1]["content"]
        if prompt.startswith("Score the following content"):
            return json.dumps({"score": 9 if "response_chosen" in prompt else 4})
        if prompt.startswith("Generate a thoughtful"):
            return "Why does gravity bend light?"
        return "An answer."

    @property
    def prompts(self) -> list[str]:
        return [m[-1]["content"] for m in self.messages]


@pytest.fixture(scope="module")
def script(tmp_path_factory):
    """Execute the script with its provider factory stubbed, inside a temp directory."""
    import os

    stub = StubModel()
    original = datafast.openrouter
    datafast.openrouter = lambda model_id, **kwargs: stub
    tmp_path = tmp_path_factory.mktemp("cookbook_42")
    previous = os.getcwd()
    os.chdir(tmp_path)
    namespace: dict = {"__name__": "cookbook_42"}
    try:
        exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), namespace)
    finally:
        os.chdir(previous)
        datafast.openrouter = original
    namespace["_stub"] = stub
    namespace["_tmp_path"] = tmp_path
    # The examples below re-run the pipeline against the same stub, so what the script
    # itself sent has to be snapshotted here.
    namespace["_messages"] = list(stub.messages)
    namespace["_prompts"] = [m[-1]["content"] for m in stub.messages]
    return namespace


@pytest.fixture
def records(script):
    return script["records"]


# --- code → docs -----------------------------------------------------------------


def test_every_step_the_script_uses_is_named_on_the_page(script):
    classes = {type(step).__name__ for step in script["pipeline"].steps}
    assert classes, "the pipeline has no steps — check the test, not the page"
    documented = {
        "SeedSource": "Seed.values",
        "LLMStep": "LLMStep",
        "Branch": "Branch",
        "JoinBranches": "JoinBranches",
        "Score": "Score",
        "Filter": "Filter",
        "JSONLSink": "Sink.jsonl",
    }
    assert classes == set(documented), f"the script's steps changed: {sorted(classes)}"
    missing = [name for name in documented.values() if f"`{name}" not in _page()]
    assert not missing, f"steps undocumented on the page: {missing}"


def test_every_step_name_the_script_sets_is_on_the_page(script):
    names = [step.name for step in script["pipeline"].steps]
    assert names == [
        "SeedSource",
        "generate_question",
        "branch_responses",
        "JoinBranches",
        "score_chosen",
        "score_rejected",
        "filter_margin",
        "JSONLSink",
    ], f"the script renamed or reordered its steps: {names}"
    missing = [name for name in names if name not in _page()]
    assert not missing, f"step names undocumented: {missing}"


def test_the_branch_path_names_are_the_ones_the_page_explains(script):
    branch = script["pipeline"].steps[2]
    assert isinstance(branch, Branch)
    assert set(branch.paths) == {"chosen", "rejected"}
    for name in branch.paths:
        assert f"`{name}`" in _page(), f"path {name} undocumented"
    assert "`response_chosen`" in _page() and "`response_rejected`" in _page()


def test_every_score_argument_the_script_passes_is_documented(script):
    source = SCRIPT.read_text()
    passed = set(re.findall(r"(\w+)=", source.split("Score(")[1].split(").as_step")[0]))
    assert passed == {
        "input_columns",
        "output_column",
        "score_range",
        "llm",
        "criteria",
    }, f"the script's Score call changed: {sorted(passed)}"
    missing = [name for name in passed if f"`{name}" not in _page()]
    assert not missing, f"Score arguments undocumented: {missing}"


def test_the_topics_and_paths_the_page_names_are_the_script_s(script):
    seed_records = list(script["pipeline"].steps[0].process(iter([])))
    topics = [record["topic"] for record in seed_records]
    assert topics == ["gravity", "photosynthesis", "machine learning"]
    for topic in topics:
        assert topic in _page(), f"topic not on the page: {topic}"


def test_the_paths_the_page_quotes_are_the_script_s_paths():
    source = SCRIPT.read_text()
    for path in (OUTPUT_PATH, CHECKPOINT_DIR):
        assert path in source, f"the script no longer writes to {path}"
        assert path in _page(), f"the page does not name {path}"
    assert SCRIPT.relative_to(ROOT).as_posix() in _page()


def test_every_output_column_is_explained_on_the_page(records):
    shown = json.loads(_code_blocks("json")[0])
    assert set(records[0]) == set(shown), (
        f"the run produced {sorted(records[0])}, the page shows {sorted(shown)}"
    )
    for column in shown:
        assert f"`{column}`" in _page(), f"column undocumented: {column}"


def test_every_relative_link_resolves():
    links = re.findall(r"\]\((?!https?://)([^)#]+)", _page())
    assert links, "the page lost its links"
    for link in links:
        assert (PAGE.parent / link).exists(), f"broken link: {link}"


# --- the examples ----------------------------------------------------------------


@pytest.mark.parametrize("block", _code_blocks(), ids=lambda b: b.split("\n")[0][:40])
def test_every_example_executes(block, tmp_path, monkeypatch, script):
    monkeypatch.chdir(tmp_path)
    if "..." in block:
        # The opening block sketches the shape; the sections below give it in full.
        compile(block, str(PAGE), "exec")
        return
    namespace = {
        "Branch": Branch,
        "Filter": Filter,
        "JoinBranches": JoinBranches,
        "LLMStep": LLMStep,
        "Score": Score,
        "Seed": Seed,
        "Sink": Sink,
        "model": script["_stub"],
        "pipeline": script["pipeline"],
        "OUTPUT_PATH": OUTPUT_PATH,
    }
    exec(compile(block, str(PAGE), "exec"), namespace)


def test_there_are_examples_to_execute():
    assert len(_code_blocks()) >= 4, "the page lost its examples"


# --- the run, measured -----------------------------------------------------------


def test_the_run_produces_the_row_count_the_page_promises(records):
    assert len(records) == 3
    assert "**3 rows** by default" in _page()


def test_each_row_costs_the_five_calls_the_page_names(script, records):
    assert len(script["_prompts"]) == 15 == 5 * len(records)
    assert "five LLM calls per row" in _page()


def test_the_step_by_step_counts_on_the_page_are_the_real_ones(script):
    manifest = json.loads(
        (script["_tmp_path"] / CHECKPOINT_DIR / "manifest.json").read_text()
    )
    measured = {s["name"]: (s["records_in"], s["records_out"]) for s in manifest["steps"]}
    assert len(measured) == 8, "the manifest lost a step"
    assert measured["branch_responses"] == (3, 6), "the branch no longer doubles"
    assert measured["JoinBranches"] == (6, 3), "the join no longer merges"
    for name, (records_in, _) in measured.items():
        row = re.search(rf"^\| `{re.escape(name)}` \| (\d+) \|", _page(), re.M)
        assert row, f"the page's table has no row for {name}"
        assert int(row.group(1)) == records_in, f"{name} reads {records_in} records in"


def test_the_checkpoint_file_names_the_page_lists_are_the_real_ones(script):
    written = {p.name for p in (script["_tmp_path"] / CHECKPOINT_DIR).iterdir()}
    quoted = set(re.findall(r"step_\d{3}_[\w.]+\.jsonl", _page()))
    assert quoted, "the page quotes no checkpoint file"
    assert not quoted - written, f"the page names files the run never wrote: {quoted - written}"
    assert "step_002_branch_responses.chosen.jsonl" in written, "one file per path"


def test_the_join_suffixes_the_columns_the_paths_added(records):
    row = records[0]
    assert "response" not in row, "the unsuffixed column should not survive the join"
    assert row["response_chosen"] and row["response_rejected"]
    assert "topic" in row and "question" in row, "pre-branch columns are copied once"
    assert not any(key.startswith("_branch") for key in row), "bookkeeping columns dropped"


def test_the_score_prompt_the_page_quotes_is_the_one_that_was_sent(script):
    quoted = [b for b in _code_blocks("text") if b.startswith("Score the following")]
    assert len(quoted) == 1, "the page no longer quotes the scoring prompt"
    sent = [p for p in script["_prompts"] if p.startswith("Score the following")]
    assert len(sent) == 6, "three rows, two scores each"
    for line in quoted[0].strip().splitlines():
        if line.strip() in ("", "Content:"):
            continue
        if line.startswith(("question:", "response_chosen:")):
            continue  # the values are this page's example, not the stub's
        assert any(line in prompt for prompt in sent), f"no prompt contained: {line}"


def test_each_score_step_sees_one_answer_not_both(script):
    scored = [p for p in script["_prompts"] if p.startswith("Score the following")]
    chosen = [p for p in scored if "response_chosen" in p]
    rejected = [p for p in scored if "response_rejected" in p]
    assert len(chosen) == len(rejected) == 3
    assert not any("response_rejected" in p for p in chosen), "the page says one at a time"


def test_only_the_chosen_path_carries_a_system_prompt(script):
    roles = [[m["role"] for m in messages] for messages in script["_messages"]]
    with_system = [r for r in roles if "system" in r]
    assert len(with_system) == 3, "one per record, from the chosen path only"
    assert "system_prompt" in _page()


# --- the claims that would burn a reader -----------------------------------------


class _Reply:
    def __init__(self, model_id: str, reply: str) -> None:
        self.model_id = model_id
        self._reply = reply

    def generate(self, messages=None, metadata=None, **kwargs) -> str:
        return self._reply


def _scored(reply: str) -> list[dict]:
    step = Score(
        input_columns=["q"],
        output_column="s",
        score_range=(1, 10),
        llm=_Reply("m", reply),
    )
    return (Source.list([{"q": "x"}]) >> step >> ListSink()).run()


SCORE_CASES = [
    (json.dumps({"score": 7}), 7.0),
    (json.dumps({"score": 99}), 10),
    (json.dumps({"score": "high"}), 1),
    (json.dumps({"explanation": "no score"}), 1),
]


@pytest.mark.parametrize("reply,expected", SCORE_CASES, ids=[c[0][:24] for c in SCORE_CASES])
def test_the_page_s_table_of_bad_score_replies_is_measured(reply, expected):
    records = _scored(reply)
    assert len(records) == 1, "the page says only unparseable replies drop the record"
    assert records[0]["s"] == expected


def test_an_unparseable_score_drops_the_record():
    assert _scored("not json at all") == []
    assert "the record is dropped" in _page()


MARGIN_CASES = [
    ({"score_chosen": 9, "score_rejected": 4}, True),
    ({"score_chosen": 7, "score_rejected": 5}, True),
    ({"score_chosen": 6, "score_rejected": 5}, False),
    ({"score_rejected": 3}, False),
]


@pytest.mark.parametrize("record,kept", MARGIN_CASES, ids=[str(c[1]) + str(c[0]) for c in MARGIN_CASES])
def test_the_margin_table_on_the_page_is_measured(record, kept, script):
    step = script["pipeline"].steps[6]
    assert isinstance(step, Filter)
    assert bool(list(step.process(iter([record])))) is kept


def test_a_path_that_rewrites_a_pre_branch_column_wins_unsuffixed():
    """The page's warning about `_model` after a join, proved on two real models."""
    pipeline = (
        Source.list([{"topic": "gravity"}])
        >> LLMStep(
            prompt="{topic}",
            input_columns=["topic"],
            output_column="question",
            model=_Reply("question-model", "Q?"),
        )
        >> Branch(
            chosen=LLMStep(
                prompt="{question}",
                input_columns=["question"],
                output_column="response",
                model=_Reply("chosen-model", "long"),
            ),
            rejected=LLMStep(
                prompt="{question}",
                input_columns=["question"],
                output_column="response",
                model=_Reply("rejected-model", "short"),
            ),
        )
        >> JoinBranches()
        >> ListSink()
    )
    row = pipeline.run()[0]
    assert "_model_chosen" not in row and "_model_rejected" not in row
    assert row["_model"] == "chosen-model", "the first path's value survives, unsuffixed"
    assert "There is no\n    `_model_chosen` or `_model_rejected`" in _page()


def test_a_score_step_stamps_model_over_whatever_the_join_left():
    """Why the `_model` that reaches the file is the scoring model's."""
    step = Score(
        input_columns=["q"],
        output_column="s",
        llm=_Reply("scorer-model", json.dumps({"score": 5})),
    )
    records = (
        Source.list([{"q": "x", "_model": "earlier-model"}]) >> step >> ListSink()
    ).run()
    assert records[0]["_model"] == "scorer-model"


def test_checkpoints_are_written_but_not_read_without_resume(tmp_path, monkeypatch):
    """The page's note: this script pays twice for a second run."""
    monkeypatch.chdir(tmp_path)
    calls = []

    def build():
        model = _Reply("m", "answer")
        original = model.generate

        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        model.generate = counted
        return (
            Source.list([{"t": "a"}, {"t": "b"}])
            >> LLMStep(
                prompt="{t}", input_columns=["t"], output_column="r", model=model
            )
            >> ListSink()
        )

    build().run(checkpoint_dir="ck", batch_size=2)
    first = len(calls)
    build().run(checkpoint_dir="ck", batch_size=2)
    assert len(calls) == 2 * first, "a second run without resume repeated every call"
    build().run(checkpoint_dir="ck", batch_size=2, resume=True)
    assert len(calls) == 2 * first, "resume=True reused the checkpoint"
    assert "resume=True" in _page()


def test_the_pipeline_is_a_source_then_seven_steps(script):
    steps = script["pipeline"].steps
    assert isinstance(steps[0], SeedSource)
    assert isinstance(steps[-1], JSONLSink)
    assert len(steps) == 8, "the page's table has one row per step"
