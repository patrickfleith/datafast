"""The sinks reference, pinned against the code it documents.

Same three checks as the sources & seed reference, in the same order of importance:

1. Every parameter the code takes is named on the page (code → docs). A parameter that
   exists and is undocumented cannot be discovered.
2. Every self-contained example executes.
3. The behavioural claims are asserted against real calls.

Nothing here touches a network: HubSink is exercised with `push_to_hub` and `HfApi`
replaced.
"""

import inspect
import re
from pathlib import Path

import pytest

from datafast import (
    CSVSink,
    HubSink,
    JSONLSink,
    ListSink,
    ParquetSink,
    Sink,
    Source,
)
from datafast.core.step import Step
from datafast.core.validation import PipelineValidationError

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "reference" / "sinks.md"

DOCUMENTED = [
    Sink.jsonl, Sink.csv, Sink.parquet, Sink.hub, Sink.list,
    JSONLSink, CSVSink, ParquetSink, HubSink, ListSink,
]


def _page() -> str:
    return PAGE.read_text()


def _code_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", _page(), re.DOTALL)


def _parameters(func) -> list[str]:
    return [
        p.name for p in inspect.signature(func).parameters.values()
        if p.kind is not p.VAR_KEYWORD and p.name != "self"
    ]


def test_the_page_documents_every_public_constructor():
    """A constructor the page forgets is one a reader never finds."""
    constructors = [n for n in set(dir(Sink)) - set(dir(Step)) if not n.startswith("_")]
    assert constructors, "no constructors found on Sink — check the test, not the page"
    for name in constructors:
        assert f"Sink.{name}(" in _page(), f"Sink.{name} undocumented"


def test_the_page_names_every_sink_class():
    for cls in (JSONLSink, CSVSink, ParquetSink, HubSink, ListSink):
        assert cls.__name__ in _page(), f"{cls.__name__} undocumented"


@pytest.mark.parametrize("func", DOCUMENTED, ids=lambda f: getattr(f, "__qualname__", str(f)))
def test_every_parameter_is_named_on_the_page(func):
    missing = [p for p in _parameters(func) if f"`{p}`" not in _page()]
    assert not missing, f"{func.__qualname__} takes {missing}, undocumented on the page"


def test_the_parameter_guard_is_not_vacuous():
    """Sink.list takes nothing, so the guard above must still see real parameters."""
    found = {p for func in DOCUMENTED for p in _parameters(func)}
    assert found >= {
        "path", "repo_id", "token", "private", "train_size", "seed", "shuffle",
        "commit_message",
    }
    assert _parameters(Sink.list) == []


@pytest.mark.parametrize("block", _code_blocks(), ids=lambda b: b.split("\n")[0][:40])
def test_every_example_executes(block, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exec(compile(block, str(PAGE), "exec"), {})


def test_a_sink_passes_its_records_through_unchanged(tmp_path):
    """The page's central claim: writing is a side effect, run() still returns records."""
    records = [{"text": "hello"}, {"text": "bye"}]
    returned = (Source.list(records) >> Sink.jsonl(tmp_path / "out.jsonl")).run()
    assert returned == records


def test_two_chained_sinks_both_write_in_one_run(tmp_path):
    pipeline = (
        Source.list([{"a": 1}, {"a": 2}])
        >> Sink.jsonl(tmp_path / "out.jsonl")
        >> Sink.csv(tmp_path / "out.csv")
    )
    returned = pipeline.run()

    assert returned == [{"a": 1}, {"a": 2}]
    assert (tmp_path / "out.jsonl").read_text() == '{"a": 1}\n{"a": 2}\n'
    assert (tmp_path / "out.csv").read_text().splitlines() == ["a", "1", "2"]


def test_nothing_may_follow_a_sink():
    from datafast import Map

    pipeline = Source.list([{"a": 1}]) >> Sink.list() >> Map(lambda r: r)
    with pytest.raises(PipelineValidationError, match="sinks must be the last steps"):
        pipeline.compile()


def test_a_sink_inside_a_branch_path_is_rejected():
    from datafast import Branch, JoinBranches, Map

    pipeline = (
        Source.list([{"a": 1}])
        >> Branch(left=Sink.list(), right=Map(lambda r: r))
        >> JoinBranches()
    )
    with pytest.raises(PipelineValidationError, match="not allowed"):
        pipeline.compile()


def test_jsonl_creates_parent_directories_and_overwrites(tmp_path):
    path = tmp_path / "nested" / "out.jsonl"
    list(Sink.jsonl(path).process(iter([{"a": 1}, {"a": 2}])))
    list(Sink.jsonl(path).process(iter([{"a": 3}])))
    assert path.read_text() == '{"a": 3}\n', "the page says overwritten, not appended"


def test_jsonl_stringifies_what_json_cannot_represent(tmp_path):
    import datetime

    path = tmp_path / "out.jsonl"
    list(Sink.jsonl(path).process(iter([{"t": datetime.date(2020, 1, 1), "s": "é"}])))
    assert path.read_text() == '{"t": "2020-01-01", "s": "é"}\n'


def test_csv_takes_its_columns_from_the_first_record(tmp_path):
    path = tmp_path / "out.csv"
    with pytest.raises(ValueError, match="fields not in fieldnames"):
        list(Sink.csv(path).process(iter([{"a": 1}, {"a": 2, "b": 3}])))

    other = tmp_path / "other.csv"
    list(Sink.csv(other).process(iter([{"a": 1, "b": 2}, {"a": 3}])))
    assert other.read_text().splitlines() == ["a,b", "1,2", "3,"]


def test_parquet_drops_a_late_column_and_nulls_a_missing_one(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")

    dropped = tmp_path / "dropped.parquet"
    list(Sink.parquet(dropped).process(iter([{"a": 1}, {"a": 2, "b": 3}])))
    assert pq.read_table(dropped).to_pylist() == [{"a": 1}, {"a": 2}]

    nulled = tmp_path / "nulled.parquet"
    list(Sink.parquet(nulled).process(iter([{"a": 1, "b": 2}, {"a": 3}])))
    assert pq.read_table(nulled).to_pylist() == [{"a": 1, "b": 2}, {"a": 3, "b": None}]


def test_an_empty_run_writes_an_empty_jsonl_and_no_csv_or_parquet(tmp_path):
    jsonl, csv, parquet = tmp_path / "a.jsonl", tmp_path / "a.csv", tmp_path / "a.parquet"
    assert list(Sink.jsonl(jsonl).process(iter([]))) == []
    assert list(Sink.csv(csv).process(iter([]))) == []
    assert list(Sink.parquet(parquet).process(iter([]))) == []

    assert jsonl.read_text() == ""
    assert not csv.exists() and not parquet.exists()


def test_the_list_sink_collects_and_never_clears(tmp_path):
    collected = Sink.list()
    pipeline = Source.list([{"a": 1}]) >> collected
    pipeline.run()
    pipeline.run()
    assert collected.records == [{"a": 1}, {"a": 1}], "the page says records is never cleared"


# --- HubSink: exercised with the network replaced -------------------------------


@pytest.fixture
def hub(monkeypatch):
    """Replace push_to_hub and HfApi, and record what HubSink asked them to do."""
    datasets = pytest.importorskip("datasets")
    huggingface_hub = pytest.importorskip("huggingface_hub")

    calls: dict = {"uploads": []}

    def fake_push(self, repo_id, **kwargs):
        calls["pushed"] = self
        calls["repo_id"] = repo_id
        calls["kwargs"] = kwargs

    monkeypatch.setattr(datasets.Dataset, "push_to_hub", fake_push)
    monkeypatch.setattr(datasets.DatasetDict, "push_to_hub", fake_push)

    class FakeApi:
        def __init__(self, token=None):
            calls["api_token"] = token

        def hf_hub_download(self, **kwargs):
            raise FileNotFoundError("no README yet")

        def upload_file(self, **kwargs):
            calls["uploads"].append(kwargs)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return calls


def test_hub_defaults_are_the_documented_ones(hub):
    records = [{"i": i} for i in range(4)]
    assert list(Sink.hub("me/ds").process(iter(records))) == records
    assert hub["repo_id"] == "me/ds"
    assert hub["kwargs"] == {
        "token": None,
        "private": True,
        "commit_message": "Upload dataset via datafast",
    }


def test_hub_reads_the_token_from_hf_token_when_none_is_passed(hub, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "from-env")
    list(Sink.hub("me/ds").process(iter([{"i": 1}])))
    assert hub["kwargs"]["token"] == "from-env"


def test_an_explicit_token_wins_over_the_environment(hub, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "from-env")
    list(Sink.hub("me/ds", token="explicit").process(iter([{"i": 1}])))
    assert hub["kwargs"]["token"] == "explicit"
    assert hub["api_token"] == "explicit", "the README upload uses the same token"


def test_train_size_pushes_two_splits(hub):
    records = [{"i": i} for i in range(10)]
    list(Sink.hub("me/ds", train_size=0.8).process(iter(records)))
    assert {name: len(split) for name, split in hub["pushed"].items()} == {
        "train": 8,
        "test": 2,
    }


def test_without_train_size_one_dataset_is_pushed(hub):
    datasets = pytest.importorskip("datasets")
    list(Sink.hub("me/ds").process(iter([{"i": i} for i in range(4)])))
    assert isinstance(hub["pushed"], datasets.Dataset)


def test_train_size_must_be_strictly_between_zero_and_one(hub):
    for bad in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="train_size must be between"):
            list(Sink.hub("me/ds", train_size=bad).process(iter([{"i": 1}])))


def test_the_same_seed_gives_the_same_split(hub):
    records = [{"i": i} for i in range(20)]

    def split_ids(**kwargs):
        list(Sink.hub("me/ds", train_size=0.5, **kwargs).process(iter(records)))
        return [record["i"] for record in hub["pushed"]["train"]]

    assert split_ids(seed=1) == split_ids(seed=1)
    assert split_ids(seed=1) != split_ids(seed=2)


def test_shuffle_changes_what_is_pushed_but_not_what_is_returned(hub):
    records = [{"i": i} for i in range(20)]

    returned = list(Sink.hub("me/ds", shuffle=True).process(iter(records)))
    shuffled = [record["i"] for record in hub["pushed"]]
    assert returned == records, "the page says records stay in pipeline order"
    assert shuffled != list(range(20))

    list(Sink.hub("me/ds", shuffle=False).process(iter(records)))
    assert [record["i"] for record in hub["pushed"]] == list(range(20))


def test_a_custom_commit_message_is_used(hub):
    list(Sink.hub("me/ds", commit_message="v2").process(iter([{"i": 1}])))
    assert hub["kwargs"]["commit_message"] == "v2"


def test_private_false_is_passed_through(hub):
    list(Sink.hub("me/ds", private=False).process(iter([{"i": 1}])))
    assert hub["kwargs"]["private"] is False


def test_the_readme_tag_is_added_in_a_second_commit(hub):
    list(Sink.hub("me/ds").process(iter([{"i": 1}])))
    (upload,) = hub["uploads"]
    assert upload["path_in_repo"] == "README.md"
    assert upload["repo_type"] == "dataset"
    assert upload["path_or_fileobj"].decode() == "---\ntags:\n- datafast\n---\n"
    assert upload["commit_message"] != hub["kwargs"]["commit_message"]


def test_an_existing_readme_is_kept_below_the_tag(hub, monkeypatch, tmp_path):
    existing = tmp_path / "README.md"
    existing.write_text("# My dataset\n")
    huggingface_hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(
        huggingface_hub.HfApi, "hf_hub_download", lambda self, **kwargs: str(existing)
    )

    list(Sink.hub("me/ds").process(iter([{"i": 1}])))
    assert hub["uploads"][0]["path_or_fileobj"].decode().endswith("# My dataset\n")


def test_an_empty_run_pushes_nothing(hub):
    assert list(Sink.hub("me/ds").process(iter([]))) == []
    assert "repo_id" not in hub and hub["uploads"] == []


# --- extras and links -----------------------------------------------------------


def test_a_missing_extra_raises_an_import_error_naming_it(monkeypatch, tmp_path):
    """sys.modules[name] = None makes the import inside process() fail."""
    monkeypatch.setitem(__import__("sys").modules, "pyarrow", None)
    with pytest.raises(ImportError, match=r"datafast\[parquet\]"):
        list(Sink.parquet(tmp_path / "out.parquet").process(iter([{"a": 1}])))

    monkeypatch.setitem(__import__("sys").modules, "datasets", None)
    with pytest.raises(ImportError, match=r"datafast\[hub\]"):
        list(Sink.hub("me/ds").process(iter([{"a": 1}])))


def test_the_extras_the_page_names_are_the_ones_the_code_names():
    source = (ROOT / "datafast" / "sinks" / "sink.py").read_text()
    for extra in ("datafast[parquet]", "datafast[hub]"):
        assert extra in source and extra in _page()


def test_every_page_linked_to_exists():
    links = re.findall(r"\]\((?!https?:)([^)#]+\.md)", _page())
    assert links, "no links found — check the test, not the page"
    missing = sorted(link for link in links if not (PAGE.parent / link).resolve().exists())
    assert not missing, f"links to pages that do not exist: {missing}"
