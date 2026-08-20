"""The contributing guide, pinned against the repository it describes.

A contributing page fails in a way reference pages do not: a command that no longer
works wastes a newcomer's first hour. So the commands are the main subject here. Every
shell command the page prints is executed — the pytest ones as collections rather than
full runs, which is enough to prove the invocation is valid and the marker expression
selects something.

Then the claims: the marker table against `pytest.ini`, the live gate against a real
pytest session, the layout table against the filesystem, and the two extension recipes
against real objects.

Nothing here makes a network call. The live-suite commands are collected, never run.
"""

import ast
import inspect
import re
import shlex
import subprocess
import sys
import tomllib
from collections.abc import Iterable
from pathlib import Path

import pytest

from datafast import Source, Step
from datafast.core.types import Record
from datafast.core.validation import PipelineValidationError
from datafast.llm.served_model import ServedModel

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "contributing.md"
PYTEST_INI = ROOT / "pytest.ini"
PYPROJECT = ROOT / "pyproject.toml"
VENV_PYTEST = ROOT / ".venv" / "bin" / "pytest"
VENV_RUFF = ROOT / ".venv" / "bin" / "ruff"
VENV_ZENSICAL = ROOT / ".venv" / "bin" / "zensical"


def _page() -> str:
    return PAGE.read_text()


def _bash_blocks() -> list[str]:
    return re.findall(r"```bash\n(.*?)```", _page(), re.DOTALL)


def _commands() -> list[str]:
    """Every shell command the page prints, one per line, comments dropped."""
    lines = []
    for block in _bash_blocks():
        for line in block.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines


def _pytest_commands() -> list[str]:
    return [c for c in _commands() if c.startswith(".venv/bin/pytest")]


def _project() -> dict:
    return tomllib.loads(PYPROJECT.read_text())["project"]


def _markers_in_ini() -> list[str]:
    """The marker names `pytest.ini` registers."""
    text = PYTEST_INI.read_text()
    body = text.split("markers =", 1)[1].split("\n\n", 1)[0]
    return re.findall(r"^\s{4}(\w+):", body, re.MULTILINE)


def _marker_id(marker: str) -> str:
    """Readable ids for the marker parametrizations below."""
    return f"marker-{marker}"


_COLLECTIONS: dict[str, subprocess.CompletedProcess] = {}


def _collect(command: str) -> subprocess.CompletedProcess:
    """Run one of the page's pytest commands in collect-only mode, once."""
    if command not in _COLLECTIONS:
        _COLLECTIONS[command] = subprocess.run(
            [str(VENV_PYTEST), *shlex.split(command)[1:], "--collect-only", "-q"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
    return _COLLECTIONS[command]


def _collected_count(result: subprocess.CompletedProcess) -> int:
    match = re.search(r"(\d+)(?:/\d+)? tests? collected", result.stdout)
    assert match, f"no collection summary in:\n{result.stdout[-2000:]}"
    return int(match.group(1))


pytestmark = pytest.mark.skipif(
    not VENV_PYTEST.exists(), reason="no .venv/bin/pytest to run the page's commands"
)


# --- the commands the page gives ------------------------------------------------


def test_the_page_gives_commands_at_all():
    """Guards every command test below from passing on an empty list."""
    assert len(_commands()) >= 5
    assert _pytest_commands()


@pytest.mark.parametrize("command", _pytest_commands())
def test_every_pytest_command_the_page_gives_collects_something(command):
    """A command that errors, or selects nothing, is a command that wastes an hour."""
    result = _collect(command)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert _collected_count(result) > 0, f"{command} selects no tests"


def test_the_default_command_leaves_the_live_suite_out():
    """`-m "not live"` is the page's default because it deselects rather than skips."""
    result = _collect('.venv/bin/pytest -m "not live"')
    assert "deselected" in result.stdout
    collected = [line for line in result.stdout.splitlines() if line.startswith("tests/")]
    assert collected
    assert not [line for line in collected if line.startswith("tests/live/")]


def test_the_single_file_command_names_a_file_that_exists():
    for command in _pytest_commands():
        for token in shlex.split(command):
            if token.startswith("tests/"):
                assert (ROOT / token).exists(), f"{token} does not exist"


def test_the_live_command_selects_only_live_tests():
    """`--run-live -m "anthropic"` must reach the live suite and nothing else."""
    result = _collect('.venv/bin/pytest --run-live -m "anthropic"')
    collected = [
        line for line in result.stdout.splitlines() if line.startswith("tests/")
    ]
    assert collected
    assert all(line.startswith("tests/live/") for line in collected), collected


@pytest.mark.parametrize(
    "command", [c for c in _commands() if c.startswith(".venv/bin/zensical")]
)
def test_every_zensical_subcommand_the_page_names_exists(command):
    if not VENV_ZENSICAL.exists():
        pytest.skip("zensical is not installed — the docs extra is optional")
    subcommand = shlex.split(command)[1]
    help_text = subprocess.run(
        [str(VENV_ZENSICAL), "--help"], capture_output=True, text=True
    ).stdout
    assert f"\n  {subcommand} " in help_text, f"zensical has no `{subcommand}` command"


def test_the_install_command_names_extras_that_exist():
    command = next(c for c in _commands() if c.startswith("pip install -e"))
    extras = re.search(r"\[([^\]]+)\]", command).group(1).split(",")
    declared = tomllib.loads(PYPROJECT.read_text())["project"][
        "optional-dependencies"
    ]
    for extra in extras:
        assert extra in declared, f"no `{extra}` extra in pyproject.toml"


def test_the_dev_extra_really_brings_what_the_page_says_it_brings():
    dev = tomllib.loads(PYPROJECT.read_text())["project"]["optional-dependencies"][
        "dev"
    ]
    joined = " ".join(dev)
    assert "pytest" in joined
    assert "ruff" in joined
    assert "parquet" in joined and "hub" in joined


def test_the_clone_url_matches_the_configured_remote():
    command = next(c for c in _commands() if c.startswith("git clone"))
    url = shlex.split(command)[-1]
    remotes = subprocess.run(
        ["git", "remote", "-v"], cwd=ROOT, capture_output=True, text=True
    ).stdout
    assert url.removesuffix(".git") in remotes


def test_the_python_version_the_page_names_matches_pyproject():
    assert _project()["requires-python"] == ">=3.10"
    assert "3.10" in _page()


# --- markers ---------------------------------------------------------------------


def test_pytest_ini_registers_markers_at_all():
    assert len(_markers_in_ini()) >= 8


@pytest.mark.parametrize("marker", _markers_in_ini(), ids=_marker_id)
def test_every_registered_marker_is_on_the_page(marker):
    assert f"`{marker}`" in _page(), f"{marker} is registered but undocumented"


@pytest.mark.parametrize("marker", _markers_in_ini(), ids=_marker_id)
def test_every_registered_marker_is_carried_by_real_tests(marker):
    """The page says so, and a marker nothing carries is a marker to retire.

    Scanned rather than collected: eight `pytest -m` subprocesses cost half a minute
    and prove the same thing, since a marker nothing writes down selects nothing.
    """
    carriers = [
        path for path in (ROOT / "tests").rglob("test_*.py")
        if f"pytest.mark.{marker}" in path.read_text()
    ]
    assert carriers, f"no test carries `{marker}`"


def test_the_marker_scan_agrees_with_pytest_on_the_gate_marker():
    """One real collection, to prove the scan above is measuring the right thing."""
    result = subprocess.run(
        [str(VENV_PYTEST), "-m", "live", "--collect-only", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    collected = [line for line in result.stdout.splitlines() if line.startswith("tests/")]
    assert collected
    assert all(line.startswith("tests/live/") for line in collected)


@pytest.mark.parametrize("marker", _markers_in_ini(), ids=_marker_id)
def test_the_page_quotes_each_markers_registered_description(marker):
    described = dict(re.findall(r"^\s{4}(\w+): (.+)$", PYTEST_INI.read_text(), re.M))
    wording = described[marker].removeprefix("marks ")
    assert wording in _page(), f"{marker}'s description drifted: {wording}"


def test_every_live_module_carries_the_gate_marker_and_a_provider_marker():
    """The page's `pytestmark = [pytest.mark.live, pytest.mark.anthropic]` shape."""
    modules = list((ROOT / "tests" / "live").rglob("test_*.py"))
    assert modules
    providers = set(_markers_in_ini()) - {"live", "multimodal"}
    for module in modules:
        marks = set(re.findall(r"pytest\.mark\.(\w+)", module.read_text()))
        assert "live" in marks, f"{module.name} is not marked live"
        assert marks & providers, f"{module.name} carries no provider marker"


# --- the gate --------------------------------------------------------------------


@pytest.fixture(scope="module")
def gate_session(tmp_path_factory):
    """A throwaway pytest project using this repo's real root conftest.

    Proving the gate against the real live suite would mean running it. This runs the
    same `conftest.py` over two fake tests instead.
    """
    root = tmp_path_factory.mktemp("gate")
    tests = root / "tests"
    tests.mkdir()
    (tests / "conftest.py").write_text((ROOT / "tests" / "conftest.py").read_text())
    (tests / "test_gated.py").write_text(
        "import pytest\n\npytestmark = pytest.mark.live\n\n\ndef test_gated():\n    pass\n"
    )
    (tests / "test_plain.py").write_text("def test_plain():\n    pass\n")
    (root / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    live: marks tests that hit a real provider endpoint\n"
    )

    def _run(*args: str) -> str:
        return subprocess.run(
            [str(VENV_PYTEST), str(root), "-q", *args],
            cwd=root,
            capture_output=True,
            text=True,
        ).stdout

    return _run


def test_a_plain_run_skips_the_live_test(gate_session):
    """The page's table, row one: collected, then skipped."""
    assert "1 passed, 1 skipped" in gate_session()


def test_not_live_deselects_the_live_test(gate_session):
    """Row two: never collected."""
    assert "1 passed, 1 deselected" in gate_session("-m", "not live")


def test_run_live_lets_the_live_test_through(gate_session):
    """Row three: the opt-in works, which is why it has to be typed."""
    assert "2 passed" in gate_session("--run-live")


def test_the_gate_ignores_a_parametrize_id_named_live(gate_session, tmp_path):
    """The gate reads the marker, so an id of "live" on a mocked test is not the gate's.

    `item.keywords` holds parametrize ids as well as markers, which is why
    `pytest_collection_modifyitems` asks for the marker instead.
    """
    module = tmp_path / "tests" / "test_ids.py"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text(
        "import pytest\n\n\n"
        '@pytest.mark.parametrize("name", ["live", "local"])\n'
        "def test_named(name):\n    pass\n"
    )
    (tmp_path / "tests" / "conftest.py").write_text(
        (ROOT / "tests" / "conftest.py").read_text()
    )
    result = subprocess.run(
        [str(VENV_PYTEST), str(tmp_path), "-q", "-p", "no:cacheprovider"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    ).stdout
    assert "2 passed" in result, result[-800:]


def test_the_gate_is_the_only_thing_standing_between_a_default_run_and_a_provider():
    """If `--run-live` ever stopped being required, the page's table would be a lie."""
    conftest = (ROOT / "tests" / "conftest.py").read_text()
    assert "--run-live" in conftest
    assert "pytest_collection_modifyitems" in conftest


# --- live tests skip themselves ---------------------------------------------------


def _fixture_function(name: str):
    """The plain function inside a live fixture, callable outside a pytest session.

    The conftest is executed with `pytest.fixture` replaced by an identity decorator,
    which is version-proof in a way that reaching into the fixture object is not.
    """
    conftest = ROOT / "tests" / "live" / "conftest.py"
    namespace = {"__file__": str(conftest), "__name__": "live_conftest"}
    # `load_dotenv()` is dropped so the repo's own `.env` cannot mask the guard
    # under test — the guard reads the environment, and that is what is being checked.
    source = (
        conftest.read_text()
        .replace('@pytest.fixture(scope="session")', "")
        .replace("load_dotenv()", "pass")
    )
    exec(compile(source, str(conftest), "exec"), namespace)
    return namespace[name]()


def test_require_api_key_skips_on_a_missing_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(pytest.skip.Exception, match="ANTHROPIC_API_KEY is not set"):
        _fixture_function("require_api_key")("ANTHROPIC_API_KEY")


def test_require_api_key_passes_when_the_key_is_there(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-not-a-real-key")
    _fixture_function("require_api_key")("ANTHROPIC_API_KEY")


def test_require_ollama_skips_when_no_daemon_answers(monkeypatch):
    """The page promises a skip, not a connection error."""
    monkeypatch.setenv("OLLAMA_API_BASE", "http://127.0.0.1:1")
    with pytest.raises(pytest.skip.Exception, match="no Ollama daemon"):
        _fixture_function("require_ollama")()


def test_both_guards_live_in_the_shared_conftest_the_page_names():
    text = (ROOT / "tests" / "live" / "conftest.py").read_text()
    assert "def require_api_key" in text
    assert "def require_ollama" in text
    assert not (ROOT / "tests" / "live" / "ollama" / "conftest.py").read_text().count(
        "def require_ollama"
    ), "require_ollama has been duplicated into the ollama package"


# --- the two config files ---------------------------------------------------------


def test_pytest_is_configured_in_one_file_and_says_so_without_warning():
    """Two config files made pytest warn on every run; the page says there is one."""
    assert "[tool.pytest.ini_options]" not in PYPROJECT.read_text()
    header = subprocess.run(
        [str(VENV_PYTEST), "--collect-only", "tests/test_public_api.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout
    assert "configfile: pytest.ini" in header
    assert "WARNING: ignoring pytest config" not in header
    assert "`pytest.ini`" in _page()


# --- the page-test convention -----------------------------------------------------


def test_most_pages_carry_a_test_that_names_them():
    """The convention the page describes, counted rather than asserted in prose."""
    pinned = set()
    for test in (ROOT / "tests").glob("test_*.py"):
        pinned.update(
            re.findall(r'"docs"((?: / "[\w.]+")+)', test.read_text())
        )
    pages = set((ROOT / "docs").rglob("*.md"))
    assert len(pinned) >= len(pages) * 0.7, "the page-test convention has decayed"


def test_the_named_example_pairing_is_real():
    assert (ROOT / "docs" / "reference" / "sinks.md").exists()
    assert (ROOT / "docs" / "reference" / "sources_and_seed.md").exists()
    assert (ROOT / "tests" / "test_reference_sinks.py").exists()
    assert (ROOT / "tests" / "test_reference_sources_and_seed.py").exists()


# --- adding a served model ---------------------------------------------------------


@pytest.mark.parametrize(
    "symbol",
    ["_SERVED_MODEL_CATALOG", "_PROVIDER_DEFAULTS", "ServedModelCapabilities",
     "HOSTED_CHAT", "MISTRAL_CHAT"],
)
def test_every_capability_symbol_the_page_names_exists(symbol):
    source = (ROOT / "datafast" / "llm" / "capabilities.py").read_text()
    assert symbol in source, f"{symbol} is gone from capabilities.py"
    assert f"`{symbol}`" in _page()


@pytest.mark.parametrize("provider", ["openai", "mistral", "ollama"])
def test_the_per_model_resolvers_the_page_names_exist(provider):
    from datafast.llm import capabilities

    assert hasattr(capabilities, f"_resolve_{provider}_capabilities")
    assert "_resolve_<provider>_capabilities" in _page()


def test_the_served_model_constructor_takes_the_arguments_the_page_lists():
    parameters = set(inspect.signature(ServedModel.__init__).parameters)
    assert parameters, "check the test, not the page"
    for name in ("model_id", "litellm_route", "env_key_name"):
        assert name in parameters, f"ServedModel no longer takes {name}"
        assert f"`{name}`" in _page()


def test_a_local_backend_really_passes_no_env_key_name():
    """The page's one exception to step 3."""
    source = (ROOT / "datafast" / "llm" / "served_model.py").read_text()
    ollama_class = source.split("class _OllamaServedModel", 1)[1].split("\n\n\n", 1)[0]
    assert "env_key_name=None" in ollama_class


def test_every_provider_factory_is_exported_as_step_five_requires():
    import datafast

    factories = ["openai", "anthropic", "gemini", "mistral", "openrouter", "ollama",
                 "openai_compatible"]
    for name in factories:
        assert name in datafast.__all__, f"{name} is not exported"
        assert callable(getattr(datafast, name))


def test_every_live_provider_directory_has_the_conftest_shape_step_six_describes():
    directories = [
        d for d in (ROOT / "tests" / "live").iterdir()
        if d.is_dir() and not d.name.startswith(("_", "."))and d.name != "assets"
    ]
    assert directories
    for directory in directories:
        conftest = directory / "conftest.py"
        assert conftest.exists(), f"{directory.name} has no conftest.py"
        text = conftest.read_text()
        assert "def served_model" in text, f"{directory.name} has no served_model fixture"
        assert "require_api_key" in text or "require_ollama" in text


def test_every_provider_with_a_live_suite_has_a_reference_page():
    """Step 7, checked in the direction that matters."""
    for directory in (ROOT / "tests" / "live").iterdir():
        if not directory.is_dir() or directory.name in {"assets", "__pycache__"}:
            continue
        page = ROOT / "docs" / "reference" / "providers" / f"{directory.name}.md"
        assert page.exists(), f"no reference page for {directory.name}"


# --- adding a step -----------------------------------------------------------------


def _page_step_class():
    """The `Shout` class exactly as the page prints it."""
    block = next(b for b in re.findall(r"```python\n(.*?)```", _page(), re.DOTALL)
                 if "class Shout" in b)
    namespace: dict = {}
    exec(compile(block, str(PAGE), "exec"), namespace)
    return namespace["Shout"]


def test_the_step_base_class_asks_for_exactly_one_method():
    """The page says "one method"; `process` is the only abstract one."""
    assert Step.__abstractmethods__ == frozenset({"process"})
    assert "`process`" in _page()


def test_the_example_step_overrides_process_with_the_base_signature():
    """A `process` that renamed its parameter would still run, and still be wrong."""
    base = list(inspect.signature(Step.process).parameters)
    override = list(inspect.signature(_page_step_class().process).parameters)
    assert base == ["self", "records"]
    assert override == base, override


@pytest.mark.parametrize("method", ["process", "as_step", "run", "compile"])
def test_every_method_the_page_names_is_real_and_on_the_page(method):
    from datafast import Pipeline

    assert hasattr(Step, method) or hasattr(Pipeline, method)
    assert f"{method}(" in _page(), f"{method} is named nowhere on the page"


def test_the_pages_example_step_runs_and_does_what_it_says():
    Shout = _page_step_class()
    records = (Source.list([{"text": "hi"}]) >> Shout("text")).run()
    assert records == [{"text": "HI"}]


def test_the_pages_second_example_block_runs_verbatim():
    blocks = re.findall(r"```python\n(.*?)```", _page(), re.DOTALL)
    setup = next(b for b in blocks if "class Shout" in b)
    usage = next(b for b in blocks if "pipeline = " in b)
    namespace: dict = {}
    exec(compile(setup + "\n" + usage, str(PAGE), "exec"), namespace)
    assert namespace["records"] == [{"text": "HI"}]


def test_a_custom_step_is_checkpointed_under_its_class_name(tmp_path):
    """The page names `step_001_Shout.jsonl` — the runner writes it."""
    Shout = _page_step_class()
    (Source.list([{"text": "hi"}]) >> Shout("text")).run(
        checkpoint_dir=str(tmp_path / "ckpt")
    )
    written = {p.name for p in (tmp_path / "ckpt").iterdir()}
    assert "step_001_Shout.jsonl" in written


def test_as_step_renames_both_the_step_and_its_checkpoint_file(tmp_path):
    Shout = _page_step_class()
    step = Shout("text")
    assert step.name == "Shout"
    assert step.as_step("shout").name == "shout"
    (Source.list([{"text": "hi"}]) >> Shout("text").as_step("shout")).run(
        checkpoint_dir=str(tmp_path / "ckpt")
    )
    assert "step_001_shout.jsonl" in {p.name for p in (tmp_path / "ckpt").iterdir()}


def test_process_is_a_generator_so_records_stream():
    Shout = _page_step_class()
    result = Shout("text").process(iter([{"text": "a"}]))
    assert inspect.isgenerator(result)


def test_input_columns_is_what_makes_compile_check_the_step():
    """The page quotes this error; the validator produces it."""
    Shout = _page_step_class()
    with pytest.raises(PipelineValidationError) as error:
        (Source.list([{"a": 1}]) >> Shout("text")).compile()
    message = str(error.value)
    assert "Step 'Shout' references column(s) ['text'] that are not available." in message
    assert "Available columns: ['a']." in message
    for line in message.split(". "):
        assert line.strip(". ") in _page(), f"the page's quoted error drifted: {line}"


def test_a_step_without_input_columns_is_silent_at_compile_time():
    class Quiet(Step):
        def process(self, records: Iterable[Record]) -> Iterable[Record]:
            yield from records

    (Source.list([{"a": 1}]) >> Quiet()).compile()


def test_a_custom_step_turns_off_column_checks_downstream_of_itself():
    """The cost the page warns about, in both directions."""
    from datafast import Group

    class Quiet(Step):
        def process(self, records: Iterable[Record]) -> Iterable[Record]:
            yield from records

    with pytest.raises(PipelineValidationError):
        (Source.list([{"a": 1}]) >> Group(by=["nope"])).compile()

    (Source.list([{"a": 1}]) >> Quiet() >> Group(by=["nope"])).compile()


def test_map_and_flatmap_exist_as_the_page_suggests_reaching_for_them():
    import datafast

    assert "Map" in datafast.__all__
    assert "FlatMap" in datafast.__all__


# --- style and CI -------------------------------------------------------------------


def test_ruff_is_configured_with_the_line_length_the_page_names():
    ruff = tomllib.loads(PYPROJECT.read_text())["tool"]["ruff"]
    assert ruff["line-length"] == 88
    assert "88" in _page()


def test_the_tree_passes_ruff_as_the_page_claims():
    """The page tells you to run this before pushing, and CI gates on it."""
    if not VENV_RUFF.exists():
        pytest.skip("ruff is not installed")
    result = subprocess.run(
        [str(VENV_RUFF), "check", "."], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout


def test_the_selected_ruff_rules_are_the_ones_the_page_names():
    """C90 is left out on purpose; the page says so and this pins it."""
    lint = tomllib.loads(PYPROJECT.read_text())["tool"]["ruff"]["lint"]
    assert lint["select"] == ["E", "F", "C4", "W"]
    assert "C90" not in lint["select"], "complexity is not enforced — see the page"
    for rule in lint["select"]:
        assert f"`{rule}`" in _page(), f"{rule} is enforced but unmentioned"


def test_ci_runs_the_suite_the_page_tells_you_to_run():
    tests = (ROOT / ".github" / "workflows" / "tests.yml").read_text()
    assert 'pytest -m "not live"' in tests
    assert 'pytest -m "not live"' in _page()


def test_publishing_is_gated_on_that_job():
    """The page promises a release cannot go out over a failing suite."""
    publish = (ROOT / ".github" / "workflows" / "publish.yml").read_text()
    assert "uses: ./.github/workflows/tests.yml" in publish
    assert "needs: test" in publish


def test_ci_builds_the_docs_the_same_way_the_page_tells_you_to():
    deploy = (ROOT / ".github" / "workflows" / "deploy-docs.yml").read_text()
    assert "zensical build --strict" in deploy
    assert "zensical build --strict" in _page()


def test_the_issue_and_discussion_links_match_pyproject():
    urls = tomllib.loads(PYPROJECT.read_text())["project"]["urls"]
    assert urls["Issue Tracker"] in _page()
    assert urls["Discussions"] in _page()


# --- layout and links ----------------------------------------------------------------


def _layout_paths() -> list[str]:
    rows = re.findall(r"^\| `([\w./<>-]+)` \| ", _page(), re.MULTILINE)
    return [r for r in rows if "/" in r]


def test_the_layout_table_lists_paths_at_all():
    assert len(_layout_paths()) >= 10


@pytest.mark.parametrize("path", _layout_paths())
def test_every_path_in_the_layout_table_exists(path):
    assert (ROOT / path).exists(), f"{path} is on the page but not on disk"


def test_the_examples_count_on_the_page_is_right():
    scripts = list((ROOT / "examples" / "scripts").glob("*.py"))
    assert f"{len(scripts)} numbered runnable scripts" in _page()


@pytest.mark.parametrize(
    "link", sorted(set(re.findall(r"\]\((?!https?:)([^)#]+\.md)", PAGE.read_text())))
)
def test_every_relative_link_resolves(link):
    assert (PAGE.parent / link).resolve().exists(), f"{link} does not exist"


def test_the_page_links_somewhere_at_all():
    assert re.findall(r"\]\((?!https?:)([^)#]+\.md)", _page())


def test_the_page_uses_no_words_the_glossary_asks_us_to_avoid():
    """`Avoid:` lines in the glossary name terms that mean something else here."""
    glossary = (ROOT / "docs" / "glossary.md").read_text()
    avoided = set()
    for line in re.findall(r"\*\*Avoid:\*\* (.+)$", glossary, re.MULTILINE):
        avoided.update(term.strip() for term in line.split(","))
    # Words that are also ordinary English are checked as whole words only. A few are
    # unavoidable here and mean something else: a pytest `fixture` is not a seed, and a
    # `variable` is an environment variable, not a dimension, and a `workflow` is a
    # file in `.github/workflows/`, not a pipeline.
    allowed = {"build", "validate", "key", "host", "template", "row", "example",
               "field", "features", "block", "flow", "engine", "backend", "endpoint",
               "provider", "served model", "target", "helper", "constructor", "item",
               "presets", "registry", "model list", "variable", "cache",
               "fixture", "workflow",
               "save file", "graph", "node", "stage", "operator"}
    for term in avoided - allowed:
        assert not re.search(rf"\b{re.escape(term)}\b", _page(), re.IGNORECASE), (
            f"the glossary asks us to avoid '{term}'"
        )


def test_the_page_is_valid_python_where_it_claims_to_be():
    for block in re.findall(r"```python\n(.*?)```", _page(), re.DOTALL):
        ast.parse(block)


def test_the_repository_tracks_no_lockfile_for_the_page_to_describe():
    """The page called `uv.lock` a stale lockfile "in the repository". It is gitignored,
    so a fresh clone has none and the warning described a file nobody would ever see.
    Commit one and this fails — then the page has to document it."""
    tracked = subprocess.run(
        ["git", "ls-files", "uv.lock"], cwd=ROOT, capture_output=True, text=True
    ).stdout
    assert not tracked.strip(), "uv.lock is tracked now — the page must describe it"
    assert "uv.lock" not in _page()
    assert "uv sync" not in _page()


def test_the_page_never_suggests_a_bare_pytest_that_could_reach_a_provider():
    for command in _pytest_commands():
        assert command.startswith(".venv/bin/pytest"), command
        if "--run-live" in command:
            assert "-m" in command, "an unfiltered --run-live would call every provider"


def test_this_test_runs_on_the_interpreter_the_page_supports():
    assert sys.version_info >= (3, 10)
