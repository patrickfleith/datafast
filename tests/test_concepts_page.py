"""The Key Concepts page teaches the execution model, so its code must actually run.

Every self-contained block on the page is executed here, and the numbers the prose
quotes are asserted against what the library returns. None of it touches a provider:
the blocks that build a served model construct it without calling it.

The page is ordered deliberately: record, step, the kinds of step, the served model and
what it can do, then how a pipeline is assembled and run. That order is pinned too, and
so is the page's promise to cover every term the glossary defines.
"""

import re
from pathlib import Path

import pytest

from datafast.core.validation import PipelineValidationError

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "concepts.md"
DATA_OPS = ROOT / "docs" / "reference" / "data_ops.md"
GLOSSARY = ROOT / "docs-agents" / "GLOSSARY.md"

# The teaching order the page is built around.
CONCEPT_ORDER = [
    "Record", "Step", "Source", "Seed", "Served model", "Capabilities", "LLM step",
    "Parse mode and structured output", "Sample", "Sink", "Data operations",
    "Branching", "Pipeline", "Compile", "The runner", "Checkpoints and the manifest",
    "Metadata columns", "Where to go next",
]


def _code_blocks(language: str, doc: Path) -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", doc.read_text(), re.DOTALL)


def _runnable_blocks() -> list[str]:
    """Blocks that stand alone — the ones that import what they use.

    The page also shows a bare `process` signature and a line lifted from the runner;
    neither is meant to run, and neither imports anything.
    """
    return [b for b in _code_blocks("python", PAGE) if "from datafast import" in b]


def _headings() -> list[str]:
    return re.findall(r"^## (.+)$", PAGE.read_text(), re.MULTILINE)


def test_the_page_has_runnable_blocks():
    assert len(_runnable_blocks()) >= 4


@pytest.mark.parametrize("block", _runnable_blocks(), ids=lambda b: b.split("\n")[0][:40])
def test_every_self_contained_block_executes(block, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exec(compile(block, str(PAGE), "exec"), {})


def _block_containing(needle: str) -> str:
    (block,) = [b for b in _runnable_blocks() if needle in b]
    return block


# --- the teaching order -------------------------------------------------------------


def test_the_page_teaches_the_concepts_in_the_intended_order():
    """Record → step → the kinds of step → how they are assembled and run."""
    headings = _headings()
    assert headings, "the page has no sections — check the test, not the page"
    assert headings == CONCEPT_ORDER


def test_the_page_is_titled_key_concepts_on_the_page_and_in_the_nav():
    assert PAGE.read_text().startswith("# Key Concepts\n")
    assert "- Key Concepts: concepts.md" in (ROOT / "mkdocs.yml").read_text()


def test_every_concept_section_says_where_to_read_more():
    """A concept section is an on-ramp, not the reference; each must hand off."""
    body = PAGE.read_text()
    for concept in ("Source", "Seed", "Served model", "Capabilities", "LLM step",
                    "Parse mode and structured output", "Sample", "Sink",
                    "Data operations", "Branching", "Checkpoints and the manifest"):
        start = body.index(f"\n## {concept}\n")
        end = body.index("\n## ", start + 1)
        assert "](" in body[start:end], f"the {concept} section links nowhere onward"


# --- claims about the pieces --------------------------------------------------------


def test_the_page_names_every_source_constructor():
    from datafast import Source

    names = [n for n in dir(Source) if not n.startswith("_")]
    assert len(names) == 8, "the page says there are eight sources"
    for name in names:
        assert f"`Source.{name}`" in PAGE.read_text()


def test_the_page_names_every_sink_constructor():
    """`Sink` is itself a Step subclass, so subtract Step to get the factories."""
    from datafast import Sink
    from datafast.core.step import Step

    names = sorted(set(dir(Sink)) - set(dir(Step)) - {n for n in dir(Sink) if n.startswith("_")})
    assert len(names) == 5, f"the page says there are five sinks, found {names}"
    for name in names:
        assert f"`Sink.{name}`" in PAGE.read_text()


def test_the_page_names_every_provider_factory():
    import datafast

    factories = ["openai", "anthropic", "gemini", "mistral", "openrouter", "ollama",
                 "openai_compatible"]
    assert all(hasattr(datafast, f) for f in factories)
    assert "Seven factories" in PAGE.read_text()
    for factory in factories:
        assert f"`{factory}`" in PAGE.read_text()


def test_the_page_names_every_sampling_strategy():
    from datafast import Sample

    strategies = Sample.VALID_STRATEGIES
    assert strategies, "Sample declares no strategies — check the test, not the page"
    assert "there are nine" in PAGE.read_text() and len(strategies) == 9
    for strategy in strategies:
        assert f"`{strategy}`" in PAGE.read_text()


def test_the_page_names_every_specialized_llm_step():
    import datafast

    for step in ("Classify", "Score", "Compare", "Rewrite", "Extract"):
        assert hasattr(datafast, step)
        assert f"`{step}`" in PAGE.read_text()


def test_a_served_model_is_built_without_a_key_as_the_page_says(monkeypatch):
    """The page promises construction does not verify the key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from datafast import openai

    openai("gpt-5.4-mini", temperature=0.2)


def test_sample_cannot_start_a_pipeline():
    from datafast import Sample, Sink

    with pytest.raises(PipelineValidationError, match="must start with a source"):
        (Sample(n=1) >> Sink.list()).compile()


# --- the data operations table, reused from the reference ---------------------------


def _table_steps(doc: Path, heading: str) -> list[str]:
    """The first column of the table under *heading*, as bare step names."""
    body = doc.read_text()
    start = body.index(f"\n## {heading}\n")
    end = body.index("\n## ", start + 1)
    rows = re.findall(r"^\| `([^`]+)`", body[start:end], re.MULTILINE)
    return [row.split("(")[0] for row in rows]


def test_the_data_ops_table_covers_the_same_steps_as_the_reference():
    """The page reuses the reference's 'At a glance' table; it must not drift from it."""
    reference = _table_steps(DATA_OPS, "At a glance")
    assert reference, "no table found in data_ops.md — check the test, not the page"
    assert _table_steps(PAGE, "Data operations") == reference


def test_every_data_op_row_carries_a_concrete_example():
    """The whole point of restating the table here rather than linking to it."""
    body = PAGE.read_text()
    start = body.index("\n## Data operations\n")
    end = body.index("\n## ", start + 1)
    rows = [r for r in body[start:end].splitlines() if r.startswith("| `")]
    assert len(rows) == 8, "the reference lists eight data operations"
    for row in rows:
        example = row.split("|")[3].strip()
        assert len(example.split()) >= 5, f"no real example on the {row.split('|')[1]} row"


def test_every_data_op_the_table_names_is_importable():
    import datafast

    for step in _table_steps(PAGE, "Data operations"):
        assert hasattr(datafast, step), f"the table names {step}, which does not exist"


# --- claims proved by running the page's own blocks ---------------------------------


def test_the_pipeline_block_returns_the_documented_records():
    namespace: dict = {}
    exec(compile(_block_containing("AddUUID()"), str(PAGE), "exec"), namespace)
    records = namespace["records"]
    assert len(records) == 2, "the page's source holds two records"
    # The prose says a step that adds a column returns a dict with one more key.
    assert set(records[0]) == {"text", "length", "id"}


def test_naming_a_step_gives_it_that_name():
    namespace: dict = {}
    exec(compile(_block_containing("as_step"), str(PAGE), "exec"), namespace)
    assert namespace["step"].name == "normalize"


def test_the_seed_block_expands_to_the_documented_four_records():
    namespace: dict = {}
    exec(compile(_block_containing("robotics"), str(PAGE), "exec"), namespace)
    # "two topics and two audiences produce four records"
    assert len(list(namespace["seed"].process(iter([])))) == 4


def test_the_sample_block_keeps_the_documented_number_of_records():
    """Nine seeded records thinned to four, repeatably."""
    namespace: dict = {}
    exec(compile(_block_containing("Sample(n=4"), str(PAGE), "exec"), namespace)
    assert len(namespace["pipeline"].run()) == 4


def test_compile_raises_what_the_page_says_it_raises():
    """The page names PipelineValidationError and the rule it enforces."""
    from datafast import Map, Sink, Source

    pipeline = Source.list([{"a": 1}]) >> Sink.list() >> Map(lambda r: r)
    with pytest.raises(PipelineValidationError, match="comes after the sink"):
        pipeline.compile()


def test_every_page_the_concepts_page_links_to_exists():
    """A link to an unwritten page fails `zensical build --strict`, so catch it here."""
    links = re.findall(r"\]\((?!https?:)([^)#]+\.md)", PAGE.read_text())
    assert links, "the page should link onward"
    missing = sorted(link for link in links if not (PAGE.parent / link).exists())
    assert not missing, f"concepts.md links to pages that do not exist: {missing}"


# --- the page covers every glossary term ---------------------------------------------


def _glossary_terms() -> list[str]:
    terms = re.findall(r"^\*\*([^*]+)\*\* — ", GLOSSARY.read_text(), re.MULTILINE)
    assert len(terms) >= 25, "no glossary terms parsed — check the test, not the page"
    return terms


@pytest.mark.parametrize("term", _glossary_terms())
def test_every_glossary_term_is_introduced_in_bold(term):
    """The page absorbs the glossary: each term is introduced where it is explained.

    A plural or a capital first letter still counts, so `**Records**` introduces Record.
    """
    prose = re.sub(r"```.*?```", "", PAGE.read_text(), flags=re.DOTALL)
    bolded = {b.lower() for b in re.findall(r"\*\*([^*\n]+)\*\*", prose)}
    assert bolded, "nothing on the page is bold — check the test, not the page"
    assert term.lower() in bolded or f"{term.lower()}s" in bolded, f"{term} not introduced"


def _avoided_words() -> set[str]:
    words = set()
    for line in re.findall(r"_avoid:_ (.+)$", GLOSSARY.read_text(), re.MULTILINE):
        words.update(w.strip() for w in line.split(","))
    assert words, "no avoid lines parsed — check the test, not the page"
    return words


def test_the_page_uses_no_word_the_glossary_asks_us_to_avoid():
    """Whole words only. The allowed ones are the names of other glossary terms, or
    ordinary English that the page uses in a sense the glossary does not cover."""
    allowed = {
        # Other glossary terms, avoided only as names for a different concept, and
        # "template", which is half of the term "prompt template".
        "provider", "served model", "target", "template",
        # Records "flow" through steps; an API key lives in an environment "variable".
        "flow", "variable",
        # `Sample` is a step's name; "key" is a dict key and a dataset join key; a
        # "schema" is a JSON schema; "build" is building records, not compiling.
        "sample", "key", "build",
        # A table row, a code block, and the "For example" column of a reused table.
        "row", "block", "example",
    }
    text = re.sub(r"`[^`]*`", "", PAGE.read_text()).lower()
    used = sorted(
        w for w in _avoided_words() - allowed
        if re.search(rf"\b{re.escape(w.lower())}s?\b", text)
    )
    assert not used, f"the page uses words the glossary avoids: {used}"


# --- claims about the new sections ---------------------------------------------------


def test_the_served_model_and_capability_blocks_print_what_the_page_says():
    import datafast

    vllm = datafast.openai_compatible(
        "qwen3", provider_id="vllm", api_base_url="http://localhost:8000/v1"
    )
    assert vllm.provider_id == "vllm"
    assert vllm.config.litellm_route == "openai", "the page says the transport is OpenAI's"

    gpt = datafast.openai("gpt-5.5")
    assert gpt.capabilities.structured_output.value == "json_schema"
    assert gpt.endpoint_mode.value == "responses"


def test_only_openai_mistral_and_ollama_match_an_uncatalogued_name():
    """The page says Anthropic, Gemini and OpenRouter fall straight to a default."""
    from datafast.llm import capabilities as caps

    for provider in ("anthropic", "gemini", "openrouter"):
        resolved = caps.resolve_capabilities(provider, "made-up-reasoning-model")
        assert resolved is caps._PROVIDER_DEFAULTS[provider]
    resolved = caps.resolve_capabilities("openai", "gpt-5.9-made-up")
    assert resolved is caps.OPENAI_RESPONSES, "OpenAI matches reasoning models by name"


def test_the_capability_resolution_order_is_the_documented_one():
    from datafast.llm import capabilities as caps

    explicit = caps.OPENAI_CHAT
    assert caps.resolve_capabilities("anthropic", "x", explicit=explicit) is explicit
    assert caps.resolve_capabilities(
        "nobody", "x", api_base_url="http://localhost:1"
    ) is caps.OPENAI_COMPATIBLE_CHAT
    unknown = caps.resolve_capabilities("nobody", "x")
    assert unknown.structured_output.value == "prompted_json"
    assert {m.value for m in unknown.modalities} == {"text"}
    assert {m.value for m in unknown.endpoint_modes} == {"chat"}


def test_the_parse_mode_table_names_every_mode():
    from datafast import LLMStep

    assert LLMStep.VALID_PARSE_MODES, "LLMStep declares no parse modes"
    for mode in LLMStep.VALID_PARSE_MODES:
        assert f'| `"{mode}"` |' in PAGE.read_text()


def test_the_page_names_every_structured_output_level():
    from datafast.llm.types import StructuredOutputMode

    levels = [m.value for m in StructuredOutputMode]
    assert len(levels) == 4, "the page says there are four levels"
    for level in levels:
        assert f"`{level}`" in PAGE.read_text()


def test_the_branch_block_prints_the_documented_record():
    namespace: dict = {}
    exec(compile(_block_containing("JoinBranches()"), str(PAGE), "exec"), namespace)
    first = namespace["records"][0]
    assert first == {"text": "hello", "shout_upper": "HELLO", "shout_length": "5"}
    assert not [c for c in first if c.startswith("_branch")], "the join removes the tags"


def test_a_branch_inside_a_branch_path_is_rejected():
    from datafast import Branch, JoinBranches, Map, Source

    inner = Branch(a=Map(lambda r: r), b=Map(lambda r: r)) >> JoinBranches()
    pipeline = Source.list([{"x": 1}]) >> Branch(c=inner, d=Map(lambda r: r)) >> JoinBranches()
    with pytest.raises(PipelineValidationError, match="nesting a Branch"):
        pipeline.compile()


def test_the_execution_strategy_table_matches_the_runner():
    import inspect

    from datafast import Pipeline
    from datafast.core.config import LLMExecutionStrategy

    strategies = [s.value for s in LLMExecutionStrategy]
    assert len(strategies) == 3
    for strategy in strategies:
        assert f"| `{strategy}`" in PAGE.read_text()
    assert inspect.signature(Pipeline.run).parameters["llm_strategy"].default == "by_model"


def test_checkpoint_every_defaults_to_the_documented_100():
    from datafast.core.config import RunConfig

    assert RunConfig().checkpoint_every == 100
    assert "100 by default" in PAGE.read_text()


def test_the_fingerprint_sees_structure_and_not_behaviour(tmp_path):
    """Both halves of the page's checkpoint claim, proved with real resumes."""
    from datafast import Filter, Map, Sink, Source
    from datafast.core.checkpoint import PipelineChangedError

    records = [{"x": 1}, {"x": 2}]
    checkpoint = str(tmp_path / "ckpt")
    (Source.list(records) >> Map(lambda r: {**r, "y": 1}) >> Sink.list()).run(
        checkpoint_dir=checkpoint
    )

    # Same structure, different function: resume trusts the checkpoint.
    resumed = (Source.list(records) >> Map(lambda r: {**r, "y": 99}) >> Sink.list()).run(
        checkpoint_dir=checkpoint, resume=True
    )
    assert [r["y"] for r in resumed] == [1, 1], "resume should not notice the new lambda"

    # A different step class changes the fingerprint.
    changed = Source.list(records) >> Filter(fn=lambda r: True) >> Sink.list()
    with pytest.raises(PipelineChangedError):
        changed.run(checkpoint_dir=checkpoint, resume=True)


def test_the_variation_column_comes_from_rewrite_only():
    """The metadata table credits `_variation` to `Rewrite` alone."""
    writers = [
        p.name for p in (ROOT / "datafast").rglob("*.py")
        if '"_variation"' in p.read_text()
    ]
    assert writers == ["llm_transform.py"], f"_variation is written in {writers}"
    assert "class Rewrite(" in (ROOT / "datafast/transforms/llm_transform.py").read_text()


class _InterruptingModel:
    """A stub served model that stops the run, like Ctrl-C, after `die_after` calls."""

    provider_id = "openai"
    model_id = "stub"

    def __init__(self, die_after: int | None = None):
        self.calls = 0
        self.die_after = die_after

    def generate(self, prompt=None, messages=None, metadata=None, response_format=None):
        if self.die_after is not None and self.calls == self.die_after:
            raise KeyboardInterrupt
        self.calls += 1
        return "ok"


def _resume_after_crash(tmp_path, die_after: int) -> int:
    """Calls a resumed run makes after a 200-call LLM step died at `die_after`."""
    from datafast import LLMStep, Sink, Source

    def pipeline(model):
        return (
            Source.list([{"t": str(i)} for i in range(200)])
            >> LLMStep(prompt="{t}", input_columns=["t"], output_column="o", model=model)
            >> Sink.list()
        )

    checkpoint = str(tmp_path / f"crash_{die_after}")
    with pytest.raises(KeyboardInterrupt):
        pipeline(_InterruptingModel(die_after)).run(checkpoint_dir=checkpoint)
    resumed = _InterruptingModel()
    records = pipeline(resumed).run(checkpoint_dir=checkpoint, resume=True)
    assert len({r["t"] for r in records}) == len(records) == 200, "no record lost or doubled"
    return resumed.calls


def test_resume_picks_up_after_the_first_progress_file(tmp_path):
    """Past the first `checkpoint_every` (100), resume makes only the missing calls."""
    assert _resume_after_crash(tmp_path, die_after=160) == 40


def test_resume_before_the_first_progress_file_reruns_the_whole_step(tmp_path):
    """The page warns about this. If it starts failing, resume got better: update the
    Checkpoints section and the troubleshooting guide, then flip this assertion."""
    assert _resume_after_crash(tmp_path, die_after=60) == 200
