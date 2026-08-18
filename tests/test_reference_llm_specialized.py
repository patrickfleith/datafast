"""The specialized LLM steps reference, pinned against the code it documents.

Same contract as `test_reference_sources_and_seed.py`:

1. Every parameter the code takes is named on the page (code → docs). A parameter that
   exists and is undocumented cannot be discovered.
2. Every mode and preset the code implements is named on the page — read out of the
   source, never hand-listed here, so a new one fails until it is documented.
3. Every self-contained example executes, with a stub in place of the served model so
   no provider is ever called.
4. The behavioural claims a reader would be burned by are asserted against real calls.
"""

import inspect
import re
from pathlib import Path

import pytest

import datafast
from datafast import Classify, Compare, Extract, Rewrite, Score
from datafast.transforms.llm_extract import _PREDEFINED_EXTRACTORS
from datafast.transforms.llm_transform import VALID_MODES

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "reference" / "llm_specialized.md"
REWRITE_SOURCE = (ROOT / "datafast" / "transforms" / "llm_transform.py").read_text()
EVAL_SOURCE = (ROOT / "datafast" / "transforms" / "llm_eval.py").read_text()

STEPS = [Classify, Score, Compare, Rewrite, Extract]

# mode → companion argument, read out of Rewrite's own validation block.
REQUIRED_COMPANIONS = dict(
    re.findall(r'if mode == "(\w+)" and not (\w+):', REWRITE_SOURCE)
)
# The values Compare accepts for output_mode, read out of its validation block.
OUTPUT_MODES = re.findall(
    r'["\'](\w+)["\']',
    " ".join(re.findall(r"if output_mode not in \(([^)]+)\)", EVAL_SOURCE)),
)


class Stub:
    """Stands in for a served model. Returns a canned reply; never touches a network."""

    provider_id = "openai"
    model_id = "stub-model"

    def __init__(self, reply: str = "rewritten text") -> None:
        self.reply = reply

    def generate(self, prompt=None, messages=None, metadata=None, response_format=None):
        return self.reply


def _page() -> str:
    return PAGE.read_text()


def _code_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", _page(), re.DOTALL)


def _params(step) -> dict[str, inspect.Parameter]:
    return {
        name: p
        for name, p in inspect.signature(step).parameters.items()
        if p.kind is not p.VAR_KEYWORD
    }


# --------------------------------------------------------------------------
# code → docs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("step", STEPS, ids=lambda s: s.__name__)
def test_every_parameter_is_named_on_the_page(step):
    """A parameter that exists and is undocumented cannot be discovered."""
    parameters = _params(step)
    assert parameters, f"{step.__name__} takes nothing — check the test, not the page"
    missing = [p for p in parameters if f"`{p}`" not in _page()]
    assert not missing, f"{step.__name__} takes {missing}, undocumented on the page"


@pytest.mark.parametrize("step", STEPS, ids=lambda s: s.__name__)
def test_every_step_is_named_on_the_page(step):
    assert f"`{step.__name__}`" in _page()


@pytest.mark.parametrize("mode", sorted(VALID_MODES))
def test_every_rewrite_mode_is_documented(mode):
    """Modes come from Rewrite's own VALID_MODES, so a new one fails until documented."""
    assert VALID_MODES, "no modes found — check the test, not the page"
    assert f'`"{mode}"`' in _page(), f"Rewrite mode {mode!r} is undocumented"


@pytest.mark.parametrize("preset", sorted(_PREDEFINED_EXTRACTORS))
def test_every_extract_preset_and_its_values_are_documented(preset):
    """Presets and the values they produce come from the extractor table itself."""
    assert _PREDEFINED_EXTRACTORS, "no presets found — check the test, not the page"
    assert f'`"{preset}"`' in _page(), f"extractor {preset!r} is undocumented"
    for value in _PREDEFINED_EXTRACTORS[preset]["fields"]:
        assert f"`{value}`" in _page(), f"{preset} produces {value!r}, undocumented"


@pytest.mark.parametrize("mode", sorted(OUTPUT_MODES))
def test_every_compare_output_mode_is_documented(mode):
    assert OUTPUT_MODES, "no output modes found — check the test, not the page"
    assert f'`"{mode}"`' in _page(), f"Compare output_mode {mode!r} is undocumented"


def _section(heading: str) -> str:
    """The page text under one heading, up to the next heading of any level."""
    body = _page().split(f"\n{heading}\n", 1)
    assert len(body) == 2, f"the page has no heading {heading!r}"
    return re.split(r"\n#+ ", body[1], maxsplit=1)[0]


def _documented_defaults(section: str) -> dict[str, str]:
    """Parameter → default, read out of the `Parameter | Type | Default` tables."""
    defaults = {}
    for line in section.splitlines():
        # Type cells escape their pipes as `\|`; hide those before splitting.
        cells = [c.strip() for c in line.replace(r"\|", "\0").split("|")]
        if len(cells) >= 5 and re.fullmatch(r"`\w+`", cells[1]) and (
            cells[3].startswith("`") or cells[3] == "required"
        ):
            defaults[cells[1].strip("`")] = cells[3]
    return defaults


@pytest.mark.parametrize("step", STEPS, ids=lambda s: s.__name__)
def test_the_documented_defaults_are_the_real_defaults(step):
    """Every default a step's own table prints must match its signature."""
    documented = _documented_defaults(_section("## Shared parameters"))
    documented |= _documented_defaults(_section(f"### `{step.__name__}`"))
    checked = 0
    for name, parameter in _params(step).items():
        if parameter.default is inspect.Parameter.empty:
            continue
        assert name in documented, f"{step.__name__}.{name} has no default in its table"
        assert documented[name] == "`%s`" % repr(parameter.default).replace("'", '"'), (
            f"{step.__name__}.{name} defaults to {parameter.default!r}, "
            f"the page says {documented[name]}"
        )
        checked += 1
    assert checked, f"no defaults checked for {step.__name__}"


# --------------------------------------------------------------------------
# the examples on the page
# --------------------------------------------------------------------------


@pytest.mark.parametrize("block", _code_blocks(), ids=lambda b: b.split("\n")[0][:40])
def test_every_example_executes(block, monkeypatch):
    """The blocks import `openai` from datafast, so the factory is patched on the
    module rather than injected into the exec namespace — an injected name would be
    overwritten by the block's own import and the test would call the real API."""
    monkeypatch.setattr(datafast, "openai", lambda *a, **k: Stub())
    exec(compile(block, str(PAGE), "exec"), {})


def test_there_is_an_example_for_every_step():
    """Guards the block list above from silently becoming empty."""
    blocks = "\n".join(_code_blocks())
    assert len(_code_blocks()) >= len(STEPS)
    for step in STEPS:
        assert step.__name__ in blocks, f"no example uses {step.__name__}"


# --------------------------------------------------------------------------
# behaviour the page claims
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode,companion", sorted(REQUIRED_COMPANIONS.items()))
def test_a_mode_without_its_companion_fails_at_construction(mode, companion):
    """The page says this is checked when you build the step, not mid-run."""
    assert len(REQUIRED_COMPANIONS) == 3, f"expected 3, got {REQUIRED_COMPANIONS}"
    with pytest.raises(ValueError, match=companion):
        Rewrite(input_column="text", llm=Stub(), mode=mode)

    assert f'`"{mode}"`' in _page() and f"`{companion}`" in _page()
    Rewrite(input_column="text", llm=Stub(), mode=mode, **{companion: "something"})


def test_an_unknown_rewrite_mode_is_rejected():
    with pytest.raises(ValueError, match="Invalid mode"):
        Rewrite(input_column="text", llm=Stub(), mode="shout")


def test_rewrite_takes_no_fn_and_no_prompt():
    """The page says Rewrite always calls an LLM and has no `prompt`."""
    assert "fn" not in _params(Rewrite)
    assert "prompt" not in _params(Rewrite)
    for step in (Classify, Score, Compare, Extract):
        assert "fn" in _params(step) and "prompt" in _params(step)


def test_rewrite_writes_the_documented_default_column():
    (record,) = Rewrite(input_column="text", llm=Stub("new")).process(
        iter([{"text": "old"}])
    )
    assert record["text_rewritten"] == "new"
    assert record["_model"] == "stub-model", "the page promises a _model column"


def test_num_variations_multiplies_records_and_adds_variation_only_above_one():
    many = list(
        Rewrite(input_column="text", llm=Stub("new"), num_variations=3).process(
            iter([{"text": "old"}])
        )
    )
    assert len(many) == 3
    assert [r["_variation"] for r in many] == [0, 1, 2]

    (one,) = Rewrite(input_column="text", llm=Stub("new")).process(iter([{"text": "x"}]))
    assert "_variation" not in one, "the page says it appears only above 1"


def test_several_served_models_multiply_records():
    a, b = Stub("x"), Stub("y")
    b.model_id = "other-model"
    out = list(
        Rewrite(input_column="text", llm=[a, b]).process(iter([{"text": "old"}]))
    )
    assert [r["_model"] for r in out] == ["stub-model", "other-model"]


def test_classify_keeps_an_unknown_single_label_but_drops_unknown_multi_labels():
    """The page's sharpest Classify trap: the two modes disagree."""
    (single,) = Classify(
        labels=["a", "b"], input_columns=["t"], llm=Stub('{"label": "zzz"}')
    ).process(iter([{"t": "x"}]))
    assert single["label"] == "zzz", "the page says an unknown label is written anyway"

    (multi,) = Classify(
        labels=["a", "b"],
        input_columns=["t"],
        multi_label=True,
        llm=Stub('{"labels": ["zzz"]}'),
    ).process(iter([{"t": "x"}]))
    assert multi["label"] == [], "the page says unknown labels are dropped"


def test_classify_explanation_and_confidence_columns_are_named_as_documented():
    (record,) = Classify(
        labels=["a"],
        input_columns=["t"],
        output_column="sentiment",
        include_explanation=True,
        include_confidence=True,
        llm=Stub('{"label": "a", "explanation": "because", "confidence": 0.9}'),
    ).process(iter([{"t": "x"}]))
    assert record["sentiment_explanation"] == "because"
    assert record["sentiment_confidence"] == 0.9


def test_labels_description_may_cover_only_some_labels():
    step = Classify(
        labels=["a", "b"],
        input_columns=["t"],
        labels_description={"a": "the first one"},
        llm=Stub("{}"),
    )
    prompt = step._build_messages({"t": "x"})[-1]["content"]
    assert "- a: the first one" in prompt and "\n- b\n" in prompt


def test_score_clamps_into_range_in_both_modes():
    """The page warns that an out-of-range answer looks like a top score."""
    (llm_record,) = Score(
        input_columns=["t"], score_range=(1, 5), llm=Stub('{"score": 99}')
    ).process(iter([{"t": "x"}]))
    assert llm_record["score"] == 5

    (fn_record,) = Score(
        input_columns=["t"], score_range=(1, 5), fn=lambda r: 99
    ).process(iter([{"t": "x"}]))
    assert fn_record["score"] == 5


def test_the_rubric_is_shown_in_ascending_order():
    step = Score(
        input_columns=["t"],
        criteria="accuracy",
        rubric={5: "Excellent", 1: "Wrong"},
        llm=Stub("{}"),
    )
    prompt = step._build_messages({"t": "x"})[-1]["content"]
    assert prompt.index("1: Wrong") < prompt.index("5: Excellent")
    assert "Criteria: accuracy" in prompt


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("winner", {"comparison"}),
        ("scores", {"comparison", "comparison_score_a", "comparison_score_b"}),
        (
            "detailed",
            {
                "comparison",
                "comparison_score_a",
                "comparison_score_b",
                "comparison_reasoning",
            },
        ),
    ],
)
def test_compare_writes_exactly_the_documented_columns(mode, expected):
    reply = '{"winner": "a", "score_a": 8, "score_b": 3, "reasoning": "clearer"}'
    (record,) = Compare(
        column_a="a", column_b="b", criteria="clarity", output_mode=mode, llm=Stub(reply)
    ).process(iter([{"a": "1", "b": "2"}]))
    assert set(record) - {"a", "b", "_model"} == expected


def test_an_unknown_compare_output_mode_is_rejected():
    with pytest.raises(ValueError, match="output_mode"):
        Compare(column_a="a", column_b="b", criteria="c", output_mode="best", fn=len)


def test_extract_flatten_makes_columns_and_ignores_output_column():
    reply = '{"name": "Widget", "price": "9"}'
    fields = {"name": "the name", "price": "the price"}

    (flat,) = Extract(
        input_column="t",
        fields=fields,
        flatten=True,
        output_column="ignored",
        llm=Stub(reply),
    ).process(iter([{"t": "x"}]))
    assert flat["name"] == "Widget" and flat["price"] == "9"
    assert "ignored" not in flat, "the page says output_column is ignored when flattened"

    (nested,) = Extract(input_column="t", fields=fields, llm=Stub(reply)).process(
        iter([{"t": "x"}])
    )
    assert nested["extracted"] == {"name": "Widget", "price": "9"}


def test_a_missing_extract_value_becomes_an_empty_string_not_a_list():
    (record,) = Extract(
        input_column="t", extractor="entities", llm=Stub('{"persons": ["Ada"]}')
    ).process(iter([{"t": "x"}]))
    assert record["extracted"]["organizations"] == "", "the page says empty string, not []"


def test_a_preset_produces_exactly_the_documented_values():
    for preset, spec in _PREDEFINED_EXTRACTORS.items():
        reply = "{}"
        (record,) = Extract(
            input_column="t", extractor=preset, llm=Stub(reply)
        ).process(iter([{"t": "x"}]))
        assert set(record["extracted"]) == set(spec["fields"])


def test_extract_needs_exactly_one_of_fields_extractor_or_fn():
    with pytest.raises(ValueError, match="requires one of"):
        Extract(input_column="t", llm=Stub("{}"))
    with pytest.raises(ValueError, match="only one of"):
        Extract(input_column="t", fields={"a": "b"}, extractor="topics", llm=Stub("{}"))
    with pytest.raises(ValueError, match="Invalid extractor"):
        Extract(input_column="t", extractor="vibes", llm=Stub("{}"))


@pytest.mark.parametrize(
    "build",
    [
        lambda **kw: Classify(labels=["a"], input_columns=["t"], **kw),
        lambda **kw: Score(input_columns=["t"], **kw),
        lambda **kw: Compare(column_a="a", column_b="b", criteria="c", **kw),
    ],
    ids=["Classify", "Score", "Compare"],
)
def test_llm_and_fn_are_mutually_exclusive_and_one_is_required(build):
    with pytest.raises(ValueError, match="requires either"):
        build()
    with pytest.raises(ValueError, match="not both"):
        build(llm=Stub("{}"), fn=lambda r: "a")


def test_fn_mode_ignores_forward_columns_except_in_extract():
    """The inconsistency the page warns about, asserted on all four steps."""
    record = {"t": "x", "extra": "y"}

    (classified,) = Classify(
        labels=["a"], input_columns=["t"], fn=lambda r: "a", forward_columns=["t"]
    ).process(iter([dict(record)]))
    assert "extra" in classified, "fn mode ignores forward_columns"
    assert "_model" not in classified, "fn mode adds no _model column"

    (extracted,) = Extract(
        input_column="t", fn=lambda r: {"a": 1}, flatten=True, forward_columns=["t"]
    ).process(iter([dict(record)]))
    assert "extra" not in extracted, "Extract's fn mode does honour forward_columns"


def test_classify_in_fn_mode_does_not_check_the_label_set():
    (record,) = Classify(
        labels=["a", "b"], input_columns=["t"], fn=lambda r: "zzz"
    ).process(iter([{"t": "x"}]))
    assert record["label"] == "zzz"


def test_skip_drops_the_record_and_raise_stops_the_run():
    """`on_parse_error` covers every error, not only parse errors — the page's warning."""

    class Broken(Stub):
        def generate(self, **kwargs):
            raise RuntimeError("provider is down")

    skipped = list(
        Classify(labels=["a"], input_columns=["t"], llm=Broken()).process(
            iter([{"t": "x"}])
        )
    )
    assert skipped == [], "the default silently drops the record"

    with pytest.raises(RuntimeError):
        list(
            Classify(
                labels=["a"], input_columns=["t"], llm=Broken(), on_parse_error="raise"
            ).process(iter([{"t": "x"}]))
        )


def test_every_page_linked_to_exists():
    links = re.findall(r"\]\((?!https?:)([^)#]+\.md)", _page())
    assert links, "no links found — check the test, not the page"
    missing = sorted(link for link in links if not (PAGE.parent / link).resolve().exists())
    assert not missing, f"links to pages that do not exist: {missing}"
