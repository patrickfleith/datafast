"""The Sample reference, pinned against the code it documents.

Same shape as ``test_reference_sources_and_seed.py``:

1. Every parameter and every strategy the code has is named on the page (code → docs).
   Something that exists and is undocumented cannot be discovered.
2. Every self-contained example executes.
3. The behavioural claims are asserted against real calls, never paraphrased from the
   source — which requirement raises, which flag is ignored, what ``seed`` buys you.

No test here makes an LLM call. The LLMStep example is constructed only.
"""

import inspect
import re
from pathlib import Path

import pytest

from datafast import Sample
from datafast.transforms.sample import Sample as SampleClass

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "reference" / "sample.md"

# The strategies that the code refuses to build without a `by`, per the source.
NEEDS_BY = ["top", "bottom", "weighted", "stratified", "gaussian"]


def _page() -> str:
    return PAGE.read_text()


def _code_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", _page(), re.DOTALL)


def _records(n: int = 10) -> list[dict]:
    return [{"i": i} for i in range(n)]


def test_every_parameter_is_named_on_the_page():
    """A parameter the page forgets is one a reader never finds."""
    parameters = [
        p.name for p in inspect.signature(Sample).parameters.values()
        if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
    ]
    assert len(parameters) > 5, "introspection found almost nothing — check the test"
    missing = [p for p in parameters if f"`{p}`" not in _page()]
    assert not missing, f"Sample takes {missing}, undocumented on the page"


def test_every_public_method_is_named_on_the_page():
    """`process` and `as_step` belong to every step; the runner calls them, not the reader."""
    inherited = {"process", "as_step"}
    methods = [
        name for name, value in inspect.getmembers(SampleClass, callable)
        if not name.startswith("_") and name not in inherited
    ]
    assert methods, "introspection found no methods — check the test"
    missing = [m for m in methods if f"`.{m}()`" not in _page() and f".{m}()" not in _page()]
    assert not missing, f"Sample has {missing}, undocumented on the page"


def test_pick_accepts_the_n_override_the_page_claims():
    assert "n" in inspect.signature(SampleClass.pick).parameters
    assert "`.pick(n)`" in _page()


def test_every_strategy_the_code_implements_is_on_the_page():
    """The list comes from the code, so a new strategy fails until it is documented."""
    strategies = SampleClass.VALID_STRATEGIES
    assert len(strategies) == 9, f"the page tables nine strategies, code has {len(strategies)}"
    missing = [s for s in strategies if f"`{s}`" not in _page()]
    assert not missing, f"undocumented strategies: {sorted(missing)}"


@pytest.mark.parametrize("block", _code_blocks(), ids=lambda b: b.split("\n")[-2][:40])
def test_every_example_executes(block):
    """Includes the LLMStep example: constructed, never run. No provider is contacted."""
    exec(compile(block, str(PAGE), "exec"), {})


def test_the_page_has_examples_to_execute():
    assert len(_code_blocks()) >= 5, "no examples found — the exec test would pass vacuously"


# --- what each strategy requires -------------------------------------------------


@pytest.mark.parametrize("strategy", NEEDS_BY)
def test_the_five_strategies_the_page_lists_require_by(strategy):
    extra = {"center": 0.0, "std": 1.0} if strategy == "gaussian" else {}
    with pytest.raises(ValueError, match="'by' parameter required"):
        Sample(n=1, strategy=strategy, **extra)


@pytest.mark.parametrize(
    "strategy", sorted(SampleClass.VALID_STRATEGIES - set(NEEDS_BY))
)
def test_no_other_strategy_requires_by(strategy):
    """Proves the page's list of five is exactly five, not a guess."""
    extra = {"step": 2} if strategy == "systematic" else {}
    Sample(n=1, strategy=strategy, **extra)


def test_systematic_requires_step():
    with pytest.raises(ValueError, match="'step' parameter required"):
        Sample(n=1, strategy="systematic")


@pytest.mark.parametrize("given", [{}, {"center": 0.0}, {"std": 1.0}])
def test_gaussian_requires_both_center_and_std(given):
    with pytest.raises(ValueError, match="'center' and 'std' parameters required"):
        Sample(n=1, strategy="gaussian", by="i", **given)


def test_requirements_are_checked_at_construction_not_at_run_time():
    """The page promises the error arrives when you write the step."""
    with pytest.raises(ValueError):
        Sample(n=1, strategy="top")  # never processes a record


def test_an_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="Invalid strategy"):
        Sample(n=1, strategy="random")


# --- n, frac, replace, seed ------------------------------------------------------


def test_n_and_frac_together_are_rejected():
    with pytest.raises(ValueError, match="Cannot specify both"):
        Sample(n=1, frac=0.5)


def test_neither_n_nor_frac_keeps_everything():
    records = _records(10)
    assert len(list(Sample().process(iter(records)))) == 10
    kept = list(Sample(strategy="top", by="i").process(iter(records)))
    assert len(kept) == 10, "the page says top then only sorts"


def test_frac_rounds_down_but_never_below_one():
    records = _records(10)
    assert len(list(Sample(frac=0.25).process(iter(records)))) == 2
    assert len(list(Sample(frac=0.01).process(iter(records)))) == 1


def test_asking_for_more_than_exists_is_capped_unless_replace():
    records = _records(10)
    assert len(list(Sample(n=99).process(iter(records)))) == 10
    assert len(list(Sample(n=15, replace=True).process(iter(records)))) == 15


def test_seed_makes_a_sample_reproducible():
    records = _records(100)
    first = list(Sample(n=5, seed=42).process(iter(records)))
    again = list(Sample(n=5, seed=42).process(iter(records)))
    assert first == again
    others = [list(Sample(n=5, seed=s).process(iter(records))) for s in range(20)]
    assert any(o != first for o in others), "seed appears to change nothing — check the test"


def test_seed_is_reproducible_for_every_random_strategy():
    records = [{"i": i, "g": "a" if i < 7 else "b"} for i in range(20)]
    cases = [
        {"strategy": "uniform"},
        {"strategy": "weighted", "by": "i"},
        {"strategy": "stratified", "by": "g"},
        {"strategy": "gaussian", "by": "i", "center": 10.0, "std": 3.0},
    ]
    for case in cases:
        a = list(Sample(n=4, seed=7, **case).process(iter(records)))
        b = list(Sample(n=4, seed=7, **case).process(iter(records)))
        assert a == b, f"{case['strategy']} is not reproducible under a seed"


# --- per-strategy behaviour ------------------------------------------------------


def test_systematic_takes_every_step_th_record():
    picked = [r["i"] for r in Sample(strategy="systematic", step=3).process(iter(_records(10)))]
    assert picked == [0, 3, 6, 9], "the page tables 0, 3, 6, 9"


def test_first_and_last_take_from_the_ends_in_order():
    assert [r["i"] for r in Sample(n=3, strategy="first").process(iter(_records(10)))] == [0, 1, 2]
    assert [r["i"] for r in Sample(n=3, strategy="last").process(iter(_records(10)))] == [7, 8, 9]


def test_top_takes_the_highest_and_bottom_the_lowest():
    records = _records(10)
    assert [r["i"] for r in Sample(n=3, strategy="top", by="i").process(iter(records))] == [9, 8, 7]
    assert [r["i"] for r in Sample(n=3, strategy="bottom", by="i").process(iter(records))] == [0, 1, 2]


def test_ascending_flips_top_and_is_ignored_by_bottom():
    """Both halves of the page's claim, which is easy to get backwards."""
    records = _records(10)
    flipped = [r["i"] for r in Sample(n=3, strategy="top", by="i", ascending=True).process(iter(records))]
    assert flipped == [0, 1, 2], "ascending=True should make top behave like bottom"
    for ascending in (True, False):
        kept = [r["i"] for r in Sample(n=3, strategy="bottom", by="i", ascending=ascending).process(iter(records))]
        assert kept == [0, 1, 2], "bottom must ignore ascending"


def test_by_may_be_a_callable():
    records = [{"text": "short"}, {"text": "a much longer piece of text"}]
    (kept,) = list(Sample(n=1, strategy="top", by=lambda r: len(r["text"])).process(iter(records)))
    assert kept["text"] == "a much longer piece of text"


def test_a_weight_list_of_the_wrong_length_is_rejected_when_the_step_runs():
    step = Sample(n=1, strategy="weighted", by=[1.0, 2.0])
    with pytest.raises(ValueError, match="must match"):
        list(step.process(iter(_records(3))))


def test_weighted_and_gaussian_fall_back_to_uniform_on_all_zero_weights():
    records = [{"i": 0} for _ in range(5)]
    assert len(list(Sample(n=3, strategy="weighted", by="i", seed=1).process(iter(records)))) == 3
    far = [{"i": None} for _ in range(5)]
    assert len(list(Sample(n=3, strategy="gaussian", by="i", center=0.0, std=1.0, seed=1).process(iter(far)))) == 3


def test_gaussian_favours_values_near_center():
    records = [{"i": i} for i in range(100)]
    kept = [r["i"] for r in Sample(n=5, strategy="gaussian", by="i", center=50.0, std=3.0, seed=42).process(iter(records))]
    assert all(abs(i - 50) < 15 for i in kept), f"gaussian strayed far from center: {kept}"


def test_stratified_keeps_each_group_share():
    records = [{"lang": "en"}] * 80 + [{"lang": "fr"}] * 20
    kept = list(Sample(n=10, strategy="stratified", by="lang", seed=42).process(iter(records)))
    french = sum(1 for r in kept if r["lang"] == "fr")
    assert 1 <= french <= 4, f"expected roughly 20% french, got {french} of {len(kept)}"


def test_stratified_gives_every_group_at_least_one_record():
    """The page's warning: rare groups are over-represented in a small sample."""
    records = [{"lang": "en"}] * 99 + [{"lang": "fr"}]
    kept = list(Sample(n=3, strategy="stratified", by="lang", seed=0).process(iter(records)))
    assert sum(1 for r in kept if r["lang"] == "fr") == 1


# --- the step / value duality ----------------------------------------------------


def test_pick_on_a_step_without_items_is_rejected():
    for call in (lambda: Sample(n=2).pick(), lambda: Sample(n=2).sample()):
        with pytest.raises(ValueError, match="without items"):
            call()


def test_sample_is_pick_with_no_arguments():
    assert Sample(list("abcd"), n=2, seed=42).sample() == Sample(list("abcd"), n=2, seed=42).pick()


def test_pick_overrides_n():
    assert len(Sample(list("abcde"), n=2, seed=1).pick(4)) == 4


def test_repeated_picks_on_one_object_differ_but_the_sequence_repeats():
    """The page's subtlest claim: seed repeats the stream, not the individual call."""
    def stream():
        s = Sample(list("abcdefgh"), n=2, seed=42)
        return [s.pick() for _ in range(5)]

    picks = stream()
    assert any(p != picks[0] for p in picks[1:]), "picks never differ — check the test"
    assert stream() == picks, "the sequence of picks is not reproducible under a seed"


def test_an_llm_step_re_picks_per_record_from_a_sample_value():
    """Construction and prompt normalization only — no provider is contacted."""
    from datafast import LLMStep, ollama

    prompts = [f"Prompt {i} about {{topic}}" for i in range(8)]
    step = LLMStep(
        prompt=Sample(prompts, n=1, seed=42),
        input_columns=["topic"],
        model=ollama("gemma3:4b"),
        output_column="explanation",
    )
    per_record = [step._normalize_prompts({"topic": "gravity"}) for _ in range(6)]
    assert all(len(p) == 1 for p in per_record)
    assert any(p != per_record[0] for p in per_record[1:]), "the page says each record re-picks"


def test_pick_freezes_the_choice_for_every_record():
    """The other half of the duality: a plain list decided once."""
    from datafast import LLMStep, ollama

    prompts = [f"Prompt {i} about {{topic}}" for i in range(8)]
    step = LLMStep(
        prompt=Sample(prompts, n=1, seed=42).pick(),
        input_columns=["topic"],
        model=ollama("gemma3:4b"),
        output_column="explanation",
    )
    per_record = [step._normalize_prompts({"topic": "gravity"}) for _ in range(6)]
    assert all(p == per_record[0] for p in per_record), "a picked list must not change"


def test_a_sample_with_items_supports_len_iteration_and_items():
    values = ["a", "b", "c"]
    holder = Sample(values, n=1)
    assert len(holder) == 3 and list(holder) == values and holder.items == values


# --- pipeline placement ----------------------------------------------------------


def test_sample_cannot_start_a_pipeline():
    from datafast import Sink

    with pytest.raises(Exception, match="must start with a source"):
        (Sample(n=1) >> Sink.list()).compile()


def test_a_string_by_is_checked_by_compile():
    """The page claims compile() refuses a `by` column that will not exist."""
    from datafast import Source

    with pytest.raises(Exception, match="score"):
        (Source.list([{"text": "a"}]) >> Sample(n=1, strategy="top", by="score")).compile()
    (Source.list([{"score": 1}]) >> Sample(n=1, strategy="top", by="score")).compile()


def test_empty_input_yields_nothing():
    assert list(Sample(n=5).process(iter([]))) == []


def test_every_page_linked_to_exists():
    links = re.findall(r"\]\((?!https?:)([^)#]+\.md)", _page())
    assert links, "no relative links found — the test would pass vacuously"
    missing = sorted(link for link in links if not (PAGE.parent / link).resolve().exists())
    assert not missing, f"links to pages that do not exist: {missing}"


def test_n_zero_keeps_nothing_whatever_the_strategy():
    """`items[-0:]` is every record; the page says n=0 keeps none."""
    records = [{"score": i} for i in range(5)]
    for strategy in ("first", "last", "uniform"):
        kept = list(Sample(n=0, strategy=strategy).process(iter(records)))
        assert kept == [], f"{strategy} kept {len(kept)} records for n=0"
    assert list(Sample(n=0, strategy="systematic", step=2).process(iter(records))) == []
    assert "`n=0` keeps nothing" in (
        Path(__file__).parent.parent / "docs" / "reference" / "sample.md"
    ).read_text()
