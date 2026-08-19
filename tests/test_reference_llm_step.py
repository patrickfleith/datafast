"""The LLMStep reference, pinned against the code it documents.

Same contract as the sources & seed reference test:

1. Every parameter the constructor takes is named on the page (code → docs). A
   parameter that exists and is undocumented cannot be discovered.
2. Every self-contained example executes.
3. The behavioural claims — the fan-out arithmetic, the metadata columns, the traps —
   are asserted against real runs.

No provider is ever reached: the served model is a stub, and the examples' own
`openai` import is patched on the `datafast` module so the block cannot re-import the
real factory.
"""

import inspect
import re
from pathlib import Path

import pytest

import datafast
from datafast import LLMStep, ServedModel, Sink, Source

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "reference" / "llm_step.md"


class StubModel:
    """Stands in for a served model: records its calls, never leaves the process."""

    provider_id = "stub"

    def __init__(self, model_id: str = "stub-model", reply: str = "an answer") -> None:
        self.model_id = model_id
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, prompt=None, messages=None, metadata=None, response_format=None):
        self.calls.append(messages)
        return self.reply


def _page() -> str:
    return PAGE.read_text()


def _code_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", _page(), re.DOTALL)


# --- code → docs guard ------------------------------------------------------


def test_every_constructor_parameter_is_named_on_the_page():
    """A parameter the page forgets is one a reader never finds."""
    parameters = [
        p.name
        for p in inspect.signature(LLMStep).parameters.values()
        if p.kind is not p.VAR_KEYWORD and p.name != "self"
    ]
    assert len(parameters) > 5, "signature looks wrong — check the test, not the page"

    missing = [p for p in parameters if f"`{p}`" not in _page()]
    assert not missing, f"LLMStep takes {missing}, undocumented on the page"


def test_the_documented_defaults_are_the_real_defaults():
    documented = {
        "output_column": "generated",
        "parse_mode": "text",
        "num_outputs": 1,
        "on_parse_error": "skip",
        "output_columns": None,
        "temperature": None,
        "max_tokens": None,
        "forward_columns": None,
        "exclude_columns": None,
        "language": None,
        "skip_if": None,
        "system_prompt": None,
    }
    actual = inspect.signature(LLMStep).parameters
    for name, default in documented.items():
        assert actual[name].default == default, f"{name} default changed"


def test_the_page_documents_every_parse_mode():
    for mode in LLMStep.VALID_PARSE_MODES:
        assert f'`"{mode}"`' in _page(), f"parse mode {mode} undocumented"


def test_only_model_and_the_two_first_parameters_are_positional():
    """The page says everything after `model` is keyword-only."""
    kinds = {
        name: p.kind for name, p in inspect.signature(LLMStep).parameters.items()
    }
    positional = [n for n, k in kinds.items() if k is inspect.Parameter.KEYWORD_ONLY]
    assert set(kinds) - set(positional) - {"self"} == {
        "prompt",
        "input_columns",
        "model",
    }


# --- the examples -----------------------------------------------------------


@pytest.mark.parametrize("block", _code_blocks(), ids=lambda b: b.split("\n")[0][:40])
def test_every_example_executes(block, tmp_path, monkeypatch):
    """Patched on the module: the block's own `from datafast import openai` would
    overwrite anything injected into the exec namespace, and hit the real API."""
    monkeypatch.setattr(datafast, "openai", lambda *args, **kwargs: StubModel())
    monkeypatch.chdir(tmp_path)
    exec(compile(block, str(PAGE), "exec"), {})


# --- the fan-out arithmetic -------------------------------------------------


def test_the_documented_multiplication_is_the_real_one():
    """2 records x 2 prompts x 2 models x 2 languages x 2 outputs = 32 calls."""
    first, second = StubModel("m1"), StubModel("m2")
    step = LLMStep(
        prompt=["A: {t}", "B: {t}"],
        input_columns=["t"],
        model=[first, second],
        output_column="out",
        language=["en", "fr"],
        num_outputs=2,
    )

    records = list(step.process(iter([{"t": "x"}, {"t": "y"}])))

    assert len(first.calls) + len(second.calls) == 32
    assert len(records) == 32, "one output record per call"


@pytest.mark.parametrize(
    "prompts,models,languages,num_outputs,expected",
    [(1, 1, None, 1, 100), (2, 1, None, 1, 200), (2, 2, None, 1, 400), (2, 2, 3, 2, 2400)],
)
def test_each_row_of_the_documented_table_holds(
    prompts, models, languages, num_outputs, expected
):
    """The table on the page counts calls for 100 records; collect_calls counts too."""
    step = LLMStep(
        prompt=[f"P{i}" for i in range(prompts)] if prompts > 1 else "P",
        input_columns=[],
        model=[StubModel(f"m{i}") for i in range(models)],
        output_column="out",
        language=[f"l{i}" for i in range(languages)] if languages else None,
        num_outputs=num_outputs,
    )

    calls, _ = step.collect_calls([{"t": i} for i in range(100)])

    assert len(calls) == expected
    assert f"| {expected} |" in _page(), "the table must carry this count"


def test_no_language_counts_as_one_not_zero():
    """The page says a step without `language` still makes one call per record."""
    model = StubModel()
    step = LLMStep(prompt="P", input_columns=[], model=model, output_column="out")
    assert len(list(step.process(iter([{}, {}])))) == 2


# --- metadata columns -------------------------------------------------------


def test_the_metadata_columns_are_the_three_the_page_names():
    model = StubModel("m1")
    step = LLMStep(
        prompt=["A", "B"],
        input_columns=[],
        model=model,
        output_column="out",
        language={"fr": "French"},
    )

    (record, _) = list(step.process(iter([{"kept": 1}])))

    assert set(record) == {"kept", "out", "_model", "_prompt_index", "_language"}
    assert record["_model"] == "m1"
    assert record["_prompt_index"] == 0
    assert record["_language"] == "fr"


def test_a_single_prompt_without_language_adds_only_model():
    """The page: with one prompt and no language, `_model` is the only added column."""
    step = LLMStep(prompt="P", input_columns=[], model=StubModel(), output_column="out")
    (record,) = list(step.process(iter([{"kept": 1}])))
    assert set(record) == {"kept", "out", "_model"}


def test_the_reserved_names_are_the_ones_compile_assumes():
    """`compile()` adds these downstream, which is why the page reserves them."""
    from datafast.core.validation import _next_columns

    step = LLMStep(prompt="P", input_columns=[], model=StubModel(), output_column="out")
    added = _next_columns(step, set()) - {"out"}
    assert added == {"_model", "_prompt_index", "_language"}
    for column in added:
        assert f"`{column}`" in _page()


# --- parse modes and output columns -----------------------------------------


def test_json_and_xml_require_output_columns():
    for mode in ("json", "xml"):
        with pytest.raises(ValueError, match="output_columns required"):
            LLMStep(prompt="P", input_columns=[], model=StubModel(), parse_mode=mode)


def test_an_unknown_parse_mode_is_rejected_at_construction():
    with pytest.raises(ValueError, match="Invalid parse_mode"):
        LLMStep(prompt="P", input_columns=[], model=StubModel(), parse_mode="yaml")


def test_json_mode_splits_the_answer_and_appends_format_instructions():
    model = StubModel(reply='```json\n{"question": "Q?", "answer": "A."}\n```')
    step = LLMStep(
        prompt="Ask about {topic}",
        input_columns=["topic"],
        model=model,
        output_columns=["question", "answer"],
        parse_mode="json",
    )

    (record,) = list(step.process(iter([{"topic": "fusion"}])))

    assert record["question"] == "Q?" and record["answer"] == "A."
    sent = model.calls[0][0]["content"]
    assert sent.startswith("Ask about fusion")
    assert "valid JSON" in sent, "the page says the mode appends its own instructions"


def test_a_missing_json_key_becomes_an_empty_string_not_a_failure():
    model = StubModel(reply='{"question": "Q?"}')
    step = LLMStep(
        prompt="P",
        input_columns=[],
        model=model,
        output_columns=["question", "answer"],
        parse_mode="json",
    )
    (record,) = list(step.process(iter([{}])))
    assert record["answer"] == ""


def test_xml_mode_matches_tags_ignoring_case():
    model = StubModel(reply="<Answer>forty two</Answer>")
    step = LLMStep(
        prompt="P",
        input_columns=[],
        model=model,
        output_columns=["answer"],
        parse_mode="xml",
    )
    (record,) = list(step.process(iter([{}])))
    assert record["answer"] == "forty two"


def test_text_mode_writes_only_the_first_of_output_columns():
    """The trap the page names: `output_column` is ignored, the rest never appear."""
    step = LLMStep(
        prompt="P",
        input_columns=[],
        model=StubModel(reply="hi"),
        output_column="ignored",
        output_columns=["first", "second"],
    )
    (record,) = list(step.process(iter([{}])))
    assert record["first"] == "hi"
    assert "second" not in record and "ignored" not in record


def test_the_default_output_column_is_the_documented_one():
    step = LLMStep(prompt="P", input_columns=[], model=StubModel(reply="hi"))
    (record,) = list(step.process(iter([{}])))
    assert record["generated"] == "hi"


# --- prompts ----------------------------------------------------------------


def test_a_prompt_that_is_an_existing_file_is_read(tmp_path):
    path = tmp_path / "summarize.txt"
    path.write_text("Summarize: {text}")
    model = StubModel()
    step = LLMStep(prompt=path, input_columns=["text"], model=model, output_column="out")

    list(step.process(iter([{"text": "a doc"}])))

    assert model.calls[0][0]["content"] == "Summarize: a doc"


def test_a_missing_prompt_path_becomes_the_prompt_text(tmp_path, monkeypatch):
    """The page's sharpest trap: no file, no error, the path is sent as the prompt."""
    monkeypatch.chdir(tmp_path)
    model = StubModel()
    step = LLMStep(
        prompt=Path("prompts/summarize.txt"),
        input_columns=[],
        model=model,
        output_column="out",
    )

    list(step.process(iter([{}])))

    assert model.calls[0][0]["content"] == "prompts/summarize.txt"


def test_a_placeholder_outside_input_columns_raises():
    step = LLMStep(
        prompt="Summarize {missing}",
        input_columns=["text"],
        model=StubModel(),
        output_column="out",
        on_parse_error="raise",
    )
    with pytest.raises(KeyError):
        list(step.process(iter([{"text": "a doc"}])))


def test_a_literal_brace_must_be_doubled():
    model = StubModel()
    step = LLMStep(
        prompt="Reply with {{}} for {t}",
        input_columns=["t"],
        model=model,
        output_column="out",
    )
    list(step.process(iter([{"t": "x"}])))
    assert model.calls[0][0]["content"] == "Reply with {} for x"


# --- languages --------------------------------------------------------------


def test_both_language_placeholders_are_filled_from_the_dict_form():
    model = StubModel()
    step = LLMStep(
        prompt="Write in {language_name} ({language})",
        input_columns=[],
        model=model,
        output_column="out",
        language={"fr": "French"},
    )

    list(step.process(iter([{}])))

    assert model.calls[0][0]["content"] == "Write in French (fr)"


def test_the_list_form_uses_the_code_as_the_name():
    model = StubModel()
    step = LLMStep(
        prompt="{language_name}/{language}",
        input_columns=[],
        model=model,
        output_column="out",
        language=["fr"],
    )
    list(step.process(iter([{}])))
    assert model.calls[0][0]["content"] == "fr/fr"


# --- columns, skipping, system prompt ---------------------------------------


def test_forward_columns_keeps_only_what_it_names():
    step = LLMStep(
        prompt="P",
        input_columns=[],
        model=StubModel(),
        output_column="out",
        forward_columns=["doc_id"],
    )
    (record,) = list(step.process(iter([{"doc_id": 1, "noise": 2}])))
    assert "noise" not in record and record["doc_id"] == 1


def test_exclude_columns_drops_only_what_it_names():
    step = LLMStep(
        prompt="P",
        input_columns=[],
        model=StubModel(),
        output_column="out",
        exclude_columns=["noise"],
    )
    (record,) = list(step.process(iter([{"doc_id": 1, "noise": 2}])))
    assert "noise" not in record and record["doc_id"] == 1


def test_forward_columns_wins_over_exclude_columns():
    step = LLMStep(
        prompt="P",
        input_columns=[],
        model=StubModel(),
        output_column="out",
        forward_columns=["a"],
        exclude_columns=["a"],
    )
    (record,) = list(step.process(iter([{"a": 1, "b": 2}])))
    assert record["a"] == 1 and "b" not in record


def test_a_generated_column_overwrites_an_input_column_of_the_same_name():
    step = LLMStep(
        prompt="P", input_columns=[], model=StubModel(reply="new"), output_column="text"
    )
    (record,) = list(step.process(iter([{"text": "old"}])))
    assert record["text"] == "new"


def test_a_skipped_record_is_dropped_not_passed_through():
    model = StubModel()
    step = LLMStep(
        prompt="P",
        input_columns=[],
        model=model,
        output_column="out",
        skip_if=lambda record: record["skip"],
    )

    records = list(step.process(iter([{"skip": True}, {"skip": False}])))

    assert len(records) == 1 and records[0]["skip"] is False
    assert len(model.calls) == 1, "no call is made for a skipped record"


def test_the_system_prompt_is_sent_as_a_message_before_the_prompt():
    model = StubModel()
    step = LLMStep(
        prompt="P",
        input_columns=[],
        model=model,
        output_column="out",
        system_prompt="You are concise.",
    )

    list(step.process(iter([{}])))

    assert [m["role"] for m in model.calls[0]] == ["system", "user"]
    assert model.calls[0][0]["content"] == "You are concise."


# --- temperature, max_tokens, errors ----------------------------------------


def test_temperature_and_max_tokens_are_stored_but_never_sent():
    """The page states these do nothing; this fails the day someone wires them up."""
    assert not {"temperature", "max_tokens"} & set(
        inspect.signature(ServedModel.generate).parameters
    ), "generate() now takes them — the page must stop saying they are ignored"

    source = (ROOT / "datafast" / "transforms" / "llm_step.py").read_text()
    for name in ("_temperature", "_max_tokens"):
        uses = [line for line in source.splitlines() if name in line]
        assert uses == [f"        self.{name} = {name[1:]}"], f"{name} is used now"


def test_on_parse_error_rejects_anything_but_skip_and_raise():
    with pytest.raises(ValueError, match="on_parse_error"):
        LLMStep(
            prompt="P", input_columns=[], model=StubModel(), on_parse_error="explode"
        )


def test_process_raises_on_a_parse_failure_when_asked_to():
    step = LLMStep(
        prompt="P",
        input_columns=[],
        model=StubModel(reply="not json"),
        output_columns=["a"],
        parse_mode="json",
        on_parse_error="raise",
    )
    with pytest.raises(ValueError, match="Failed to parse JSON"):
        list(step.process(iter([{}])))


def test_process_skips_a_parse_failure_by_default():
    step = LLMStep(
        prompt="P",
        input_columns=[],
        model=StubModel(reply="not json"),
        output_columns=["a"],
        parse_mode="json",
    )
    assert list(step.process(iter([{}]))) == []


def test_on_parse_error_raise_stops_the_run():
    """`run()` used to catch the error and finish short, so the same step behaved
    differently depending on whether it was driven by `run()` or `process()`."""
    pipeline = (
        Source.list([{"t": "x"}])
        >> LLMStep(
            prompt="Ask {t}",
            input_columns=["t"],
            model=StubModel(reply="not json"),
            output_columns=["a"],
            parse_mode="json",
            on_parse_error="raise",
        )
        >> Sink.list()
    )

    with pytest.raises(Exception, match="(?i)json"):
        pipeline.run()


# --- links ------------------------------------------------------------------


def test_every_page_linked_to_exists():
    links = re.findall(r"\]\((?!https?:)([^)#]+\.md)", _page())
    assert links, "the page must link somewhere"
    missing = sorted(link for link in links if not (PAGE.parent / link).resolve().exists())
    assert not missing, f"links to pages that do not exist: {missing}"
