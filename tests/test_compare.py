"""Tests for the Compare step."""

import json
from unittest.mock import MagicMock

import pytest

from datafast import Compare


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RECORDS = [
    {
        "id": 1,
        "response_a": "The capital of France is Paris.",
        "response_b": "France capital is paris i think.",
    },
    {
        "id": 2,
        "response_a": "Water boils at 100°C at sea level.",
        "response_b": "Water boils at 100 degrees Celsius under standard atmospheric pressure.",
    },
]


def _make_mock_model(model_id: str = "mock-model") -> MagicMock:
    model = MagicMock()
    model.model_id = model_id
    return model


# ---------------------------------------------------------------------------
# Init validation
# ---------------------------------------------------------------------------


class TestCompareInit:
    def test_requires_llm_or_fn(self):
        with pytest.raises(ValueError, match="requires either"):
            Compare(column_a="a", column_b="b", criteria="quality")

    def test_rejects_both_llm_and_fn(self):
        with pytest.raises(ValueError, match="not both"):
            Compare(
                column_a="a",
                column_b="b",
                criteria="quality",
                llm=_make_mock_model(),
                fn=lambda r: "a",
            )

    def test_rejects_invalid_output_mode(self):
        with pytest.raises(ValueError, match="output_mode"):
            Compare(
                column_a="a",
                column_b="b",
                criteria="quality",
                fn=lambda r: "a",
                output_mode="invalid",
            )

    def test_rejects_invalid_on_parse_error(self):
        with pytest.raises(ValueError, match="on_parse_error"):
            Compare(
                column_a="a",
                column_b="b",
                criteria="quality",
                fn=lambda r: "a",
                on_parse_error="ignore",
            )

    def test_uses_llm_property(self):
        fn_step = Compare(
            column_a="a", column_b="b", criteria="q", fn=lambda r: "a"
        )
        assert fn_step.uses_llm is False

        llm_step = Compare(
            column_a="a",
            column_b="b",
            criteria="q",
            llm=_make_mock_model(),
        )
        assert llm_step.uses_llm is True


# ---------------------------------------------------------------------------
# Function-based mode
# ---------------------------------------------------------------------------


class TestCompareFn:
    def test_winner_mode_with_dict(self):
        step = Compare(
            column_a="response_a",
            column_b="response_b",
            criteria="accuracy",
            fn=lambda r: {"winner": "a"},
        )
        results = list(step.process(iter(RECORDS)))
        assert len(results) == 2
        for r in results:
            assert r["comparison"] == "a"
            assert "comparison_score_a" not in r
            assert "comparison_reasoning" not in r

    def test_winner_mode_with_string(self):
        step = Compare(
            column_a="response_a",
            column_b="response_b",
            criteria="accuracy",
            fn=lambda r: "b",
        )
        results = list(step.process(iter(RECORDS)))
        assert all(r["comparison"] == "b" for r in results)

    def test_scores_mode(self):
        step = Compare(
            column_a="response_a",
            column_b="response_b",
            criteria="accuracy",
            output_mode="scores",
            fn=lambda r: {"winner": "a", "score_a": 8, "score_b": 5},
        )
        results = list(step.process(iter(RECORDS)))
        assert len(results) == 2
        for r in results:
            assert r["comparison"] == "a"
            assert r["comparison_score_a"] == 8
            assert r["comparison_score_b"] == 5
            assert "comparison_reasoning" not in r

    def test_detailed_mode(self):
        step = Compare(
            column_a="response_a",
            column_b="response_b",
            criteria="accuracy",
            output_mode="detailed",
            fn=lambda r: {
                "winner": "tie",
                "score_a": 7,
                "score_b": 7,
                "reasoning": "Both are equally good.",
            },
        )
        results = list(step.process(iter(RECORDS)))
        for r in results:
            assert r["comparison"] == "tie"
            assert r["comparison_score_a"] == 7
            assert r["comparison_score_b"] == 7
            assert r["comparison_reasoning"] == "Both are equally good."

    def test_custom_output_column(self):
        step = Compare(
            column_a="response_a",
            column_b="response_b",
            criteria="accuracy",
            output_column="preference",
            fn=lambda r: {"winner": "a"},
        )
        results = list(step.process(iter(RECORDS)))
        assert all("preference" in r for r in results)
        assert all("comparison" not in r for r in results)

    def test_preserves_input_fields(self):
        step = Compare(
            column_a="response_a",
            column_b="response_b",
            criteria="accuracy",
            fn=lambda r: "a",
        )
        results = list(step.process(iter(RECORDS)))
        for r in results:
            assert "id" in r
            assert "response_a" in r
            assert "response_b" in r


# ---------------------------------------------------------------------------
# LLM-based mode — prompt construction
# ---------------------------------------------------------------------------


class TestComparePrompt:
    def test_default_prompt_winner_mode(self):
        model = _make_mock_model()
        step = Compare(
            column_a="response_a",
            column_b="response_b",
            criteria="accuracy",
            llm=model,
        )
        messages = step._build_messages(RECORDS[0])
        user_text = messages[-1]["content"]
        assert "accuracy" in user_text
        assert "Response A" in user_text
        assert "Response B" in user_text
        assert '"winner"' in user_text
        assert '"score_a"' not in user_text
        assert '"reasoning"' not in user_text

    def test_default_prompt_scores_mode(self):
        model = _make_mock_model()
        step = Compare(
            column_a="response_a",
            column_b="response_b",
            criteria="accuracy",
            output_mode="scores",
            llm=model,
        )
        messages = step._build_messages(RECORDS[0])
        user_text = messages[-1]["content"]
        assert '"score_a"' in user_text
        assert '"score_b"' in user_text
        assert '"reasoning"' not in user_text

    def test_default_prompt_detailed_mode(self):
        model = _make_mock_model()
        step = Compare(
            column_a="response_a",
            column_b="response_b",
            criteria="accuracy",
            output_mode="detailed",
            llm=model,
        )
        messages = step._build_messages(RECORDS[0])
        user_text = messages[-1]["content"]
        assert '"reasoning"' in user_text

    def test_custom_prompt_placeholders(self):
        model = _make_mock_model()
        step = Compare(
            column_a="response_a",
            column_b="response_b",
            criteria="accuracy",
            llm=model,
            prompt="Compare {text_a} vs {text_b} on {criteria}. ID={id}",
        )
        messages = step._build_messages(RECORDS[0])
        user_text = messages[-1]["content"]
        assert RECORDS[0]["response_a"] in user_text
        assert RECORDS[0]["response_b"] in user_text
        assert "accuracy" in user_text
        assert "ID=1" in user_text

    def test_system_prompt(self):
        model = _make_mock_model()
        step = Compare(
            column_a="response_a",
            column_b="response_b",
            criteria="accuracy",
            llm=model,
            system_prompt="You are an expert evaluator.",
        )
        messages = step._build_messages(RECORDS[0])
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are an expert evaluator."
        assert messages[1]["role"] == "user"


# ---------------------------------------------------------------------------
# LLM-based mode — result parsing
# ---------------------------------------------------------------------------


class TestCompareParsingLLM:
    def test_parse_winner_only(self):
        model = _make_mock_model()
        step = Compare(
            column_a="a",
            column_b="b",
            criteria="q",
            output_mode="winner",
            llm=model,
        )
        fields = step._parse_llm_result('{"winner": "a"}')
        assert fields == {"comparison": "a"}

    def test_parse_scores(self):
        model = _make_mock_model()
        step = Compare(
            column_a="a",
            column_b="b",
            criteria="q",
            output_mode="scores",
            score_range=(1, 10),
            llm=model,
        )
        raw = json.dumps({"winner": "b", "score_a": 4, "score_b": 9})
        fields = step._parse_llm_result(raw)
        assert fields["comparison"] == "b"
        assert fields["comparison_score_a"] == 4.0
        assert fields["comparison_score_b"] == 9.0

    def test_parse_detailed(self):
        model = _make_mock_model()
        step = Compare(
            column_a="a",
            column_b="b",
            criteria="q",
            output_mode="detailed",
            llm=model,
        )
        raw = json.dumps(
            {
                "winner": "tie",
                "score_a": 7,
                "score_b": 7,
                "reasoning": "Equal quality.",
            }
        )
        fields = step._parse_llm_result(raw)
        assert fields["comparison"] == "tie"
        assert fields["comparison_reasoning"] == "Equal quality."

    def test_parse_clamps_scores(self):
        model = _make_mock_model()
        step = Compare(
            column_a="a",
            column_b="b",
            criteria="q",
            output_mode="scores",
            score_range=(1, 5),
            llm=model,
        )
        raw = json.dumps({"winner": "a", "score_a": 99, "score_b": -5})
        fields = step._parse_llm_result(raw)
        assert fields["comparison_score_a"] == 5.0
        assert fields["comparison_score_b"] == 1.0

    def test_parse_strips_json_fences(self):
        model = _make_mock_model()
        step = Compare(
            column_a="a",
            column_b="b",
            criteria="q",
            llm=model,
        )
        raw = '```json\n{"winner": "b"}\n```'
        fields = step._parse_llm_result(raw)
        assert fields["comparison"] == "b"


# ---------------------------------------------------------------------------
# LLM-based mode — direct execution
# ---------------------------------------------------------------------------


class TestCompareLLMProcess:
    def test_process_llm_winner(self):
        model = _make_mock_model()
        model.generate.return_value = '{"winner": "a"}'

        step = Compare(
            column_a="response_a",
            column_b="response_b",
            criteria="accuracy",
            llm=model,
        )
        results = list(step.process(iter(RECORDS)))
        assert len(results) == 2
        assert all(r["comparison"] == "a" for r in results)
        assert all(r["_model"] == "mock-model" for r in results)

    def test_process_llm_detailed(self):
        model = _make_mock_model()
        model.generate.return_value = json.dumps(
            {
                "winner": "b",
                "score_a": 3,
                "score_b": 8,
                "reasoning": "B is better.",
            }
        )

        step = Compare(
            column_a="response_a",
            column_b="response_b",
            criteria="accuracy",
            output_mode="detailed",
            llm=model,
        )
        results = list(step.process(iter(RECORDS)))
        assert len(results) == 2
        assert results[0]["comparison"] == "b"
        assert results[0]["comparison_score_a"] == 3.0
        assert results[0]["comparison_reasoning"] == "B is better."

    def test_process_llm_skips_on_error(self):
        model = _make_mock_model()
        model.generate.side_effect = RuntimeError("API error")

        step = Compare(
            column_a="response_a",
            column_b="response_b",
            criteria="accuracy",
            llm=model,
            on_parse_error="skip",
        )
        results = list(step.process(iter(RECORDS)))
        assert len(results) == 0

    def test_process_llm_raises_on_error(self):
        model = _make_mock_model()
        model.generate.side_effect = RuntimeError("API error")

        step = Compare(
            column_a="response_a",
            column_b="response_b",
            criteria="accuracy",
            llm=model,
            on_parse_error="raise",
        )
        with pytest.raises(RuntimeError, match="API error"):
            list(step.process(iter(RECORDS)))

    def test_forward_columns(self):
        model = _make_mock_model()
        model.generate.return_value = '{"winner": "a"}'

        step = Compare(
            column_a="response_a",
            column_b="response_b",
            criteria="accuracy",
            llm=model,
            forward_columns=["id"],
        )
        results = list(step.process(iter(RECORDS)))
        assert results[0]["id"] == 1
        assert "response_a" not in results[0]

    def test_exclude_columns(self):
        model = _make_mock_model()
        model.generate.return_value = '{"winner": "a"}'

        step = Compare(
            column_a="response_a",
            column_b="response_b",
            criteria="accuracy",
            llm=model,
            exclude_columns=["response_b"],
        )
        results = list(step.process(iter(RECORDS)))
        assert "response_a" in results[0]
        assert "response_b" not in results[0]

    def test_multiple_models(self):
        model_a = _make_mock_model("model-a")
        model_a.generate.return_value = '{"winner": "a"}'
        model_b = _make_mock_model("model-b")
        model_b.generate.return_value = '{"winner": "b"}'

        step = Compare(
            column_a="response_a",
            column_b="response_b",
            criteria="accuracy",
            llm=[model_a, model_b],
        )
        results = list(step.process(iter(RECORDS)))
        assert len(results) == 4  # 2 records × 2 models


# ---------------------------------------------------------------------------
# Runner integration — collect_calls / apply_result
# ---------------------------------------------------------------------------


class TestCompareRunnerIntegration:
    def test_collect_calls(self):
        model = _make_mock_model()
        step = Compare(
            column_a="response_a",
            column_b="response_b",
            criteria="accuracy",
            llm=model,
        )
        calls, models_map = step.collect_calls(RECORDS)
        assert len(calls) == 2
        assert "mock-model" in models_map
        assert calls[0].call_id == "0_mock-model"
        assert calls[1].call_id == "1_mock-model"

    def test_collect_calls_skips(self):
        model = _make_mock_model()
        step = Compare(
            column_a="response_a",
            column_b="response_b",
            criteria="accuracy",
            llm=model,
        )
        calls, _ = step.collect_calls(RECORDS, skip_call_ids={"0_mock-model"})
        assert len(calls) == 1
        assert calls[0].call_id == "1_mock-model"

    def test_collect_calls_fn_mode_returns_empty(self):
        step = Compare(
            column_a="response_a",
            column_b="response_b",
            criteria="accuracy",
            fn=lambda r: "a",
        )
        calls, models_map = step.collect_calls(RECORDS)
        assert calls == []
        assert models_map == {}

    def test_apply_result(self):
        model = _make_mock_model()
        step = Compare(
            column_a="response_a",
            column_b="response_b",
            criteria="accuracy",
            output_mode="detailed",
            llm=model,
        )
        calls, _ = step.collect_calls(RECORDS)
        raw = json.dumps(
            {
                "winner": "a",
                "score_a": 9,
                "score_b": 4,
                "reasoning": "A is more accurate.",
            }
        )
        result = step.apply_result(calls[0], raw, model)
        assert result["comparison"] == "a"
        assert result["comparison_score_a"] == 9.0
        assert result["comparison_reasoning"] == "A is more accurate."
        assert result["_model"] == "mock-model"
