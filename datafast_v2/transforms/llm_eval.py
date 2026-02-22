"""LLM Evaluation steps: Classify and Score."""

import json
from collections.abc import Callable, Iterable
from typing import Any

from loguru import logger

from datafast_v2.core.config import LLMCall
from datafast_v2.core.step import Step
from datafast_v2.core.types import Record
from datafast_v2.llm.provider import LLMProvider
from datafast_v2.transforms.sample import Sample


def _strip_json_fences(content: str) -> str:
    """Strip markdown code fences from JSON content."""
    content = content.strip()
    if content.startswith("```"):
        first_newline = content.find("\n")
        if first_newline != -1:
            content = content[first_newline + 1 :]
        else:
            content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


def _format_input_content(record: Record, input_columns: list[str]) -> str:
    """Format input columns from a record into readable text."""
    if len(input_columns) == 1:
        return str(record.get(input_columns[0], ""))
    parts = []
    for col in input_columns:
        parts.append(f"{col}: {record.get(col, '')}")
    return "\n".join(parts)


def _normalize_models(
    model: LLMProvider | list[LLMProvider] | Sample,
) -> list[LLMProvider]:
    """Get list of models, resolving Sample if needed."""
    if isinstance(model, Sample):
        return list(model.pick())
    if isinstance(model, list):
        return model
    return [model]


def _build_output_record(
    input_record: Record,
    new_fields: dict[str, Any],
    model: LLMProvider | None,
    forward_columns: list[str] | None,
    exclude_columns: list[str] | None,
) -> Record:
    """Build output record with forwarded columns, new fields, and metadata."""
    output: Record = {}

    if forward_columns is not None:
        for col in forward_columns:
            if col in input_record:
                output[col] = input_record[col]
    elif exclude_columns is not None:
        exclude_set = set(exclude_columns)
        for col, value in input_record.items():
            if col not in exclude_set:
                output[col] = value
    else:
        output.update(input_record)

    output.update(new_fields)

    if model is not None:
        output["_model"] = model.model_id

    return output


# ---------------------------------------------------------------------------
# Classify
# ---------------------------------------------------------------------------


class Classify(Step):
    """Assign one or more labels from a fixed set.

    Supports two modes:
        - **LLM-based**: Uses an LLM to classify content.
        - **Function-based**: Uses a Python function for rule-based classification.

    In LLM mode, integrates with the Runner for batched execution and
    checkpointing via ``collect_calls`` / ``apply_result``.

    Examples:
        >>> # LLM-based sentiment classification
        >>> Classify(
        ...     labels=["positive", "negative", "neutral"],
        ...     input_columns=["review_text"],
        ...     output_column="sentiment",
        ...     llm=openrouter("openai/gpt-4o-mini"),
        ...     include_explanation=True,
        ... )

        >>> # Multi-label topic classification
        >>> Classify(
        ...     labels=["politics", "sports", "tech", "entertainment"],
        ...     input_columns=["article"],
        ...     output_column="topics",
        ...     multi_label=True,
        ...     llm=gpt4,
        ...     labels_description={
        ...         "politics": "Government, elections, policy",
        ...         "sports": "Athletic events, teams, players",
        ...     },
        ... )

        >>> # Function-based classification
        >>> Classify(
        ...     labels=["short", "medium", "long"],
        ...     input_columns=["text"],
        ...     output_column="length_class",
        ...     fn=lambda r: "short" if len(r["text"]) < 100 else "long",
        ... )
    """

    def __init__(
        self,
        labels: list[str],
        input_columns: list[str],
        *,
        output_column: str = "label",
        multi_label: bool = False,
        include_explanation: bool = False,
        include_confidence: bool = False,
        llm: LLMProvider | list[LLMProvider] | Sample | None = None,
        prompt: str | None = None,
        fn: Callable[[Record], str | list[str]] | None = None,
        labels_description: dict[str, str] | None = None,
        system_prompt: str | None = None,
        forward_columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
        on_parse_error: str = "skip",
    ) -> None:
        """
        Initialize a Classify step.

        Args:
            labels: Valid label set.
            input_columns: Columns to feed into classification.
            output_column: Column name for the assigned label(s).
            multi_label: If True, allows multiple labels per record.
            include_explanation: Add ``{output_column}_explanation`` column.
            include_confidence: Add ``{output_column}_confidence`` column.
            llm: LLM provider(s) for LLM-based classification.
            prompt: Custom prompt override for LLM mode. Use ``{content}``
                as placeholder for the formatted input columns.
            fn: Callable for function-based classification. Receives a record
                and must return a label string (or list of strings if
                multi_label).
            labels_description: Mapping of label → human-readable description.
            system_prompt: System prompt prepended in LLM mode.
            forward_columns: Columns to keep from input (LLM mode).
            exclude_columns: Columns to drop from input (LLM mode).
            on_parse_error: ``"skip"`` or ``"raise"`` on LLM parse failure.
        """
        super().__init__()

        if llm is None and fn is None:
            raise ValueError("Classify requires either 'llm' or 'fn'")
        if llm is not None and fn is not None:
            raise ValueError("Classify accepts 'llm' or 'fn', not both")
        if on_parse_error not in ("skip", "raise"):
            raise ValueError("on_parse_error must be 'skip' or 'raise'")

        self._labels = labels
        self._input_columns = input_columns
        self._output_column = output_column
        self._multi_label = multi_label
        self._include_explanation = include_explanation
        self._include_confidence = include_confidence
        self._llm = llm
        self._prompt = prompt
        self._fn = fn
        self._labels_description = labels_description
        self._system_prompt = system_prompt
        self._forward_columns = forward_columns
        self._exclude_columns = exclude_columns
        self._on_parse_error = on_parse_error

    @property
    def uses_llm(self) -> bool:
        """Whether this step uses LLM (vs function-based) mode."""
        return self._llm is not None

    # -- prompt construction ------------------------------------------------

    def _build_default_prompt(self, content: str) -> str:
        """Build the default classification prompt."""
        if self._multi_label:
            task = (
                "Classify the following content. "
                "Select ALL applicable labels from the list below."
            )
        else:
            task = (
                "Classify the following content. "
                "Select exactly ONE label from the list below."
            )

        label_lines: list[str] = []
        for label in self._labels:
            if self._labels_description and label in self._labels_description:
                label_lines.append(f"- {label}: {self._labels_description[label]}")
            else:
                label_lines.append(f"- {label}")
        label_block = "\n".join(label_lines)

        json_fields: list[str] = []
        if self._multi_label:
            json_fields.append('"labels": ["label1", "label2"]')
        else:
            json_fields.append(f'"label": "one of: {", ".join(self._labels)}"')
        if self._include_explanation:
            json_fields.append('"explanation": "your reasoning"')
        if self._include_confidence:
            json_fields.append('"confidence": 0.95')
        json_example = "{" + ", ".join(json_fields) + "}"

        return (
            f"{task}\n\n"
            f"Labels:\n{label_block}\n\n"
            f"Content:\n{content}\n\n"
            f"Respond with valid JSON: {json_example}\n"
            "Return only the JSON object, no additional text."
        )

    def _build_messages(self, record: Record) -> list[dict[str, str]]:
        """Build messages list for LLM call."""
        content = _format_input_content(record, self._input_columns)

        if self._prompt:
            user_text = self._prompt.format(content=content)
        else:
            user_text = self._build_default_prompt(content)

        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": user_text})
        return messages

    # -- result parsing -----------------------------------------------------

    def _parse_llm_result(self, raw: str) -> dict[str, Any]:
        """Parse LLM JSON response into output fields."""
        cleaned = _strip_json_fences(raw)
        data = json.loads(cleaned)

        fields: dict[str, Any] = {}

        if self._multi_label:
            raw_labels = data.get("labels", [])
            if isinstance(raw_labels, str):
                raw_labels = [raw_labels]
            valid = [lbl for lbl in raw_labels if lbl in self._labels]
            fields[self._output_column] = valid
        else:
            raw_label = data.get("label", "")
            if raw_label not in self._labels:
                logger.warning(
                    f"Classify: LLM returned invalid label '{raw_label}', "
                    f"expected one of {self._labels}"
                )
            fields[self._output_column] = raw_label

        if self._include_explanation:
            fields[f"{self._output_column}_explanation"] = data.get("explanation", "")
        if self._include_confidence:
            try:
                fields[f"{self._output_column}_confidence"] = float(
                    data.get("confidence", 0.0)
                )
            except (TypeError, ValueError):
                fields[f"{self._output_column}_confidence"] = 0.0

        return fields

    # -- Runner integration (LLM mode) -------------------------------------

    def collect_calls(
        self,
        records: list[Record],
        skip_call_ids: set[str] | None = None,
    ) -> tuple[list[LLMCall], dict[str, LLMProvider]]:
        """Collect LLM calls for batched execution by the Runner."""
        if self._llm is None:
            return [], {}

        skip_call_ids = skip_call_ids or set()
        calls: list[LLMCall] = []
        models_map: dict[str, LLMProvider] = {}

        models = _normalize_models(self._llm)

        for record_idx, record in enumerate(records):
            for model in models:
                models_map[model.model_id] = model
                call_id = f"{record_idx}_{model.model_id}"
                if call_id in skip_call_ids:
                    continue

                messages = self._build_messages(record)
                call = LLMCall(
                    call_id=call_id,
                    record=record,
                    record_index=record_idx,
                    prompt_template="",
                    prompt_index=0,
                    model_id=model.model_id,
                    language_code="",
                    language_name="",
                    messages=messages,
                    output_index=0,
                )
                calls.append(call)

        return calls, models_map

    def apply_result(
        self, call: LLMCall, result: str, model: LLMProvider
    ) -> Record:
        """Convert an LLM result into an output record."""
        fields = self._parse_llm_result(result)
        return _build_output_record(
            call.record,
            fields,
            model,
            self._forward_columns,
            self._exclude_columns,
        )

    # -- process (direct execution) -----------------------------------------

    def _process_fn(self, records: Iterable[Record]) -> Iterable[Record]:
        """Function-based classification."""
        classified = 0
        for record in records:
            result = self._fn(record)
            output = dict(record)
            output[self._output_column] = result
            classified += 1
            yield output
        logger.info(f"Classify (fn): classified {classified} records")

    def _process_llm(self, records: Iterable[Record]) -> Iterable[Record]:
        """LLM-based classification (direct, without Runner batching)."""
        models = _normalize_models(self._llm)
        generated = 0
        errors = 0

        for record in records:
            for model in models:
                try:
                    messages = self._build_messages(record)
                    raw = model.generate(messages)
                    fields = self._parse_llm_result(raw)
                    output = _build_output_record(
                        record,
                        fields,
                        model,
                        self._forward_columns,
                        self._exclude_columns,
                    )
                    generated += 1
                    yield output
                except Exception as e:
                    errors += 1
                    logger.warning(
                        f"Classify error | Model: {model.model_id} | Error: {e}"
                    )
                    if self._on_parse_error == "raise":
                        raise

        logger.info(f"Classify (llm): generated {generated}, errors {errors}")

    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        """Classify records using either fn or LLM mode."""
        if self._fn is not None:
            yield from self._process_fn(records)
        else:
            yield from self._process_llm(records)


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------


class Score(Step):
    """Assign a numeric score within a range.

    Supports two modes:
        - **LLM-based**: Uses an LLM to score content against criteria/rubric.
        - **Function-based**: Uses a Python function for programmatic scoring.

    In LLM mode, integrates with the Runner for batched execution and
    checkpointing via ``collect_calls`` / ``apply_result``.

    Examples:
        >>> # LLM-based quality scoring with rubric
        >>> Score(
        ...     input_columns=["question", "answer"],
        ...     output_column="quality",
        ...     score_range=(1, 10),
        ...     llm=gpt4,
        ...     criteria="helpfulness, accuracy, and completeness",
        ...     rubric={
        ...         1: "Completely wrong or unhelpful",
        ...         5: "Adequate response with some gaps",
        ...         10: "Excellent, comprehensive, accurate",
        ...     },
        ...     include_explanation=True,
        ... )

        >>> # Function-based scoring
        >>> Score(
        ...     input_columns=["text"],
        ...     output_column="length_score",
        ...     score_range=(0, 100),
        ...     fn=lambda r: min(len(r["text"]) / 10, 100),
        ... )
    """

    def __init__(
        self,
        input_columns: list[str],
        *,
        output_column: str = "score",
        score_range: tuple[float, float] = (1, 10),
        include_explanation: bool = False,
        llm: LLMProvider | list[LLMProvider] | Sample | None = None,
        prompt: str | None = None,
        fn: Callable[[Record], float] | None = None,
        criteria: str | None = None,
        rubric: dict[int, str] | None = None,
        system_prompt: str | None = None,
        forward_columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
        on_parse_error: str = "skip",
    ) -> None:
        """
        Initialize a Score step.

        Args:
            input_columns: Columns to feed into scoring.
            output_column: Column name for the numeric score.
            score_range: ``(min, max)`` valid score range.
            include_explanation: Add ``{output_column}_explanation`` column.
            llm: LLM provider(s) for LLM-based scoring.
            prompt: Custom prompt override for LLM mode. Use ``{content}``
                as placeholder for the formatted input columns.
            fn: Callable for function-based scoring. Receives a record and
                must return a float.
            criteria: Scoring criteria description for LLM mode.
            rubric: Mapping of score value → description for LLM mode.
            system_prompt: System prompt prepended in LLM mode.
            forward_columns: Columns to keep from input (LLM mode).
            exclude_columns: Columns to drop from input (LLM mode).
            on_parse_error: ``"skip"`` or ``"raise"`` on LLM parse failure.
        """
        super().__init__()

        if llm is None and fn is None:
            raise ValueError("Score requires either 'llm' or 'fn'")
        if llm is not None and fn is not None:
            raise ValueError("Score accepts 'llm' or 'fn', not both")
        if on_parse_error not in ("skip", "raise"):
            raise ValueError("on_parse_error must be 'skip' or 'raise'")

        self._input_columns = input_columns
        self._output_column = output_column
        self._score_range = score_range
        self._include_explanation = include_explanation
        self._llm = llm
        self._prompt = prompt
        self._fn = fn
        self._criteria = criteria
        self._rubric = rubric
        self._system_prompt = system_prompt
        self._forward_columns = forward_columns
        self._exclude_columns = exclude_columns
        self._on_parse_error = on_parse_error

    @property
    def uses_llm(self) -> bool:
        """Whether this step uses LLM (vs function-based) mode."""
        return self._llm is not None

    # -- prompt construction ------------------------------------------------

    def _build_default_prompt(self, content: str) -> str:
        """Build the default scoring prompt."""
        lo, hi = self._score_range
        task = f"Score the following content on a scale of {lo} to {hi}."

        parts = [task]

        if self._criteria:
            parts.append(f"\nCriteria: {self._criteria}")

        if self._rubric:
            rubric_lines = []
            for score_val in sorted(self._rubric.keys()):
                rubric_lines.append(f"  {score_val}: {self._rubric[score_val]}")
            parts.append("\nRubric:\n" + "\n".join(rubric_lines))

        parts.append(f"\nContent:\n{content}")

        json_fields = [f'"score": <number between {lo} and {hi}>']
        if self._include_explanation:
            json_fields.append('"explanation": "your reasoning"')
        json_example = "{" + ", ".join(json_fields) + "}"

        parts.append(
            f"\nRespond with valid JSON: {json_example}\n"
            "Return only the JSON object, no additional text."
        )
        return "\n".join(parts)

    def _build_messages(self, record: Record) -> list[dict[str, str]]:
        """Build messages list for LLM call."""
        content = _format_input_content(record, self._input_columns)

        if self._prompt:
            user_text = self._prompt.format(content=content)
        else:
            user_text = self._build_default_prompt(content)

        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": user_text})
        return messages

    # -- result parsing -----------------------------------------------------

    def _parse_llm_result(self, raw: str) -> dict[str, Any]:
        """Parse LLM JSON response into output fields."""
        cleaned = _strip_json_fences(raw)
        data = json.loads(cleaned)

        fields: dict[str, Any] = {}

        lo, hi = self._score_range
        try:
            score = float(data.get("score", 0))
        except (TypeError, ValueError):
            score = lo
        score = max(lo, min(hi, score))
        fields[self._output_column] = score

        if self._include_explanation:
            fields[f"{self._output_column}_explanation"] = data.get("explanation", "")

        return fields

    # -- Runner integration (LLM mode) -------------------------------------

    def collect_calls(
        self,
        records: list[Record],
        skip_call_ids: set[str] | None = None,
    ) -> tuple[list[LLMCall], dict[str, LLMProvider]]:
        """Collect LLM calls for batched execution by the Runner."""
        if self._llm is None:
            return [], {}

        skip_call_ids = skip_call_ids or set()
        calls: list[LLMCall] = []
        models_map: dict[str, LLMProvider] = {}

        models = _normalize_models(self._llm)

        for record_idx, record in enumerate(records):
            for model in models:
                models_map[model.model_id] = model
                call_id = f"{record_idx}_{model.model_id}"
                if call_id in skip_call_ids:
                    continue

                messages = self._build_messages(record)
                call = LLMCall(
                    call_id=call_id,
                    record=record,
                    record_index=record_idx,
                    prompt_template="",
                    prompt_index=0,
                    model_id=model.model_id,
                    language_code="",
                    language_name="",
                    messages=messages,
                    output_index=0,
                )
                calls.append(call)

        return calls, models_map

    def apply_result(
        self, call: LLMCall, result: str, model: LLMProvider
    ) -> Record:
        """Convert an LLM result into an output record."""
        fields = self._parse_llm_result(result)
        return _build_output_record(
            call.record,
            fields,
            model,
            self._forward_columns,
            self._exclude_columns,
        )

    # -- process (direct execution) -----------------------------------------

    def _process_fn(self, records: Iterable[Record]) -> Iterable[Record]:
        """Function-based scoring."""
        scored = 0
        lo, hi = self._score_range
        for record in records:
            raw_score = self._fn(record)
            clamped = max(lo, min(hi, float(raw_score)))
            output = dict(record)
            output[self._output_column] = clamped
            scored += 1
            yield output
        logger.info(f"Score (fn): scored {scored} records")

    def _process_llm(self, records: Iterable[Record]) -> Iterable[Record]:
        """LLM-based scoring (direct, without Runner batching)."""
        models = _normalize_models(self._llm)
        generated = 0
        errors = 0

        for record in records:
            for model in models:
                try:
                    messages = self._build_messages(record)
                    raw = model.generate(messages)
                    fields = self._parse_llm_result(raw)
                    output = _build_output_record(
                        record,
                        fields,
                        model,
                        self._forward_columns,
                        self._exclude_columns,
                    )
                    generated += 1
                    yield output
                except Exception as e:
                    errors += 1
                    logger.warning(
                        f"Score error | Model: {model.model_id} | Error: {e}"
                    )
                    if self._on_parse_error == "raise":
                        raise

        logger.info(f"Score (llm): generated {generated}, errors {errors}")

    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        """Score records using either fn or LLM mode."""
        if self._fn is not None:
            yield from self._process_fn(records)
        else:
            yield from self._process_llm(records)


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


class Compare(Step):
    """Pairwise comparison of two fields.

    Supports two modes:
        - **LLM-based**: Uses an LLM to compare content from two columns.
        - **Function-based**: Uses a Python function for rule-based comparison.

    Output depends on ``output_mode``:
        - ``"winner"``: Adds ``{output_column}`` with value ``"a"``, ``"b"``,
          or ``"tie"``.
        - ``"scores"``: Winner plus ``{output_column}_score_a`` and
          ``{output_column}_score_b``.
        - ``"detailed"``: Scores plus ``{output_column}_reasoning``.

    In LLM mode, integrates with the Runner for batched execution and
    checkpointing via ``collect_calls`` / ``apply_result``.

    Examples:
        >>> # LLM-based comparison of two responses
        >>> Compare(
        ...     column_a="response_chosen",
        ...     column_b="response_rejected",
        ...     criteria="helpfulness and accuracy",
        ...     output_mode="detailed",
        ...     llm=openrouter("openai/gpt-4o-mini"),
        ... )

        >>> # With individual scores
        >>> Compare(
        ...     column_a="summary_model_a",
        ...     column_b="summary_model_b",
        ...     criteria="faithfulness to the source text",
        ...     output_mode="scores",
        ...     score_range=(1, 5),
        ...     llm=gpt4,
        ... )

        >>> # Function-based comparison
        >>> Compare(
        ...     column_a="summary_a",
        ...     column_b="summary_b",
        ...     criteria="length",
        ...     fn=lambda r: {
        ...         "winner": "a" if len(r["summary_a"]) > len(r["summary_b"]) else "b",
        ...     },
        ... )
    """

    def __init__(
        self,
        column_a: str,
        column_b: str,
        criteria: str,
        *,
        output_column: str = "comparison",
        output_mode: str = "winner",
        score_range: tuple[float, float] = (1, 10),
        llm: LLMProvider | list[LLMProvider] | Sample | None = None,
        prompt: str | None = None,
        fn: Callable[[Record], dict[str, Any] | str] | None = None,
        system_prompt: str | None = None,
        forward_columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
        on_parse_error: str = "skip",
    ) -> None:
        """
        Initialize a Compare step.

        Args:
            column_a: Column name for the first item to compare.
            column_b: Column name for the second item to compare.
            criteria: Description of comparison criteria (e.g.
                ``"helpfulness and accuracy"``).
            output_column: Column name prefix for comparison results.
            output_mode: Level of detail in output:
                - ``"winner"``: just the winner (``"a"``, ``"b"``, or
                  ``"tie"``).
                - ``"scores"``: winner plus per-item scores.
                - ``"detailed"``: scores plus reasoning text.
            score_range: ``(min, max)`` range for scores when
                ``output_mode`` is ``"scores"`` or ``"detailed"``.
            llm: LLM provider(s) for LLM-based comparison.
            prompt: Custom prompt override for LLM mode. Available
                placeholders: ``{text_a}``, ``{text_b}``, ``{criteria}``,
                ``{column_a}``, ``{column_b}``, and any record field.
            fn: Callable for function-based comparison. Receives a record
                and must return a dict with at least a ``"winner"`` key
                (value ``"a"``, ``"b"``, or ``"tie"``), and optionally
                ``"score_a"``, ``"score_b"``, ``"reasoning"``. May also
                return a plain string (``"a"``, ``"b"``, or ``"tie"``).
            system_prompt: System prompt prepended in LLM mode.
            forward_columns: Columns to keep from input (LLM mode).
            exclude_columns: Columns to drop from input (LLM mode).
            on_parse_error: ``"skip"`` or ``"raise"`` on LLM parse failure.
        """
        super().__init__()

        if llm is None and fn is None:
            raise ValueError("Compare requires either 'llm' or 'fn'")
        if llm is not None and fn is not None:
            raise ValueError("Compare accepts 'llm' or 'fn', not both")
        if output_mode not in ("winner", "scores", "detailed"):
            raise ValueError(
                "output_mode must be 'winner', 'scores', or 'detailed'"
            )
        if on_parse_error not in ("skip", "raise"):
            raise ValueError("on_parse_error must be 'skip' or 'raise'")

        self._column_a = column_a
        self._column_b = column_b
        self._criteria = criteria
        self._output_column = output_column
        self._output_mode = output_mode
        self._score_range = score_range
        self._llm = llm
        self._prompt = prompt
        self._fn = fn
        self._system_prompt = system_prompt
        self._forward_columns = forward_columns
        self._exclude_columns = exclude_columns
        self._on_parse_error = on_parse_error

    @property
    def uses_llm(self) -> bool:
        """Whether this step uses LLM (vs function-based) mode."""
        return self._llm is not None

    # -- prompt construction ------------------------------------------------

    def _build_default_prompt(self, text_a: str, text_b: str) -> str:
        """Build the default comparison prompt."""
        lo, hi = self._score_range

        parts: list[str] = [
            "Compare the following two responses based on this criteria: "
            f"{self._criteria}",
            f"\nResponse A ({self._column_a}):\n{text_a}",
            f"\nResponse B ({self._column_b}):\n{text_b}",
        ]

        json_fields: list[str] = ['"winner": "a" or "b" or "tie"']
        if self._output_mode in ("scores", "detailed"):
            json_fields.append(f'"score_a": <number between {lo} and {hi}>')
            json_fields.append(f'"score_b": <number between {lo} and {hi}>')
        if self._output_mode == "detailed":
            json_fields.append(
                '"reasoning": "your reasoning for the comparison"'
            )
        json_example = "{" + ", ".join(json_fields) + "}"

        parts.append(
            f"\nRespond with valid JSON: {json_example}\n"
            "Return only the JSON object, no additional text."
        )
        return "\n".join(parts)

    def _build_messages(self, record: Record) -> list[dict[str, str]]:
        """Build messages list for LLM call."""
        text_a = str(record.get(self._column_a, ""))
        text_b = str(record.get(self._column_b, ""))

        if self._prompt:
            format_kwargs: dict[str, Any] = {
                "text_a": text_a,
                "text_b": text_b,
                "criteria": self._criteria,
                "column_a": self._column_a,
                "column_b": self._column_b,
            }
            format_kwargs.update(record)
            user_text = self._prompt.format(**format_kwargs)
        else:
            user_text = self._build_default_prompt(text_a, text_b)

        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": user_text})
        return messages

    # -- result parsing -----------------------------------------------------

    def _parse_llm_result(self, raw: str) -> dict[str, Any]:
        """Parse LLM JSON response into output fields."""
        cleaned = _strip_json_fences(raw)
        data = json.loads(cleaned)

        fields: dict[str, Any] = {}

        winner = str(data.get("winner", "")).lower().strip()
        if winner not in ("a", "b", "tie"):
            logger.warning(
                f"Compare: LLM returned invalid winner '{winner}', "
                f"expected 'a', 'b', or 'tie'"
            )
        fields[self._output_column] = winner

        if self._output_mode in ("scores", "detailed"):
            lo, hi = self._score_range
            try:
                score_a = float(data.get("score_a", 0))
            except (TypeError, ValueError):
                score_a = lo
            try:
                score_b = float(data.get("score_b", 0))
            except (TypeError, ValueError):
                score_b = lo
            fields[f"{self._output_column}_score_a"] = max(lo, min(hi, score_a))
            fields[f"{self._output_column}_score_b"] = max(lo, min(hi, score_b))

        if self._output_mode == "detailed":
            fields[f"{self._output_column}_reasoning"] = data.get(
                "reasoning", ""
            )

        return fields

    # -- Runner integration (LLM mode) -------------------------------------

    def collect_calls(
        self,
        records: list[Record],
        skip_call_ids: set[str] | None = None,
    ) -> tuple[list[LLMCall], dict[str, LLMProvider]]:
        """Collect LLM calls for batched execution by the Runner."""
        if self._llm is None:
            return [], {}

        skip_call_ids = skip_call_ids or set()
        calls: list[LLMCall] = []
        models_map: dict[str, LLMProvider] = {}

        models = _normalize_models(self._llm)

        for record_idx, record in enumerate(records):
            for model in models:
                models_map[model.model_id] = model
                call_id = f"{record_idx}_{model.model_id}"
                if call_id in skip_call_ids:
                    continue

                messages = self._build_messages(record)
                call = LLMCall(
                    call_id=call_id,
                    record=record,
                    record_index=record_idx,
                    prompt_template="",
                    prompt_index=0,
                    model_id=model.model_id,
                    language_code="",
                    language_name="",
                    messages=messages,
                    output_index=0,
                )
                calls.append(call)

        return calls, models_map

    def apply_result(
        self, call: LLMCall, result: str, model: LLMProvider
    ) -> Record:
        """Convert an LLM result into an output record."""
        fields = self._parse_llm_result(result)
        return _build_output_record(
            call.record,
            fields,
            model,
            self._forward_columns,
            self._exclude_columns,
        )

    # -- process (direct execution) -----------------------------------------

    def _process_fn(self, records: Iterable[Record]) -> Iterable[Record]:
        """Function-based comparison."""
        compared = 0
        for record in records:
            result = self._fn(record)
            output = dict(record)

            if isinstance(result, dict):
                output[self._output_column] = result.get("winner", "")
                if self._output_mode in ("scores", "detailed"):
                    output[f"{self._output_column}_score_a"] = result.get(
                        "score_a"
                    )
                    output[f"{self._output_column}_score_b"] = result.get(
                        "score_b"
                    )
                if self._output_mode == "detailed":
                    output[f"{self._output_column}_reasoning"] = result.get(
                        "reasoning", ""
                    )
            else:
                output[self._output_column] = str(result)

            compared += 1
            yield output
        logger.info(f"Compare (fn): compared {compared} records")

    def _process_llm(self, records: Iterable[Record]) -> Iterable[Record]:
        """LLM-based comparison (direct, without Runner batching)."""
        models = _normalize_models(self._llm)
        generated = 0
        errors = 0

        for record in records:
            for model in models:
                try:
                    messages = self._build_messages(record)
                    raw = model.generate(messages)
                    fields = self._parse_llm_result(raw)
                    output = _build_output_record(
                        record,
                        fields,
                        model,
                        self._forward_columns,
                        self._exclude_columns,
                    )
                    generated += 1
                    yield output
                except Exception as e:
                    errors += 1
                    logger.warning(
                        f"Compare error | Model: {model.model_id} | Error: {e}"
                    )
                    if self._on_parse_error == "raise":
                        raise

        logger.info(f"Compare (llm): generated {generated}, errors {errors}")

    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        """Compare records using either fn or LLM mode."""
        if self._fn is not None:
            yield from self._process_fn(records)
        else:
            yield from self._process_llm(records)
