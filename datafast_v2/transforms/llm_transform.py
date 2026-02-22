"""LLM Transformation steps: Rewrite."""

from collections.abc import Iterable
from typing import Any

from loguru import logger

from datafast_v2.core.config import LLMCall
from datafast_v2.core.step import Step
from datafast_v2.core.types import Record
from datafast_v2.llm.provider import LLMProvider
from datafast_v2.transforms.llm_eval import (
    _build_output_record,
    _normalize_models,
)
from datafast_v2.transforms.sample import Sample

# ---------------------------------------------------------------------------
# Rewrite mode prompts
# ---------------------------------------------------------------------------

_MODE_INSTRUCTIONS: dict[str, str] = {
    "paraphrase": (
        "Rewrite (Paraphrase) the following text using different words and sentence structures "
        "while preserving the original meaning."
    ),
    "simplify": (
        "Rewrite (Simplify) the following text in simpler language that is easier to understand. "
        "Use shorter sentences and common vocabulary."
    ),
    "formalize": (
        "Rewrite (Formalize) the following text in a formal, professional tone. "
        "Use proper grammar, avoid slang, and maintain a polished style."
    ),
    "informalize": (
        "Rewrite (Informalize) the following text in a casual, conversational tone. "
        "Use everyday language and a friendly style."
    ),
    "elaborate": (
        "Expand (Elaborate) and elaborate on the following text. "
        "Add more detail, examples, and explanation while preserving the core message."
    ),
}

VALID_MODES = frozenset([
    "paraphrase", "simplify", "formalize", "informalize",
    "custom", "audience", "length", "elaborate",
])


# ---------------------------------------------------------------------------
# Rewrite
# ---------------------------------------------------------------------------


class Rewrite(Step):
    """Generate variations of text while preserving meaning.

    Supports several rewriting modes and integrates with the Runner for
    batched execution and checkpointing via ``collect_calls`` / ``apply_result``.

    Modes:
        - ``"paraphrase"``: Different words, same meaning (default).
        - ``"simplify"``: Simpler language.
        - ``"formalize"``: Professional tone.
        - ``"informalize"``: Casual tone.
        - ``"custom"``: Rewrite following a custom instruction (requires ``custom_instruction``).
        - ``"audience"``: Rewrite for a target audience (requires ``target_audience``).
        - ``"length"``: Adjust length (requires ``target_length``).
        - ``"elaborate"``: Expand with more detail.

    Examples:
        >>> # Paraphrase for data augmentation
        >>> Rewrite(
        ...     input_column="text",
        ...     mode="paraphrase",
        ...     num_variations=3,
        ...     llm=openrouter("openai/gpt-4o-mini"),
        ... )

        >>> # Simplify for different audience
        >>> Rewrite(
        ...     input_column="technical_doc",
        ...     mode="simplify",
        ...     target_audience="high school student",
        ...     llm=gpt4,
        ... )

        >>> # Formalize casual text
        >>> Rewrite(
        ...     input_column="casual_email",
        ...     mode="formalize",
        ...     llm=gpt4,
        ... )

        >>> # Custom rewrite instruction
        >>> Rewrite(
        ...     input_column="text",
        ...     mode="custom",
        ...     custom_instruction="Rewrite as a haiku",
        ...     llm=gpt4,
        ... )
    """

    def __init__(
        self,
        input_column: str,
        llm: LLMProvider | list[LLMProvider] | Sample,
        *,
        output_column: str | None = None,
        mode: str = "paraphrase",
        preserve: list[str] | None = None,
        num_variations: int = 1,
        custom_instruction: str | None = None,
        target_audience: str | None = None,
        target_length: str | None = None,
        system_prompt: str | None = None,
        forward_columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
        on_parse_error: str = "skip",
    ) -> None:
        """
        Initialize a Rewrite step.

        Args:
            input_column: Column containing text to rewrite.
            llm: LLM provider(s) to use.
            output_column: Column for rewritten text. Defaults to
                ``{input_column}_rewritten``.
            mode: Rewriting mode. One of: ``paraphrase``, ``simplify``,
                ``formalize``, ``informalize``, ``custom``, ``audience``,
                ``length``, ``elaborate``.
            preserve: Aspects to preserve (e.g. ``["technical terms", "tone"]``).
            num_variations: Number of rewritten variations per input.
            custom_instruction: Free-form rewrite instruction (required for
                ``custom`` mode).
            target_audience: Target audience (required/useful for ``audience``
                and ``simplify`` modes).
            target_length: Length directive such as ``"shorter"``, ``"longer"``,
                ``"2x"``, ``"half"`` (required for ``length`` mode).
            system_prompt: System prompt prepended to messages.
            forward_columns: Columns to keep from input record.
            exclude_columns: Columns to drop from input record.
            on_parse_error: ``"skip"`` or ``"raise"`` on LLM failure.
        """
        super().__init__()

        if mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode: {mode}. Valid modes: {sorted(VALID_MODES)}"
            )
        if mode == "custom" and not custom_instruction:
            raise ValueError("custom_instruction is required for mode='custom'")
        if mode == "audience" and not target_audience:
            raise ValueError("target_audience is required for mode='audience'")
        if mode == "length" and not target_length:
            raise ValueError("target_length is required for mode='length'")
        if on_parse_error not in ("skip", "raise"):
            raise ValueError("on_parse_error must be 'skip' or 'raise'")

        self._input_column = input_column
        self._output_column = output_column or f"{input_column}_rewritten"
        self._mode = mode
        self._preserve = preserve
        self._num_variations = num_variations
        self._custom_instruction = custom_instruction
        self._target_audience = target_audience
        self._target_length = target_length
        self._llm = llm
        self._system_prompt = system_prompt
        self._forward_columns = forward_columns
        self._exclude_columns = exclude_columns
        self._on_parse_error = on_parse_error

    @property
    def uses_llm(self) -> bool:
        """Rewrite always uses an LLM."""
        return True

    # -- prompt construction ------------------------------------------------

    def _build_prompt(self, text: str) -> str:
        """Build the rewrite prompt for the given text."""
        # Core instruction
        if self._mode in _MODE_INSTRUCTIONS:
            instruction = _MODE_INSTRUCTIONS[self._mode]
        elif self._mode == "custom":
            instruction = (
                f"{self._custom_instruction}\n"
                "Preserve the original meaning unless instructed otherwise."
            )
        elif self._mode == "audience":
            instruction = (
                f"Rewrite the following text for this audience: {self._target_audience}. "
                "Adapt vocabulary and complexity accordingly."
            )
        elif self._mode == "length":
            instruction = (
                f"Rewrite the following text to be {self._target_length}. "
                "Preserve the key information."
            )
        else:
            instruction = "Rewrite the following text."

        # Audience hint for simplify mode
        if self._mode == "simplify" and self._target_audience:
            instruction += f" Target audience: {self._target_audience}."

        # Preservation constraints
        if self._preserve:
            aspects = ", ".join(self._preserve)
            instruction += f"\n\nPreserve these aspects: {aspects}."

        return (
            f"{instruction}\n\n"
            f"Text:\n{text}\n\n"
            f"Respond with the rewritten text only, no preamble or explanation."
        )

    def _build_messages(self, record: Record) -> list[dict[str, str]]:
        """Build messages list for LLM call."""
        text = str(record.get(self._input_column, ""))
        user_text = self._build_prompt(text)

        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": user_text})
        return messages

    # -- Runner integration -------------------------------------------------

    def collect_calls(
        self,
        records: list[Record],
        skip_call_ids: set[str] | None = None,
    ) -> tuple[list[LLMCall], dict[str, LLMProvider]]:
        """Collect LLM calls for batched execution by the Runner."""
        skip_call_ids = skip_call_ids or set()
        calls: list[LLMCall] = []
        models_map: dict[str, LLMProvider] = {}

        models = _normalize_models(self._llm)

        for record_idx, record in enumerate(records):
            for model in models:
                models_map[model.model_id] = model
                for var_idx in range(self._num_variations):
                    call_id = f"{record_idx}_{model.model_id}_{var_idx}"
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
                        output_index=var_idx,
                    )
                    calls.append(call)

        return calls, models_map

    def apply_result(
        self, call: LLMCall, result: str, model: LLMProvider
    ) -> Record:
        """Convert an LLM result into an output record."""
        fields: dict[str, Any] = {self._output_column: result.strip()}
        if self._num_variations > 1:
            fields["_variation"] = call.output_index

        return _build_output_record(
            call.record,
            fields,
            model,
            self._forward_columns,
            self._exclude_columns,
        )

    # -- process (direct execution without Runner) --------------------------

    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        """Rewrite text for each record × model × variation."""
        models = _normalize_models(self._llm)
        generated = 0
        errors = 0

        for record in records:
            for model in models:
                for var_idx in range(self._num_variations):
                    try:
                        messages = self._build_messages(record)
                        raw = model.generate(messages)

                        fields: dict[str, Any] = {
                            self._output_column: raw.strip(),
                        }
                        if self._num_variations > 1:
                            fields["_variation"] = var_idx

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
                            f"Rewrite error | Model: {model.model_id} | Error: {e}"
                        )
                        if self._on_parse_error == "raise":
                            raise

        logger.info(f"Rewrite: generated {generated}, errors {errors}")
