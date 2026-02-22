"""LLM Extraction step: Extract structured information from unstructured text."""

import json
from collections.abc import Callable, Iterable
from typing import Any

from loguru import logger

from datafast_v2.core.config import LLMCall
from datafast_v2.core.step import Step
from datafast_v2.core.types import Record
from datafast_v2.llm.provider import LLMProvider
from datafast_v2.transforms.llm_eval import (
    _build_output_record,
    _normalize_models,
    _strip_json_fences,
)
from datafast_v2.transforms.sample import Sample

# ---------------------------------------------------------------------------
# Predefined extractors
# ---------------------------------------------------------------------------

_PREDEFINED_EXTRACTORS: dict[str, dict[str, Any]] = {
    "entities": {
        "fields": {
            "persons": "List of person names mentioned",
            "organizations": "List of organization/company names mentioned",
            "locations": "List of geographic locations mentioned",
            "dates": "List of dates or time references mentioned",
        },
        "instruction": (
            "Extract all named entities from the following text. "
            "For each category, return a JSON list of strings. "
            "If no entities are found for a category, return an empty list."
        ),
    },
    "keywords": {
        "fields": {
            "keywords": "List of the most important keywords and key phrases",
        },
        "instruction": (
            "Extract the most important keywords and key phrases from "
            "the following text. Return them as a JSON list of strings, "
            "ordered by importance."
        ),
    },
    "topics": {
        "fields": {
            "topics": "List of main topics discussed",
        },
        "instruction": (
            "Identify the main topics discussed in the following text. "
            "Return them as a JSON list of concise topic labels."
        ),
    },
    "facts": {
        "fields": {
            "facts": "List of factual claims or statements",
        },
        "instruction": (
            "Extract all factual claims and key statements from the "
            "following text. Return them as a JSON list of strings, "
            "each being a self-contained factual statement."
        ),
    },
    "metadata": {
        "fields": {
            "title": "Title or headline (empty string if not found)",
            "author": "Author name (empty string if not found)",
            "date": "Publication date (empty string if not found)",
            "source": "Source or publisher (empty string if not found)",
            "language": "Language of the text",
        },
        "instruction": (
            "Extract document metadata from the following text. "
            "If a field cannot be determined, return an empty string."
        ),
    },
    "summary_fields": {
        "fields": {
            "title": "A concise title for the text",
            "summary": "A 1-2 sentence summary",
            "key_points": "List of main points",
        },
        "instruction": (
            "Summarise the following text by extracting a title, "
            "a brief summary, and the key points."
        ),
    },
}

VALID_EXTRACTORS = frozenset(_PREDEFINED_EXTRACTORS.keys())


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------


class Extract(Step):
    """Pull structured information from unstructured text.

    Supports three extraction approaches:
        - **Custom fields** (``fields``): Define exactly which fields to extract
          with descriptions.
        - **Predefined extractor** (``extractor``): Use a built-in extraction
          template (``"entities"``, ``"keywords"``, ``"topics"``, ``"facts"``,
          ``"metadata"``, ``"summary_fields"``).
        - **Function-based** (``fn``): Use a Python callable for programmatic
          extraction.

    In LLM mode, integrates with the Runner for batched execution and
    checkpointing via ``collect_calls`` / ``apply_result``.

    When ``flatten=True``, each extracted field becomes its own column in the
    output record. When ``flatten=False`` (default), the entire extraction
    result is stored as a dict in a single ``output_column``.

    Examples:
        >>> # Extract custom fields from product descriptions
        >>> Extract(
        ...     input_column="product_description",
        ...     fields={
        ...         "product_name": "The name of the product",
        ...         "price": "Price in dollars (number only)",
        ...         "features": "List of key features",
        ...     },
        ...     flatten=True,
        ...     llm=openrouter("openai/gpt-4o-mini"),
        ... )

        >>> # Named entity extraction
        >>> Extract(
        ...     input_column="news_article",
        ...     extractor="entities",
        ...     llm=gpt4,
        ... )

        >>> # Keyword extraction with flattening
        >>> Extract(
        ...     input_column="document",
        ...     extractor="keywords",
        ...     flatten=True,
        ...     llm=gpt4,
        ... )

        >>> # Function-based extraction
        >>> import re
        >>> Extract(
        ...     input_column="text",
        ...     fn=lambda r: {
        ...         "emails": re.findall(r"[\\w.]+@[\\w.]+", r["text"]),
        ...         "urls": re.findall(r"https?://\\S+", r["text"]),
        ...     },
        ...     flatten=True,
        ... )
    """

    def __init__(
        self,
        input_column: str,
        *,
        fields: dict[str, str] | None = None,
        extractor: str | None = None,
        fn: Callable[[Record], dict[str, Any]] | None = None,
        flatten: bool = False,
        output_column: str = "extracted",
        llm: LLMProvider | list[LLMProvider] | Sample | None = None,
        prompt: str | None = None,
        system_prompt: str | None = None,
        forward_columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
        on_parse_error: str = "skip",
    ) -> None:
        """
        Initialize an Extract step.

        Args:
            input_column: Column containing text to extract from.
            fields: Mapping of ``field_name`` → ``description`` for custom
                extraction. The LLM will be asked to populate each field.
            extractor: Name of a predefined extractor. One of:
                ``"entities"``, ``"keywords"``, ``"topics"``, ``"facts"``,
                ``"metadata"``, ``"summary_fields"``.
            fn: Callable for function-based extraction. Receives a record and
                must return a ``dict[str, Any]``.
            flatten: If ``True``, each extracted field becomes a separate
                column. If ``False``, the full dict is stored in
                ``output_column``.
            output_column: Column name for the extraction result when
                ``flatten=False``. Ignored when ``flatten=True``.
            llm: LLM provider(s) for LLM-based extraction.
            prompt: Custom prompt override for LLM mode. Use ``{content}``
                as placeholder for the input text and ``{fields}`` for the
                field descriptions.
            system_prompt: System prompt prepended in LLM mode.
            forward_columns: Columns to keep from input (LLM mode).
            exclude_columns: Columns to drop from input (LLM mode).
            on_parse_error: ``"skip"`` or ``"raise"`` on LLM parse failure.
        """
        super().__init__()

        # Validate extraction method
        methods_given = sum([fields is not None, extractor is not None, fn is not None])
        if methods_given == 0:
            raise ValueError(
                "Extract requires one of 'fields', 'extractor', or 'fn'"
            )
        if methods_given > 1:
            raise ValueError(
                "Extract accepts only one of 'fields', 'extractor', or 'fn'"
            )

        if extractor is not None and extractor not in VALID_EXTRACTORS:
            raise ValueError(
                f"Invalid extractor: {extractor}. "
                f"Valid extractors: {sorted(VALID_EXTRACTORS)}"
            )

        if fn is None and llm is None:
            raise ValueError("LLM-based Extract requires 'llm' to be set")
        if fn is not None and llm is not None:
            raise ValueError(
                "Function-based Extract does not use 'llm'; pass only 'fn'"
            )

        if on_parse_error not in ("skip", "raise"):
            raise ValueError("on_parse_error must be 'skip' or 'raise'")

        self._input_column = input_column
        self._flatten = flatten
        self._output_column = output_column
        self._fn = fn
        self._llm = llm
        self._prompt = prompt
        self._system_prompt = system_prompt
        self._forward_columns = forward_columns
        self._exclude_columns = exclude_columns
        self._on_parse_error = on_parse_error

        # Resolve fields from extractor or direct specification
        if extractor is not None:
            preset = _PREDEFINED_EXTRACTORS[extractor]
            self._fields: dict[str, str] = preset["fields"]
            self._base_instruction: str = preset["instruction"]
        elif fields is not None:
            self._fields = fields
            self._base_instruction = (
                "Extract the following fields from the text below. "
                "Follow the description for each field carefully."
            )
        else:
            # fn mode — fields not used
            self._fields = {}
            self._base_instruction = ""

    @property
    def uses_llm(self) -> bool:
        """Whether this step uses LLM (vs function-based) mode."""
        return self._llm is not None

    # -- prompt construction ------------------------------------------------

    def _format_field_descriptions(self) -> str:
        """Format field names and descriptions for the prompt."""
        lines = []
        for name, desc in self._fields.items():
            lines.append(f'- "{name}": {desc}')
        return "\n".join(lines)

    def _build_json_example(self) -> str:
        """Build a JSON example showing expected output structure."""
        example: dict[str, str] = {}
        for name in self._fields:
            example[name] = f"<{name} value>"
        return json.dumps(example, indent=2)

    def _build_prompt(self, text: str) -> str:
        """Build the extraction prompt for the given text."""
        field_desc = self._format_field_descriptions()
        json_example = self._build_json_example()

        return (
            f"{self._base_instruction}\n\n"
            f"Fields to extract:\n{field_desc}\n\n"
            f"Text:\n{text}\n\n"
            f"Respond with valid JSON matching this structure:\n"
            f"{json_example}\n\n"
            "Return only the JSON object, no additional text or markdown "
            "code fences."
        )

    def _build_messages(self, record: Record) -> list[dict[str, str]]:
        """Build messages list for LLM call."""
        text = str(record.get(self._input_column, ""))

        if self._prompt:
            user_text = self._prompt.format(
                content=text,
                fields=self._format_field_descriptions(),
            )
        else:
            user_text = self._build_prompt(text)

        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": user_text})
        return messages

    # -- result parsing -----------------------------------------------------

    def _parse_llm_result(self, raw: str) -> dict[str, Any]:
        """Parse LLM JSON response into extracted fields."""
        cleaned = _strip_json_fences(raw)
        data = json.loads(cleaned)

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected JSON object, got {type(data).__name__}"
            )

        result: dict[str, Any] = {}
        for field_name in self._fields:
            result[field_name] = data.get(field_name, "")
        return result

    def _apply_extracted(
        self,
        input_record: Record,
        extracted: dict[str, Any],
        model: LLMProvider | None,
    ) -> Record:
        """Build output record from extracted fields."""
        if self._flatten:
            new_fields = extracted
        else:
            new_fields = {self._output_column: extracted}

        return _build_output_record(
            input_record,
            new_fields,
            model,
            self._forward_columns,
            self._exclude_columns,
        )

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
        extracted = self._parse_llm_result(result)
        return self._apply_extracted(call.record, extracted, model)

    # -- process (direct execution) -----------------------------------------

    def _process_fn(self, records: Iterable[Record]) -> Iterable[Record]:
        """Function-based extraction."""
        processed = 0
        for record in records:
            extracted = self._fn(record)
            output = self._apply_extracted(record, extracted, model=None)
            processed += 1
            yield output
        logger.info(f"Extract (fn): processed {processed} records")

    def _process_llm(self, records: Iterable[Record]) -> Iterable[Record]:
        """LLM-based extraction (direct, without Runner batching)."""
        models = _normalize_models(self._llm)
        generated = 0
        errors = 0

        for record in records:
            for model in models:
                try:
                    messages = self._build_messages(record)
                    raw = model.generate(messages)
                    extracted = self._parse_llm_result(raw)
                    output = self._apply_extracted(record, extracted, model)
                    generated += 1
                    yield output
                except Exception as e:
                    errors += 1
                    logger.warning(
                        f"Extract error | Model: {model.model_id} | Error: {e}"
                    )
                    if self._on_parse_error == "raise":
                        raise

        logger.info(f"Extract (llm): generated {generated}, errors {errors}")

    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        """Extract information from records using either fn or LLM mode."""
        if self._fn is not None:
            yield from self._process_fn(records)
        else:
            yield from self._process_llm(records)
