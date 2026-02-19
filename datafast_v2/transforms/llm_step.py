"""LLMStep - Core step for LLM-based generation in pipelines."""

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from loguru import logger

from datafast_v2.core.config import LLMCall
from datafast_v2.core.step import Step
from datafast_v2.core.types import Record
from datafast_v2.llm.parsing import get_parser, OutputParser
from datafast_v2.llm.provider import LLMProvider
from datafast_v2.transforms.sample import Sample


class LLMStep(Step):
    """
    Core step for free-form LLM generation.

    For each input record, LLMStep generates outputs for:
        all prompts × all models × all languages × num_outputs

    Supports three output parsing modes:
        - "text": Direct write of LLM output to single column
        - "json": Parse structured JSON into multiple columns
        - "xml": Parse <tag>content</tag> patterns into columns

    Examples:
        >>> # Simple text generation
        >>> LLMStep(
        ...     prompt="Summarize: {text}",
        ...     input_columns=["text"],
        ...     output_column="summary",
        ...     model=openrouter("openai/gpt-4o-mini"),
        ... )

        >>> # Structured JSON output
        >>> LLMStep(
        ...     prompt="Generate a Q&A about: {text}",
        ...     input_columns=["text"],
        ...     output_columns=["question", "answer"],
        ...     parse_mode="json",
        ...     model=ollama("llama3"),
        ... )

        >>> # Multiple prompts and models
        >>> LLMStep(
        ...     prompt=["Summarize: {text}", "List key points: {text}"],
        ...     input_columns=["text"],
        ...     output_column="result",
        ...     model=[model1, model2],
        ...     num_outputs=2,
        ... )
    """

    VALID_PARSE_MODES = frozenset(["text", "json", "xml"])

    def __init__(
        self,
        prompt: str | Path | list[str | Path] | Sample,
        input_columns: list[str],
        model: LLMProvider | list[LLMProvider] | Sample,
        *,
        output_columns: list[str] | None = None,
        output_column: str = "generated",
        parse_mode: str = "text",
        temperature: float | None = None,
        max_tokens: int | None = None,
        num_outputs: int = 1,
        forward_columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
        language: str | list[str] | dict[str, str] | Sample | None = None,
        skip_if: Callable[[Record], bool] | None = None,
        system_prompt: str | None = None,
        on_parse_error: str = "skip",
    ) -> None:
        """
        Initialize an LLMStep.

        Args:
            prompt: Prompt template(s) with {field} placeholders. Can also be a
                filepath (or list of filepaths) to load the prompt content.
            input_columns: Columns to inject into prompt template.
            model: LLM provider(s) to use for generation.
            output_columns: Output column names (for json/xml modes).
            output_column: Single output column name (for text mode).
            parse_mode: Output parsing mode: "text", "json", or "xml".
            temperature: Override model temperature for this step.
            max_tokens: Override model max_tokens for this step.
            num_outputs: Number of outputs per prompt×model×language combo.
            forward_columns: Columns to keep from input record.
            exclude_columns: Columns to drop from input record.
            language: Language(s) for generation. If dict, keys are codes,
                values are names (adds {language} and {language_name} to context).
            skip_if: Function returning True to skip a record.
            system_prompt: System prompt to prepend to messages.
            on_parse_error: Action on parse failure: "skip" or "raise".
        """
        super().__init__()

        if parse_mode not in self.VALID_PARSE_MODES:
            raise ValueError(
                f"Invalid parse_mode: {parse_mode}. "
                f"Valid modes: {sorted(self.VALID_PARSE_MODES)}"
            )

        if parse_mode in ("json", "xml") and not output_columns:
            raise ValueError(
                f"output_columns required for parse_mode='{parse_mode}'"
            )

        if on_parse_error not in ("skip", "raise"):
            raise ValueError("on_parse_error must be 'skip' or 'raise'")

        self._prompt = prompt
        self._input_columns = input_columns
        self._model = model
        self._output_columns = output_columns or [output_column]
        self._output_column = output_column
        self._parse_mode = parse_mode
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._num_outputs = num_outputs
        self._forward_columns = forward_columns
        self._exclude_columns = exclude_columns
        self._language = language
        self._skip_if = skip_if
        self._system_prompt = system_prompt
        self._on_parse_error = on_parse_error

        self._parser: OutputParser = get_parser(parse_mode)
        self._prompt_cache: dict[str, str] = {}

    def _load_prompt_if_file(self, prompt: str | Path) -> str:
        """Load prompt content if the value is a file path."""
        path = Path(prompt)
        if not path.is_file():
            return str(prompt)

        cache_key = str(path)
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]

        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.error(f"Failed to read prompt file: {path} | Error: {exc}")
            raise

        self._prompt_cache[cache_key] = content
        return content

    def _normalize_prompts(self, record: Record) -> list[str]:
        """Get list of prompts, resolving Sample if needed."""
        if isinstance(self._prompt, Sample):
            return [self._load_prompt_if_file(item) for item in self._prompt.pick()]
        if isinstance(self._prompt, list):
            return [self._load_prompt_if_file(item) for item in self._prompt]
        return [self._load_prompt_if_file(self._prompt)]

    def _normalize_models(self, record: Record) -> list[LLMProvider]:
        """Get list of models, resolving Sample if needed."""
        if isinstance(self._model, Sample):
            return list(self._model.pick())
        elif isinstance(self._model, list):
            return self._model
        else:
            return [self._model]

    def _normalize_languages(self, record: Record) -> list[tuple[str, str]]:
        """
        Get list of (code, name) tuples for languages.

        Returns:
            List of (language_code, language_name) tuples.
        """
        if self._language is None:
            return [("", "")]

        if isinstance(self._language, Sample):
            items = list(self._language.pick())
            if items and isinstance(items[0], tuple):
                return items
            return [(str(item), str(item)) for item in items]

        if isinstance(self._language, dict):
            return list(self._language.items())

        if isinstance(self._language, list):
            return [(lang, lang) for lang in self._language]

        return [(self._language, self._language)]

    def _build_context(
        self,
        record: Record,
        language_code: str,
        language_name: str,
    ) -> dict[str, Any]:
        """Build prompt context from record and language."""
        context: dict[str, Any] = {}

        for col in self._input_columns:
            context[col] = record.get(col, "")

        if language_code:
            context["language"] = language_code
            context["language_name"] = language_name

        return context

    def _build_messages(
        self,
        prompt: str,
        context: dict[str, Any],
    ) -> list[dict[str, str]]:
        """Build messages list for LLM call."""
        formatted_prompt = prompt.format(**context)

        format_instructions = self._parser.get_format_instructions(
            self._output_columns
        )
        if format_instructions:
            formatted_prompt += format_instructions

        messages: list[dict[str, str]] = []

        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        messages.append({"role": "user", "content": formatted_prompt})

        return messages

    def _build_output_record(
        self,
        input_record: Record,
        parsed_output: dict[str, str],
        model: LLMProvider,
        prompt_index: int,
        language_code: str,
    ) -> Record:
        """Build output record with forwarded columns and metadata."""
        output: Record = {}

        if self._forward_columns is not None:
            for col in self._forward_columns:
                if col in input_record:
                    output[col] = input_record[col]
        elif self._exclude_columns is not None:
            exclude_set = set(self._exclude_columns)
            for col, value in input_record.items():
                if col not in exclude_set:
                    output[col] = value
        else:
            output.update(input_record)

        output.update(parsed_output)

        output["_model"] = model.model_id
        if prompt_index > 0 or isinstance(self._prompt, list):
            output["_prompt_index"] = prompt_index
        if language_code:
            output["_language"] = language_code

        return output

    def collect_calls(
        self,
        records: list[Record],
        skip_call_ids: set[str] | None = None,
    ) -> tuple[list[LLMCall], dict[str, LLMProvider]]:
        """
        Collect all LLM calls without executing them.

        This method is used by the Runner for batching and reordering calls.

        Args:
            records: List of input records.
            skip_call_ids: Set of call IDs to skip (for resume).

        Returns:
            Tuple of (list of LLMCall objects, dict mapping model_id to provider).
        """
        skip_call_ids = skip_call_ids or set()
        calls: list[LLMCall] = []
        models_map: dict[str, LLMProvider] = {}

        for record_idx, record in enumerate(records):
            if self._skip_if and self._skip_if(record):
                continue

            prompts = self._normalize_prompts(record)
            models = self._normalize_models(record)
            languages = self._normalize_languages(record)

            for prompt_idx, prompt_template in enumerate(prompts):
                for model in models:
                    models_map[model.model_id] = model

                    for lang_code, lang_name in languages:
                        context = self._build_context(record, lang_code, lang_name)

                        for output_idx in range(self._num_outputs):
                            call_id = f"{record_idx}_{prompt_idx}_{model.model_id}_{lang_code}_{output_idx}"

                            if call_id in skip_call_ids:
                                continue

                            messages = self._build_messages(prompt_template, context)

                            call = LLMCall(
                                call_id=call_id,
                                record=record,
                                record_index=record_idx,
                                prompt_template=prompt_template,
                                prompt_index=prompt_idx,
                                model_id=model.model_id,
                                language_code=lang_code,
                                language_name=lang_name,
                                messages=messages,
                                output_index=output_idx,
                            )
                            calls.append(call)

        return calls, models_map

    def apply_result(self, call: LLMCall, result: str, model: LLMProvider) -> Record:
        """
        Convert an LLM result into an output record.

        Args:
            call: The LLMCall that was executed.
            result: Raw LLM output text.
            model: The LLM provider that was used.

        Returns:
            Processed output record.

        Raises:
            ValueError: If parsing fails and on_parse_error is "raise".
        """
        parsed = self._parser.parse(result, self._output_columns)
        return self._build_output_record(
            call.record,
            parsed,
            model,
            call.prompt_index,
            call.language_code,
        )

    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        """
        Process records through LLM generation.

        For each input record, generates:
            all prompts × all models × all languages × num_outputs

        Yields:
            Output records with generated content and metadata.
        """
        total_generated = 0
        total_skipped = 0
        total_errors = 0

        for record in records:
            if self._skip_if and self._skip_if(record):
                total_skipped += 1
                continue

            prompts = self._normalize_prompts(record)
            models = self._normalize_models(record)
            languages = self._normalize_languages(record)

            for prompt_idx, prompt_template in enumerate(prompts):
                for model in models:
                    for lang_code, lang_name in languages:
                        context = self._build_context(record, lang_code, lang_name)

                        for _ in range(self._num_outputs):
                            try:
                                messages = self._build_messages(prompt_template, context)

                                raw_output = model.generate(messages)

                                parsed = self._parser.parse(
                                    raw_output, self._output_columns
                                )

                                output_record = self._build_output_record(
                                    record,
                                    parsed,
                                    model,
                                    prompt_idx,
                                    lang_code,
                                )

                                total_generated += 1
                                yield output_record

                            except Exception as e:
                                total_errors += 1
                                logger.warning(
                                    f"LLMStep error | Model: {model.model_id} | "
                                    f"Error: {e}"
                                )
                                if self._on_parse_error == "raise":
                                    raise

        logger.info(
            f"LLMStep: generated {total_generated}, "
            f"skipped {total_skipped}, errors {total_errors}"
        )
