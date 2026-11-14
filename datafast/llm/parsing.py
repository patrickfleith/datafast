"""Output parsers for LLM responses: text, JSON, and XML modes."""

import re
from abc import ABC, abstractmethod
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field, create_model


class OutputParser(ABC):
    """Abstract base class for output parsers."""

    @abstractmethod
    def parse(self, raw_output: str, output_columns: list[str]) -> dict[str, str]:
        """
        Parse raw LLM output into a dictionary of column values.

        Args:
            raw_output: The raw text output from the LLM.
            output_columns: List of expected output column names.

        Returns:
            Dictionary mapping column names to extracted values.
        """
        pass

    @abstractmethod
    def get_format_instructions(self, output_columns: list[str]) -> str:
        """
        Get format instructions to append to the prompt.

        Args:
            output_columns: List of expected output column names.

        Returns:
            Instructions string to guide LLM output format.
        """
        pass


class TextParser(OutputParser):
    """Direct text output parser - writes LLM output to a single column."""

    def parse(self, raw_output: str, output_columns: list[str]) -> dict[str, str]:
        """Return the raw output mapped to the first output column."""
        if not output_columns:
            return {"generated": raw_output.strip()}
        return {output_columns[0]: raw_output.strip()}

    def get_format_instructions(self, output_columns: list[str]) -> str:
        """No special instructions needed for text mode."""
        return ""


class JSONParser(OutputParser):
    """Parse structured JSON output into multiple columns."""

    def __init__(self, strip_code_fences: bool = True) -> None:
        """
        Initialize the JSON parser.

        Args:
            strip_code_fences: Whether to strip markdown code fences from output.
        """
        self._strip_code_fences = strip_code_fences

    def _strip_fences(self, content: str) -> str:
        """Strip markdown code fences if present."""
        if not content:
            return content

        content = content.strip()

        if content.startswith("```"):
            first_newline = content.find("\n")
            if first_newline != -1:
                content = content[first_newline + 1:]
            else:
                content = content[3:]

        if content.endswith("```"):
            content = content[:-3]

        return content.strip()

    def _create_response_model(self, output_columns: list[str]) -> type[BaseModel]:
        """Dynamically create a Pydantic model for the expected output columns."""
        fields: dict[str, Any] = {}
        for col in output_columns:
            fields[col] = (str, Field(..., description=f"Value for {col}"))
        return create_model("LLMResponse", **fields)

    def parse(self, raw_output: str, output_columns: list[str]) -> dict[str, str]:
        """
        Parse JSON output into column values.

        Args:
            raw_output: Raw LLM output (potentially with code fences).
            output_columns: Expected column names.

        Returns:
            Dictionary mapping column names to values.

        Raises:
            ValueError: If JSON parsing fails.
        """
        import json

        content = raw_output
        if self._strip_code_fences:
            content = self._strip_fences(content)

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e} | Content: {content[:200]}")
            raise ValueError(f"Failed to parse JSON: {e}") from e

        result: dict[str, str] = {}
        for col in output_columns:
            if col in data:
                value = data[col]
                result[col] = str(value) if not isinstance(value, str) else value
            else:
                logger.warning(f"Missing column in JSON output: {col}")
                result[col] = ""

        return result

    def get_format_instructions(self, output_columns: list[str]) -> str:
        """Generate JSON format instructions."""
        fields = ", ".join(f'"{col}": "<{col} value>"' for col in output_columns)
        return (
            f"\n\nRespond with valid JSON containing these fields: {{{fields}}}\n"
            "Return only the JSON object, no additional text or markdown code fences."
        )


class XMLParser(OutputParser):
    """Parse XML tag-based output into multiple columns."""

    def parse(self, raw_output: str, output_columns: list[str]) -> dict[str, str]:
        """
        Extract content from XML-style tags.

        Args:
            raw_output: Raw LLM output with <tag>content</tag> patterns.
            output_columns: Expected column/tag names.

        Returns:
            Dictionary mapping column names to extracted content.
        """
        result: dict[str, str] = {}

        for col in output_columns:
            pattern = rf"<{re.escape(col)}>(.*?)</{re.escape(col)}>"
            match = re.search(pattern, raw_output, re.DOTALL | re.IGNORECASE)

            if match:
                result[col] = match.group(1).strip()
            else:
                logger.warning(f"Missing XML tag in output: <{col}>")
                result[col] = ""

        return result

    def get_format_instructions(self, output_columns: list[str]) -> str:
        """Generate XML format instructions."""
        tags = "\n".join(f"<{col}>your {col} here</{col}>" for col in output_columns)
        return (
            f"\n\nRespond using these XML tags:\n{tags}\n"
            "Include all tags in your response."
        )


def get_parser(parse_mode: str) -> OutputParser:
    """
    Get the appropriate parser for the given mode.

    Args:
        parse_mode: One of "text", "json", "xml".

    Returns:
        Configured OutputParser instance.

    Raises:
        ValueError: If parse_mode is invalid.
    """
    parsers = {
        "text": TextParser,
        "json": JSONParser,
        "xml": XMLParser,
    }

    if parse_mode not in parsers:
        raise ValueError(
            f"Invalid parse_mode: {parse_mode}. "
            f"Valid modes: {list(parsers.keys())}"
        )

    return parsers[parse_mode]()
