"""Source steps for loading data from external sources."""

from collections.abc import Iterable
from pathlib import Path

from loguru import logger

from datafast_v2.core.step import Step
from datafast_v2.core.types import Record


class ListSource(Step):
    """Load data from a Python list."""

    def __init__(self, records: list[Record]) -> None:
        """
        Initialize a list source.

        Args:
            records: List of records to load.
        """
        super().__init__()
        self._records = records

    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        """Yield records from the list."""
        logger.info(f"Loading {len(self._records)} records from list")
        yield from self._records


class FileSource(Step):
    """Load data from local files (JSONL, CSV, Parquet, TSV, TXT)."""

    def __init__(
        self,
        path: str | Path,
        format: str | None = None,
        text_column: str = "text",
        **kwargs,
    ) -> None:
        """
        Initialize a file source.

        Args:
            path: Path to the file to load.
            format: File format. If None, auto-detected from extension.
                Supported: "jsonl", "csv", "parquet", "tsv", "txt"
            text_column: Column name for TXT files (each line becomes a record).
            **kwargs: Additional arguments passed to the underlying reader.
        """
        super().__init__()
        self._path = Path(path)
        self._format = format or self._detect_format()
        self._text_column = text_column
        self._kwargs = kwargs

        if self._format not in ("jsonl", "csv", "parquet", "tsv", "txt"):
            raise ValueError(
                f"Unsupported format: {self._format}. "
                "Supported formats: jsonl, csv, parquet, tsv, txt"
            )

    def _detect_format(self) -> str:
        """Detect file format from extension."""
        suffix = self._path.suffix.lower()
        format_map = {
            ".jsonl": "jsonl",
            ".json": "jsonl",
            ".csv": "csv",
            ".tsv": "tsv",
            ".parquet": "parquet",
            ".pq": "parquet",
            ".txt": "txt",
        }
        if suffix not in format_map:
            raise ValueError(
                f"Cannot auto-detect format from extension: {suffix}. "
                "Please specify format explicitly."
            )
        return format_map[suffix]

    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        """Load and yield records from the file."""
        logger.info(f"Loading data from {self._path} (format: {self._format})")

        if self._format == "jsonl":
            yield from self._load_jsonl()
        elif self._format == "csv":
            yield from self._load_csv()
        elif self._format == "tsv":
            yield from self._load_tsv()
        elif self._format == "parquet":
            yield from self._load_parquet()
        elif self._format == "txt":
            yield from self._load_txt()

    def _load_jsonl(self) -> Iterable[Record]:
        """Load records from a JSONL file."""
        import json

        with open(self._path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")

    def _load_csv(self) -> Iterable[Record]:
        """Load records from a CSV file."""
        import csv

        with open(self._path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, **self._kwargs)
            for row in reader:
                yield dict(row)

    def _load_tsv(self) -> Iterable[Record]:
        """Load records from a TSV file."""
        import csv

        with open(self._path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t", **self._kwargs)
            for row in reader:
                yield dict(row)

    def _load_txt(self) -> Iterable[Record]:
        """Load records from a TXT file (one record per line)."""
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    yield {self._text_column: line}

    def _load_parquet(self) -> Iterable[Record]:
        """Load records from a Parquet file."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "pyarrow is required for reading Parquet files. "
                "Install it with: pip install pyarrow"
            )

        table = pq.read_table(self._path, **self._kwargs)
        for batch in table.to_batches():
            for row in batch.to_pylist():
                yield row


class Source:
    """Factory class for creating source steps."""

    @staticmethod
    def file(
        path: str | Path,
        format: str | None = None,
        **kwargs,
    ) -> FileSource:
        """
        Load data from a local file.

        Args:
            path: Path to the file to load.
            format: File format. If None, auto-detected from extension.
                Supported: "jsonl", "csv", "parquet"
            **kwargs: Additional arguments passed to the underlying reader.

        Returns:
            A FileSource step.

        Examples:
            >>> Source.file("data.jsonl")
            >>> Source.file("data.csv")
            >>> Source.file("data.parquet")
        """
        return FileSource(path=path, format=format, **kwargs)

    @staticmethod
    def jsonl(path: str | Path, **kwargs) -> FileSource:
        """
        Load data from a JSONL file.

        Args:
            path: Path to the JSONL file.
            **kwargs: Additional arguments passed to the reader.

        Returns:
            A FileSource step configured for JSONL.
        """
        return FileSource(path=path, format="jsonl", **kwargs)

    @staticmethod
    def csv(path: str | Path, **kwargs) -> FileSource:
        """
        Load data from a CSV file.

        Args:
            path: Path to the CSV file.
            **kwargs: Additional arguments passed to csv.DictReader.

        Returns:
            A FileSource step configured for CSV.
        """
        return FileSource(path=path, format="csv", **kwargs)

    @staticmethod
    def parquet(path: str | Path, **kwargs) -> FileSource:
        """
        Load data from a Parquet file.

        Args:
            path: Path to the Parquet file.
            **kwargs: Additional arguments passed to pyarrow.parquet.read_table.

        Returns:
            A FileSource step configured for Parquet.
        """
        return FileSource(path=path, format="parquet", **kwargs)

    @staticmethod
    def tsv(path: str | Path, **kwargs) -> FileSource:
        """
        Load data from a TSV file.

        Args:
            path: Path to the TSV file.
            **kwargs: Additional arguments passed to csv.DictReader.

        Returns:
            A FileSource step configured for TSV.
        """
        return FileSource(path=path, format="tsv", **kwargs)

    @staticmethod
    def txt(path: str | Path, text_column: str = "text", **kwargs) -> FileSource:
        """
        Load data from a TXT file (one record per line).

        Args:
            path: Path to the TXT file.
            text_column: Column name for the text content.
            **kwargs: Additional arguments.

        Returns:
            A FileSource step configured for TXT.
        """
        return FileSource(path=path, format="txt", text_column=text_column, **kwargs)

    @staticmethod
    def list(records: list[Record]) -> ListSource:
        """
        Load data from a Python list.

        Args:
            records: List of record dictionaries.

        Returns:
            A ListSource step.

        Example:
            >>> Source.list([
            ...     {"text": "Document 1", "category": "science"},
            ...     {"text": "Document 2", "category": "history"},
            ... ])
        """
        return ListSource(records)
