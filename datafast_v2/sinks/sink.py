"""Sink classes for saving pipeline output."""

import json
from collections.abc import Iterable
from pathlib import Path

from loguru import logger

from datafast_v2.core.step import Step
from datafast_v2.core.types import Record


class Sink(Step):
    """Factory class for creating sink steps."""

    @staticmethod
    def jsonl(path: str | Path) -> "JSONLSink":
        """Create a JSONL file sink."""
        return JSONLSink(path)

    @staticmethod
    def csv(path: str | Path) -> "CSVSink":
        """Create a CSV file sink."""
        return CSVSink(path)

    @staticmethod
    def list() -> "ListSink":
        """Create a list sink that collects records in memory."""
        return ListSink()


class JSONLSink(Step):
    """Write records to a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self._path = Path(path)

    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        """Write records to JSONL file and pass them through."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        count = 0

        with open(self._path, "w") as f:
            for record in records:
                f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
                count += 1
                yield record

        logger.info(f"Saved {count} records to {self._path}")


class CSVSink(Step):
    """Write records to a CSV file."""

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self._path = Path(path)

    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        """Write records to CSV file and pass them through."""
        import csv

        self._path.parent.mkdir(parents=True, exist_ok=True)
        records_list = list(records)

        if not records_list:
            logger.warning(f"No records to save to {self._path}")
            return

        fieldnames = list(records_list[0].keys())

        with open(self._path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in records_list:
                writer.writerow(record)
                yield record

        logger.info(f"Saved {len(records_list)} records to {self._path}")


class ListSink(Step):
    """Collect records into a list (for testing)."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[Record] = []

    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        """Collect records and pass them through."""
        for record in records:
            self.records.append(record)
            yield record

        logger.info(f"Collected {len(self.records)} records")
