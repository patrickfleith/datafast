"""Sink classes for saving pipeline output."""

import json
from collections.abc import Iterable
from pathlib import Path

from loguru import logger

from datafast_v2.core.step import Step
from datafast_v2.core.types import Record

_DATAFAST_README_TEMPLATE = """\
---
tags:
- datafast
---
"""


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
    def parquet(path: str | Path) -> "ParquetSink":
        """Create a Parquet file sink."""
        return ParquetSink(path)

    @staticmethod
    def hub(
        repo_id: str,
        token: str | None = None,
        private: bool = True,
        train_size: float | None = None,
        seed: int = 42,
        shuffle: bool = True,
        commit_message: str | None = None,
    ) -> "HubSink":
        """
        Create a HuggingFace Hub dataset sink.

        Args:
            repo_id: HuggingFace repo ID (e.g., "username/my-dataset").
            token: HF API token. If None, uses HF_TOKEN env var or cached login.
            private: Whether to create a private dataset.
            train_size: If set (0.0–1.0), splits into train/test splits.
            seed: Random seed for train/test split.
            shuffle: Whether to shuffle before splitting.
            commit_message: Commit message for the dataset push.

        Returns:
            A HubSink step.

        Examples:
            >>> Sink.hub("username/my-dataset")
            >>> Sink.hub("username/my-dataset", train_size=0.9, private=False)
        """
        return HubSink(
            repo_id=repo_id,
            token=token,
            private=private,
            train_size=train_size,
            seed=seed,
            shuffle=shuffle,
            commit_message=commit_message,
        )

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


class ParquetSink(Step):
    """Write records to a Parquet file."""

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self._path = Path(path)

    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        """Write records to Parquet file and pass them through."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "pyarrow is required for ParquetSink. "
                "Install it with: pip install pyarrow"
            )

        self._path.parent.mkdir(parents=True, exist_ok=True)
        records_list = list(records)

        if not records_list:
            logger.warning(f"No records to save to {self._path}")
            return

        table = pa.Table.from_pylist(records_list)
        pq.write_table(table, self._path)

        logger.info(f"Saved {len(records_list)} records to {self._path}")
        yield from records_list


class HubSink(Step):
    """Push records to HuggingFace Hub as a dataset."""

    def __init__(
        self,
        repo_id: str,
        token: str | None = None,
        private: bool = True,
        train_size: float | None = None,
        seed: int = 42,
        shuffle: bool = True,
        commit_message: str | None = None,
    ) -> None:
        """
        Initialize a HuggingFace Hub sink.

        Args:
            repo_id: HuggingFace repo ID (e.g., "username/my-dataset").
            token: HF API token. If None, uses HF_TOKEN env var or cached login.
            private: Whether to create a private dataset.
            train_size: If set (0.0–1.0), splits into train/test splits.
            seed: Random seed for train/test split.
            shuffle: Whether to shuffle before splitting.
            commit_message: Commit message for the dataset push.
        """
        super().__init__()
        self._repo_id = repo_id
        self._token = token
        self._private = private
        self._train_size = train_size
        self._seed = seed
        self._shuffle = shuffle
        self._commit_message = commit_message or "Upload dataset via datafast"

    def _get_token(self) -> str | None:
        """Resolve HF token from argument or environment."""
        if self._token:
            return self._token
        import os
        return os.getenv("HF_TOKEN")

    def _ensure_readme(self, token: str | None) -> None:
        """Create README.md with datafast-dataset tag if not already present."""
        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for HubSink. "
                "Install it with: pip install huggingface-hub"
            )

        api = HfApi(token=token)

        try:
            existing = api.hf_hub_download(
                repo_id=self._repo_id,
                filename="README.md",
                repo_type="dataset",
                token=token,
            )
            with open(existing, "r", encoding="utf-8") as f:
                content = f.read()
            if "datafast-dataset" in content:
                return
            updated = _DATAFAST_README_TEMPLATE + content
        except Exception:
            updated = _DATAFAST_README_TEMPLATE

        api.upload_file(
            path_or_fileobj=updated.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=self._repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Add datafast-dataset tag",
        )

    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        """Push records to HuggingFace Hub and pass them through."""
        try:
            from datasets import Dataset, DatasetDict
        except ImportError:
            raise ImportError(
                "datasets is required for HubSink. "
                "Install it with: pip install datasets"
            )

        token = self._get_token()
        records_list = list(records)

        if not records_list:
            logger.warning(f"No records to push to {self._repo_id}")
            return

        logger.info(f"Pushing {len(records_list)} records to '{self._repo_id}'")

        dataset = Dataset.from_list(records_list)

        if self._shuffle:
            dataset = dataset.shuffle(seed=self._seed)

        if self._train_size is not None:
            if not (0.0 < self._train_size < 1.0):
                raise ValueError("train_size must be between 0.0 and 1.0 (exclusive)")
            split = dataset.train_test_split(
                train_size=self._train_size, seed=self._seed
            )
            dataset_to_push = DatasetDict({"train": split["train"], "test": split["test"]})
            logger.info(
                f"Split: {len(split['train'])} train, {len(split['test'])} test records"
            )
        else:
            dataset_to_push = dataset

        dataset_to_push.push_to_hub(
            self._repo_id,
            token=token,
            private=self._private,
            commit_message=self._commit_message,
        )

        logger.info(f"Successfully pushed to '{self._repo_id}'")

        self._ensure_readme(token)

        yield from records_list
