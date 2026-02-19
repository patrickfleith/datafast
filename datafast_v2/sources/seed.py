"""Seed builders for creating initial records from configuration."""

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product as itertools_product

from loguru import logger

from datafast_v2.core.step import Step
from datafast_v2.core.types import Record


@dataclass
class SeedDimension:
    """A single dimension of variation for seed generation."""

    columns: list[str]
    values: list[dict[str, any]]

    def __len__(self) -> int:
        return len(self.values)


class SeedSource(Step):
    """A source step that generates records from seed dimensions."""

    def __init__(self, records: list[Record]) -> None:
        super().__init__()
        self._records = records

    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        """Yield the seed records."""
        logger.info(f"Generating {len(self._records)} seed records")
        yield from self._records

    def __len__(self) -> int:
        return len(self._records)


class Seed:
    """Factory class for building initial records from configuration."""

    @staticmethod
    def values(column: str, values: list) -> SeedDimension:
        """
        Create a dimension with explicit values.

        Args:
            column: The column name for this dimension.
            values: List of values for this dimension.

        Returns:
            A SeedDimension representing this axis of variation.

        Example:
            >>> Seed.values("language", ["en", "fr", "de"])
            # Creates 3 potential records: {"language": "en"}, {"language": "fr"}, {"language": "de"}
        """
        return SeedDimension(
            columns=[column],
            values=[{column: v} for v in values],
        )

    @staticmethod
    def expand(parent: str, child: str, mapping: dict[str, list]) -> SeedDimension:
        """
        Create a dimension from nested structure.

        Args:
            parent: The parent column name.
            child: The child column name.
            mapping: Dictionary mapping parent values to lists of child values.

        Returns:
            A SeedDimension with parent-child combinations.

        Example:
            >>> Seed.expand("topic", "subtopic", {
            ...     "Physics": ["Quantum", "Relativity"],
            ...     "Biology": ["Genetics", "Evolution"],
            ... })
            # Creates 4 potential records:
            # {"topic": "Physics", "subtopic": "Quantum"}
            # {"topic": "Physics", "subtopic": "Relativity"}
            # {"topic": "Biology", "subtopic": "Genetics"}
            # {"topic": "Biology", "subtopic": "Evolution"}
        """
        values = []
        for parent_value, child_values in mapping.items():
            for child_value in child_values:
                values.append({parent: parent_value, child: child_value})
        return SeedDimension(columns=[parent, child], values=values)

    @staticmethod
    def range(column: str, start: int, end: int, step: int = 1) -> SeedDimension:
        """
        Create a dimension from numeric range.

        Args:
            column: The column name for this dimension.
            start: Start of range (inclusive).
            end: End of range (inclusive).
            step: Step between values.

        Returns:
            A SeedDimension with numeric values.

        Example:
            >>> Seed.range("grade_level", 1, 12)
            # Creates 12 potential records with grade_level 1 through 12
        """
        values = list(range(start, end + 1, step))
        return SeedDimension(
            columns=[column],
            values=[{column: v} for v in values],
        )

    @staticmethod
    def product(*dimensions: SeedDimension) -> SeedSource:
        """
        Create cartesian product of all dimensions.

        Args:
            *dimensions: Variable number of SeedDimension objects to combine.

        Returns:
            A SeedSource step that can be used to start a pipeline.

        Example:
            >>> Seed.product(
            ...     Seed.values("persona", ["student", "teacher"]),
            ...     Seed.values("language", ["en", "fr"]),
            ... )
            # Creates 4 records (2 × 2):
            # {"persona": "student", "language": "en"}
            # {"persona": "student", "language": "fr"}
            # {"persona": "teacher", "language": "en"}
            # {"persona": "teacher", "language": "fr"}
        """
        if not dimensions:
            return SeedSource([])

        dim_values = [dim.values for dim in dimensions]
        records = []
        for combo in itertools_product(*dim_values):
            record: Record = {}
            for item in combo:
                record.update(item)
            records.append(record)

        logger.debug(
            f"Seed.product created {len(records)} records from "
            f"{len(dimensions)} dimensions"
        )
        return SeedSource(records)

    @staticmethod
    def zip(*dimensions: SeedDimension) -> SeedSource:
        """
        Zip dimensions together (must be same length).

        Args:
            *dimensions: Variable number of SeedDimension objects to zip.
                All dimensions must have the same length.

        Returns:
            A SeedSource step that can be used to start a pipeline.

        Raises:
            ValueError: If dimensions have different lengths.

        Example:
            >>> Seed.zip(
            ...     Seed.values("question", ["Q1", "Q2", "Q3"]),
            ...     Seed.values("answer", ["A1", "A2", "A3"]),
            ... )
            # Creates 3 records: (Q1,A1), (Q2,A2), (Q3,A3)
        """
        if not dimensions:
            return SeedSource([])

        lengths = [len(dim) for dim in dimensions]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"All dimensions must have the same length for zip. "
                f"Got lengths: {lengths}"
            )

        records = []
        for i in range(lengths[0]):
            record: Record = {}
            for dim in dimensions:
                record.update(dim.values[i])
            records.append(record)

        logger.debug(
            f"Seed.zip created {len(records)} records from "
            f"{len(dimensions)} dimensions"
        )
        return SeedSource(records)
