"""Data operation steps: Map, FlatMap, Filter, Group, Pair, Concat."""

import itertools
import random
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

from loguru import logger

from datafast_v2.core.step import Step
from datafast_v2.core.types import Record


class Map(Step):
    """Transform each record one-to-one."""

    def __init__(self, fn: Callable[[Record], Record]) -> None:
        """
        Initialize a Map step.

        Args:
            fn: Function that takes a record and returns a transformed record.

        Example:
            >>> Map(lambda r: {**r, "text_length": len(r["text"])})
            >>> Map(lambda r: {"id": r["id"], "content": r["text"].upper()})
        """
        super().__init__()
        self._fn = fn

    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        """Apply the transformation to each record."""
        for record in records:
            yield self._fn(record)


class FlatMap(Step):
    """Transform each record into zero or more records."""

    def __init__(self, fn: Callable[[Record], list[Record]]) -> None:
        """
        Initialize a FlatMap step.

        Args:
            fn: Function that takes a record and returns a list of records.

        Example:
            >>> # Explode a list field
            >>> FlatMap(lambda r: [{"id": r["id"], "q": q} for q in r["questions"]])

            >>> # Duplicate with variations
            >>> FlatMap(lambda r: [{**r, "style": "formal"}, {**r, "style": "casual"}])
        """
        super().__init__()
        self._fn = fn

    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        """Apply the transformation and flatten results."""
        for record in records:
            yield from self._fn(record)


class Filter(Step):
    """Keep or drop records based on conditions."""

    TYPE_MAP = {
        "list": list,
        "dict": dict,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "none": type(None),
    }

    def __init__(
        self,
        fn: Callable[[Record], bool] | None = None,
        where: dict | None = None,
        keep: bool = True,
    ) -> None:
        """
        Initialize a Filter step.

        Args:
            fn: Function that returns True for records to keep (or drop if keep=False).
            where: Declarative filter conditions using MongoDB-style operators.
            keep: If True, keep matching records. If False, drop matching records.

        Examples:
            >>> # Function-based
            >>> Filter(fn=lambda r: len(r["text"]) > 100)

            >>> # Declarative
            >>> Filter(where={"score": {"$gte": 7}, "category": "science"})
            >>> Filter(where={"text": {"$len_gt": 100, "$len_lt": 5000}})
            >>> Filter(where={"category": {"$in": ["science", "tech"]}})

            >>> # Drop instead of keep
            >>> Filter(where={"quality": {"$lt": 3}}, keep=False)

            >>> # Logical operators
            >>> Filter(where={"$or": [{"category": "science"}, {"score": {"$gte": 9}}]})
        """
        super().__init__()

        if fn is None and where is None:
            raise ValueError("Must specify either 'fn' or 'where'")
        if fn is not None and where is not None:
            raise ValueError("Cannot specify both 'fn' and 'where'")

        self._fn = fn
        self._where = where
        self._keep = keep

    def _evaluate(self, record: Record) -> bool:
        """Evaluate whether a record matches the filter condition."""
        if self._fn is not None:
            return self._fn(record)
        return self._evaluate_where(record, self._where)

    def _evaluate_where(self, record: Record, condition: dict) -> bool:
        """Evaluate a declarative where condition."""
        if "$or" in condition:
            return any(
                self._evaluate_where(record, sub) for sub in condition["$or"]
            )

        if "$and" in condition:
            return all(
                self._evaluate_where(record, sub) for sub in condition["$and"]
            )

        for key, value in condition.items():
            if key.startswith("$"):
                continue

            field_value = record.get(key)

            if isinstance(value, dict):
                if not self._evaluate_operators(field_value, value):
                    return False
            else:
                if field_value != value:
                    return False

        return True

    def _evaluate_operators(self, field_value: Any, operators: dict) -> bool:
        """Evaluate operator conditions on a field value."""
        for op, expected in operators.items():
            if not self._evaluate_operator(field_value, op, expected):
                return False
        return True

    def _evaluate_operator(self, value: Any, op: str, expected: Any) -> bool:
        """Evaluate a single operator."""
        if op == "$eq":
            return value == expected

        if op == "$ne":
            return value != expected

        if op == "$gt":
            return value is not None and value > expected

        if op == "$gte":
            return value is not None and value >= expected

        if op == "$lt":
            return value is not None and value < expected

        if op == "$lte":
            return value is not None and value <= expected

        if op == "$in":
            return value in expected

        if op == "$nin":
            return value not in expected

        if op == "$contains":
            return value is not None and expected in value

        if op == "$startswith":
            return value is not None and isinstance(value, str) and value.startswith(expected)

        if op == "$endswith":
            return value is not None and isinstance(value, str) and value.endswith(expected)

        if op == "$regex":
            return value is not None and isinstance(value, str) and bool(re.search(expected, value))

        if op == "$len_gt":
            return value is not None and hasattr(value, "__len__") and len(value) > expected

        if op == "$len_lt":
            return value is not None and hasattr(value, "__len__") and len(value) < expected

        if op == "$len_eq":
            return value is not None and hasattr(value, "__len__") and len(value) == expected

        if op == "$len_gte":
            return value is not None and hasattr(value, "__len__") and len(value) >= expected

        if op == "$len_lte":
            return value is not None and hasattr(value, "__len__") and len(value) <= expected

        if op == "$exists":
            return (value is not None) == expected

        if op == "$type":
            expected_type = self.TYPE_MAP.get(expected.lower() if isinstance(expected, str) else expected)
            if expected_type is None:
                raise ValueError(f"Unknown type: {expected}")
            return isinstance(value, expected_type)

        if op == "$all":
            if not isinstance(value, (list, tuple, set)):
                return False
            return all(item in value for item in expected)

        if op == "$any":
            if not isinstance(value, (list, tuple, set)):
                return False
            return any(item in value for item in expected)

        raise ValueError(f"Unknown operator: {op}")

    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        """Filter records based on the condition."""
        kept = 0
        dropped = 0

        for record in records:
            match = self._evaluate(record)
            if match == self._keep:
                kept += 1
                yield record
            else:
                dropped += 1

        logger.info(f"Filter: kept {kept}, dropped {dropped}")


class Group(Step):
    """Aggregate records by key columns."""

    AGG_FUNCTIONS = frozenset([
        "count", "sum", "mean", "min", "max", "first", "last", "collect", "concat"
    ])

    def __init__(
        self,
        by: str | list[str],
        collect: str | list[str] | None = None,
        output_column: str | None = None,
        agg: dict[str, str] | None = None,
        min_per_group: int | None = None,
        max_per_group: int | None = None,
    ) -> None:
        """
        Initialize a Group step.

        Args:
            by: Column(s) to group by.
            collect: Column(s) to collect into lists (creates {col}_list columns).
            output_column: Custom name for collected list (only if collect is single column).
            agg: Aggregations as {"new_col": "source_col:func"}.
                 Functions: count, sum, mean, min, max, first, last, collect, concat.
            min_per_group: Drop groups with fewer records than this.
            max_per_group: Limit records per group before aggregating.

        Examples:
            >>> # Collect chunks per document
            >>> Group(by="document_id", collect="text")
            >>> # Output: document_id, text_list

            >>> # Multiple aggregations
            >>> Group(
            ...     by="product_id",
            ...     collect="review",
            ...     agg={"avg_rating": "rating:mean", "num_reviews": "rating:count"},
            ... )

            >>> # Filter by group size
            >>> Group(by="doc_id", collect="chunk", min_per_group=3)
        """
        super().__init__()

        self._by = [by] if isinstance(by, str) else list(by)
        self._collect = None
        if collect is not None:
            self._collect = [collect] if isinstance(collect, str) else list(collect)
        self._output_column = output_column
        self._agg = agg or {}
        self._min_per_group = min_per_group
        self._max_per_group = max_per_group

        self._validate_agg()

    def _validate_agg(self) -> None:
        """Validate aggregation specifications."""
        for new_col, spec in self._agg.items():
            parts = spec.split(":")
            if len(parts) < 2:
                raise ValueError(
                    f"Invalid aggregation spec '{spec}'. "
                    f"Expected format: 'column:function' or 'column:concat:separator'"
                )
            func = parts[1]
            if func not in self.AGG_FUNCTIONS:
                raise ValueError(
                    f"Unknown aggregation function '{func}'. "
                    f"Valid functions: {sorted(self.AGG_FUNCTIONS)}"
                )

    def _get_group_key(self, record: Record) -> tuple:
        """Extract group key from record."""
        return tuple(record.get(col) for col in self._by)

    def _apply_aggregation(
        self, func: str, values: list[Any], separator: str = "\n"
    ) -> Any:
        """Apply an aggregation function to a list of values."""
        if func == "count":
            return len(values)
        if func == "sum":
            return sum(v for v in values if v is not None)
        if func == "mean":
            valid = [v for v in values if v is not None]
            return sum(valid) / len(valid) if valid else None
        if func == "min":
            valid = [v for v in values if v is not None]
            return min(valid) if valid else None
        if func == "max":
            valid = [v for v in values if v is not None]
            return max(valid) if valid else None
        if func == "first":
            return values[0] if values else None
        if func == "last":
            return values[-1] if values else None
        if func == "collect":
            return values
        if func == "concat":
            return separator.join(str(v) for v in values if v is not None)
        raise ValueError(f"Unknown aggregation function: {func}")

    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        """Group records and apply aggregations."""
        groups: dict[tuple, list[Record]] = defaultdict(list)

        for record in records:
            key = self._get_group_key(record)
            group_records = groups[key]

            if self._max_per_group is None or len(group_records) < self._max_per_group:
                group_records.append(record)

        total_groups = len(groups)
        output_count = 0

        for key, group_records in groups.items():
            if self._min_per_group and len(group_records) < self._min_per_group:
                continue

            output: Record = {}

            for col, val in zip(self._by, key):
                output[col] = val

            if self._collect:
                for col in self._collect:
                    values = [r.get(col) for r in group_records]
                    if self._output_column and len(self._collect) == 1:
                        output[self._output_column] = values
                    else:
                        output[f"{col}_list"] = values

            for new_col, spec in self._agg.items():
                parts = spec.split(":")
                source_col = parts[0]
                func = parts[1]
                separator = parts[2] if len(parts) > 2 else "\n"

                values = [r.get(source_col) for r in group_records]
                output[new_col] = self._apply_aggregation(func, values, separator)

            output_count += 1
            yield output

        logger.info(f"Group: {total_groups} groups, {output_count} output records")


class Pair(Step):
    """Create pairs (or n-tuples) from records."""

    VALID_STRATEGIES = frozenset(["random", "sequential", "sliding", "all"])
    VALID_OUTPUT_FORMATS = frozenset(["columns", "list"])

    def __init__(
        self,
        n: int = 2,
        strategy: str = "random",
        within: str | list[str] | None = None,
        across: str | list[str] | None = None,
        output_format: str = "columns",
        max_pairs: int | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Initialize a Pair step.

        Args:
            n: Tuple size (2 for pairs, 3 for triplets, etc.).
            strategy: Pairing strategy.
                - "random": Random combinations
                - "sequential": Consecutive records (0,1), (2,3), ...
                - "sliding": Sliding window (0,1), (1,2), (2,3), ...
                - "all": All possible combinations
            within: Records must share these column values to be paired.
            across: Records must differ on these column values to be paired.
            output_format: Output format.
                - "columns": chunk_1_*, chunk_2_* prefixed columns
                - "list": chunks list and {col}_list for each column
            max_pairs: Maximum number of pairs to generate.
            seed: Random seed for reproducibility (only for random strategy).

        Examples:
            >>> # Random pairs from same document
            >>> Pair(n=2, within="document_id")

            >>> # Sliding window triplets
            >>> Pair(n=3, strategy="sliding")

            >>> # Pairs with different categories
            >>> Pair(n=2, within="topic", across="author")

            >>> # List output format
            >>> Pair(n=2, output_format="list")
        """
        super().__init__()

        if n < 2:
            raise ValueError("n must be at least 2")

        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{strategy}'. "
                f"Valid strategies: {sorted(self.VALID_STRATEGIES)}"
            )

        if output_format not in self.VALID_OUTPUT_FORMATS:
            raise ValueError(
                f"Invalid output_format '{output_format}'. "
                f"Valid formats: {sorted(self.VALID_OUTPUT_FORMATS)}"
            )

        self._n = n
        self._strategy = strategy
        self._within = [within] if isinstance(within, str) else (within or [])
        self._across = [across] if isinstance(across, str) else (across or [])
        self._output_format = output_format
        self._max_pairs = max_pairs
        self._seed = seed

    def _get_within_key(self, record: Record) -> tuple:
        """Get the 'within' grouping key for a record."""
        if not self._within:
            return ()
        return tuple(record.get(col) for col in self._within)

    def _get_across_key(self, record: Record) -> tuple:
        """Get the 'across' key for a record."""
        if not self._across:
            return ()
        return tuple(record.get(col) for col in self._across)

    def _is_valid_tuple(self, records: list[Record]) -> bool:
        """Check if a tuple of records satisfies the 'across' constraint."""
        if not self._across:
            return True
        across_keys = [self._get_across_key(r) for r in records]
        return len(set(across_keys)) == len(across_keys)

    def _format_output(self, records: list[Record]) -> Record:
        """Format a tuple of records into output format."""
        if self._output_format == "list":
            output: Record = {"chunks": list(records)}
            if records:
                columns = set()
                for r in records:
                    columns.update(r.keys())
                for col in columns:
                    output[f"{col}_list"] = [r.get(col) for r in records]
            return output

        output = {}
        for i, record in enumerate(records, start=1):
            for key, value in record.items():
                output[f"chunk_{i}_{key}"] = value
        return output

    def _generate_pairs_random(
        self, records: list[Record], rng: random.Random
    ) -> Iterable[list[Record]]:
        """Generate random combinations."""
        if len(records) < self._n:
            return

        indices = list(range(len(records)))
        generated = 0

        max_attempts = (self._max_pairs or 10000) * 10
        attempts = 0

        while attempts < max_attempts:
            if self._max_pairs and generated >= self._max_pairs:
                break

            selected_indices = rng.sample(indices, self._n)
            selected = [records[i] for i in selected_indices]

            if self._is_valid_tuple(selected):
                generated += 1
                yield selected

            attempts += 1

    def _generate_pairs_sequential(
        self, records: list[Record]
    ) -> Iterable[list[Record]]:
        """Generate sequential pairs (0,1), (2,3), ..."""
        generated = 0
        for i in range(0, len(records) - self._n + 1, self._n):
            if self._max_pairs and generated >= self._max_pairs:
                break

            selected = records[i:i + self._n]
            if self._is_valid_tuple(selected):
                generated += 1
                yield selected

    def _generate_pairs_sliding(
        self, records: list[Record]
    ) -> Iterable[list[Record]]:
        """Generate sliding window pairs (0,1), (1,2), (2,3), ..."""
        generated = 0
        for i in range(len(records) - self._n + 1):
            if self._max_pairs and generated >= self._max_pairs:
                break

            selected = records[i:i + self._n]
            if self._is_valid_tuple(selected):
                generated += 1
                yield selected

    def _generate_pairs_all(
        self, records: list[Record]
    ) -> Iterable[list[Record]]:
        """Generate all possible combinations."""
        generated = 0
        for combo in itertools.combinations(records, self._n):
            if self._max_pairs and generated >= self._max_pairs:
                break

            selected = list(combo)
            if self._is_valid_tuple(selected):
                generated += 1
                yield selected

    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        """Create pairs/tuples from records."""
        all_records = list(records)

        if len(all_records) > 100000:
            logger.warning(
                f"Pair: buffering {len(all_records)} records. "
                "Consider filtering first for large datasets."
            )

        groups: dict[tuple, list[Record]] = defaultdict(list)
        for record in all_records:
            key = self._get_within_key(record)
            groups[key].append(record)

        rng = random.Random(self._seed)
        total_pairs = 0

        for group_key, group_records in groups.items():
            if len(group_records) < self._n:
                continue

            if self._strategy == "random":
                generator = self._generate_pairs_random(group_records, rng)
            elif self._strategy == "sequential":
                generator = self._generate_pairs_sequential(group_records)
            elif self._strategy == "sliding":
                generator = self._generate_pairs_sliding(group_records)
            elif self._strategy == "all":
                generator = self._generate_pairs_all(group_records)
            else:
                raise ValueError(f"Unknown strategy: {self._strategy}")

            for pair_records in generator:
                if self._max_pairs and total_pairs >= self._max_pairs:
                    break

                total_pairs += 1
                yield self._format_output(pair_records)

            if self._max_pairs and total_pairs >= self._max_pairs:
                break

        logger.info(
            f"Pair: created {total_pairs} {self._n}-tuples "
            f"from {len(all_records)} records ({len(groups)} groups)"
        )


class Concat(Step):
    """Stack multiple data sources/pipelines vertically.

    Executes each source pipeline and yields all their records sequentially,
    followed by any records received from upstream.

    Examples:
        >>> # Combine results from different sources
        >>> science = Source.file("science.jsonl") >> LLMStep(...)
        >>> history = Source.file("history.jsonl") >> LLMStep(...)
        >>> Concat(science, history) >> Sink.jsonl("combined.jsonl")

        >>> # Combine different generation runs
        >>> Concat(
        ...     data >> LLMStep(model=gpt4),
        ...     data >> LLMStep(model=claude),
        ... ) >> Sink.jsonl("multi_model.jsonl")
    """

    def __init__(self, *sources: Step) -> None:
        """
        Initialize a Concat step.

        Args:
            *sources: Steps or pipelines whose outputs will be concatenated.
        """
        super().__init__()
        if not sources:
            raise ValueError("Concat requires at least one source")
        self._sources = sources

    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        """Run each source and yield all records sequentially."""
        total = 0
        for i, source in enumerate(self._sources):
            count = 0
            for record in source.process(iter([])):
                count += 1
                total += 1
                yield record
            logger.debug(f"Concat: source {i} produced {count} records")

        logger.info(f"Concat: {total} total records from {len(self._sources)} sources")


class Join(Step):
    """Merge two datasets horizontally by key.

    The *left* side is the upstream pipeline (records flowing through
    ``process``).  The *right* side is a separate ``Step`` or
    ``Pipeline`` whose records are materialized once when ``process``
    is called.

    Overlapping column names (other than the join key) are disambiguated
    with configurable suffixes.

    Examples:
        >>> # Inner join on user_id
        >>> users >> Join(actions, on="user_id")
        >>> # Output: {user_id, name, action}

        >>> # Left join — keep all left records even without a match
        >>> users >> Join(actions, on="user_id", how="left")

        >>> # Custom suffixes for overlapping columns
        >>> chosen >> Join(rejected, on="question_id",
        ...               suffixes=("_chosen", "_rejected"))
    """

    VALID_HOW = frozenset(["inner", "left", "right", "outer"])

    def __init__(
        self,
        right: Step,
        on: str | list[str],
        how: str = "inner",
        suffixes: tuple[str, str] = ("_left", "_right"),
    ) -> None:
        """
        Initialize a Join step.

        Args:
            right: Step or pipeline providing the right-side records.
            on: Column name(s) used as the join key.
            how: Join type — ``"inner"``, ``"left"``, ``"right"``, or
                ``"outer"``.
            suffixes: A 2-tuple of suffixes applied to overlapping
                column names from the left and right sides respectively.

        Raises:
            ValueError: If *how* is not a recognised join type.
        """
        super().__init__()

        if how not in self.VALID_HOW:
            raise ValueError(
                f"how must be one of {sorted(self.VALID_HOW)}, got '{how}'"
            )

        self._right = right
        self._on = [on] if isinstance(on, str) else list(on)
        self._how = how
        self._left_suffix, self._right_suffix = suffixes

    def _key(self, record: Record) -> tuple:
        """Extract the join key from a record."""
        return tuple(record.get(col) for col in self._on)

    def _merge(
        self,
        left: Record,
        right: Record | None,
        overlapping: set[str],
    ) -> Record:
        """Merge a left and right record, handling overlapping columns."""
        output: Record = {}

        # Join key columns — always unsuffixed.
        for col in self._on:
            output[col] = left.get(col) if left is not None else (
                right.get(col) if right is not None else None
            )

        # Left columns.
        if left is not None:
            for key, value in left.items():
                if key in self._on:
                    continue
                if key in overlapping:
                    output[f"{key}{self._left_suffix}"] = value
                else:
                    output[key] = value

        # Right columns.
        if right is not None:
            for key, value in right.items():
                if key in self._on:
                    continue
                if key in overlapping:
                    output[f"{key}{self._right_suffix}"] = value
                else:
                    output[key] = value

        return output

    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        """Join left (upstream) records with right-side records."""
        left_records = list(records)
        right_records = list(self._right.process(iter([])))

        # Build right-side index (key → list of records).
        right_index: dict[tuple, list[Record]] = defaultdict(list)
        for rec in right_records:
            right_index[self._key(rec)].append(rec)

        # Determine overlapping non-key columns.
        left_cols: set[str] = set()
        for rec in left_records:
            left_cols.update(rec.keys())
        right_cols: set[str] = set()
        for rec in right_records:
            right_cols.update(rec.keys())
        key_cols = set(self._on)
        overlapping = (left_cols - key_cols) & (right_cols - key_cols)

        matched_right_keys: set[tuple] = set()
        total = 0

        # Left side iteration.
        for left_rec in left_records:
            lkey = self._key(left_rec)
            right_matches = right_index.get(lkey)

            if right_matches:
                matched_right_keys.add(lkey)
                for right_rec in right_matches:
                    total += 1
                    yield self._merge(left_rec, right_rec, overlapping)
            elif self._how in ("left", "outer"):
                total += 1
                yield self._merge(left_rec, None, overlapping)

        # Right-only records (for right / outer joins).
        if self._how in ("right", "outer"):
            for right_rec in right_records:
                rkey = self._key(right_rec)
                if rkey not in matched_right_keys:
                    total += 1
                    yield self._merge(None, right_rec, overlapping)

        logger.info(
            f"Join ({self._how}): {len(left_records)} left × "
            f"{len(right_records)} right → {total} output records"
        )
