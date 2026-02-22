"""Branch and JoinBranches steps for parallel processing paths."""

import copy
import itertools
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from loguru import logger

from datafast_v2.core.step import Step
from datafast_v2.core.types import Record

# Internal metadata keys used to communicate between Branch and JoinBranches.
_BRANCH_ID = "_branch_id"
_BRANCH_NAME = "_branch_name"
_BRANCH_INPUT_KEYS = "_branch_input_keys"
_BRANCH_META_KEYS = frozenset({_BRANCH_ID, _BRANCH_NAME, _BRANCH_INPUT_KEYS})


class Branch(Step):
    """
    Split data into named parallel processing paths.

    Each input record is independently processed by every named path.
    Paths are executed **sequentially** (one at a time) to keep execution
    simple and deterministic.  The output is a flat stream of tagged
    records that a downstream :class:`JoinBranches` step can merge back
    together.

    Every output record carries three metadata fields consumed by
    ``JoinBranches``:

    * ``_branch_id``  – integer index linking back to the input record
    * ``_branch_name`` – name of the path that produced this record
    * ``_branch_input_keys`` – list of column names present *before*
      the branch step, so ``JoinBranches`` can distinguish original
      columns from columns added by the branch.

    Examples:
        >>> # Preference-data generation
        >>> Branch(
        ...     chosen=LLMStep(
        ...         prompt="Expert answer: {question}",
        ...         output_columns=["response"],
        ...         model=gpt4,
        ...     ),
        ...     rejected=LLMStep(
        ...         prompt="Brief answer: {question}",
        ...         output_columns=["response"],
        ...         model=gpt4_mini,
        ...     ),
        ... )

        >>> # Style variations
        >>> Branch(
        ...     formal=Rewrite(mode="formalize"),
        ...     casual=Rewrite(mode="informalize"),
        ... )
    """

    def __init__(self, **paths: Step) -> None:
        """
        Initialize a Branch step.

        Args:
            **paths: Named steps (or pipelines) to run on each input
                record.  At least two paths are required.

        Raises:
            ValueError: If fewer than two paths are provided.
        """
        super().__init__()

        if len(paths) < 2:
            raise ValueError(
                "Branch requires at least 2 named paths, "
                f"got {len(paths)}: {list(paths.keys())}"
            )

        self._paths: dict[str, Step] = paths

    @property
    def path_names(self) -> list[str]:
        """Return the ordered list of branch path names."""
        return list(self._paths.keys())

    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        """
        Run each named path on the input records sequentially.

        For every ``(input_record, path)`` combination the path receives
        a deep-copy of the input record (so paths cannot interfere with
        each other).  Each output record is tagged with branch metadata
        before being yielded.
        """
        all_records = list(records)

        if not all_records:
            logger.info("Branch: no input records")
            return

        # Tag every input record with an ID and its original column names.
        for idx, record in enumerate(all_records):
            record[_BRANCH_ID] = idx
            record[_BRANCH_INPUT_KEYS] = [
                k for k in record.keys()
                if k not in _BRANCH_META_KEYS
            ]

        total_yielded = 0

        for branch_name, step in self._paths.items():
            # Deep-copy so each path works on independent data.
            branch_input = copy.deepcopy(all_records)

            count = 0
            for output_record in step.process(iter(branch_input)):
                output_record[_BRANCH_NAME] = branch_name

                # Ensure metadata survives even if the path step
                # rebuilt the record from scratch.
                if _BRANCH_ID not in output_record:
                    logger.warning(
                        f"Branch path '{branch_name}' dropped "
                        f"_branch_id; record will be skipped by "
                        f"JoinBranches"
                    )
                if _BRANCH_INPUT_KEYS not in output_record:
                    output_record[_BRANCH_INPUT_KEYS] = []

                count += 1
                total_yielded += 1
                yield output_record

            logger.debug(
                f"Branch path '{branch_name}': produced {count} records"
            )

        logger.info(
            f"Branch: {len(all_records)} input records × "
            f"{len(self._paths)} paths → {total_yielded} tagged records"
        )


class JoinBranches(Step):
    """
    Merge records produced by a preceding :class:`Branch` step.

    Groups tagged records by ``_branch_id``, then for each group builds
    a single output record containing:

    * **Original columns** (present before branching) — kept unsuffixed,
      taken from the first available branch.
    * **New columns** (added by branch processing) — suffixed with
      ``_{branch_name}`` (or a custom suffix if provided).
    * All ``_branch_*`` metadata is stripped from the output.

    When a branch produces multiple records per input (e.g. an LLMStep
    with ``num_outputs > 1``), ``JoinBranches`` creates the cartesian
    product across branches for that ``_branch_id``.

    Examples:
        >>> # Default suffixes (_{branch_name})
        >>> Branch(
        ...     chosen=LLMStep(..., output_columns=["response"]),
        ...     rejected=LLMStep(..., output_columns=["response"]),
        ... ) >> JoinBranches()
        >>> # → response_chosen, response_rejected

        >>> # Custom suffixes
        >>> Branch(
        ...     formal=Rewrite(mode="formalize"),
        ...     casual=Rewrite(mode="informalize"),
        ... ) >> JoinBranches(suffixes={"formal": "_formal", "casual": "_casual"})
    """

    def __init__(
        self,
        suffixes: dict[str, str] | None = None,
        how: str = "inner",
    ) -> None:
        """
        Initialize a JoinBranches step.

        Args:
            suffixes: Per-branch column suffixes.  Keys are branch names,
                values are suffix strings.  If ``None``, defaults to
                ``_{branch_name}`` for every branch.
            how: Join behaviour when a branch is missing for a given
                ``_branch_id``.

                * ``"inner"`` (default) — skip that ``_branch_id``
                  entirely.
                * ``"outer"`` — include the record with ``None`` for
                  columns from the missing branch.

        Raises:
            ValueError: If *how* is not ``"inner"`` or ``"outer"``.
        """
        super().__init__()

        if how not in ("inner", "outer"):
            raise ValueError(
                f"how must be 'inner' or 'outer', got '{how}'"
            )

        self._suffixes = suffixes
        self._how = how

    def _get_suffix(self, branch_name: str) -> str:
        """Return the column suffix for *branch_name*."""
        if self._suffixes and branch_name in self._suffixes:
            return self._suffixes[branch_name]
        return f"_{branch_name}"

    def _merge_group(
        self,
        branch_records: dict[str, list[Record]],
        branch_names: list[str],
    ) -> Iterable[Record]:
        """Merge records from one ``_branch_id`` group."""
        # Determine original columns from the first available record.
        first_rec = next(
            rec
            for recs in branch_records.values()
            for rec in recs
        )
        input_keys: set[str] = set(
            first_rec.get(_BRANCH_INPUT_KEYS, [])
        )
        input_keys -= _BRANCH_META_KEYS

        # Collect per-branch record lists for cartesian product.
        per_branch: list[list[Record]] = []
        per_branch_names: list[str] = []
        for bname in branch_names:
            recs = branch_records.get(bname)
            if recs:
                per_branch.append(recs)
                per_branch_names.append(bname)
            elif self._how == "outer":
                per_branch.append([None])  # type: ignore[list-item]
                per_branch_names.append(bname)

        for combo in itertools.product(*per_branch):
            output: Record = {}

            # Populate original (shared) columns from the first
            # non-None record in the combo.
            for rec in combo:
                if rec is not None:
                    for key in input_keys:
                        if key in rec and key not in output:
                            output[key] = rec[key]
                    break

            # Populate branch-specific columns with suffixes.
            for bname, rec in zip(per_branch_names, combo):
                suffix = self._get_suffix(bname)
                if rec is None:
                    continue
                for key, value in rec.items():
                    if key in _BRANCH_META_KEYS:
                        continue
                    if key in input_keys:
                        continue
                    output[f"{key}{suffix}"] = value

            yield output

    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        """
        Collect tagged records, group by ``_branch_id``, and merge.
        """
        all_records = list(records)

        if not all_records:
            logger.info("JoinBranches: no input records")
            return

        # ---- discover branch names (preserving insertion order) ----
        branch_names_seen: dict[str, None] = {}
        for rec in all_records:
            name = rec.get(_BRANCH_NAME)
            if name is not None and name not in branch_names_seen:
                branch_names_seen[name] = None
        branch_names: list[str] = list(branch_names_seen.keys())

        if not branch_names:
            logger.warning(
                "JoinBranches: no _branch_name metadata found; "
                "passing records through unchanged"
            )
            yield from all_records
            return

        # ---- group by _branch_id ----
        groups: dict[Any, dict[str, list[Record]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for rec in all_records:
            bid = rec.get(_BRANCH_ID)
            bname = rec.get(_BRANCH_NAME)
            if bid is None or bname is None:
                continue
            groups[bid][bname].append(rec)

        total_merged = 0
        skipped_incomplete = 0

        for bid in sorted(groups.keys()):
            branch_recs = groups[bid]

            if self._how == "inner":
                if not all(bn in branch_recs for bn in branch_names):
                    skipped_incomplete += 1
                    continue

            for merged in self._merge_group(branch_recs, branch_names):
                total_merged += 1
                yield merged

        logger.info(
            f"JoinBranches: {total_merged} merged records"
            + (
                f" ({skipped_incomplete} incomplete groups skipped)"
                if skipped_incomplete
                else ""
            )
        )
