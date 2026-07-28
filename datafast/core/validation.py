"""Static pipeline validation, run before execution by ``Pipeline.compile()``.

Checks are conservative: column references are only flagged when the schema is
statically known. Steps that reshape records opaquely (Map/FlatMap/Group/Pair/
Join/Concat and the LLM-eval steps) reset the tracked schema to "unknown" so a
later reference is never wrongly reported as missing.
"""

from datafast.core.step import Step
from datafast.sinks.sink import CSVSink, HubSink, JSONLSink, ListSink, ParquetSink
from datafast.sources.seed import SeedSource
from datafast.sources.source import FileSource, HuggingFaceSource, ListSource
from datafast.transforms.branch import Branch, JoinBranches
from datafast.transforms.data_ops import AddUUID, Concat, Filter, Group, Pair
from datafast.transforms.llm_step import LLMStep
from datafast.transforms.sample import Sample


class PipelineValidationError(Exception):
    """Raised when a pipeline fails structural or column validation."""


_SINKS = (JSONLSink, CSVSink, ListSink, ParquetSink, HubSink)
# Steps that generate records from nothing, ignoring upstream input. Concat is
# included because it reads only its own sub-sources (see Concat.process).
_PRODUCERS = (SeedSource, ListSource, FileSource, HuggingFaceSource, Concat)


def validate_pipeline(steps: list[Step]) -> None:
    """Validate a pipeline's steps, raising PipelineValidationError on the first
    problem found."""
    if not steps:
        raise PipelineValidationError("Pipeline is empty.")
    _check_structure(steps)
    _check_branches(steps)
    _check_columns(steps)


def _check_structure(steps: list[Step]) -> None:
    first = steps[0]
    if not isinstance(first, _PRODUCERS):
        raise PipelineValidationError(
            f"Pipeline must start with a source; step 0 is "
            f"'{first.name}' ({type(first).__name__})."
        )

    last_index = len(steps) - 1
    for i, step in enumerate(steps):
        if i > 0 and isinstance(step, _PRODUCERS):
            raise PipelineValidationError(
                f"Source '{step.name}' at position {i} discards upstream records; "
                f"a source may only be the first step."
            )
        if isinstance(step, _SINKS) and i != last_index:
            raise PipelineValidationError(
                f"Sink '{step.name}' at position {i} must be the last step."
            )


def _check_branches(steps: list[Step]) -> None:
    open_at: int | None = None
    for i, step in enumerate(steps):
        if isinstance(step, Branch):
            if open_at is not None:
                raise PipelineValidationError(
                    f"Branch at position {i} opens before the Branch at position "
                    f"{open_at} is closed by a JoinBranches."
                )
            open_at = i
        elif isinstance(step, JoinBranches):
            if open_at is None:
                raise PipelineValidationError(
                    f"JoinBranches at position {i} has no matching Branch."
                )
            open_at = None
    if open_at is not None:
        raise PipelineValidationError(
            f"Branch at position {open_at} is never closed by a JoinBranches."
        )


def _check_columns(steps: list[Step]) -> None:
    known = _initial_columns(steps[0])
    for step in steps[1:]:
        if known is not None:
            missing = [col for col in _required_columns(step) if col not in known]
            if missing:
                raise PipelineValidationError(
                    f"Step '{step.name}' references column(s) {missing} that are "
                    f"not available. Available columns: {sorted(known)}."
                )
        known = _next_columns(step, known)


def _initial_columns(source: Step) -> set[str] | None:
    """Columns produced by the source, or None if not knowable statically."""
    records = getattr(source, "_records", None)
    if isinstance(source, (SeedSource, ListSource)) and records:
        return {key for record in records for key in record}
    hf_columns = getattr(source, "_columns", None)
    if isinstance(source, HuggingFaceSource) and hf_columns:
        return set(hf_columns)
    return None


def _required_columns(step: Step) -> list[str]:
    """Input columns a step reads that must already exist (empty if opaque)."""
    cols: list[str] = list(getattr(step, "_input_columns", None) or [])

    single = getattr(step, "_input_column", None)
    if isinstance(single, str):
        cols.append(single)
    for attr in ("_column_a", "_column_b"):
        value = getattr(step, attr, None)
        if isinstance(value, str):
            cols.append(value)

    cols += getattr(step, "_forward_columns", None) or []

    if isinstance(step, Group):
        cols += step._by
    elif isinstance(step, Pair):
        cols += step._within + step._across
    elif isinstance(step, Sample) and isinstance(step._by, str):
        cols.append(step._by)

    return cols


def _next_columns(step: Step, known: set[str] | None) -> set[str] | None:
    """Schema after ``step``, or None once it can no longer be tracked exactly."""
    if known is None:
        return None
    if isinstance(step, LLMStep):
        base = set(known)
        if step._forward_columns is not None:
            base = set(step._forward_columns)
        elif step._exclude_columns is not None:
            base -= set(step._exclude_columns)
        return base | set(step._output_columns) | {"_model", "_prompt_index", "_language"}
    if isinstance(step, AddUUID):
        return known | {step._column}
    if isinstance(step, (Filter, Sample)):
        return known
    return None  # opaque: Map/FlatMap/Group/Pair/Join/Concat/eval steps/Branch/...
