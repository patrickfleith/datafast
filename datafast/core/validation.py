"""Static pipeline validation, run before execution by ``Pipeline.compile()``.

Checks are conservative: column references are only flagged when the schema is
statically known. Steps that reshape records opaquely (Map/FlatMap/Group/Pair/
Join/Concat and the LLM-eval steps) reset the tracked schema to "unknown" so a
later reference is never wrongly reported as missing.

Validation recurses into sub-pipelines, which come in two flavours:

* **inherited** — a ``Branch`` path, fed by the records flowing into the branch.
  It may not contain a source (that would discard the branch input) and its
  column references are checked against the incoming schema.
* **sourced** — a ``Concat`` source or a ``Join`` right side, executed with empty
  input. It must start with a source of its own.

Neither flavour may contain a sink.
"""

from datafast.core.step import Pipeline, Step
from datafast.sinks.sink import CSVSink, HubSink, JSONLSink, ListSink, ParquetSink
from datafast.sources.seed import SeedSource
from datafast.sources.source import FileSource, HuggingFaceSource, ListSource
from datafast.transforms.branch import Branch, JoinBranches
from datafast.transforms.data_ops import AddUUID, Concat, Filter, Group, Join, Pair
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
    _validate_sequence(
        steps,
        known=None,
        context="",
        needs_source=True,
        allow_sink=True,
    )


def _validate_sequence(
    steps: list[Step],
    *,
    known: set[str] | None,
    context: str,
    needs_source: bool,
    allow_sink: bool,
) -> None:
    """Validate one linear sequence of steps and everything nested inside it.

    Args:
        steps: The steps to validate.
        known: Columns available on entry, or None if not statically knowable.
            Ignored when *needs_source* is set (the source defines the schema).
        context: Human-readable location, empty for the top-level pipeline.
        needs_source: Whether the sequence must begin with a source.
        allow_sink: Whether sinks may appear as the trailing steps.
    """
    if not steps:
        raise PipelineValidationError(f"{context or 'Pipeline'} is empty.")

    _check_structure(steps, context, needs_source, allow_sink)
    _check_branches(steps, context)

    start = 0
    if needs_source:
        # The source defines the schema and reads no upstream columns, but it
        # may still own sub-pipelines (Concat).
        known = _initial_columns(steps[0])
        _validate_children(steps[0], known, context)
        start = 1

    for step in steps[start:]:
        if known is not None:
            missing = [col for col in _required_columns(step) if col not in known]
            if missing:
                raise PipelineValidationError(
                    f"Step '{step.name}'{_inside(context)} references column(s) "
                    f"{missing} that are not available. "
                    f"Available columns: {sorted(known)}."
                )
        _validate_children(step, known, context)
        known = _next_columns(step, known)


def _inside(context: str) -> str:
    """Render a context suffix for error messages ('' at the top level)."""
    return f" inside {context}" if context else ""


def _nested(context: str, child: str) -> str:
    """Compose a child context under *context*."""
    return f"{context} → {child}" if context else child


def _as_steps(step: Step) -> list[Step]:
    """View a step or pipeline as a flat list of steps."""
    return list(step.steps) if isinstance(step, Pipeline) else [step]


def _check_structure(
    steps: list[Step], context: str, needs_source: bool, allow_sink: bool
) -> None:
    first = steps[0]
    if needs_source and not isinstance(first, _PRODUCERS):
        raise PipelineValidationError(
            f"{context or 'Pipeline'} must start with a source; step 0 is "
            f"'{first.name}' ({type(first).__name__})."
        )

    # Sinks pass their records through, so a pipeline may end in several of them
    # — writing the same dataset to a file and to the Hub is one run, not two.
    # They must still come last together: a sink is a side effect, and a step
    # after one would write before the pipeline had finished shaping the records.
    first_sink: int | None = None
    for i, step in enumerate(steps):
        if isinstance(step, _PRODUCERS) and (i > 0 or not needs_source):
            raise PipelineValidationError(
                f"Source '{step.name}' at position {i}{_inside(context)} discards "
                f"upstream records; "
                + (
                    "a source may only be the first step."
                    if needs_source
                    else f"{context} receives records from upstream."
                )
            )
        if isinstance(step, _SINKS):
            if not allow_sink:
                raise PipelineValidationError(
                    f"Sink '{step.name}' at position {i} is not allowed"
                    f"{_inside(context)}."
                )
            if first_sink is None:
                first_sink = i
        elif first_sink is not None:
            raise PipelineValidationError(
                f"Step '{step.name}' at position {i}{_inside(context)} comes after "
                f"the sink at position {first_sink}; sinks must be the last steps. "
                f"Several sinks may be chained, but nothing may follow them."
            )


def _check_branches(steps: list[Step], context: str) -> None:
    open_at: int | None = None
    for i, step in enumerate(steps):
        if isinstance(step, Branch):
            if open_at is not None:
                raise PipelineValidationError(
                    f"Branch at position {i}{_inside(context)} opens before the "
                    f"Branch at position {open_at} is closed by a JoinBranches."
                )
            open_at = i
        elif isinstance(step, JoinBranches):
            if open_at is None:
                raise PipelineValidationError(
                    f"JoinBranches at position {i}{_inside(context)} has no "
                    f"matching Branch."
                )
            open_at = None
    if open_at is not None:
        raise PipelineValidationError(
            f"Branch at position {open_at}{_inside(context)} is never closed by "
            f"a JoinBranches."
        )


def _validate_children(step: Step, known: set[str] | None, context: str) -> None:
    """Recurse into a step's sub-pipelines."""
    if isinstance(step, Branch):
        for name, path in step.paths.items():
            path_context = _nested(context, f"Branch path '{name}'")
            path_steps = _as_steps(path)
            _reject_nested_branch(path_steps, path_context)
            _validate_sequence(
                path_steps,
                known=known,
                context=path_context,
                needs_source=False,
                allow_sink=False,
            )
    elif isinstance(step, Concat):
        for i, source in enumerate(step._sources):
            _validate_sequence(
                _as_steps(source),
                known=None,
                context=_nested(context, f"Concat source {i}"),
                needs_source=True,
                allow_sink=False,
            )
    elif isinstance(step, Join):
        _validate_sequence(
            _as_steps(step._right),
            known=None,
            context=_nested(context, f"Join right side of '{step.name}'"),
            needs_source=True,
            allow_sink=False,
        )


def _reject_nested_branch(steps: list[Step], context: str) -> None:
    """A Branch inside a Branch path would overwrite the outer branch metadata,
    so JoinBranches silently drops every record. Reject it outright."""
    for i, step in enumerate(steps):
        if isinstance(step, Branch):
            raise PipelineValidationError(
                f"Branch at position {i}{_inside(context)}: nesting a Branch "
                f"inside a branch path is not supported — the inner branch "
                f"overwrites the outer branch metadata. Close the outer Branch "
                f"with a JoinBranches first."
            )


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
    elif isinstance(step, Join):
        cols += step._on  # read from the upstream (left) records
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
