"""Base Step and Pipeline classes for datafast v2."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from datafast.core.types import Record


class Step(ABC):
    """Base class for all pipeline steps."""

    def __init__(self) -> None:
        self._name: str | None = None

    @abstractmethod
    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        """Process input records and yield output records."""
        ...

    def as_step(self, name: str) -> "Step":
        """Assign a name to this step for checkpointing and debugging."""
        self._name = name
        return self

    def __rshift__(self, other: "Step") -> "Pipeline":
        """Enable >> syntax for chaining steps."""
        return Pipeline([self, other])

    @property
    def name(self) -> str:
        """Return step name (auto-generated if not set)."""
        return self._name or self.__class__.__name__


class Pipeline(Step):
    """A sequence of steps that form a pipeline."""

    def __init__(self, steps: list[Step]) -> None:
        super().__init__()
        self._steps: list[Step] = steps

    def __rshift__(self, other: Step) -> "Pipeline":
        """Enable >> syntax for appending steps to pipeline."""
        if isinstance(other, Pipeline):
            return Pipeline(self._steps + other._steps)
        return Pipeline(self._steps + [other])

    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        """Process records through all steps in sequence."""
        current = records
        for step in self._steps:
            current = step.process(current)
        return current

    @property
    def steps(self) -> list[Step]:
        """Return the list of steps in this pipeline."""
        return self._steps

    def compile(self) -> "Pipeline":
        """Validate pipeline structure and column references before execution.

        Raises PipelineValidationError on the first problem found; returns self
        so it can be chained. Called automatically by run().
        """
        from datafast.core.validation import validate_pipeline

        validate_pipeline(self._steps)
        return self

    def run(
        self,
        checkpoint_dir: str | None = None,
        resume: bool = False,
        batch_size: int = 4,
        llm_strategy: str = "by_model",
        resume_from: str | None = None,
        limit: int | None = None,
        stop_after: int | str | None = None,
        **kwargs: Any,
    ) -> list[Record]:
        """
        Execute the pipeline and return all records.

        Args:
            checkpoint_dir: Directory for checkpoint files (None to disable).
            resume: Whether to resume from existing checkpoint.
            batch_size: Number of LLM calls per batch.
            llm_strategy: LLM execution strategy.
                - "by_model": All calls for model A, then B (default)
                - "round_robin": Interleave models
                - "by_record": Process each record completely
            resume_from: Re-run from this step name, discarding it and later
                steps (reuses completed upstream steps; requires a checkpoint).
            limit: Process only first N source records.
            stop_after: Stop after step (index or name).
            **kwargs: Additional RunConfig parameters.

        Returns:
            List of output records.

        Raises:
            PipelineValidationError: If the pipeline is built wrong.
            ValueError: If resume_from or stop_after names no step in the
                pipeline, or stop_after is an out-of-range index.
        """
        self.compile()

        from datafast.core.runner import run_pipeline

        return run_pipeline(
            self,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
            batch_size=batch_size,
            llm_strategy=llm_strategy,
            resume_from=resume_from,
            limit=limit,
            stop_after=stop_after,
            **kwargs,
        )
