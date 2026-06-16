"""Pipeline execution engine with checkpointing and LLM batching."""

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from datafast.core.checkpoint import (
    CheckpointManager,
    Manifest,
    PipelineChangedError,
    compute_pipeline_hash,
)
from datafast.core.config import LLMCall, LLMStepProgress, RunConfig
from datafast.core.types import Record
from datafast.tracing import build_trace_metadata

if TYPE_CHECKING:
    from datafast.core.step import Pipeline, Step
    from datafast.transforms.llm_step import LLMStep


def chunked(iterable: list, size: int):
    """Yield successive chunks of size from iterable."""
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


@dataclass
class _LLMBatchStats:
    generated: int = 0
    errors: int = 0


class Runner:
    """
    Execution engine for pipelines.

    Handles:
    - Step-by-step execution with materialization between steps
    - Checkpointing and resume
    - LLM call batching and execution strategies
    - Progress tracking
    """

    def __init__(self, pipeline: "Pipeline", config: RunConfig | None = None) -> None:
        """
        Initialize the runner.

        Args:
            pipeline: The pipeline to execute.
            config: Execution configuration. Uses defaults if None.
        """
        self.pipeline = pipeline
        self.config = config or RunConfig()
        self._checkpoint_mgr: CheckpointManager | None = None
        self._trace_session_id = f"datafast-run-{uuid.uuid4().hex}"

        if self.config.checkpoint_dir:
            self._checkpoint_mgr = CheckpointManager(self.config.checkpoint_dir)

    def execute(self) -> list[Record]:
        """
        Execute the pipeline.

        Returns:
            List of output records.
        """
        steps = self.pipeline.steps
        step_names = [s.name for s in steps]
        step_types = [s.__class__.__name__ for s in steps]

        records: list[Record] = []
        start_step = 0
        llm_progress: LLMStepProgress | None = None
        manifest: Manifest | None = None

        if self._checkpoint_mgr:
            manifest = self._setup_checkpoint(step_names, step_types)
            if self.config.resume and manifest:
                start_step, records, llm_progress = self._checkpoint_mgr.find_resume_point(
                    manifest
                )
                if start_step == -1:
                    logger.info("All steps already complete, returning cached results")
                    last_step = manifest.steps[-1]
                    return self._checkpoint_mgr.load_step_records(
                        last_step.index, last_step.name
                    )
                if start_step > 0:
                    logger.info(f"Resuming from step {start_step}: {step_names[start_step]}")

        for i, step in enumerate(steps):
            if i < start_step:
                continue

            step_name = step.name
            records_in = len(records)

            if manifest and self._checkpoint_mgr:
                self._checkpoint_mgr.mark_step_in_progress(manifest, i)

            logger.info(f"Executing step {i}: {step_name}")
            start_time = time.time()

            if self._is_llm_step(step):
                records = self._execute_llm_step(
                    step,
                    records,
                    i,
                    step_name,
                    manifest,
                    llm_progress if i == start_step else None,
                )
                llm_progress = None
            else:
                records = list(step.process(iter(records)))

            elapsed = time.time() - start_time
            logger.info(
                f"Step {i} ({step_name}): {records_in} → {len(records)} records "
                f"({elapsed:.2f}s)"
            )

            if manifest and self._checkpoint_mgr:
                # LLM steps already write incrementally, skip duplicate save
                if not self._is_llm_step(step):
                    self._checkpoint_mgr.save_step_records(i, step_name, records)
                self._checkpoint_mgr.mark_step_complete(
                    manifest, i, records_in, len(records)
                )

            if self.config.stop_after is not None:
                if i == self.config.stop_after or step_name == self.config.stop_after:
                    logger.info(f"Stopping after step: {step_name}")
                    break

        return records

    def _setup_checkpoint(
        self, step_names: list[str], step_types: list[str]
    ) -> Manifest | None:
        """Set up checkpointing, handling resume logic."""
        if not self._checkpoint_mgr:
            return None

        pipeline_hash = compute_pipeline_hash(step_names, step_types)

        if self._checkpoint_mgr.has_checkpoint():
            existing = self._checkpoint_mgr.load_manifest()
            if existing:
                if existing.pipeline_hash != pipeline_hash:
                    if self.config.resume:
                        raise PipelineChangedError(
                            "Pipeline structure has changed since checkpoint. "
                            "Use resume=False to start fresh."
                        )
                    logger.warning("Pipeline changed, clearing old checkpoint")
                    self._checkpoint_mgr.clear()
                else:
                    return existing

        return self._checkpoint_mgr.create_manifest(
            pipeline_hash, step_names, self.config
        )

    def _is_llm_step(self, step: "Step") -> bool:
        """Check if a step supports LLM batching (has collect_calls/apply_result).

        For dual-mode steps (e.g. Classify, Score) that support both fn and
        LLM modes, also checks the ``uses_llm`` property.
        """
        if not (hasattr(step, "collect_calls") and hasattr(step, "apply_result")):
            return False
        # Dual-mode steps expose a uses_llm flag; respect it.
        if hasattr(step, "uses_llm"):
            return step.uses_llm
        return True

    def _execute_llm_step(
        self,
        step: "LLMStep",
        records: list[Record],
        step_index: int,
        step_name: str,
        manifest: Manifest | None,
        resume_progress: LLMStepProgress | None,
    ) -> list[Record]:
        """Execute an LLM step with batching and optional checkpointing."""
        skip_call_ids: set[str] = set()
        output_records: list[Record] = []

        if resume_progress:
            skip_call_ids = set(resume_progress.completed_call_ids)
            _, partial_records = self._checkpoint_mgr.load_llm_progress(
                step_index, step_name
            )
            output_records = partial_records
            logger.info(
                f"Resuming LLM step: {resume_progress.completed_calls}/"
                f"{resume_progress.total_calls} calls already done"
            )

        calls, models_map = step.collect_calls(records, skip_call_ids)

        if not calls:
            logger.info("No LLM calls to execute")
            return output_records

        calls = self._order_calls(calls)

        total_calls = len(calls) + len(skip_call_ids)
        progress = LLMStepProgress(
            step_index=step_index,
            step_name=step_name,
            total_calls=total_calls,
            completed_call_ids=list(skip_call_ids),
        )

        # Clear the step file if starting fresh (not resuming)
        if self._checkpoint_mgr and not skip_call_ids:
            self._checkpoint_mgr.clear_step_file(step_index, step_name)

        completed_since_checkpoint = 0
        errors = 0
        generated_total = len(skip_call_ids)

        for batch in chunked(calls, self.config.batch_size):
            batch_start = time.perf_counter()
            batch_model_id = (
                batch[0].model_id
                if len({call.model_id for call in batch}) == 1
                else "mixed"
            )
            stats = self._execute_llm_batch(
                step=step,
                step_name=step_name,
                step_index=step_index,
                batch=batch,
                models_map=models_map,
                progress=progress,
                output_records=output_records,
            )
            completed_since_checkpoint += stats.generated
            errors += stats.errors
            generated_total += stats.generated

            batch_duration = time.perf_counter() - batch_start
            logger.info(
                f"Generated {stats.generated} samples (total: {generated_total}) | "
                f"model: {batch_model_id} | duration: {batch_duration:.2f}s"
            )

            if (
                self._checkpoint_mgr
                and manifest
                and completed_since_checkpoint >= self.config.checkpoint_every
            ):
                self._checkpoint_mgr.save_llm_progress(
                    step_index, step_name, progress, output_records
                )
                completed_since_checkpoint = 0

        logger.info(
            f"LLMStep complete: {len(output_records)} outputs, {errors} errors"
        )
        return output_records

    def _execute_llm_batch(
        self,
        *,
        step: "LLMStep",
        step_name: str,
        step_index: int,
        batch: list[LLMCall],
        models_map: dict[str, object],
        progress: LLMStepProgress,
        output_records: list[Record],
    ) -> _LLMBatchStats:
        """Execute and apply one runner batch, preserving input order."""
        batch_results, errors = self._collect_llm_batch_results(
            step=step,
            step_name=step_name,
            batch=batch,
            models_map=models_map,
        )
        stats = self._apply_llm_batch_results(
            step=step,
            step_name=step_name,
            step_index=step_index,
            batch=batch,
            batch_results=batch_results,
            models_map=models_map,
            progress=progress,
            output_records=output_records,
        )
        stats.errors += errors
        return stats

    def _collect_llm_batch_results(
        self,
        *,
        step: "LLMStep",
        step_name: str,
        batch: list[LLMCall],
        models_map: dict[str, object],
    ) -> tuple[list[object | None], int]:
        batch_results: list[object | None] = [None] * len(batch)
        errors = 0
        grouped_indexes: dict[str, list[int]] = defaultdict(list)

        for index, call in enumerate(batch):
            grouped_indexes[call.model_id].append(index)

        for model_id, indexes in grouped_indexes.items():
            group_calls = [batch[index] for index in indexes]
            model = models_map[model_id]
            try:
                group_results = self._generate_llm_group(
                    step=step,
                    step_name=step_name,
                    model=model,
                    group_calls=group_calls,
                )
            except Exception as e:
                errors += len(group_calls)
                self._log_llm_failures(group_calls, e)
                continue

            for result_index, result in zip(indexes, group_results):
                batch_results[result_index] = result

        return batch_results, errors

    def _generate_llm_group(
        self,
        *,
        step: "LLMStep",
        step_name: str,
        model: object,
        group_calls: list[LLMCall],
    ) -> list[object]:
        group_metadata = [
            self._build_llm_call_metadata(step, step_name, call, model)
            for call in group_calls
        ]
        if hasattr(model, "generate_batch"):
            return list(
                model.generate_batch(  # type: ignore[attr-defined]
                    [call.messages for call in group_calls],
                    metadata=group_metadata,
                )
            )
        return [
            model.generate(  # type: ignore[attr-defined]
                messages=call.messages,
                metadata=metadata,
            )
            for call, metadata in zip(group_calls, group_metadata)
        ]

    def _apply_llm_batch_results(
        self,
        *,
        step: "LLMStep",
        step_name: str,
        step_index: int,
        batch: list[LLMCall],
        batch_results: list[object | None],
        models_map: dict[str, object],
        progress: LLMStepProgress,
        output_records: list[Record],
    ) -> _LLMBatchStats:
        stats = _LLMBatchStats()

        for call, result in zip(batch, batch_results):
            if result is None:
                continue
            try:
                output_record = step.apply_result(call, result, models_map[call.model_id])
                output_records.append(output_record)
                progress.completed_call_ids.append(call.call_id)
                stats.generated += 1

                if self._checkpoint_mgr:
                    self._checkpoint_mgr.append_record(
                        step_index, step_name, output_record
                    )
            except Exception as e:
                stats.errors += 1
                self._log_llm_failures([call], e)

        return stats

    def _build_llm_call_metadata(
        self,
        step: "LLMStep",
        step_name: str,
        call: LLMCall,
        model: object,
    ) -> dict[str, object]:
        return build_trace_metadata(
            model=model,
            component="pipeline.step",
            trace_name=f"datafast.{step_name}",
            session_id=self._trace_session_id,
            step_name=step_name,
            step_type=step.__class__.__name__,
            record_index=call.record_index,
            prompt_index=call.prompt_index,
            output_index=call.output_index,
            language_code=call.language_code or None,
            call_id=call.call_id,
        )

    @staticmethod
    def _log_llm_failures(calls: list[LLMCall], error: Exception) -> None:
        for call in calls:
            logger.warning(
                f"LLM call failed | Model: {call.model_id} | "
                f"Call: {call.call_id} | Error: {error}"
            )

    def _order_calls(self, calls: list[LLMCall]) -> list[LLMCall]:
        """Order calls according to execution strategy."""
        strategy = self.config.llm_strategy

        if strategy == "by_record":
            return calls

        if strategy == "by_model":
            by_model: dict[str, list[LLMCall]] = defaultdict(list)
            for call in calls:
                by_model[call.model_id].append(call)

            ordered = []
            for model_calls in by_model.values():
                ordered.extend(model_calls)
            return ordered

        if strategy == "round_robin":
            by_model: dict[str, list[LLMCall]] = defaultdict(list)
            for call in calls:
                by_model[call.model_id].append(call)

            model_ids = list(by_model.keys())
            iterators = {m: iter(by_model[m]) for m in model_ids}
            ordered = []
            active = set(model_ids)

            while active:
                for model_id in model_ids:
                    if model_id not in active:
                        continue
                    try:
                        call = next(iterators[model_id])
                        ordered.append(call)
                    except StopIteration:
                        active.discard(model_id)

            return ordered

        return calls


def run_pipeline(
    pipeline: "Pipeline",
    checkpoint_dir: str | None = None,
    resume: bool = False,
    batch_size: int = 4,
    llm_strategy: str = "by_model",
    rate_limits: dict[str, int] | None = None,
    limit: int | None = None,
    stop_after: int | str | None = None,
    **kwargs,
) -> list[Record]:
    """
    Execute a pipeline with the runner.

    Args:
        pipeline: Pipeline to execute.
        checkpoint_dir: Directory for checkpoints (None to disable).
        resume: Whether to resume from checkpoint.
        batch_size: LLM calls per batch.
        llm_strategy: "by_model", "round_robin", or "by_record".
        rate_limits: Requests per minute per model ID.
        limit: Process only first N source records.
        stop_after: Stop after step (index or name).
        **kwargs: Additional RunConfig parameters.

    Returns:
        List of output records.
    """
    config = RunConfig(
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        batch_size=batch_size,
        llm_strategy=llm_strategy,
        rate_limits=rate_limits,
        limit=limit,
        stop_after=stop_after,
        **kwargs,
    )
    runner = Runner(pipeline, config)
    return runner.execute()
