"""Checkpoint management for pipeline execution."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from datafast.core.config import LLMStepProgress, RunConfig, StepStatus
from datafast.core.types import Record


@dataclass
class Manifest:
    """Checkpoint manifest tracking pipeline execution state."""

    pipeline_hash: str
    created_at: str
    updated_at: str
    steps: list[StepStatus]
    current_step: int
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to dictionary for JSON serialization."""
        return {
            "pipeline_hash": self.pipeline_hash,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "steps": [asdict(s) for s in self.steps],
            "current_step": self.current_step,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Manifest":
        """Create manifest from dictionary."""
        steps = [StepStatus(**s) for s in data["steps"]]
        return cls(
            pipeline_hash=data["pipeline_hash"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            steps=steps,
            current_step=data["current_step"],
            config=data["config"],
        )


class CheckpointManager:
    """Manages checkpoint save/load operations."""

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self.checkpoint_dir / "manifest.json"

    def _step_file_path(self, step_index: int, step_name: str) -> Path:
        safe_name = step_name.replace("/", "_").replace("\\", "_")
        return self.checkpoint_dir / f"step_{step_index:03d}_{safe_name}.jsonl"

    def _progress_file_path(self, step_index: int, step_name: str) -> Path:
        safe_name = step_name.replace("/", "_").replace("\\", "_")
        return self.checkpoint_dir / f"step_{step_index:03d}_{safe_name}.progress.json"

    def _calls_file_path(self, step_index: int, step_name: str) -> Path:
        """One completed call id per line, appended after the record it produced."""
        safe_name = step_name.replace("/", "_").replace("\\", "_")
        return self.checkpoint_dir / f"step_{step_index:03d}_{safe_name}.calls"

    def has_checkpoint(self) -> bool:
        """Check if a checkpoint exists."""
        return self.manifest_path.exists()

    def load_manifest(self) -> Manifest | None:
        """Load the checkpoint manifest."""
        if not self.manifest_path.exists():
            return None
        with open(self.manifest_path, "r") as f:
            data = json.load(f)
        return Manifest.from_dict(data)

    def save_manifest(self, manifest: Manifest) -> None:
        """Save the checkpoint manifest."""
        manifest.updated_at = datetime.utcnow().isoformat() + "Z"
        with open(self.manifest_path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)

    def create_manifest(
        self,
        pipeline_hash: str,
        step_names: list[str],
        config: RunConfig,
    ) -> Manifest:
        """Create a new manifest for a pipeline."""
        now = datetime.utcnow().isoformat() + "Z"
        steps = [
            StepStatus(index=i, name=name, status="pending")
            for i, name in enumerate(step_names)
        ]
        config_dict = {
            "batch_size": config.batch_size,
            "llm_strategy": config.llm_strategy,
            "checkpoint_every": config.checkpoint_every,
        }

        manifest = Manifest(
            pipeline_hash=pipeline_hash,
            created_at=now,
            updated_at=now,
            steps=steps,
            current_step=0,
            config=config_dict,
        )
        self.save_manifest(manifest)
        return manifest

    def save_step_records(
        self,
        step_index: int,
        step_name: str,
        records: list[Record],
    ) -> None:
        """Save records after a step completes."""
        path = self._step_file_path(step_index, step_name)
        with open(path, "w") as f:
            for record in records:
                f.write(json.dumps(record, default=str) + "\n")
        logger.debug(f"Saved {len(records)} records to {path}")

    def append_record(
        self,
        step_index: int,
        step_name: str,
        record: Record,
        call_id: str | None = None,
    ) -> None:
        """Append a single record to the step JSONL file.

        The call id is appended to its own file immediately afterwards, so the two
        files can only ever disagree by the one call a crash interrupted. Writing the
        ids periodically instead is what made resume re-run and duplicate everything
        finished since the last save.
        """
        path = self._step_file_path(step_index, step_name)
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

        if call_id is not None:
            with open(self._calls_file_path(step_index, step_name), "a") as f:
                f.write(call_id + "\n")

    def clear_step_file(self, step_index: int, step_name: str) -> None:
        """Clear/create an empty step JSONL file for fresh writes."""
        self._step_file_path(step_index, step_name).write_text("")
        self._calls_file_path(step_index, step_name).write_text("")

    def load_step_records(self, step_index: int, step_name: str) -> list[Record]:
        """Load records from a step checkpoint."""
        path = self._step_file_path(step_index, step_name)
        if not path.exists():
            return []
        records = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        logger.debug(f"Loaded {len(records)} records from {path}")
        return records

    def save_llm_progress(
        self,
        step_index: int,
        step_name: str,
        progress: LLMStepProgress,
        partial_records: list[Record],
    ) -> None:
        """Save in-progress LLM step state."""
        path = self._progress_file_path(step_index, step_name)
        data = {
            "step_index": progress.step_index,
            "step_name": progress.step_name,
            "total_calls": progress.total_calls,
            "completed_call_ids": progress.completed_call_ids,
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        # The records file is append-only and authoritative; rewriting it here from
        # memory is what let it drift ahead of the completed-call list.
        logger.debug(
            f"Saved LLM progress: {progress.completed_calls}/{progress.total_calls}"
        )

    def load_llm_progress(
        self, step_index: int, step_name: str
    ) -> tuple[LLMStepProgress | None, list[Record]]:
        """Load in-progress LLM step state."""
        path = self._progress_file_path(step_index, step_name)
        if not path.exists():
            return None, []

        with open(path, "r") as f:
            data = json.load(f)

        records = self.load_step_records(step_index, step_name)
        call_ids = self._load_completed_call_ids(step_index, step_name, data)

        # A crash between the two appends leaves one more record than id. Dropping it
        # costs one repeated call; keeping it would write that record twice.
        kept = min(len(records), len(call_ids))
        if len(records) != kept:
            logger.warning(
                f"Discarding {len(records) - kept} record(s) written after the last "
                "completed call id; their calls will run again"
            )

        progress = LLMStepProgress(
            step_index=data["step_index"],
            step_name=data["step_name"],
            total_calls=data["total_calls"],
            completed_call_ids=call_ids[:kept],
        )
        return progress, records[:kept]

    def _load_completed_call_ids(
        self, step_index: int, step_name: str, data: dict[str, Any]
    ) -> list[str]:
        """Ids from the append-only file, falling back to a checkpoint written before
        that file existed."""
        path = self._calls_file_path(step_index, step_name)
        if not path.exists():
            return list(data.get("completed_call_ids", []))
        return [line for line in path.read_text().splitlines() if line.strip()]

    def mark_step_complete(
        self,
        manifest: Manifest,
        step_index: int,
        records_in: int,
        records_out: int,
    ) -> None:
        """Mark a step as complete in the manifest."""
        manifest.steps[step_index].status = "complete"
        manifest.steps[step_index].records_in = records_in
        manifest.steps[step_index].records_out = records_out
        manifest.current_step = step_index + 1
        self.save_manifest(manifest)

        # Glob rather than name the files: a Branch step also owns one progress and
        # one calls file per nested LLM step (step_NNN_Branch.<path>....progress.json).
        # Both are resume state, so a completed step leaves only its records behind.
        for suffix in ("progress.json", "calls"):
            for path in self.checkpoint_dir.glob(f"step_{step_index:03d}_*.{suffix}"):
                path.unlink(missing_ok=True)

    def mark_step_in_progress(self, manifest: Manifest, step_index: int) -> None:
        """Mark a step as in-progress."""
        manifest.steps[step_index].status = "in_progress"
        manifest.current_step = step_index
        self.save_manifest(manifest)

    def reset_from_step(self, manifest: Manifest, step_name: str) -> int:
        """Mark the named step and all later steps pending, discarding their
        saved records so execution re-runs from that step.

        Returns the index of the reset step. Raises ValueError if the name is
        not a step in the pipeline.
        """
        index = next(
            (s.index for s in manifest.steps if s.name == step_name), None
        )
        if index is None:
            names = ", ".join(s.name for s in manifest.steps)
            raise ValueError(
                f"resume_from step '{step_name}' not found. Steps: {names}"
            )

        for step in manifest.steps:
            if step.index >= index:
                step.status = "pending"
                step.records_in = None
                step.records_out = None
                # Glob rather than name the files: a Branch step also owns
                # per-path files (step_NNN_Branch.<path>....jsonl).
                for path in self.checkpoint_dir.glob(f"step_{step.index:03d}_*"):
                    path.unlink(missing_ok=True)

        manifest.current_step = index
        self.save_manifest(manifest)
        return index

    def find_resume_point(
        self, manifest: Manifest
    ) -> tuple[int, list[Record], LLMStepProgress | None]:
        """
        Find where to resume execution.

        Returns:
            Tuple of (step_index, records, llm_progress).
            If all steps complete, returns (-1, [], None).
        """
        for step_status in manifest.steps:
            if step_status.status == "complete":
                continue

            if step_status.status == "in_progress":
                prev_records = []
                if step_status.index > 0:
                    prev_step = manifest.steps[step_status.index - 1]
                    prev_records = self.load_step_records(prev_step.index, prev_step.name)

                llm_progress, partial_records = self.load_llm_progress(
                    step_status.index, step_status.name
                )
                return step_status.index, prev_records, llm_progress

            if step_status.status == "pending":
                prev_records = []
                if step_status.index > 0:
                    prev_step = manifest.steps[step_status.index - 1]
                    prev_records = self.load_step_records(prev_step.index, prev_step.name)
                return step_status.index, prev_records, None

        return -1, [], None

    def clear(self) -> None:
        """Remove all checkpoint files."""
        if self.checkpoint_dir.exists():
            for f in self.checkpoint_dir.iterdir():
                f.unlink()
            logger.info(f"Cleared checkpoint directory: {self.checkpoint_dir}")


def compute_pipeline_hash(step_names: list[str], step_types: list[str]) -> str:
    """
    Compute a hash for pipeline structure.

    Args:
        step_names: List of step names.
        step_types: List of step class names.

    Returns:
        Short hash string for pipeline identification.
    """
    content = "|".join(f"{name}:{typ}" for name, typ in zip(step_names, step_types))
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class PipelineChangedError(Exception):
    """Raised when pipeline structure has changed since checkpoint."""

    pass
