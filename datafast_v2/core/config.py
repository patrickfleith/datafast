"""Configuration dataclasses for pipeline execution."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LLMExecutionStrategy(Enum):
    """Strategy for ordering LLM calls during execution."""
    
    BY_MODEL = "by_model"
    """Execute all calls for model A first, then model B, etc."""
    
    ROUND_ROBIN = "round_robin"
    """Interleave models: A₁, B₁, A₂, B₂, ..."""
    
    BY_RECORD = "by_record"
    """Process each record completely before moving to next (default LLMStep behavior)."""


@dataclass
class RunConfig:
    """Configuration for pipeline execution."""
    
    checkpoint_dir: str | None = None
    """Directory for checkpoint files. None disables checkpointing."""
    
    resume: bool = False
    """Whether to resume from existing checkpoint."""
    
    resume_from: str | None = None
    """Step name to resume from (discards later steps)."""
    
    stop_after: int | str | None = None
    """Stop after this step (index or name)."""
    
    limit: int | None = None
    """Process only first N records from source."""
    
    batch_size: int = 4
    """Number of LLM calls per batch."""
    
    max_concurrent: int = 1
    """Maximum concurrent batches (for async execution)."""
    
    llm_strategy: str = "by_model"
    """LLM execution strategy: 'by_model', 'round_robin', or 'by_record'."""
    
    rate_limits: dict[str, int] | None = None
    """Requests per minute per model ID. Example: {"gpt-4o-mini": 60}."""
    
    checkpoint_every: int = 100
    """Checkpoint LLM progress every N calls."""
    
    show_progress: bool = True
    """Show progress bar during execution."""
    
    log_level: str = "INFO"
    """Logging level."""

    def get_strategy(self) -> LLMExecutionStrategy:
        """Get the LLM execution strategy enum value."""
        return LLMExecutionStrategy(self.llm_strategy)


@dataclass
class LLMCall:
    """Represents a single LLM call to be executed."""
    
    call_id: str
    """Unique identifier for this call."""
    
    record: dict[str, Any]
    """Source record this call is derived from."""
    
    record_index: int
    """Index of the source record."""
    
    prompt_template: str
    """Original prompt template."""
    
    prompt_index: int
    """Index of prompt in prompt list."""
    
    model_id: str
    """Model identifier."""
    
    language_code: str
    """Language code (empty string if no language)."""
    
    language_name: str
    """Language name (empty string if no language)."""
    
    messages: list[dict[str, str]]
    """Pre-built messages for LLM API."""
    
    output_index: int
    """Index for num_outputs (0 to num_outputs-1)."""


@dataclass
class StepStatus:
    """Status of a single step in checkpoint manifest."""
    
    index: int
    """Step index in pipeline."""
    
    name: str
    """Step name."""
    
    status: str
    """Status: 'pending', 'in_progress', 'complete'."""
    
    records_in: int | None = None
    """Number of input records."""
    
    records_out: int | None = None
    """Number of output records."""


@dataclass
class LLMStepProgress:
    """Progress tracking within an LLM step."""
    
    step_index: int
    """Index of the LLM step."""
    
    step_name: str
    """Name of the LLM step."""
    
    total_calls: int
    """Total number of LLM calls."""
    
    completed_call_ids: list[str] = field(default_factory=list)
    """IDs of completed calls."""
    
    @property
    def completed_calls(self) -> int:
        """Number of completed calls."""
        return len(self.completed_call_ids)
    
    @property
    def progress_fraction(self) -> float:
        """Fraction of calls completed (0.0 to 1.0)."""
        if self.total_calls == 0:
            return 1.0
        return self.completed_calls / self.total_calls
