# Checkpointing

Datafast checkpoints after each executed step when `checkpoint_dir` is set.

## Basic Usage

```python
results = pipeline.run(
    checkpoint_dir="examples/checkpoints/demo",
    resume=True,
    batch_size=4,
)
```

## Useful Run Controls

- `checkpoint_dir`: enable checkpoint files
- `resume=True`: continue from the latest valid checkpoint
- `batch_size`: number of LLM calls per batch
- `llm_strategy`: ordering strategy for LLM calls
- `limit`: process only the first `N` records
- `stop_after`: stop after a given step index or step name

## Branching

Checkpointing works with branching pipelines as well. Each step still materializes in sequence, which makes intermediate outputs inspectable and resumable.

## Examples

See:

- `examples/scripts/36_checkpointing_and_resume.py`
- `examples/scripts/37_limit_stop_and_strategies.py`
