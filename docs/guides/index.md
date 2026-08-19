# Guides

Task-shaped pages. Each one takes a job you are trying to do and walks it end to end.

- [Pipelines & execution](pipelines_and_execution.md) — composing steps with `>>`, what
  `compile()` catches, and every option `run()` takes: checkpointing, resume, batching
  and call ordering.
- [Structured output](structured_output.md) — getting named columns out of a model,
  and the difference between parsing a reply and constraining it.
- [Multimodal input](multimodal_input.md) — sending images, audio, video and files, and
  how a served model decides what it accepts.
- [Calling a served model](calling_a_served_model.md) — using a model directly, without
  a pipeline.
- [Error handling & troubleshooting](troubleshooting.md) — what each error means, what
  survives a crash, and the failures that are silent.
- [Langfuse tracing](langfuse_tracing.md) — recording every call to Langfuse.
