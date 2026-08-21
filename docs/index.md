# Datafast
Generate high-quality and diverse synthetic data for your project with LLMs.

In datafast, you assemble building blocks of dataset engineering operations would otherwise do manually. It designed to be flexible and modular, so you get a custom pipeline that generates a dataset matching your actual project needs. It works with open-weights and proprietary models.

### Intended Use Cases

- Get initial evaluation text data instead of starting your LLM project blind.
- Increase diversity and coverage of another dataset by generating additional data.
- Experiment and test quickly LLM-based applications, and before shipping to production.
- Make your own datasets to fine-tune and evaluate LLMs and agents.
- Model distillation

🌟 Star this repo if you find this useful!

### Example Pipeline

The simple pipeline below combines seeds topics and difficulty levels to generate questiosn and answers, written directly to a JSONL file.

```python
from datafast import LLMStep, Seed, Sink, openai

pipeline = (
    # Create two columns combining topic x level (the 'seeds')
    Seed.product(
        Seed.values("topic", ["photosynthesis", "plate tectonics", "virus"]),
        Seed.values("level", ["beginner", "advanced"]),
    )
    # Use seeds in LLM prompt to generate structured outputs of Q&As.
    >> LLMStep(
        prompt="Write one {level} exam question about {topic}, with its answer. "
               "Return JSON with fields question and answer.",
        input_columns=["topic", "level"],
        output_columns=["question", "answer"],
        parse_mode="json",
        model=openai(),
    )
    # Save locally to a file
    >> Sink.jsonl("questions.jsonl")
)

# Run the pipeline
pipeline.run()
```

## Why pipelines

AI projects all needs something special in their datasets. Wether this is for training, fine tuning, evaluation, or just to test the application coverage in realistic conditions. So we need something highly flexible and modular: Like assembling LEGOs, datafast provides building blocks so you can build your synthetic data generation pipleine that matches your need.

## Where to start?

- Follow the [Quickstart](quickstart.md) - from install to your first dataset.
- Read [Concepts](concepts.md) for the execution model.
- Read [Sources & seed](reference/sources_and_seed.md) and [Data
  ops](reference/data_ops.md) for sources, transforms and sinks.
- Read [LLM step](reference/llm_step.md) for generation and evaluation steps.
- Read [Pipelines & execution](guides/pipelines_and_execution.md) for resume and
  execution controls.
- Read [Served models](llms.md) for provider configuration, reasoning and multimodal
  input.
- Browse the [Cookbook](cookbook/index.md) for complete recipes.
- Look up anything in the [API reference](api.md).

### Coming soon

- An `llms.txt` so you can ask questions about of docs to your favorite AI.
- A `datafast` SKILL so your coding agent can build the pipeline the right way under your supervision.
- Integration with llama.cpp, vLLM