<div align="center">

<img src="https://raw.githubusercontent.com/patrickfleith/datafast/main/assets/Datafast%20Logo.png" alt="Datafast logo" width="160">

<h3>Generate diverse synthetic datasets with LLMs, from a pipeline you can read.</h3>

[![PyPI version](https://img.shields.io/pypi/v/datafast?color=f5c518)](https://pypi.org/project/datafast/)
[![Python versions](https://img.shields.io/pypi/pyversions/datafast)](https://pypi.org/project/datafast/)
[![License](https://img.shields.io/github/license/patrickfleith/datafast)](https://github.com/patrickfleith/datafast/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://patrickfleith.github.io/datafast/)
[![GitHub stars](https://img.shields.io/github/stars/patrickfleith/datafast?style=social)](https://github.com/patrickfleith/datafast)

**[Documentation](https://patrickfleith.github.io/datafast/)**
· [Quickstart](https://patrickfleith.github.io/datafast/quickstart/)
· [Cookbook](https://patrickfleith.github.io/datafast/cookbook/)
· [Examples](https://patrickfleith.github.io/datafast/cookbook/examples/)
· [API reference](https://patrickfleith.github.io/datafast/api/)

</div>

Datafast is a Python library for developers and ML engineers who need a dataset for an
LLM project and do not have one yet.

A plain script that loops over prompts gets hard to control. You must write the
nested loops that cover each combination of inputs. Without checkpoints, a crash
halfway means you pay for the same LLM calls again. Without metadata, you cannot tell
which input and which model made each record.

With datafast, you declare the dimensions that your dataset must cover, connect steps
into a pipeline, and run it. The pipeline writes the dataset to JSONL, CSV, Parquet or
the Hugging Face Hub.

## Install

```bash
pip install datafast
```

Datafast needs Python 3.10 or newer. Extras add Parquet, the Hugging Face Hub and
Langfuse tracing: see [Installation](https://patrickfleith.github.io/datafast/installation/).

## Quick example

This example generates exam questions and answers for 3 science topics at 2 levels,
beginner and advanced. The result is 6 example records.

Set the key for your provider, here OpenAI:

```bash
export OPENAI_API_KEY="sk-..."
```

```python
from datafast import LLMStep, Seed, Sink, openai

pipeline = (
    Seed.product(
        Seed.values("topic", ["photosynthesis", "plate tectonics", "vaccines"]),
        Seed.values("level", ["beginner", "advanced"]),
    )
    >> LLMStep(
        prompt=(
            "Write one {level} exam question about {topic}, with its answer. "
            "Return JSON with fields question and answer."
        ),
        input_columns=["topic", "level"],
        output_columns=["question", "answer"],
        parse_mode="json",
        model=openai(),
    )
    >> Sink.jsonl("questions.jsonl")
)

pipeline.run()
```

The pipeline has 3 steps:

1. `Seed.product` makes 1 record for each combination of the values in its dimensions:
   3 topics × 2 levels = 6 records.
2. `LLMStep` makes 1 LLM call for each record. With `parse_mode="json"`, it splits the
   response into the `question` and `answer` columns.
3. `Sink.jsonl` writes the records to `questions.jsonl`.

Each line of the file is 1 record. The seed columns stay beside the generated columns,
and `_model` tells you which model wrote the record:

```json
{
  "topic": "photosynthesis",
  "level": "beginner",
  "question": "What process do plants use to convert light into chemical energy?",
  "answer": "Photosynthesis, which occurs in the chloroplasts.",
  "_model": "gpt-6-luna"
}
```

The [Quickstart](https://patrickfleith.github.io/datafast/quickstart/) explains each
step in more detail. [Key concepts](https://patrickfleith.github.io/datafast/concepts/)
and the [API reference](https://patrickfleith.github.io/datafast/api/) list all the
steps.

## Use cases

- Make evaluation data for an LLM project before you have real user data.
- Add records to an existing dataset to cover more cases.
- Test an LLM application on realistic inputs before you ship it.
- Make datasets to fine-tune LLMs and agents.
- Distill a large model into a smaller one: the large model writes the training data.

## Why datafast

- **You declare the coverage.** You give the values of each dimension, and
  `Seed.product` makes all the combinations. You do not write nested loops.
- **A long run can resume after a crash.** Set `checkpoint_dir` to save progress,
  then restart with the same directory and `resume=True`. See
  [Checkpoints and resume](https://patrickfleith.github.io/datafast/guides/pipelines_and_execution/#checkpoints-and-resume) for recovery limits.
- **You can change the provider in 1 line.** The same provider factories work for
  OpenAI, Anthropic, Gemini, Mistral, OpenRouter, Ollama and all OpenAI-compatible
  servers such as vLLM and llama.cpp. See
  [Served models](https://patrickfleith.github.io/datafast/llms/).
- **Datafast finds mistakes before you pay.** `Pipeline.compile()` checks the step
  order and all column references before the first LLM call.
- **Each record shows its origin.** The seed values and the model id stay in the
  record, so you can filter or audit the dataset later.

## Cookbook

Each cookbook is a complete pipeline that makes a real dataset. The script is in
`examples/scripts/`, and the page explains each step.

| Cookbook | Dataset | Script |
|----------|---------|--------|
| [Text classification](https://patrickfleith.github.io/datafast/cookbook/text_classification/) | Trail reports from hikers in 2 languages, with labels | [`45`](https://github.com/patrickfleith/datafast/blob/main/examples/scripts/45_cookbook_text_classification.py) |
| [Personas](https://patrickfleith.github.io/datafast/cookbook/persona_generation/) | Personas made from real articles | [`43`](https://github.com/patrickfleith/datafast/blob/main/examples/scripts/43_cookbook_persona_generation.py) |
| [Domain text corpus](https://patrickfleith.github.io/datafast/cookbook/space_text_generation/) | Spacecraft engineering text in 2 languages | [`44`](https://github.com/patrickfleith/datafast/blob/main/examples/scripts/44_cookbook_space_text_generation.py) |
| [Preference pairs](https://patrickfleith.github.io/datafast/cookbook/preference_with_scoring/) | Chosen and rejected answers with scores, for RLHF | [`42`](https://github.com/patrickfleith/datafast/blob/main/examples/scripts/42_pipeline_preference_with_scoring.py) |

The [examples page](https://patrickfleith.github.io/datafast/cookbook/examples/) lists
all 45 scripts. Scripts 01 to 14 do not use an API key.

## Contributing

Bug reports, ideas and pull requests are welcome. Open an
[issue](https://github.com/patrickfleith/datafast/issues) or read the
[contributing guide](https://patrickfleith.github.io/datafast/contributing/).

## License

Datafast is released under the [Apache 2.0 license](https://github.com/patrickfleith/datafast/blob/main/LICENSE).

If datafast helps you, give the repo a star. It helps other people find it.
