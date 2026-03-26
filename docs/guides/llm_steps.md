# LLM Steps

## `LLMStep`

`LLMStep` is the generic generation primitive.

```python
from datafast import LLMStep, Sink, Source, openrouter

pipeline = (
    Source.list([{"topic": "fusion power"}])
    >> LLMStep(
        prompt="Write one short question about {topic}",
        input_columns=["topic"],
        output_column="question",
        model=openrouter("z-ai/glm-4.6"),
    )
    >> Sink.list()
)
```

## Parse Modes

- `text`: write the raw model output into one column
- `json`: parse a JSON object into named columns
- `xml`: parse XML-style tags into named columns

## Specialized Steps

- `Classify`: assign labels
- `Score`: assign numeric or categorical scores
- `Compare`: compare two candidate columns
- `Rewrite`: transform style or wording
- `Extract`: extract structured fields from text

## Common Patterns

Use multiple prompts or multiple models:

```python
LLMStep(
    prompt=["Summarize {text}", "List key points from {text}"],
    input_columns=["text"],
    output_column="result",
    model=[openrouter("z-ai/glm-4.6"), openrouter("qwen/qwen3-next-80b-a3b-instruct")],
)
```

Use `Branch` and `JoinBranches` when different LLM paths should be compared or merged later.
