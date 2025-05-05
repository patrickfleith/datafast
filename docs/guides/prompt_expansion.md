# Prompt Expansion

Prompt expansion is a powerful technique in datafast that enables the generation of diverse and varied outputs by creating multiple versions of prompt templates. This document explains how to use prompt expansion effectively across different dataset types.

## What is Prompt Expansion?

Prompt expansion allows you to create multiple variants of your base prompts by substituting placeholders with different values. Instead of writing dozens of similar prompts manually, you can define a template with placeholders and automatically expand it into many unique prompts.

For example, a single template like:

```
"Explain how to use {{programming_language}} for {{application_type}} development"
```

Could expand to prompts like:
- "Explain how to use Python for web development"
- "Explain how to use Python for data analysis"
- "Explain how to use JavaScript for web development"
- "Explain how to use JavaScript for mobile development"

This dramatically increases the diversity of your generated data with minimal configuration.

## Key Concepts

### Placeholder Types

Datafast supports two types of placeholders:

1. **Mandatory Placeholders** (`{single_curly_braces}`): These are required placeholders specific to each dataset type and are filled in automatically by datafast based on your configuration.

2. **Optional Placeholders** (`{{double_curly_braces}}`): These are user-defined placeholders that expand into multiple variants based on your expansion configuration.

### Expansion Modes

Datafast offers two modes for expanding prompts:

1. **Combinatorial Mode** (default): Generates all possible combinations of placeholder values, resulting in comprehensive coverage of the prompt space.

2. **Random Sampling Mode**: Creates a specified number of random variations, useful when the total number of possible combinations would be too large.

## How to Configure Prompt Expansion

Prompt expansion is configured using the `PromptExpansionConfig` class, which is included in your dataset configuration.

```python
from datafast.schema.config import PromptExpansionConfig

expansion_config = PromptExpansionConfig(
    # Dictionary mapping placeholder names to lists of possible values
    placeholders={
        "difficulty_level": ["beginner", "intermediate", "advanced"],
        "topic_area": ["theory", "application", "history"]
    },
    # Whether to generate all combinations (True) or random samples (False)
    combinatorial=True,
    # Number of random samples to generate if combinatorial=False
    num_random_samples=100,
    # Safety limit to prevent generating too many combinations
    max_samples=1000
)
```

## Mandatory Placeholders by Dataset Type

Each dataset type in datafast requires specific mandatory placeholders to be included in your prompts:

### RawDataset

- `{num_samples}`: Number of examples to generate per prompt
- `{language_name}`: Language to generate content in
- `{document_type}`: Type of document to generate
- `{topic}`: Topic to generate content about

### MCQDataset

- `{num_samples}`: Number of questions to generate per document
- `{language_name}`: Language to generate questions in
- `{document}`: Source text to generate questions from

### UltraChatDataset

- `{num_samples}`: Number of conversations to generate
- `{language_name}`: Language to generate conversations in
- `{topic}`: Topic for the generated conversations

### TextClassificationDataset

- `{num_samples}`: Number of examples to generate per prompt
- `{language_name}`: Language to generate content in
- `{label}`: Label for classification
- `{label_description}`: Description of the label

## Usage Examples

### Basic Example with RawDataset

```python
from datafast.datasets import RawDataset
from datafast.schema.config import RawDatasetConfig, PromptExpansionConfig

config = RawDatasetConfig(
    # Basic configuration
    document_types=["research paper", "blog post"],
    topics=["machine learning", "data science"],
    num_samples_per_prompt=1,
    output_file="output.jsonl",
    languages={"en": "English"},
    
    # Custom prompt with both mandatory and optional placeholders
    prompts=[
        "Generate {num_samples} {document_type} in {language_name} " 
        "about {topic} with {{tone}} tone and targeted at {{audience}}."
    ],
    
    # Prompt expansion configuration
    expansion=PromptExpansionConfig(
        placeholders={
            "tone": ["formal", "casual", "technical"],
            "audience": ["beginners", "experts", "general public"]
        },
        combinatorial=True
    )
)
```

This configuration would generate 18 different prompt variations (2 document types × 2 topics × 3 tones × 3 audiences) which would then be used to generate the dataset.

### Random Sampling Example with MCQDataset

When the number of potential combinations is very large, use random sampling instead:

```python
expansion_config = PromptExpansionConfig(
    placeholders={
        "difficulty": ["elementary", "high-school", "undergraduate", "graduate", "expert"],
        "question_type": ["factual", "conceptual", "analytical", "applied"],
        "context_specificity": ["general", "specific"],
        "subject_area": ["history", "science", "literature", "mathematics", "art"]
    },
    combinatorial=False,  # Use random sampling
    num_random_samples=100  # Generate 100 random combinations
)
```

## How Many Outputs Will Be Generated?

The number of outputs depends on:

1. **Base Dataset Configuration**: Number of iterations from the dataset-specific parameters
2. **Number of LLM Providers**: Each provider will generate its own set of outputs
3. **Prompt Expansion**: Multiplies outputs based on combinatorial expansion or random sampling

For example, with `RawDataset`:
```
Total outputs = document_types × topics × languages × num_samples_per_prompt × num_providers × expansion_combinations
```

For `MCQDataset`:
```
Total questions = source_documents × num_samples_per_prompt × languages × num_providers × expansion_combinations
```

## Performance Considerations

Keep these tips in mind when using prompt expansion:

- **Memory Usage**: Large combinatorial expansions can consume significant memory
- **API Costs**: More expansions mean more API calls to LLM providers
- **Safety Limits**: The `max_samples` parameter prevents accidental generation of too many prompts
- **Prefer Random Sampling**: For very large expansion spaces, use random sampling instead of combinatorial expansion

## Under the Hood: The `expand_prompts` Function

Datafast implements prompt expansion using the `expand_prompts` function. This function is used internally by dataset classes, but you can also use it directly for custom applications:

```python
from datafast.expanders import expand_prompts

templates = ["Tell me about {{animal}} in {{environment}}"]
placeholders = {
    "animal": ["lions", "dolphins", "eagles"],
    "environment": ["desert", "ocean", "mountain"]
}

# Combinatorial expansion (all combinations)
expanded = expand_prompts(
    prompt_templates=templates,
    placeholders=placeholders,
    combinatorial=True
)

# Random sampling (5 random combinations)
expanded_random = expand_prompts(
    prompt_templates=templates,
    placeholders=placeholders,
    combinatorial=False,
    num_random_samples=5
)
```

Each expansion returns both the expanded prompt and metadata about which values were used for each placeholder.

## Best Practices

- **Start Small**: Begin with a small number of placeholders and values to understand the expansion
- **Monitor Expansion Size**: Check the expected number of outputs before running the full generation
- **Use Meaningful Variations**: Choose placeholder values that create meaningfully different prompts
- **Balance Coverage vs. Volume**: Decide whether complete coverage (combinatorial) or sampling is more appropriate
- **Use Multiple Providers**: Combine prompt expansion with multiple LLM providers for maximum diversity

## Common Pitfalls

- **Too Many Combinations**: Avoid creating too many combinations which can lead to memory issues or excessive API costs
- **Missing Mandatory Placeholders**: Ensure all required placeholders for your dataset type are included
- **Inconsistent Placeholder Format**: Use single curly braces for mandatory placeholders and double for optional ones
- **Unclear Expansions**: Having too many placeholders can make it hard to interpret the results

By using prompt expansion effectively, you can generate diverse, high-quality datasets that better represent the variety of real-world data, leading to more robust model training and evaluation.
