# How to Create a Text Classification Dataset

This guide walks you through the process of creating a custom text classification dataset using Datafast.

## Use case overview

We'll create a sentiment classification dataset using synthetic data generated from LLMs. Specifically, to classify travel outdoor activities reviews with the following characteristics:

* Multi-class: the review belongs to one of the following classes (positive, negative, neutral)
* Multi-lingual: the reviews in the dataset will be in several languages
* Multi-LLM: we generate examples using multiple LLM providers to boost diversity
* We'll push the dataset to your Hugging Face Hub.

## Step 1: Import Required Modules

Generating a dataset with `datafast` requires 3 types of imports:

* Dataset
* Configs
* LLM Providers

```python
from datafast.datasets import TextClassificationDataset
from datafast.schema.config import ClassificationConfig, PromptExpansionConfig
from datafast.llms import OpenAIProvider, AnthropicProvider, GoogleProvider
```

In addition, we'll use `dotenv` to load environment variables containing API keys.
```python
from dotenv import load_dotenv

# Load environment variables containing API keys
load_dotenv("secrets.env")
```

## Step 2: Configure Your Dataset

The `ClassificationConfig` class defines all parameters for your text classification dataset.

- **`classes`**: List of dictionaries defining your classification labels. Each dictionary represent a class and should include:
    - `name`: Label identifier (required)
    - `description`: Detailed description of what this class represents (required)

- **`num_samples_per_prompt`**: Number of examples to generate in a single LLM call. 

!!! note
    My recommendation is to use a larger number (like 10-20 if you generate very short texts of 10 to 30 words).
    Use a smaller number (like 5) if you generate longer texts like 100 to 300 words.
    Use an even smaller number (like 2-3) if you generate very long texts like 300+ words.


- **`output_file`**: Path where the generated dataset will be saved (JSONL format). 

- **`languages`**: Dictionary mapping language codes to their names (e.g., `{"en": "English"}`)

- **`prompts`**: (Optional, but highly recommended) Custom prompt templates.
    - **Mandatory placeholders**: When providing a custom prompt for `ClassificationConfig`, you must always include the following placeholders, using single curly braces:
        - `{num_samples}`: it uses the `num_samples_per_prompt` parameter defined above)
        - `{language_name}`: it uses the `languages` parameter defined above)
        - `{label_name}`: it comes from the `classes` parameter defined above)
        - `{label_description}`: it comes from the `classes` parameter defined above)
    - **Optional placeholders**: These placeholders can be used to expand the diversity of your dataset. They are optional, but can help you create a more diverse dataset. They must be written using double curly braces. For example:
        - `{{style}}`: In our example we want to generate reviews in different writting style
        - `{{context}}`: In our example we want to generate reviews of different outdoor activities

```python
config = ClassificationConfig(
    # Define your classification classes
    classes=[
        {
            "name": "positive",
            "description": "Text expressing positive emotions, approval, or favorable opinions"
        },
        {
            "name": "negative",
            "description": "Text expressing negative emotions, criticism, or unfavorable opinions"
        },
        {
            "name": "neutral",
            "description": "Text expressing factual, objective, or balanced statements without strong emotions"
        }
    ],
    # Number of examples to generate per prompt
    num_samples_per_prompt=5,
    
    # Output file path
    output_file="sentiment_dataset.jsonl",
    
    # Languages to generate data for
    languages={"en": "English", "fr": "French"},
    
    # Custom prompts (optional - otherwise defaults will be used)
    prompts=[
        (
            "Generate {num_samples} reviews in {language_name} which are diverse "
            "and representative of a '{label_name}' sentiment class. "
            "{label_description}. The reviews should be {{style}} and in the "
            "context of {{context}}."
        )
    ]
)
```

## Step 3: Prompt Expansion for Diverse Examples

Datafast's `PromptExpansionConfig` allows you to generate diverse examples by creating variations of your base prompts:

```python
config = ClassificationConfig(
    # Basic configuration as above
    # ...
    
    # Add prompt expansion configuration
    expansion=PromptExpansionConfig(
        placeholders={
            "context": [
                "product review",
                "movie review", 
                "restaurant experience",
                "customer service"
            ],
            "style": ["brief", "detailed"]
        },
        combinatorial=True  # Generate all combinations
    )
)
```

This expansion creates prompt variations by replacing `{style}` and `{context}` with all possible combinations of the provided values, dramatically increasing the diversity of your dataset.

