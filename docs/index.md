# Datafast: Synthetic Text Dataset Generator

Datafast is a powerful Python package designed to generate synthetic text datasets, designed to:

* Experiment and test LLM-based applications
* Fine-tune and evaluate language models (LLMs / NLP)

!!! warning
    This library is in its early stages of development and might change significantly.

### Key Features

🚀 **Easy-to-use** and simple interface

🌍 **Multi-lingual** datasets generation 

🤖 **Multiple LLMs** used to boost dataset **diversity**

📝 **Flexible prompt**: default or custom

🔄 **Prompt expansion** to maximize diversity

🤗 **Hugging Face Integration**: Push generated datasets to the Hub, soon to argilla

## Quick Start

### 1. Configuration
```python
from datafast import ClassificationConfig, TextClassificationDataset
from datafast.schema.config import PromptExpansionConfig

config = ClassificationConfig(
    classes=[
        {"name": "positive", "description": "Text expressing positive emotions or approval"},
        {"name": "negative", "description": "Text expressing negative emotions or criticism"}
    ],
    num_samples_per_prompt=5,
    output_file="sentiment_dataset.jsonl",
    languages={"en": "English"},
    expansion=PromptExpansionConfig(
        placeholders={
            "context": ["product", "movie", "restaurant"],
            "style": ["brief", "detailed"]
        },
        combinatorial=True
    )
)
```

### 2. LLM Providers
```python
from datafast.llms import OpenAIProvider, AnthropicProvider, GoogleProvider

providers = [
    OpenAIProvider(model_id="gpt-4o-mini"),
    AnthropicProvider(model_id="claude-3-5-haiku-latest"),
    GoogleProvider(model_id="gemini-1.5-flash")
]
```

### 3. Dataset Generation
```python
# Generate dataset
dataset = TextClassificationDataset(config)
dataset.generate(providers)

# Optional: Push to Hugging Face Hub
dataset.push_to_hub(
    repo_id="YOUR_USERNAME/sentiment-dataset",
    train_size=0.8
)
```

## Supported Dataset Types

Currently supported dataset types:

* ✅ Text Classification
* 📋 More coming soon!

## Next Steps

* Visit our [GitHub repository](https://github.com/patrickfleith/datafast) for the latest updates
