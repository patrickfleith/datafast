# Welcome to Datafast

**datafast** is a Python package for synthetic text dataset generation.
- Ideal to get the data you need to *experiment and test LLM-based applications*.
- Made to generate diverse datasets to *fine-tune or evaluate* LLMs / NLPs models.

> [!WARNING]
> This library is in its early stages of development and might change significantly.

## Supported Dataset Types
- [X] Text Classification Dataset generation

> [!NOTE]
> We'll add more as the API design choices prove to be effective.

## Features:
- Easy to use API (see examples)
- Multi-Lingual dataset generation
- Multiple LLMs to boost diversity
- Default or custom prompt templates
- Prompt expansion to ensure diversity
- Push the dataset to Hugging Face Hub


## Installation
```bash
pip install datafast
```

## Usage
```python
from datafast.datasets import TextClassificationDataset
from datafast.schema.config import ClassificationConfig
from datafast.llms import OpenAIProvider, AnthropicProvider, GoogleProvider
from dotenv import load_dotenv

# Load environment variables
load_dotenv("secrets.env") # <--- your API keys

# Configure the dataset for text classification
config = ClassificationConfig(
    classes=[
        {"name": "concise", "description": "Concise text - clear and precise."},
        {"name": "verbose", "description": "Verbose text - detailed and redundant."}
    ],
    num_samples_per_prompt=5,
    output_file="concise_vs_verbose.jsonl",
    languages={"en": "English", "fr": "French"},
)

# Create LLM providers
providers = [
    OpenAIProvider(model_id="gpt-4o-mini"),
    AnthropicProvider(model_id="claude-3-5-sonnet-latest"),
    GoogleProvider(model_id="gemini-1.5-flash"),
]

# Generate the dataset
dataset = TextClassificationDataset(config)
dataset.generate(providers)

# Optionally, push the dataset to Hugging Face Hub
repo_url = dataset.push_to_hub(
    repo_id="YOUR_USERNAME/YOUR_DATASET_NAME",
    train_size=0.75
)
```

## Testing
No tests available yet.

## Project Details
- **Status:** Work in Progress (APIs may change)
- **License:** [GNU AGPL v3](LICENSE) -> This may change to MIT or Apache 2.0.