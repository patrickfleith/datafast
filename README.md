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

Make sure you have created an `secrets.env` file with your API keys.
HF token is needed if you want to push the dataset to your HF hub.
Other keys depends on which LLM providers you use.
```
GOOGLE_API_KEY=XXXX
OPENAI_API_KEY=sk-XXXX
ANTHROPIC_API_KEY=sk-ant-XXXXX
HF_TOKEN=hf_XXXXX
```

```python
from datafast.datasets import TextClassificationDataset
from datafast.schema.config import ClassificationConfig, PromptExpansionConfig
from datafast.llms import OpenAIProvider, AnthropicProvider, GoogleProvider
from dotenv import load_dotenv

# Load environment variables
load_dotenv("secrets.env") # <--- your API keys

# Configure the dataset for text classification
config = ClassificationConfig(
    classes=[
        {"name": "positive", "description": "Text expressing positive emotions or approval"},
        {"name": "negative", "description": "Text expressing negative emotions or criticism"}
    ],
    num_samples_per_prompt=5,
    output_file="outdoor_activities_sentiments.jsonl",
    languages={
        "en": "English", 
        "fr": "French"
    },
    prompts=[
        (
            "Generate {num_samples} reviews in {language_name} which are diverse "
            "and representative of a '{label_name}' sentiment class. "
            "{label_description}. The reviews should be {{style}} and in the "
            "context of {{context}}."
        )
    ],
    expansion=PromptExpansionConfig(
        placeholders={
            "context": ["hike review", "speedboat tour review", "outdoor climbing experience"],
            "style": ["brief", "detailed"]
        },
        combinatorial=True
    )
)

# Create LLM providers
providers = [
    OpenAIProvider(model_id="gpt-4o-mini"),
    AnthropicProvider(model_id="claude-3-5-haiku-latest"),
    GoogleProvider(model_id="gemini-1.5-flash")
]

# Generate dataset
dataset = TextClassificationDataset(config)
dataset.generate(providers)

# Optional: Push to Hugging Face Hub
dataset.push_to_hub(
    repo_id="YOUR_USERNAME/YOUR_DATASET_NAME",
    train_size=0.6
)
```

## Testing
No tests available yet.

## Project Details
- **Status:** Work in Progress (APIs may change)
- **License:** [GNU AGPL v3](LICENSE) -> This may change to MIT or Apache 2.0.