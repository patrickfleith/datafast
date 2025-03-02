# Welcome to Datafast

Datafast is a Python package for high-quality and diverse synthetic text dataset generation. 

It is designed **to help you get the data you need** to:

* Experiment and test LLM-based applications
* Fine-tune and evaluate language models (LLMs / NLP)

!!! warning
    This library is in its early stages of development and might change significantly.

### Key Features

* **Easy-to-use** and simple interface 🚀
* **Multi-lingual** datasets generation 🌍
* **Multiple LLMs** used to boost dataset diversity 🤖
* **Flexible prompt**: default or custom 📝
* **Prompt expansion** to maximize diversity 🔄
* **Hugging Face Integration**: Push generated datasets to the Hub, soon to argilla 🤗

## Supported Dataset Types

Currently we support the following dataset types:

- ✅ Text Classification
- ✅ Raw Text Generation
- ✅ Instruction Dataset
    - UltraChat
- 📋 More coming soon!

## Quick Start

### 1. Environment Setup

Make sure you have created a `secrets.env` file with your API keys.
HF token is needed if you want to push the dataset to your HF hub.
Other keys depends on which LLM providers you use.
```
GOOGLE_API_KEY=XXXX
OPENAI_API_KEY=sk-XXXX
ANTHROPIC_API_KEY=sk-ant-XXXXX
HF_TOKEN=hf_XXXXX
```

### 2. Import Dependencies
```python
from datafast.datasets import TextClassificationDataset
from datafast.schema.config import ClassificationConfig, PromptExpansionConfig
from datafast.llms import OpenAIProvider, AnthropicProvider, GoogleProvider
from dotenv import load_dotenv

# Load environment variables
load_dotenv("secrets.env") # <--- your API keys
```

### 3. Configure Dataset
```python
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
```

### 4. Setup LLM Providers
```python
# Create LLM providers
providers = [
    OpenAIProvider(model_id="gpt-4o-mini"),
    AnthropicProvider(model_id="claude-3-5-haiku-latest"),
    GoogleProvider(model_id="gemini-1.5-flash")
]
```

### 5. Generate and Push Dataset
```python
# Generate dataset
dataset = TextClassificationDataset(config)
dataset.generate(providers)

# Optional: Push to Hugging Face Hub
dataset.push_to_hub(
    repo_id="YOUR_USERNAME/YOUR_DATASET_NAME",
    train_size=0.6
)
```

## Next Steps

* Check out our full example guide on [How to Generate a Text Classification Dataset](guides/generating_text_classification_datasets.md)
* Visit our [GitHub repository](https://github.com/patrickfleith/datafast) for the latest updates

## Creator

Made with ❤️ by [Patrick Fleith](https://www.linkedin.com/in/patrick-fleith/).
