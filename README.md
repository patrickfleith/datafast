<div align="left">
  <img src="./assets/Datafast Logo.png" alt="Datafast Logo" width="300px">
</div>
    
> *Generate text datasets for LLMs in minutes, not weeks.*

## Intended use cases
- Get initial evaluation text data instead of starting your LLM project blind.
- Increase diversity and coverage of an existing dataset by generating more data.
- Experiment and test quickly LLM-based application PoCs.
- Make your own datasets to fine-tune and evaluate language models for your application.

🌟 Star this repo if you find this useful! 

## Supported Dataset Types

- ✅ Text Classification Dataset
- ✅ Raw Text Generation Dataset
- ✅ Instruction Dataset (Ultrachat-like)
- ✅ Multiple Choice Question (MCQ) Dataset
- ✅ Preference Dataset
- ⏳ more to come...

## Supported LLM Providers

Currently we support the following LLM providers:

- ✔︎ OpenAI
- ✔︎ Anthropic
- ✔︎ Google Gemini
- ✔︎ Ollama (local LLM server)
- ⏳ more to come...


Try it in Colab: 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/patrickfleith/datapipes/blob/main/notebooks/Datafast_Quickstart_Text_Classification.ipynb)


## Installation
```bash
pip install datafast
```

## Quick Start

### 1. Environment Setup

Make sure you have created a `secrets.env` file with your API keys.
HF token is needed if you want to push the dataset to your HF hub.
Other keys depends on which LLM providers you use.
```
GEMINI_API_KEY=XXXX
OPENAI_API_KEY=sk-XXXX
ANTHROPIC_API_KEY=sk-ant-XXXXX
HF_TOKEN=hf_XXXXX
```

### 2. Import Dependencies
```python
from datafast.datasets import ClassificationDataset
from datafast.schema.config import ClassificationDatasetConfig, PromptExpansionConfig
from datafast.llms import OpenAIProvider, AnthropicProvider, GeminiProvider
from dotenv import load_dotenv

# Load environment variables
load_dotenv("secrets.env") # <--- your API keys
```

### 3. Configure Dataset
```python
# Configure the dataset for text classification
config = ClassificationDatasetConfig(
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
    OpenAIProvider(model_id="gpt-4.1-mini-2025-04-14"),
    AnthropicProvider(model_id="claude-3-5-haiku-latest"),
    GeminiProvider(model_id="gemini-2.0-flash")
]
```

### 5. Generate and Push Dataset
```python
# Generate dataset and local save
dataset = ClassificationDataset(config)
dataset.generate(providers)

# Optional: Push to Hugging Face Hub
dataset.push_to_hub(
    repo_id="YOUR_USERNAME/YOUR_DATASET_NAME",
    train_size=0.6
)
```

## Next Steps

Check out our guides for different dataset types:

* [How to Generate a Text Classification Dataset](https://patrickfleith.github.io/datafast/guides/generating_text_classification_datasets/)
* [How to Create a Raw Text Dataset](https://patrickfleith.github.io/datafast/guides/generating_text_datasets/)
* [How to Create a Preference Dataset](https://patrickfleith.github.io/datafast/guides/generating_preference_datasets/)
* [How to Create a Multiple Choice Question (MCQ) Dataset](https://patrickfleith.github.io/datafast/guides/generating_mcq_datasets/)
* [How to Create an Instruction (Ultrachat) Dataset](https://patrickfleith.github.io/datafast/guides/generating_ultrachat_datasets/)
* Star and watch this github repo to get updates 🌟

## Key Features

* **Easy-to-use** and simple interface 🚀
* **Multi-lingual** datasets generation 🌍
* **Multiple LLMs** used to boost dataset diversity 🤖
* **Flexible prompt**: use our default prompts or provide your own custom prompts 📝
* **Prompt expansion**: Combinatorial variation of prompts to maximize diversity 🔄
* **Hugging Face Integration**: Push generated datasets to the Hub 🤗

> [!WARNING]
> This library is in its early stages of development and might change significantly.

## Roadmap:

- RAG datasets
- Personas
- Seeds
- More types of instructions datasets (not just ultrachat)
- More LLM providers
- Deduplication, filtering
- Dataset cards generation

## Creator

Made with ❤️ by [Patrick Fleith](https://www.linkedin.com/in/patrick-fleith/).

<hr> 

This is volunteer work, star this repo to show your support! 🙏


## Project Details
- **Status:** Work in Progress (APIs may change)
- **License:** [Apache 2.0](LICENSE)
