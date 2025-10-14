# Welcome to Datafast

Create high-quality and diverse synthetic text datasets in minutes, not weeks.

## Intended use cases
- Get initial evaluation text data instead of starting your LLM project blind.
- Increase diversity and coverage of another dataset by generating additional data.
- Experiment and test quickly LLM-based application PoCs
- Make your own datasets to fine-tune and evaluate language models for your application.

🌟 Star this repo if you find this useful! 

## Supported Dataset Types

- ✅ Text Classification Dataset
- ✅ Raw Text Generation Dataset
- ✅ Instruction Dataset (Ultrachat-like)
- ✅ Multiple Choice Question (MCQ) Dataset
- ✅ Preference Dataset
- ✅ Generic Pipeline Dataset
- ⏳ more to come...

## Supported LLM Providers

Currently we support the following LLM providers:

- ✔︎ OpenAI
- ✔︎ Anthropic
- ✔︎ Google Gemini
- ✔︎ Ollama (your local LLM server)
- ✔︎ OpenRouter (almost any LLM including open-source models)
- ⏳ more to come...

## Quick Start

### 1. Environment Setup

Make sure you have created a `secrets.env` file with your API keys.
HF token is needed if you want to push the dataset to your HF hub.
Other keys depends on which LLM providers you use.
```
GEMINI_API_KEY=XXXX
OPENAI_API_KEY=sk-XXXX
ANTHROPIC_API_KEY=sk-ant-XXXXX
OPENROUTER_API_KEY=XXXXX
HF_TOKEN=hf_XXXXX
```

### 2. Import Dependencies
```python
from datafast.datasets import ClassificationDataset
from datafast.schema.config import ClassificationDatasetConfig, PromptExpansionConfig
from datafast.llms import OpenAIProvider, AnthropicProvider, GeminiProvider, OpenRouterProvider
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
    OpenAIProvider(model_id="gpt-5-mini-2025-08-07"),
    AnthropicProvider(model_id="claude-sonnet-4-5-20250929"),
    GeminiProvider(model_id="gemini-2.5-flash"),
    OpenRouterProvider(model_id="z-ai/glm-4.6")
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

Check out our comprehensive guides for different dataset types:

- [Text Classification](guides/generating_text_classification_datasets.md) - Generate labeled datasets for text classification tasks
- [Text Generation](guides/generating_text_datasets.md) - Create datasets for general text generation tasks
- [Multiple Choice Questions](guides/generating_mcq_datasets.md) - Build datasets with multiple choice questions and answers
- [Instruction Following](guides/generating_ultrachat_datasets.md) - Develop instruction-following conversation datasets
- [Preference Pairs](guides/generating_preference_datasets.md) - Generate datasets for preference-based learning
- [Generic Pipeline](guides/generating_generic_pipeline_datasets.md) - Build custom input-output LLM synthetic data generation pipelines

To understand the core concepts behind Datafast, visit our [Concepts](concepts.md) page.

Star this package to send positive vibes and support 🌟

## Key Features

* **Easy-to-use** and simple interface 🚀
* **Multi-lingual** datasets generation 🌍
* **Multiple LLMs** used to boost dataset diversity 🤖
* **Flexible prompt**: use our default prompts or provide your own custom prompts 📝
* **Prompt expansion**: Combinatorial variation of prompts to maximize diversity 🔄
* **Hugging Face Integration**: Push generated datasets to the Hub 🤗

!!! warning
    This library is in its early stages of development and might change significantly.

## Roadmap

- RAG datasets
- Integrate personas
- Integrate seeds
- More types of instructions datasets (not just ultrachat)
- More LLM providers
- Deduplication, filtering
- Dataset cards generation

## License

[Apache 2.0](LICENSE)

## Creator

Made with ❤️ by [Patrick Fleith](https://www.linkedin.com/in/patrick-fleith/).
