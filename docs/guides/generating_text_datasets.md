# How to Create a Raw Text Dataset

!!! example "Use Case"
    Let's say you're an AI researcher at a *Space Agency* and you need to improve the text generation capabilities of language models specialized for **space engineering**. 
    You're tasked with creating a diverse corpus of text covering various space engineering topics to pre-train your foundation model before fine-tuning it on proprietary data.

We'll demonstrate datafast's capabilities by creating a space engineering text dataset with the following characteristics:

* Multi-document types: generate different types of documents
* Multi-topic: generate texts on various space environment-related topics
* Multi-lingual: generate texts in several languages
* Multi-LLM: generate texts using multiple LLM providers to boost diversity
* Publish the dataset to your Hugging Face Hub (optional)

??? note
    In this guide we are generating raw text without using personas or seed texts. 
    We are only specifying document types, topics, and languages. 
    Generating synthetic data using personas or seed texts is a common use case which is on our roadmap but not yet available.

## Step 1: Import Required Modules

Generating a dataset with `datafast` requires 3 types of imports:

* Dataset
* Configs
* LLM Providers

```python
from datafast.datasets import RawDataset
from datafast.schema.config import RawDatasetConfig, PromptExpansionConfig
from datafast.llms import OpenAIProvider, AnthropicProvider, GeminiProvider
```

In addition, we'll use `dotenv` to load environment variables containing API keys.
```python
from dotenv import load_dotenv

# Load environment variables containing API keys
load_dotenv("secrets.env")
```

Make sure you have created a secrets.env file with your API keys. HF token is needed if you want to push the dataset to your HF hub. Other keys depend on which LLM providers you use. In our example, we use OpenAI and Anthropic.

```
GEMINI_API_KEY=XXXX
OPENAI_API_KEY=sk-XXXX
ANTHROPIC_API_KEY=sk-ant-XXXXX
HF_TOKEN=hf_XXXXX
```

## Step 2: Configure Your Dataset

The `RawDatasetConfig` class defines all parameters for your text generation dataset.

- **`document_types`**: List of document types to generate (e.g., "tech journalism blog", "personal blog", "MSc lecture notes").

- **`topics`**: List of topics to generate content about (e.g., "technology", "artificial intelligence", "cloud computing").

- **`num_samples_per_prompt`**: Number of examples to generate in a single LLM call.

!!! note
    My recommendation is to use a smaller number (like 5) for text generation as these outputs tend to be longer.
    Use an even smaller number (like 2-3) if you generate very long texts like 300+ words.

- **`output_file`**: Path where the generated dataset will be saved (JSONL format).

- **`languages`**: Dictionary mapping language codes to their names (e.g., `{"en": "English", "fr": "French"}`).
    - You can use any language code and name you want. However, make sure that the underlying LLM provider you'll be using supports the language you're requesting.

- **`prompts`**: (Optional) Custom prompt templates.
    - **Mandatory placeholders**: When providing a custom prompt for `RawDatasetConfig`, you must always include the following variable placeholders in your prompt, using **single curly braces**:
        - `{num_samples}`: it uses the `num_samples_per_prompt` parameter defined above
        - `{language_name}`: it uses the `languages` parameter defined above
        - `{document_type}`: it comes from the `document_types` parameter defined above
        - `{topic}`: it comes from the `topics` parameter defined above
    - **Optional placeholders**: These placeholders can be used to expand the diversity of your dataset. They are optional, but can help you create a more diverse dataset. They must be written **using double curly braces**.

Here's a basic configuration example:

```python
config = RawDatasetConfig(
    # Types of documents to generate
    document_types=[
        "space engineering textbook", 
        "spacecraft design justification document", 
        "personal blog of a space engineer"
    ],
    
    # Topics to generate content about
    topics=[
        "Microgravity",
        "Vacuum",
        "Heavy Ions",
        "Thermal Extremes",
        "Atomic Oxygen",
        "Debris Impact",
        "Electrostatic Charging",
        "Propellant Boil-off",
        # ... You can pour hundreds of topics here. 8 is enough for this example
    ],
    
    # Number of examples to generate per prompt
    num_samples_per_prompt=1,
    
    # Output file path
    output_file="space_engineering_environment_effects_texts.jsonl",
    
    # Languages to generate data for
    languages={"en": "English", "fr": "French"},
    
    # Custom prompts (optional - otherwise defaults will be used)
    # ...
)
```

## Step 3: Prompt Expansion for Diverse Examples (Optional)

Prompt expansion is a key concept in the `datafast` library. It helps generate multiple variations of a base prompt to increase the diversity of the generated data.

For example, we added one optional placeholder using double curly braces:

* `{{expertise_level}}`: To generate texts that elaborate on space engineering topics for different reader expertise levels (e.g., "executives", " engineers", "senior scientists")

You can configure prompt expansion like this:

```python
config = RawDatasetConfig(
    # Basic configuration as above
    # ...
    
    # Custom prompt with placeholders (this will overwrite the default one). Watch out, you have to include the mandatory placeholders defined above.
    prompts=[
        (
            "Generate {num_samples} section of a {document_type} in {language_name} " 
            "about the topic of {topic} in Space Engineering "
            "Target the content for {{expertise_level}} level readers."
        )
    ],
    
    # Add prompt expansion configuration
    expansion=PromptExpansionConfig(
        placeholders={
            "expertise_level": ["executives", "senior engineers", "PhD candidates"]
        },
        combinatorial=True,  # Generate all combinations
        num_random_samples=100 # Only needed if combinatorial is False. Then samples 100 at random.
    )
)
```

This expansion creates prompt variations by replacing `{{expertise_level}}` with all possible combinations of the provided values, dramatically increasing the diversity of your dataset.

## Step 4: Set Up LLM Providers

Configure one or more LLM providers to generate your dataset:

```python
providers = [
    OpenAIProvider(model_id="gpt-4.1-mini-2025-04-14"), # You may want to use stronger models
    AnthropicProvider(model_id="claude-3-5-haiku-latest"),
]
```

Using multiple providers helps create more diverse and robust datasets.

## Step 5: How Many Instances Will It Generate?

The number of generated instances in your dataset in combinatorial mode can be calculated by multiplying the following:

- number of document types (3 in our example)
- number of topics (8 in our example)
- number of languages (2 in our example)
- number of samples per prompt (1 in our example)
- number of LLM providers (2 in our example)
- number of variations for each optional placeholder (if using prompt expansion)
  - For example: 3 for `{{expertise_level}}` (executives, senior engineers, PhD candidates).

With these numbers, and without prompt expansion, we'd generate: 3 x 8 × 2 × 1 × 2 = 96 instances.

With prompt expansion we further multiply by the number of combinations from the optional placeholders (here 3): 3 x 8 × 2 × 1 × 2 × 3 = 288 instances.

If that seems sufficient and representative of your use case, we can proceed to generate the dataset.

In the real world this is a too small dataset for pre-training. But it can be one tiny slice of a much larger pre-training text corpus.

## Step 6: Generate the Dataset

Now you can create and generate your dataset:

```python
# Initialize dataset with your configuration
dataset = RawDataset(config)

# Get expected number of rows (useful to know before generating)
num_expected_rows = dataset.get_num_expected_rows(providers)
print(f"Expected number of rows: {num_expected_rows}")

# Generate examples using configured providers
dataset.generate(providers)
```

This will:
1. Initialize a dataset with your configuration
2. For each document type, topic, and language combination:
   - Create base prompts
   - Expand prompts with configured variations (if provided)
   - Call each LLM provider with each expanded prompt
3. Save the dataset to the specified output file

## Step 7: Push to Hugging Face Hub (Optional)

After generating your dataset, you can push it to the Hugging Face Hub for sharing and version control:

```python
USERNAME = "your_huggingface_username"  # <--- Your Hugging Face username
DATASET_NAME = "your_dataset_name"  # <--- Your Hugging Face dataset name
url = dataset.push_to_hub(
    repo_id=f"{USERNAME}/{DATASET_NAME}",
    train_size=0.8,  # for a 80/20 train/test split, otherwise omit
    seed=20250304,
    shuffle=True,
)
print(f"\nDataset pushed to Hugging Face Hub: {url}")
```

You don't need to specify a seed if you don't want to use train/test splitting: 
- If not provided, will push the entire dataset with train/test splits.

Make sure you have set your `HF_TOKEN` in the environment variables.

!!! info
    Check the resulting dataset [here](https://huggingface.co/datasets/patrickfleith/space_engineering_environment_effects_texts).

## Complete Example

Here's a complete example script that generates a text dataset across multiple document types, topics, and languages:

```python
from datafast.datasets import RawDataset
from datafast.schema.config import RawDatasetConfig, PromptExpansionConfig
from datafast.llms import OpenAIProvider, AnthropicProvider


def main():
    # 1. Configure the dataset generation
    config = RawDatasetConfig(
        document_types=[
            "space engineering textbook", 
            "spacecraft design justification document", 
            "personal blog of a space engineer"
        ],
        topics=[
            "Microgravity",
            "Vacuum",
            "Heavy Ions",
            "Thermal Extremes",
            "Atomic Oxygen",
            "Debris Impact",
            "Electrostatic Charging",
            "Propellant Boil-off",
            # ... You can pour hundreds of topics here. 8 is enough for this example
        ],
        num_samples_per_prompt=1,
        output_file="space_engineering_environment_effects_texts.jsonl",
        languages={"en": "English", "fr": "French"},
        prompts=[
            (
                "Generate {num_samples} section of a {document_type} in {language_name} " 
                "about the topic of {topic} in Space Engineering "
                "Target the content for {{expertise_level}} level readers."
            )
        ],
        expansion=PromptExpansionConfig(
            placeholders={
                "expertise_level": ["executives", "senior engineers", "PhD candidates"]
            },
            combinatorial=True,
        )
    )

    # 2. Create LLM providers with specific models
    providers = [
        OpenAIProvider(model_id="gpt-4.1-mini-2025-04-14"), # You may want to use stronger models
        AnthropicProvider(model_id="claude-3-5-haiku-latest"),
    ]

    # 3. Generate the dataset
    dataset = RawDataset(config)
    num_expected_rows = dataset.get_num_expected_rows(providers)
    print(f"Expected number of rows: {num_expected_rows}")
    dataset.generate(providers)

    # 4. Push to HF hub (optional)
    # USERNAME = "your_huggingface_username"
    # DATASET_NAME = "your_dataset_name"
    # url = dataset.push_to_hub(
    #     repo_id=f"{USERNAME}/{DATASET_NAME}",
    #     train_size=0.8,  # for a 80/20 train/test split, otherwise omit
    #     seed=20250304,
    #     shuffle=True,
    # )
    # print(f"\nDataset pushed to Hugging Face Hub: {url}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv("secrets.env")
    main()
```

## Conclusion

`datafast` simplifies generation of diverse technical datasets across document types, topics, and languages using multiple LLM providers. Generated data is saved in JSONL format with Hugging Face Hub integration for sharing.

🚀 Coming soon: seed text-based generation and persona features for enhanced dataset diversity.
