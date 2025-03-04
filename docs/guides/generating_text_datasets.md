# How to Create a Raw Text Dataset

We'll create a raw text dataset for usage as part of a pre-training with the following characteristics:

* Multi-document types: generate different types of documents (blogs, lecture notes, etc.)
* Multi-topic: generate texts on various topics (AI, cloud computing, etc.)
* Multi-lingual: generate texts in several languages
* Multi-LLM: generate texts using multiple LLM providers to boost diversity
* Push the dataset to your Hugging Face Hub (optional)

!!! note
    In this guide we are generating raw text without using personas or seed texts. We are only specifying document types, topics, and languages. Generating synthetic data using personas or seed texts is a common use case which is on our roadmap but not yet available.

## Step 1: Import Required Modules

Generating a dataset with `datafast` requires 3 types of imports:

* Dataset
* Configs
* LLM Providers

```python
from datafast.datasets import TextDataset
from datafast.schema.config import TextDatasetConfig, PromptExpansionConfig
from datafast.llms import OpenAIProvider, AnthropicProvider, GoogleProvider
```

In addition, we'll use `dotenv` to load environment variables containing API keys.
```python
from dotenv import load_dotenv

# Load environment variables containing API keys
load_dotenv("secrets.env")
```

Make sure you have created a secrets.env file with your API keys. HF token is needed if you want to push the dataset to your HF hub. Other keys depend on which LLM providers you use. In our example, we use Google, OpenAI, and Anthropic.

```
GOOGLE_API_KEY=XXXX
OPENAI_API_KEY=sk-XXXX
ANTHROPIC_API_KEY=XXXXX
HF_TOKEN=XXXXX
```

## Step 2: Configure Your Dataset

The `TextDatasetConfig` class defines all parameters for your text generation dataset.

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
    - **Mandatory placeholders**: When providing a custom prompt for `TextDatasetConfig`, you must always include the following variable placeholders in your prompt, using **single curly braces**:
        - `{num_samples}`: it uses the `num_samples_per_prompt` parameter defined above
        - `{language_name}`: it uses the `languages` parameter defined above
        - `{document_type}`: it comes from the `document_types` parameter defined above
        - `{topic}`: it comes from the `topics` parameter defined above
    - **Optional placeholders**: These placeholders can be used to expand the diversity of your dataset. They are optional, but can help you create a more diverse dataset. They must be written **using double curly braces**.

Here's a basic configuration example:

```python
config = TextDatasetConfig(
    # Types of documents to generate
    document_types=["tech journalism blog", "personal blog", "MSc lecture notes"],
    
    # Topics to generate content about
    topics=["technology", "artificial intelligence", "cloud computing"],
    
    # Number of examples to generate per prompt
    num_samples_per_prompt=3,
    
    # Output file path
    output_file="tech_posts.jsonl",
    
    # Languages to generate data for
    languages={"en": "English", "fr": "French"},
    
    # Custom prompts (optional - otherwise defaults will be used)
    # ...
)
```

## Step 3: Prompt Expansion for Diverse Examples (Optional)

Prompt expansion is a key concept in the `datafast` library. It helps generate multiple variations of a base prompt to increase the diversity of the generated data.

For example, we added one optional placeholder using double curly braces:

* `{{country}}`: To generate texts that elaborate on cloud computing or AI from different perspectives (e.g., "United States", "Canada", "Europe")

You can configure prompt expansion like this:

```python
config = TextDatasetConfig(
    # Basic configuration as above
    # ...
    
    # Custom prompt with placeholders (this will overwrite the default one). Watch out, you have to include the mandatory placeholders defined above.
    prompts=[
        (
            "Generate {num_samples} {document_type} entries in {language_name} " "about {topic}. "
            "The emphasis should be a perspective from {{country}}."
        )
    ],
    
    # Add prompt expansion configuration
    expansion=PromptExpansionConfig(
        placeholders={
            "country": ["United States", "Europe", "Japan", "India", "China", "Australia"]
        },
        combinatorial=True,  # Generate all combinations
        num_random_samples=100 # Only needed if combinatorial is False. Then samples 100 at random.
    )
)
```

This expansion creates prompt variations by replacing `{{country}}` with all possible combinations of the provided values, dramatically increasing the diversity of your dataset.

## Step 4: Set Up LLM Providers

Configure one or more LLM providers to generate your dataset:

```python
providers = [
    OpenAIProvider(model_id="gpt-4o-mini"),
    AnthropicProvider(model_id="claude-3-5-haiku-latest"),
    GoogleProvider(model_id="gemini-1.5-flash")
]
```

Using multiple providers helps create more diverse and robust datasets.

## Step 5: How Many Instances Will It Generate?

The number of generated instances in your dataset in combinatorial mode can be calculated by multiplying the following:

- number of document types (3 in our example)
- number of topics (3 in our example)
- number of languages (2 in our example)
- number of samples per prompt (3 in our example)
- number of LLM providers (3 in our example)
- number of variations for each optional placeholder (if using prompt expansion)
  - For example: 6 for `{{country}}`.

With these numbers, and without prompt expansion, we'd generate: 3 × 3 × 2 × 3 × 3 = 162 instances.

With prompt expansion we further by the number of combinations from the optional placeholders (here 6): 3 × 3 × 2 × 3 × 3 × 6 = 972 instances.

If that seems sufficient and representative of your use case, we can proceed to generate the dataset.

In the real world this is a too small dataset for pre-training. But it can be one tiny slice of a much larger pre-training text corpus.

## Step 6: Generate the Dataset

Now you can create and generate your dataset:

```python
# Initialize dataset with your configuration
dataset = TextDataset(config)

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

## Complete Example

Here's a complete example script that generates a text dataset across multiple document types, topics, and languages:

```python
from datafast.datasets import TextDataset
from datafast.schema.config import TextDatasetConfig, PromptExpansionConfig
from datafast.llms import OpenAIProvider, AnthropicProvider, GoogleProvider


def main():
    # 1. Configure the dataset generation
    config = TextDatasetConfig(
        document_types=["tech journalism blog", "personal blog", "MSc lecture notes"],
        topics=["technology", "artificial intelligence", "cloud computing"],
        num_samples_per_prompt=3,
        output_file="tech_posts.jsonl",
        languages={"en": "English", "fr": "French"},
        prompts=[
            (
                "Generate {num_samples} {document_type} entries in {language_name} "about {topic}. "
                "The emphasis should be a perspective from {{country}}"
            )
        ],
        expansion=PromptExpansionConfig(
            placeholders={
                "country": ["United States", "Europe", "Japan", "India", "China", "Australia"]
            },
            combinatorial=True,
        )
    )

    # 2. Create LLM providers with specific models
    providers = [
        OpenAIProvider(model_id="gpt-4o-mini"),
        AnthropicProvider(model_id="claude-3-5-haiku-latest"),
        GoogleProvider(model_id="gemini-1.5-flash"),
    ]

    # 3. Generate the dataset
    dataset = TextDataset(config)
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

With `datafast`, you can easily generate diverse text datasets across multiple document types, topics, languages, and using multiple LLM providers. The generated datasets are saved in JSONL format and can be pushed to the Hugging Face Hub for sharing and version control.

The `TextDataset` class provides a simple interface for generating text data, while the `TextDatasetConfig` class allows you to configure the generation process in detail. By using prompt expansion, you can create even more diverse datasets by generating multiple variations of your base prompts.

🚀 There is more to come with new feature for generating raw text datasets on the basis of seed texts, and also using personas for more diversity.
