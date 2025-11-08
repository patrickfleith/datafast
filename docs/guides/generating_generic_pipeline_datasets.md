# How to Create a Generic Pipeline Dataset

!!! example "Use Case"
    You have an existing dataset with structured information and you want to **transform or augment it using LLMs** with custom prompts.
    For example, you might have a dataset of personas and want to generate tweets and CVs for each persona, or you might have product descriptions and want to generate marketing copy in multiple styles.

The Generic Pipeline Dataset is designed for **maximum flexibility**. Unlike other dataset types in Datafast that have predefined structures (like classification labels or MCQ questions), the Generic Pipeline Dataset allows you to:

- **Process any existing dataset** (from Hugging Face Hub or local files)
- **Define custom input-output transformations** using your own prompts
- **Specify exactly which columns** to use as input, to forward, or to generate as output

This makes it ideal for creating custom input-output LLM synthetic data generation steps which you can even chain together.

## Why Use Generic Pipeline Dataset?

The Generic Pipeline Dataset is perfect when:

1. **You need custom transformations**: The predefined dataset types (classification, MCQ, etc.) don't fit your use case
2. **You have some existing data**: You have some seeds rows (like some topics), and you want to transform or generate more data for each seed row
3. **You need flexibility**: You want full control over the input columns, output structure, and processing logic

## Step 1: Import Required Modules

Generating a dataset with `datafast` requires 3 types of imports:

* Dataset
* Configs
* LLM Providers

```python
from datafast.datasets import GenericPipelineDataset
from datafast.schema.config import GenericPipelineDatasetConfig
from datafast.llms import OpenRouterProvider
```

In addition, we'll use `dotenv` to load environment variables containing API keys and configure logging to monitor the generation process:

```python
from dotenv import load_dotenv
from datafast.logger_config import configure_logger

# Load environment variables containing API keys
load_dotenv()

# Configure logger to see progress, warnings, and success messages
configure_logger()
```

Make sure you have created a `.env` file with your API keys:

```
OPENROUTER_API_KEY=XXXX
HF_TOKEN=hf_XXXXX
```

## Step 2: Understand the Configuration

The `GenericPipelineDatasetConfig` class provides flexible configuration for processing any dataset:

### Key Parameters

- **`hf_dataset_name`** or **`local_file_path`**: Specify your data source
    - Use `hf_dataset_name` for Hugging Face datasets (e.g., `"organization/dataset-name"`)
    - Use `local_file_path` for local files (CSV, TXT, PARQUET, or JSONL)
    - You must provide exactly one of these

- **`input_columns`**: List of column names from your source dataset to use as input
    - These columns will be available as placeholders in your prompts
    - At least one column is required
    - Example: `["persona", "background"]`

- **`forward_columns`**: (Optional) Columns to pass through unchanged to the output
    - Useful for IDs, labels, or metadata you want to preserve
    - Example: `["user_id", "category"]`

- **`output_columns`**: (Optional) Names for the generated data fields
    - If specified, your LLM will generate JSON with these field names
    - If not specified, defaults to a single `"generated_text"` field
    - Example: `["tweet", "cv"]`

- **`prompts`**: List of custom prompt templates
    - **Mandatory placeholders** (use single curly braces):
        - `{num_samples}`: Number of samples to generate per input
        - `{language}`: Language name from the languages config
        - At least one input column as placeholder (e.g., `{persona}`, `{background}`)
        - You can use multiple input columns in your prompt; unused columns will trigger a warning
    - **Optional placeholders** (use double curly braces):
        - Any custom placeholders for prompt expansion (e.g., `{{style}}`)

- **`num_samples_per_prompt`**: Number of outputs to generate for each input row
    - Default: 1
    - Recommended: Keep this low (1-3) for complex outputs

- **`sample_count`**: (Optional) Limit the number of rows to process from source dataset
    - Useful for testing before running on full dataset

- **`skip_function`**: (Optional) Custom function to skip certain rows
    - Must be a callable that takes a row dict (with all source dataset columns) and returns `True` to skip
    - Example use cases: skip rows with certain keywords, filter by length, exclude based on metadata
    - Example: `lambda row: len(row.get("text", "")) < 100` (skip short texts)

## Step 3: Configure Your Dataset

Let's create a simple example that generates tweets from personas:

```python
# Define a simple prompt template
PROMPT_TEMPLATE = """I will give you a persona description.
Generate {num_samples} authentic tweets in {language} that this person might write.
Make each tweet engaging and true to their character.

Persona: {persona}

Your response should be formatted in valid JSON."""

# Configure the dataset
config = GenericPipelineDatasetConfig(
    # Data source
    hf_dataset_name="patrickfleith/FinePersonas-v0.1-100k-space-filtered",
    
    # Define input/output structure
    input_columns=["persona"],              # Use persona column as input
    forward_columns=["summary_label"],      # Keep summary_label in output
    output_columns=["tweet"],               # Generate a tweet
    
    # Processing parameters
    sample_count=10,                        # Process only 10 rows for testing
    num_samples_per_prompt=1,               # Generate 1 tweet per persona
    
    # Prompt configuration
    prompts=[PROMPT_TEMPLATE],              # We could use define a variety of prompts here
    
    # Output file
    output_file="persona_tweets.jsonl",
    
    # Languages
    languages={"en": "English", "fr", "French"}   # languages to generate tweets in
)
```


!!! note
    If you need to generate multiple fields at once, specify them in `output_columns`. Make sure your prompt clearly instructs the LLM to generate valid JSON with those exact field names.

    ```python
    MULTI_OUTPUT_PROMPT = """Given this persona, generate {num_samples} complete profiles in {language}:
    1. A tweet they might write
    2. A brief CV/resume

    Persona: {persona}

    Respond with valid JSON containing both fields."""
    ```

## Step 4: Set Up LLM Providers

Configure one LLM provider or more for diversity. Read more about LLM providers [here](../llms.md).

```python
providers = [
    OpenRouterProvider(model_id="z-ai/glm-4.6")
]
```

## Step 5: Generate the Dataset

Now generate your dataset:

```python
# Initialize dataset
dataset = GenericPipelineDataset(config)

# Check expected output size
num_expected = dataset.get_num_expected_rows(providers)
print(f"Expected rows: {num_expected}")

# Generate
dataset.generate(providers)

# Print summary
print(f"Generated {len(dataset.data_rows)} examples")
print(f"Saved to {config.output_file}")
```

The generation process:
1. Loads the source dataset
2. For each row in the source dataset:
   - Extracts input columns
   - Formats prompts with input data
   - Applies prompt expansion (if configured)
   - Calls each LLM provider
   - Saves generated outputs to file


## Step 6: Publishing to Hugging Face Hub (Optional)

Push your generated dataset to Hugging Face:

```python
url = dataset.push_to_hub(
    repo_id="YOUR_USERNAME/persona-tweets",
    train_size=0.8,
    seed=42,
    shuffle=True
)
print(f"Dataset published at: {url}")
```

!!! warning
    Ensure your `HF_TOKEN` environment variable is set before pushing to the Hub.

## Complete Example

Here's a complete working example:

```python
from datafast.datasets import GenericPipelineDataset
from datafast.schema.config import GenericPipelineDatasetConfig
from datafast.llms import OpenRouterProvider
from datafast.logger_config import configure_logger
from dotenv import load_dotenv

# Load API keys
load_dotenv()

# Configure logger
configure_logger()

# Define prompt
PROMPT = """I will give you a persona description.
Generate {num_samples} authentic tweets in {language} that this person might write.
Make each tweet engaging and true to their character.

Persona: {persona}

Your response should be formatted in valid JSON."""

# Configure dataset
config = GenericPipelineDatasetConfig(
    hf_dataset_name="patrickfleith/FinePersonas-v0.1-100k-space-filtered",
    input_columns=["persona"],
    forward_columns=["summary_label"],
    output_columns=["tweet"],
    sample_count=10,
    num_samples_per_prompt=1,
    prompts=[PROMPT],
    output_file="persona_tweets.jsonl",
    languages={"en": "English", "fr": "French"}
)

# Set up provider
providers = [
    OpenRouterProvider(model_id="z-ai/glm-4.6")
]

# Generate dataset
dataset = GenericPipelineDataset(config)
num_expected = dataset.get_num_expected_rows(providers)
print(f"Expected rows: {num_expected}")
dataset.generate(providers)
print(f"Generated {len(dataset.data_rows)} examples")
print(f"Saved to {config.output_file}")

# Optional: Push to hub
url = dataset.push_to_hub(
    repo_id="YOUR_USERNAME/persona-tweets",     # <--- Your hugging face username
    train_size=0.8,
    seed=42,
    shuffle=True
)
print(f"Dataset published at: {url}")
```

## Understanding the Generated Data

Each generated example is stored as a `GenericPipelineRow` with:

- **Input columns**: Original data from your source (e.g., `persona`)
- **Forward columns**: Pass-through data without modification (e.g., `summary_label`)
- **Output columns**: Generated text data (e.g., `tweet`, `cv`)
- **`model_id`**: The LLM model that generated this output
- **`pipeline_source`**: Source type (always `SYNTHETIC`)
- **`language`**: Language code for the generated content
- **`metadata`**: Additional info like prompt index and source row index
- **`uuid`**: Unique identifier

## Advanced: Skip Function

Filter which rows to process using a skip function:

```python
def skip_short_personas(row):
    """Skip personas with very short descriptions"""
    persona = row.get("persona", "")
    return len(persona) < 50

config = GenericPipelineDatasetConfig(
    hf_dataset_name="patrickfleith/FinePersonas-v0.1-100k-space-filtered",
    input_columns=["persona"],
    output_columns=["tweet"],
    skip_function=skip_short_personas,  # Apply filter
    prompts=["Generate tweet for: {persona}"],
    output_file="filtered_tweets.jsonl",
    languages={"en": "English"}
)
```

## Next Steps

- Explore [Prompt Expansion](prompt_expansion.md) for more advanced diversity techniques
- Learn about other dataset types in the [Guides Index](index.md)
- Check out the [Concepts](../concepts.md) page for understanding Datafast's architecture
