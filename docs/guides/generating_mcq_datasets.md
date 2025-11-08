# How to Create a Multiple Choice Question (MCQ) Dataset

!!! example "Use Case"
    Let's say you want to **fine-tune a small and efficient model for quizz generations**. You decide to create a *multiple choice question (MCQ) dataset*. You have various text passages from textbooks and want to generate high-quality multiple choice questions from them.

**The process:**

1. Loads text documents from your specified source
2. For each document that meets some criteria:
   - Generates questions and correct answers using the configured prompt(s)
   - Generates three plausible but incorrect answers for each question
3. Saves the dataset to the specified output file

!!! note
    Note that:
    * Works with multiple input sources (Hugging Face datasets or local files)
    * Supports multi-lingual question generation
    * Uses multiple LLM providers to boost diversity and quality

## Step 1: Import Required Modules

Generating an MCQ dataset with `datafast` requires 3 types of imports:

* Dataset
* Configs
* LLM Providers

```python
from datafast.datasets import MCQDataset
from datafast.schema.config import MCQDatasetConfig, PromptExpansionConfig
from datafast.llms import OpenAIProvider, AnthropicProvider, GeminiProvider
```

In addition, we'll use `dotenv` to load environment variables containing API keys and configure logging to monitor the generation process.
```python
from dotenv import load_dotenv
from datafast.logger_config import configure_logger

# Load environment variables containing API keys
load_dotenv()

# Configure logger to see progress, warnings, and success messages
configure_logger()
```

Make sure you have created a `.env` file with your API keys. HF token is needed if you want to push the dataset to your HF hub. Other keys depend on which LLM providers you use.

```
GEMINI_API_KEY=XXXX
OPENAI_API_KEY=sk-XXXX
ANTHROPIC_API_KEY=sk-ant-XXXXX
HF_TOKEN=hf_XXXXX
```

## Step 2: Prepare Your Input Data

The `MCQDataset` will create multiple-choices questions based on text documents.
This means you have to start with some text sources. Two types are supported:

1. **Hugging Face datasets**: Any dataset on the Hugging Face Hub that contains a column with string values
2. **Local files**: Any file in CSV, TXT, PARQUET, or JSONL format with text content

For local files, make sure they're formatted properly:

* **CSV**: Must have a column matching your specified `text_column` name
* **PARQUET**: Must have a column matching your specified `text_column` name
* **TXT**: Each line will be treated as a separate document
* **JSONL**: Each line must be a valid JSON object with a field matching your specified `text_column` name

In our example we will start from a local JSONL file.

## Step 3: Configure Your Dataset

The `MCQDatasetConfig` class defines all parameters for your MCQ dataset generation:

- **`hf_dataset_name`**: (Optional) Name of a Hugging Face dataset to use as source material
- **`local_file_path`**: (Optional) Path to a local file to use as source material
- **`text_column`**: (Required) Column name containing the text to generate questions from
- **`context_column`**: (Optional) Column name containing contextual information to enhance question generation with domain-specific context
- **`num_samples_per_prompt`**: Number of questions to generate for each text document
- **`sample_count`**: (Optional) Number of samples to process from the source text dataset (useful for testing)
- **`min_document_length`**: Minimum text length (in characters) for processing (skips shorter documents)
- **`max_document_length`**: Maximum text length (in characters) for processing (skips longer documents)
- **`output_file`**: Path where the generated dataset will be saved (JSONL format)
- **`prompts`**: (Optional) Custom prompt templates for question generation
- **`distractor_prompt`**: (Optional) Custom prompt for generating incorrect answers
- **`languages`**: Dictionary mapping language codes to their names (e.g., `{"en": "English"}`)

!!! note
    You must provide either `hf_dataset_name` or `local_file_path`, but not both.

Here's a basic configuration example:

```python
config = MCQDatasetConfig(
    # Source data configuration - use ONE of these options:
    # Option 1: Hugging Face dataset
    # hf_dataset_name="patrickfleith/space_engineering_environment_effects_texts",
    
    # Option 2: Local file (CSV, TXT, PARQUET, or JSONL)
    local_file_path="data/sample_texts.jsonl",
    
    # Column containing the text to generate questions from
    text_column="text",
    
    # Process 5 only from the JSONL containing source texts
    sample_count=5,
    
    # Generate 3 questions per document
    num_samples_per_prompt=3,
    
    # Skip documents shorter than 100 characters
    min_document_length=100,
    
    # Skip documents longer than 20000 characters
    max_document_length=20000,
    
    # Where to save the results
    output_file="mcq_dataset.jsonl",
    
    # Language(s) to generate questions in
    languages={"en": "English"}
)
```

## Step 4: Custom Prompts (Optional)

You can customize the prompts used for generating questions and incorrect answers.

### Question Generation Prompts

When providing custom prompts for the `prompts` parameter, you must include these mandatory placeholders:

- `{num_samples}`: Number of questions to generate per document
- `{language_name}`: Language to generate questions in
- `{document}`: The source text used to generate questions

Optional placeholders can also be included using double curly braces (e.g., `{{difficulty_level}}` to generate questions of varying difficulty levels).

### Distractor Prompt

For the `distractor_prompt` parameter (used to generate incorrect answers), include these mandatory placeholders:

- `{language_name}`: Language to generate incorrect answers in
- `{question}`: The generated question for which we want to generate incorrect answers
- `{correct_answer}`: The correct answer (for reference)

## Step 5: Prompt Expansion for Diverse Questions (Optional)

Just like with other Datafast dataset types, you can use prompt expansion to generate more diverse questions:

```python
config = MCQDatasetConfig(
    # Basic configuration as above
    # ...
    
    # Custom prompts with placeholders
    prompts=[
        "Generate {num_samples} multiple choice questions in {language_name} "
        "based on the following document. Each question should be at "
        "{{difficulty_level}} difficulty level and focus on {{question_type}} "
        "aspects of the content: {document}. "
        "The question should be self-contained, short and answerable. "
        "The answer must be short. "
        "The question should not specifically refer to the document. "
        "The question contain enough context to be answerable for a person who may "
        "have previously read this document. "
    ],
    
    # Add prompt expansion configuration
    expansion=PromptExpansionConfig(
        placeholders={
            "difficulty_level": ["high-school", "university"],
            "question_type": ["factual", "conceptual"]
        },
        combinatorial=True,  # Generate all combinations
        num_random_samples=100  # Only needed if combinatorial is False
    )
)
```

This expansion creates prompt variations by replacing `{{difficulty_level}}` and `{{question_type}}` with all possible combinations of the provided values, dramatically increasing the diversity of your questions.

## Step 6: Set Up LLM Providers

Configure one or more LLM providers to generate your dataset:

```python
providers = [
    OpenAIProvider(model_id="gpt-5-mini-2025-08-07"),
    AnthropicProvider(model_id="claude-haiku-4-5-20251001"),
    GeminiProvider(model_id="gemini-2.0-flash")
]
```

Using multiple providers helps create more diverse and robust question sets.

## Step 7: How Many Questions Will Be Generated?

Before generating the dataset, you can calculate the expected number of questions:

```python
dataset = MCQDataset(config)
num_expected_rows = dataset.get_num_expected_rows(providers, source_data_num_rows=5)
print(f"Expected number of rows: {num_expected_rows}")
```

Note that is only an estimate: if some generation fails, the actual number of questions will be lower. But it gives you at least a rough idea of how many rows in the final dataset you can expect.

If you're using prompt expansion with `combinatorial=True`, the number of questions will be:

- Number of text row used from source documents × Number of languages × Number of LLM providers × Number of questions per prompt × Number of prompt variations

For example, with 5 document text rows, 1 language, 3 LLM providers, 3 questions per prompt, and 4 prompt variations (2 difficulty levels × 2 question types), you'd get:
5 × 1 × 3 × 3 × 4 = 180 questions (hence rows) in the final dataset.

## Step 8: Generate the Dataset

Now you can create and generate your MCQ dataset:

```python
# Initialize dataset with your configuration
dataset = MCQDataset(config)

# Generate examples using configured providers
dataset.generate(providers)

# Print results summary
print(f"Generated {len(dataset.data_rows)} MCQs")
print(f"Results saved to {config.output_file}")
```

## Step 9: Push to Hugging Face Hub (Optional)

After generating your dataset, you can push it to the Hugging Face Hub for sharing:

```python
url = dataset.push_to_hub(
    repo_id="YOUR_USERNAME/your-mcq-dataset-name",
    train_size=0.7,  # Split 70% for training
    seed=42,         # Set seed for split reproducibility
    shuffle=True     # Shuffle before splitting
)
print(f"Dataset published at: {url}")
```

This automatically splits your dataset into training and test sets and uploads it to Hugging Face.

!!! warning
    Don't forget to set and load your HF_TOKEN environment variable before running this example.

## Complete Example

Here's a complete example for creating an MCQ dataset from a local JSONL file:

```python
from datafast.datasets import MCQDataset
from datafast.schema.config import MCQDatasetConfig, PromptExpansionConfig
from datafast.llms import OpenAIProvider, AnthropicProvider, GeminiProvider
from datafast.logger_config import configure_logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logger
configure_logger()

def main():
    # 1. Define the configuration
    config = MCQDatasetConfig(
        local_file_path="datafast/examples/data/mcq/sample.jsonl",
        text_column="text",    # Column containing the text to generate questions from
        sample_count=3,        # Process only 3 samples for testing
        num_samples_per_prompt=2, # Generate 2 questions per document
        min_document_length=100,  # Skip documents shorter than 100 chars
        max_document_length=20000, # Skip documents longer than 20000 chars
        output_file="mcq_test_dataset.jsonl",
    )

    # 2. Initialize LLM providers
    providers = [
        OpenAIProvider(model_id="gpt-5-mini-2025-08-07"),
        # Add more providers as needed
        # AnthropicProvider(model_id="claude-haiku-4-5-20251001"),
        # GeminiProvider(model_id="gemini-2.0-flash"),
    ]

    # 3. Generate the dataset
    dataset = MCQDataset(config)
    num_expected_rows = dataset.get_num_expected_rows(providers, source_data_num_rows=3)
    print(f"\nExpected number of rows: {num_expected_rows}")
    dataset.generate(providers)

    # 4. Print results summary
    print(f"\nGenerated {len(dataset.data_rows)} MCQs")
    print(f"Results saved to {config.output_file}")

    # 5. Optional: Push to HF hub
    # USERNAME = "your_username"  # <--- Your hugging face username
    # DATASET_NAME = "mcq_test_dataset"  # <--- Your hugging face dataset name
    # url = dataset.push_to_hub(
    #     repo_id=f"{USERNAME}/{DATASET_NAME}",
    #     train_size=0.7,
    #     seed=42,
    #     shuffle=True,
    #     upload_card=True,
    # )
    # print(f"\nDataset pushed to Hugging Face Hub: {url}")

if __name__ == "__main__":
    main()
```

## Understanding the Generated Data

Each generated question is stored as an `MCQRow` with these properties:

- `source_document`: The original text used to generate the question
- `question`: The generated question
- `correct_answer`: The correct answer to the question
- `incorrect_answer_1`, `incorrect_answer_2`, `incorrect_answer_3`: Three plausible but incorrect answers
- `model_id`: The LLM model that generated this question
- `mcq_source`: How the question was created (typically `SYNTHETIC` for generated data)
- `metadata`: Additional information like language code and source dataset
- `uuid`: Unique identifier

## Best Practices

1. **Text Length**: Ensure your source texts are substantive enough for meaningful questions (at least one or two paragraphs).
2. **Question Diversity**: Use prompt expansion with different difficulty levels and question types.
3. **Model Selection**: Larger, more capable models generally produce better questions and answers.
4. **Validation**: Review a sample of the generated questions to ensure quality and accuracy, then edit prompt.
5. **Start Small**: Begin with a small sample_count to test the configuration before scaling up.
6. **Use Context**: When available, use the `context_column` parameter to provide additional domain-specific context that helps generate more self-contained questions. Good contexts include document summaries, topic descriptions.
