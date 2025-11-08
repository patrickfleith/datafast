# How to Create a Preference Dataset

!!! example "Use Case"
    Let's say you work at **NASA** and want to *fine-tune an LLM on decades of engineering lessons learned from past missions successes and failures*:
    
    - You're developing a training dataset for fine-tuning and/or aligning an AI assistant and need to create preferences between high-quality and lower-quality responses. 
    - These **preference pairs** will help the model learn which types of responses are more helpful and should be ranked higher during training using techniques like DPO (Direct Preference Optimization) or ORPO.

We'll demonstrate datafast's capabilities by creating a preference dataset based on NASA lessons learned documents with the following characteristics:

* Question-based: generate questions from NASA lessons learned documents
* Response pairs: generate "chosen" (high-quality) and "rejected" (slightly lower quality) responses for each question
* Multi-LLM: use different LLM providers for questions and responses
* Optional LLM judge scoring: use an LLM to score and evaluate response quality

## Step 1: Import Required Modules

Generating a preference dataset with `datafast` requires these imports:

```python
from datafast.datasets import PreferenceDataset
from datafast.schema.config import PreferenceDatasetConfig
from datafast.llms import OpenAIProvider, GeminiProvider, AnthropicProvider
from datafast.logger_config import configure_logger
from dotenv import load_dotenv
import json
from pathlib import Path
```

You'll need to load environment variables containing API keys and configure logging:

```python
# Load environment variables containing API keys
load_dotenv()

# Configure logger to see progress, warnings, and success messages
configure_logger()
```

Make sure you have created a `.env` file with your API keys for the LLM providers you plan to use:

```
OPENAI_API_KEY=sk-XXXX
GEMINI_API_KEY=XXXX
ANTHROPIC_API_KEY=sk-ant-XXXXX
HF_TOKEN=hf_XXXXX  # If publishing to Hugging Face Hub
```

## Step 2: Load NASA Lessons Learned Documents

First, create a function to load NASA lessons learned documents from a JSONL file:

```python
def load_documents_from_jsonl(jsonl_path: str | Path) -> list[str]:
    """Load documents from a JSONL file where the document text is stored in the 'text' key."""
    documents = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if 'text' in data:
                documents.append(data['text'])
    return documents

# Path to the NASA lessons learned sample data
jsonl_path = Path("data/preferences/nasa_lsi_sample.jsonl")
documents = load_documents_from_jsonl(jsonl_path)
```

## Step 3: Configure Your Dataset

The `PreferenceDatasetConfig` class defines all parameters for your preference dataset:

```python
config = PreferenceDatasetConfig(
    # List of NASA lessons learned documents (list[str])
    input_documents=documents,
    
    # Number of questions to generate per document
    num_samples_per_prompt=3,
    
    # Languages to generate data for (in this case, just English)
    languages={"en": "English"},
    
    # Enable LLM judge to score responses (optional)
    llm_as_judge=True,
    
    # Output file path
    output_file="nasa_lessons_learned_dataset.jsonl"
)
```

### Configuration Parameters

- **`input_documents`**: List of NASA lessons learned documents from which questions will be generated.

- **`num_samples_per_prompt`**: Number of questions to generate per document in each language.

- **`languages`**: Dictionary mapping language codes to their names (e.g., `{"en": "English"}`).
    - Make sure the LLM providers you use support the specified languages.

- **`output_file`**: Path where the generated dataset will be saved (JSONL format).

- **`llm_as_judge`** (optional): When set to `True`, uses an additional LLM to score and evaluate the quality of both chosen and rejected responses.

- **`judge_prompt`** (optional): Custom prompt template for the LLM judge (required when `llm_as_judge=True`).

### Custom Prompts (Optional)

You can customize the prompts used for generation:

```python
config = PreferenceDatasetConfig(
    # Basic configuration as above
    # ...
    
    # Custom prompts for question generation
    question_generation_prompts=[
        "Based on the following document, generate {num_samples} clear and specific questions in {language_name}:\n\n{document}",
        "Generate {num_samples} questions in {language_name} that would require detailed responses about this document:\n\n{document}"
    ],
    
    # Custom prompt for chosen (high-quality) responses
    chosen_response_generation_prompt="""
    You are an expert assistant. Create a comprehensive answer in {language_name} to this question:
    Document: {document}
    Question: {question}
    """,
    
    # Custom prompt for rejected (lower-quality) responses
    rejected_response_generation_prompt="""
    Briefly answer in {language_name}:
    Document: {document}
    Question: {question}
    """
)
```

#### Required Placeholders

When providing custom prompts for `PreferenceDatasetConfig`, you must include these variable placeholders:

- **For question generation prompts**:
  - `{num_samples}`: Number of questions to generate per document
  - `{language_name}`: Name of the language to generate in
  - `{document}`: The source document content

- **For response generation prompts**:
  - `{language_name}`: Name of the language to generate in
  - `{document}`: The source document content
  - `{question}`: The generated question

## Step 4: Set Up LLM Providers

Configure LLM providers for different aspects of dataset generation:

```python
# For generating questions from NASA lessons learned documents
question_gen_llm = OpenAIProvider(model_id="gpt-5-mini-2025-08-07")

# For generating high-quality (chosen) responses
chosen_response_gen_llm = AnthropicProvider(model_id="claude-3-7-sonnet-latest")

# For generating lower-quality (rejected) responses
rejected_response_gen_llm = GeminiProvider(model_id="gemini-2.0-flash")

# For scoring responses (only needed if llm_as_judge=True)
judge_llm = OpenAIProvider(model_id="gpt-5-mini-2025-08-07")
```

Using different providers for different aspects of generation helps create more diverse and realistic preference pairs.

## Step 5: How Many Instances Will It Generate?

The number of generated preference pairs in your dataset can be calculated:

Number of preference pairs = Number of documents × Number of languages × Number of samples per prompt

For example, with 10 NASA lessons learned documents, 1 language, and 3 samples per prompt:
10 × 1 × 3 = 30 preference pairs

You can check this programmatically:

```python
dataset = PreferenceDataset(config)
num_expected_rows = dataset.get_num_expected_rows(llms=[question_gen_llm])
print(f"Expected number of rows: {num_expected_rows}")
```

## Step 6: Generate the Dataset

Now you can create and generate your dataset:

```python
# Initialize dataset with your configuration
dataset = PreferenceDataset(config)

# Generate the dataset with specified LLM providers
dataset.generate(
    question_gen_llm=question_gen_llm,
    chosen_response_gen_llm=chosen_response_gen_llm,
    rejected_response_gen_llm=rejected_response_gen_llm,
    judge_llm=judge_llm  # Only needed if llm_as_judge=True
)

# Print summary of results
print(f"Generated {len(dataset.data_rows)} preference pairs")
print(f"Results saved to {config.output_file}")
```

## Step 7: Examine the Results

The generated dataset contains preference pairs with these components:

- **Question**: Generated from the input document
- **Chosen response**: High-quality response generated by chosen_response_gen_llm
- **Rejected response**: Lower-quality response generated by rejected_response_gen_llm
- **Scores** (if llm_as_judge=True): Numeric scores assigned by the judge_llm

You can examine a sample:

```python
if dataset.data_rows:
    sample = dataset.data_rows[0]
    print(f"Question: {sample.question}")
    print(f"Chosen model: {sample.chosen_model_id}")
    print(f"Rejected model: {sample.rejected_model_id}")
    if sample.chosen_response_score is not None:
        print(f"Chosen response score: {sample.chosen_response_score}")
        print(f"Rejected response score: {sample.rejected_response_score}")
```

## Complete Example

Here's a complete example of generating a preference dataset:

```python
import json
from pathlib import Path
from datafast.schema.config import PreferenceDatasetConfig
from datafast.datasets import PreferenceDataset 
from datafast.llms import OpenAIProvider, GeminiProvider, AnthropicProvider
from datafast.logger_config import configure_logger
from dotenv import load_dotenv

# Load environment variables with API keys
load_dotenv()

# Configure logger
configure_logger()

# Load NASA lessons learned documents from JSONL file
def load_documents_from_jsonl(jsonl_path: str | Path) -> list[str]:
    """Load documents from a JSONL file where the document text is stored in the 'text' key."""
    documents = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if 'text' in data:
                documents.append(data['text'])
    return documents

# Path to the NASA lessons learned sample data
jsonl_path = Path("data/preferences/nasa_lsi_sample.jsonl")
documents = load_documents_from_jsonl(jsonl_path)

# 1. Define the configuration
config = PreferenceDatasetConfig(
    input_documents=documents,
    num_samples_per_prompt=3,  # Generate 3 questions per document
    languages={"en": "English"},  # Generate in English only
    llm_as_judge=True,  # Use LLM to judge and score responses
    output_file="nasa_lessons_learned_dataset.jsonl",
)

# 2. Initialize LLM providers
question_gen_llm = OpenAIProvider(model_id="gpt-5-mini-2025-08-07")
chosen_response_gen_llm = AnthropicProvider(model_id="claude-3-7-sonnet-latest")
rejected_response_gen_llm = GeminiProvider(model_id="gemini-2.0-flash")
judge_llm = OpenAIProvider(model_id="gpt-5-mini-2025-08-07")

# 3. Generate the dataset
dataset = PreferenceDataset(config)
num_expected_rows = dataset.get_num_expected_rows(llms=[question_gen_llm])
print(f"Expected number of rows: {num_expected_rows}")

dataset.generate(
    question_gen_llm=question_gen_llm,
    chosen_response_gen_llm=chosen_response_gen_llm,
    rejected_response_gen_llm=rejected_response_gen_llm,
    judge_llm=judge_llm
)

# 4. Print results summary
print(f"Generated {len(dataset.data_rows)} preference pairs")
print(f"Results saved to {config.output_file}")

# 5. Display a sample of the generated data
if dataset.data_rows:
    sample = dataset.data_rows[0]
    print("\nSample preference pair:")
    print(f"Question: {sample.question}")
    print(f"Chosen model: {sample.chosen_model_id}")
    print(f"Rejected model: {sample.rejected_model_id}")
    if sample.chosen_response_score is not None:
        print(f"Chosen response score: {sample.chosen_response_score}")
        print(f"Rejected response score: {sample.rejected_response_score}")
```

## Using Your Preference Dataset

The generated preference dataset can be used for:

1. Fine-tuning models using RLHF (Reinforcement Learning from Human Feedback)
2. DPO or ORPO training
3. Evaluating model response quality and preferences
4. Studying what makes responses more helpful and preferred by users

The dataset is saved in JSONL format, compatible with most fine-tuning frameworks.

