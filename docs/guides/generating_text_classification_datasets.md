# How to Create a Text Classification Dataset

!!! example "Use Case"
    Let's say you work at the *Department of Conservation* and you are tasked
    to **categorize hiker reports about trail conditions** from social media posts and hiking reviews.
    This will allow to generate real-time trail status updates and help hikers plan their trips.

We'll demonstrate datafast's capabilities by creating a trail conditions classification dataset with the following characteristics:

* Multi-class: the report belongs to one of four trail condition categories
* Multi-lingual: the reports in the dataset will be in 2 different languages
* Multi-LLM: we generate examples using multiple LLM providers to boost diversity
* Publish the dataset to your Hugging Face Hub.

## Step 1: Import Required Modules

Generating a dataset with `datafast` requires 3 types of imports:

* Dataset
* Configs
* LLM Providers

```python
from datafast.datasets import TextClassificationDataset
from datafast.schema.config import ClassificationConfig, PromptExpansionConfig
from datafast.llms import OpenAIProvider, AnthropicProvider
```

In addition, we'll use `dotenv` to load environment variables containing API keys.
```python
from dotenv import load_dotenv

# Load environment variables containing API keys
load_dotenv("secrets.env")
```

Make sure you have created a `secrets.env` file with your API keys. HF token is needed if you want to push the dataset to your HF hub. Other keys depends on which LLM providers you use. In our example, we use OpenAI and Anthropic.

```
OPENAI_API_KEY=sk-XXXX
ANTHROPIC_API_KEY=XXXXX
HF_TOKEN=XXXXX
```

## Step 2: Configure Your Dataset

The `ClassificationConfig` class defines all parameters for your text classification dataset.

- **`classes`**: List of dictionaries defining your classification labels. Each dictionary represent a class and should include:
    - `name`: Label identifier (required)
    - `description`: Detailed description of what this class represents (required)

- **`num_samples_per_prompt`**: Number of examples to generate in a single LLM call. 

!!! note
    My recommendation is to use a larger number (like 10-20 if you generate very short texts of 10 to 30 words).
    Use a smaller number (like 5) if you generate longer texts like 100 to 300 words.
    Use an even smaller number (like 2-3) if you generate very long texts like 300+ words.

- **`output_file`**: Path where the generated dataset will be saved (JSONL format). 

- **`languages`**: Dictionary mapping language codes to their names (e.g., `{"en": "English"}`).
    - You can use any language code and name you want. However, make sure that the underlying LLM provider you'll be using supports the language you're using.

- **`prompts`**: (Optional, but highly recommended) Custom prompt templates.
    - **Mandatory placeholders**: When providing a custom prompt for `ClassificationConfig`, you must always include the following variable placeholders in your prompt, using **single curly braces**:
        - `{num_samples}`: it uses the `num_samples_per_prompt` parameter defined above)
        - `{language_name}`: it uses the `languages` parameter defined above)
        - `{label_name}`: it comes from the `classes` parameter defined above)
        - `{label_description}`: it comes from the `classes` parameter defined above)
    - **Optional placeholders**: These placeholders can be used to expand the diversity of your dataset. They are optional, but can help you create a more diverse dataset. They must be written **using double curly braces**. For example:
        - `{{style}}`: In our example we want to generate hiker reports in different writing styles
        - `{{trail_type}}`: In our example we want to generate reports about different types of trails

```python
config = ClassificationConfig(
    # Define your classification classes
    classes=[
        {
            "name": "trail_obstruction",
            "description": "Conditions where the trail is partially or fully blocked by obstacles like fallen trees, landslides, snow, or dense vegetation, and other forms of obstruction."
        },
        {
            "name": "infrastructure_issues",
            "description": "Problems related to trail structures and amenities, including damaged bridges, signage, stairs, handrails, or markers, and other forms of infrastructure issues."
        },
        {
            "name": "hazards",
            "description": "Trail conditions posing immediate risks to hiker safety, such as slippery surfaces, dangerous wildlife, hazardous crossings, or unstable terrain, and other forms of hazards."
        },
        {
            "name": "positive_conditions",
            "description": "Conditions highlighting clear, safe, and enjoyable hiking experiences, including well-maintained trails, reliable infrastructure, clear markers, or scenic features, and other forms of positive conditions."
        }
    ],
    # Number of examples to generate per prompt
    num_samples_per_prompt=5,
    
    # Output file path
    output_file="trail_conditions_classification.jsonl",
    
    # Languages to generate data for
    languages={
        "en": "English", 
        "fr": "French"
    },
    
    # Custom prompts (optional - otherwise defaults will be used)
    prompts=[
        (
            "Generate {num_samples} hiker reports in {language_name} which are diverse "
            "and representative of a '{label_name}' trail condition category. "
            "{label_description}. The reports should be {{style}} and about a {{trail_type}}."
        )
    ]
)
```

## Step 3: Prompt Expansion for Diverse Examples

Prompt expansion is key concept in the `datafast` library. Prompt expansion helps generating multiple variations of a base prompt to increase the diversity of the generated data.

We are using two optional placholders for prompt expansion:

* `{{style}}`: In our example we want to generate hiker reports in different writing styles
* `{{trail_type}}`: In our example we want to generate reports about different types of trails


You can use different variables depending on your actual use case. For example, you can use `{{experience_level}}` to generate reports from hikers with different experience levels or `{{season}}` to generate reports across different seasons.

!!! Note
    Prompt expansion will automatically generate all possible combinations of style and trail type to maximize diversity. You can also limit the number of combinations by setting the `combinatorial` parameter to `False` and providing a value for `num_random_samples` instead to your `PromptExpansionConfig`.

Datafast's `PromptExpansionConfig` allows you to generate diverse examples by creating variations of your base prompts:

```python
config = ClassificationConfig(
    # Basic configuration as above
    # ...
    
    # Add prompt expansion configuration
    expansion=PromptExpansionConfig(
        placeholders={
            "trail_type": [
                "mountain trail",
                "coastal path",
                "forest walk",
            ],
            "style": [
                "brief social media post",
                "trail review"
            ]
        },
        combinatorial=True,  # Generate all combinations
        num_random_samples=200 # Only needed if combinatorial is False. Then samples 200 at random.
    )
)
```

This expansion creates prompt variations by replacing `{{style}}` and `{{trail_type}}` with all possible combinations of the provided values, dramatically increasing the diversity of your dataset.

## Step 4: Set Up LLM Providers

Configure one or more LLM providers to generate your dataset:

```python
providers = [
    OpenAIProvider(model_id="gpt-4o-mini"),
    AnthropicProvider(model_id="claude-3-5-haiku-latest")
]
```

Using multiple providers helps create more diverse and robust datasets.

## Step 5: How many instances will it generate?

The number of generated instances in your dataset in combinatorial mode can be calculated by multiplying the following:

- number of classes (here 4)
- number of languages (here 2)
- number of samples per prompt (here 5)
- number of LLM providers (here 2)
- number of variations for each optional placeholders 
    - 3 for ``{{trail_type}}`` here
    - 2 for ``{{style}}``

This means: 4 x 2 x 5 x 2 x 3 x 2 = 480 instances.
 
If that seems sufficient, and representative of your use case, we can proceed to generate the dataset.

## Step 6: Generate the Dataset

Now you can create and generate your dataset:

```python
# Initialize dataset with your configuration
dataset = TextClassificationDataset(config)

# Generate examples using configured providers
dataset.generate(providers)
```

This will:
1. Initialize a dataset with your configuration
2. For each label and language combination:
   - Create base prompts
   - Expand prompts with configured variations
   - Call each LLM provider with each expanded prompt
3. Save the dataset to the specified output file

## Step 7: Publishing to Hugging Face Hub (Optional)

You can easily publish your dataset to Hugging Face Hub for sharing or use in other projects:

```python
url = dataset.push_to_hub(
    repo_id="YOUR_USERNAME/your-dataset-name",
    train_size=0.8,  # Split 80% for training
    seed=42,         # Set seed for split reproducibility
    shuffle=True     # Shuffle before splitting
)
print(f"Dataset published at: {url}")
```

This automatically splits your dataset into training and test sets and uploads it to Hugging Face.

!!! warning
    Do not forget to set and load your HF_TOKEN environment variable before running this example.

## Complete Example

Here's a complete example for creating a trail conditions classification dataset:

```python
from datafast.datasets import TextClassificationDataset
from datafast.schema.config import ClassificationConfig, PromptExpansionConfig
from datafast.llms import OpenAIProvider, AnthropicProvider
from dotenv import load_dotenv

# Load API keys
load_dotenv("secrets.env")

# Configure dataset
config = ClassificationConfig(
    classes=[
        {
            "name": "trail_obstruction",
            "description": "Conditions where the trail is partially or fully blocked by obstacles like fallen trees, landslides, snow, or dense vegetation, and other forms of obstruction."
        },
        {
            "name": "infrastructure_issues",
            "description": "Problems related to trail structures and amenities, including damaged bridges, signage, stairs, handrails, or markers, and other forms of infrastructure issues."
        },
        {
            "name": "hazards",
            "description": "Trail conditions posing immediate risks to hiker safety, such as slippery surfaces, dangerous wildlife, hazardous crossings, or unstable terrain, and other forms of hazards."
        },
        {
            "name": "positive_conditions",
            "description": "Conditions highlighting clear, safe, and enjoyable hiking experiences, including well-maintained trails, reliable infrastructure, clear markers, or scenic features, and other forms of positive conditions."
        }
    ],
    num_samples_per_prompt=5,
    output_file="trail_conditions_classification.jsonl",
    languages={
        "en": "English", 
        "fr": "French"
    },
    prompts=[
        (
            "Generate {num_samples} hiker reports in {language_name} which are diverse "
            "and representative of a '{label_name}' trail condition category. "
            "{label_description}. The reports should be {{style}} and about a {{trail_type}}."
        )
    ],
    expansion=PromptExpansionConfig(
        placeholders={
            "trail_type": [
                "mountain trail",
                "coastal path",
                "forest walk"
            ],
            "style": [
                "brief social media post",
                "detailed trail review"
            ]
        },
        combinatorial=True,  # Generate all combinations
        num_random_samples=200  # Only needed if combinatorial is False
    )
)

# Set up providers
providers = [
    OpenAIProvider(model_id="gpt-4o-mini"),
    AnthropicProvider(model_id="claude-3-5-haiku-latest")
]

# Generate dataset
dataset = TextClassificationDataset(config)
dataset.generate(providers)

# Optional: Push to Hugging Face Hub
dataset.push_to_hub(
    repo_id="YOUR_USERNAME/trail-conditions-classification",
    train_size=0.8,
    seed=42,
    shuffle=True
)
```

## Understanding the Generated Data

Each generated example is stored as a `TextClassificationRow` with these properties:

- `text`: The generated text content
- `label`: The assigned classification label
- `model_id`: The LLM model that generated this example
- `label_source`: How the label was assigned (typically `SYNTHETIC` for generated data)
- `metadata`: Additional information like language code
- `uuid`: Unique identifier
