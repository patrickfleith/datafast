# How to Create an Ultrachat Dataset

!!! example "Use Case"
    Suppose you are a data engineer at a large wind turbine manufacturer. You want to **fine-tune a conversational AI model on the specific domain of materials science**. You decide to create an *Ultrachat Instruction Dataset* with synthetic conversations spanning multiple topics and subtopics. This dataset will contain realistic back-and-forth exchanges based on your company's user personas.

**The process:**

1. Define a domain (Materials Science), topics, and subtopics for conversation generation.
2. Create a pool of user personas to simulate different types of users.
3. For each topic-subtopic pair:
   - Generate initial questions.
   - Reformulate them based on random personas.
   - Simulate assistant responses.
   - Optionally, continue the conversation with follow-up exchanges for multi-turn conversations.
4. Save the dataset to the specified output file

!!! note
    What is an Ultrachat instruction dataset? It is a standard instruction dataset consisting of conversations between a user and an assistant. The term `Ultrachat` refers to previous from which this method is inspired. Conversations are generated based on a domain, topics, and subtopics [Ultrachat method](https://arxiv.org/abs/2305.14233).

## Step 1: Import Required Modules

Generating an Ultrachat dataset with `datafast` requires three types of imports:

* Dataset
* Configs
* LLM Providers

```python
from datafast.datasets import UltrachatDataset
from datafast.schema.config import UltrachatDatasetConfig, PromptExpansionConfig
from datafast.llms import OpenAIProvider, AnthropicProvider, GeminiProvider
```

In addition, use `dotenv` to load environment variables containing API keys:
```python
from dotenv import load_dotenv

# Load environment variables containing API keys
load_dotenv("secrets.env")
```

Make sure you have created a `secrets.env` file with your API keys. A Hugging Face token (HF_TOKEN) is needed if you want to push the dataset to your HF hub. Other keys depend on which LLM providers you use.

```
GEMINI_API_KEY=XXXX
OPENAI_API_KEY=sk-XXXX
ANTHROPIC_API_KEY=sk-ant-XXXXX
HF_TOKEN=hf_XXXXX
```

## Step 2: Define Your Domain Knowledge Structure

The `UltrachatDataset` generates conversations based on a hierarchical knowledge structure:

1. **Domain**: The high-level field of knowledge (in our case, "Materials Science" for wind turbine manufacturing)
2. **Topics**: Major areas within the domain (like "Composite Materials", "Metal Alloys", "Polymers")
3. **Subtopics**: Specific subjects within each topic (like "Fatigue Testing", "Corrosion Resistance", "Thermal Properties")

This structure helps generate focused, coherent conversations about particular subjects. You'll need to define this structure as part of your configuration.

!!! tip
    To find topics and subtopics: look for domain-specific lexicon, glossaries, or existing ontologies. Ask an LLM or expert in your field. For a wind turbine manufacturer, consider reviewing materials engineering textbooks, industry standards, or consulting with your materials science department.

## Step 3: Create User Personas

To generate realistic conversations, you need to define a list of user personas. These personas will be used to reformulate generic questions into more contextual, persona-specific inquiries relevant to your wind turbine manufacturing company.

Good personas for your use case might include:

- *Junior Engineer with 2 years of experience in the technical department*
- *Senior Materials Scientist specializing in durability under extreme conditions*
- *R&D Director evaluating materials for next-generation products*

## Step 4: Configure Your Dataset

The `UltrachatDatasetConfig` class defines all parameters for your Ultrachat dataset generation:

- **`domain`**: The high-level field of knowledge for the conversations
- **`topics_and_subtopics`**: Dictionary mapping topics to lists of subtopics
- **`personas`**: List of user personas for question reformulation
- **`num_samples`**: Number of initial questions to generate for each topic-subtopic pair
- **`max_turns`**: Maximum number of turns in each conversation (default: 1 turn = A user question and and the assistant response)
- **`conversation_continuation_prob`**: Probability of continuing the conversation after each turn (default: 0.5)
- **`output_file`**: Path where the generated dataset will be saved (JSONL format)
- **`question_generation_prompts`**: (Optional) Custom prompt templates for generating initial questions
- **`languages`**: Dictionary mapping language codes to their names (e.g., `{"en": "English"}`)

Here's a basic configuration example:

```python
config = UltrachatDatasetConfig(
    # Domain of knowledge
    domain="Materials Science",
    
    # Topics and subtopics structure
    topics_and_subtopics={
        "Composite Materials": ["Fiber Reinforcement", "Resin Systems", "Fatigue Testing"],
        "Metal Alloys": ["Corrosion Resistance", "Structural Integrity", "Welding Properties"],
        "Polymers": ["UV Degradation", "Thermal Properties", "Impact Resistance"]
    },
    
    # User personas specific to wind turbine manufacturing
    personas=[
        "Junior Engineer with 2 years of experience in the technical department",
        "Senior Materials Scientist specializing in durability under extreme conditions",
        "R&D Director evaluating materials for next-generation products",
    ],
    
    # Generate 4 questions per topic-subtopic pair
    num_samples=3,
    
    # Allow up to 3 back-and-forth exchanges
    max_turns=3,
    
    # 25% chance of continuing conversation after each exchange
    conversation_continuation_prob=0.25,
    
    # Where to save the results
    output_file="wind_turbine_materials_conversations.jsonl",
    
    # Language(s) to generate conversations in
    languages={"en": "English"}
)
```

## Step 5: Custom Prompts (Optional)

You can customize the prompts used for generating questions and responses in the conversation.

### Question Generation Prompts

When providing custom prompts for the `question_generation_prompts` parameter, you **must include these mandatory placeholders**:

- `{num_samples}`: Number of questions to generate per topic-subtopic pair
- `{language_name}`: Language to generate questions in
- `{domain}`: The domain specified in the configuration
- `{topic}`: The current topic being processed
- `{subtopic}`: The current subtopic being processed


## Step 6: Prompt Expansion for Diverse Questions (Optional)

Just like with other Datafast dataset types, you can use prompt expansion to generate more diverse questions. 
We won't do that in this example to keep it simple. Prompt expansion would tremendeously increase the number of prompt variation
and we have enough already with the list of topics, subtopics and personas.

!!! warning
    When adding prompt expansion, we have to be carefull not to add potentially conflicting placeholders with respect to the topics, subtopics, or personas. So use with caution.

## Step 7: Set Up LLM Providers

Configure one or more LLM providers to generate your dataset:

```python
providers = [
    AnthropicProvider(model_id="claude-3-5-haiku-latest"),
]
```

Using multiple LLM providers helps create more diverse and robust conversation datasets. In our case, we just use one for demonstration purpose.
If you choose N providers, each prompt variation will be passed to all providers you would get N times more generated rows.

## Step 8: How Many Conversations Will Be Generated?

Before generating the dataset, you can calculate the expected number of conversations:

```python
dataset = UltrachatDataset(config)
num_expected_rows = dataset.get_num_expected_rows(providers)
print(f"Expected number of rows: {num_expected_rows}")
```

The number of conversations will be calculated as:

- Number of topics × Number of subtopics × Number of languages × Number of LLM providers × Number of initial questions per topic-subtopic pair

For our wind turbine materials example:

- 3 topics (Composite Materials, Metal Alloys, Polymers)
- 3 subtopics per topic (so a total of 9 topic-subtopic pairs)
- 1 language (English)
- 1 LLM provider (Claude)
- 3 questions generated for each topic-subtopic pair

You'd get: 3 × 3 × 1 × 1 × 3 = 27 initial conversations in the dataset.

However, since we set `max_turns=3` and `conversation_continuation_prob=0.25`, some conversations will continue with follow-up exchanges. The actual number of conversational turns will vary based on the random continuation probability. A multi-turn conversation is still one single row in the dataset.

## Step 9: Generate the Dataset

Now you can create and generate your Ultrachat dataset:

```python
# Initialize dataset with your configuration
dataset = UltrachatDataset(config)

# Generate conversations using configured providers
dataset.generate(providers)

# Print results summary
print(f"Generated {len(dataset.data_rows)} conversations")
print(f"Results saved to {config.output_file}")
```

## Step 10: Push to Hugging Face Hub (Optional)

After generating your dataset, you can push it to the Hugging Face Hub for sharing:

```python
YOUR_USERNAME = "your-username"
DATASET_NAME = "your-dataset-name"
url = dataset.push_to_hub(
    repo_id=f"{YOUR_USERNAME}/{DATASET_NAME}",
    train_size=0.8,  # Split 80% for training
    seed=42,         # Set seed for split reproducibility
    shuffle=True     # Shuffle before splitting
)
print(f"Dataset published at: {url}")
```

This automatically splits your dataset into training and test sets and uploads it to Hugging Face.

!!! warning
    Don't forget to set and load your HF_TOKEN environment variable before running this example.

## Complete Example

Here's a complete example for creating an Ultrachat dataset:

```python
from datafast.datasets import UltrachatDataset
from datafast.schema.config import UltrachatDatasetConfig
from datafast.llms import AnthropicProvider
from dotenv import load_dotenv

# Load environment variables
load_dotenv("secrets.env")

def main():
    # 1. Define the configuration
    config = UltrachatDatasetConfig(
        domain="Materials Science for Wind Turbines",
        topics_and_subtopics={
            "Composite Materials": ["Fiber Reinforcement", "Resin Systems", "Fatigue Testing"],
            "Metal Alloys": ["Corrosion Resistance", "Structural Integrity", "Welding Properties"],
            "Polymers": ["UV Degradation", "Thermal Properties", "Impact Resistance"]
        },
        personas=[
            "Junior Engineer with 2 years of experience in the technical department",
            "Senior Materials Scientist specializing in durability under extreme conditions",
            "R&D Director evaluating materials for next-generation products"
        ],
        num_samples=3,
        max_turns=3,
        conversation_continuation_prob=0.25,
        output_file="wind_turbine_materials_conversations.jsonl",
        languages={"en": "English"}
    )

    # 2. Initialize LLM providers - using just one for simplicity
    providers = [
        AnthropicProvider(model_id="claude-3-5-haiku-latest"),
    ]

    # 3. Get expected number of rows
    dataset = UltrachatDataset(config)
    num_expected_rows = dataset.get_num_expected_rows(providers)
    print(f"\nExpected number of rows: {num_expected_rows}")
    
    # 4. Generate the dataset
    dataset.generate(providers)

    # 5. Print results summary
    print(f"\nGenerated {len(dataset.data_rows)} conversations")
    print(f"Results saved to {config.output_file}")

    # 6. Optional: Push to HF hub
    # USERNAME = "your_username"  # <--- Your hugging face username
    # DATASET_NAME = "wind_turbine_materials_dataset"  # <--- Your hugging face dataset name
    # url = dataset.push_to_hub(
    #     repo_id=f"{USERNAME}/{DATASET_NAME}",
    #     train_size=0.8,
    #     seed=42,
    #     shuffle=True,
    # )
    # print(f"\nDataset pushed to Hugging Face Hub: {url}")

if __name__ == "__main__":
    main()
```

## Understanding the Generated Data

Each generated conversation is stored as a `ChatRow` with these properties:

- `opening_question`: The initial reformulated user question
- `messages`: A list of message objects with `role` (user/assistant) and `content`
- `model_id`: The LLM model that generated this conversation
- `metadata`: Additional information like language code, domain, topic, and subtopic
- `persona`: The persona used for the user in this conversation
- `uuid`: Unique identifier

## Beyond Ultrachat

There are different approaches in generating an instruction dataset. I plan to implement different strategy than the specific Ultrachat method used here.
If you have some idea or request, feel free to open a feature request.
