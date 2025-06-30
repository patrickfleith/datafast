"""
Example script showing how to generate a dataset and launch the visual inspector.

Run with:
    python -m datafast.examples.inspect_dataset_example

Requires:
    - OpenAI API key in secrets.env or environment
    - gradio package (pip install gradio)
"""
from datafast.datasets import ClassificationDataset
from datafast.schema.config import ClassificationDatasetConfig, PromptExpansionConfig
from dotenv import load_dotenv

# Load API keys from environment or secrets.env
load_dotenv("secrets.env")

# Configure the dataset generation
config = ClassificationDatasetConfig(
    classes=[
        {"name": "positive", "description": "Text expressing positive emotions or approval"},
        {"name": "negative", "description": "Text expressing negative emotions or criticism"},
    ],
    num_samples_per_prompt=2,  # Small number for quick demo
    output_file="outdoor_activities_sentiments.jsonl",  # Optional, will save generated data
    languages={
        "en": "English",
    },
    prompts=[
        (
            "Generate {num_samples} reviews in {language_name} which are diverse "
            "and representative of a '{label_name}' sentiment class. "
            "{label_description}. The reviews should be brief and in the "
            "context of {{context}}."
        )
    ],
    expansion=PromptExpansionConfig(
        placeholders={
            "context": ["hiking trail review", "kayaking trip review"],
        },
        combinatorial=True  # Will generate combinations of all placeholders
    )
)

# Set up LLM providers
from datafast.llms import OpenAIProvider, AnthropicProvider, GeminiProvider

providers = [
    OpenAIProvider(model_id="gpt-4.1-nano"),
    # Uncomment to use additional providers
    # AnthropicProvider(model_id="claude-3-5-haiku-latest"),
    # GeminiProvider(model_id="gemini-2.0-flash"),
]

def main():
    # Generate the dataset
    dataset = ClassificationDataset(config)
    num_expected_rows = dataset.get_num_expected_rows(providers)
    print(f"Expected number of rows to generate: {num_expected_rows}")
    
    # Generate data (comment out if loading existing data)
    print("Generating dataset...")
    dataset.generate(providers)
    print(f"Generated {len(dataset.data_rows)} examples")
    
    # Launch the interactive inspector
    print("\nLaunching dataset inspector...")
    print("(Close the browser window or press Ctrl+C to exit)")
    print("Showing examples in random order")
    dataset.inspect(random=True)
    
if __name__ == "__main__":
    main()
