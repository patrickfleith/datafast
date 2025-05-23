from datafast.datasets import ClassificationDataset
from datafast.schema.config import ClassificationDatasetConfig, PromptExpansionConfig
from dotenv import load_dotenv
load_dotenv("secrets.env")

config = ClassificationDatasetConfig(
    classes=[
        {"name": "positive", "description": "Text expressing positive emotions or approval"},
        {"name": "negative", "description": "Text expressing negative emotions or criticism"},
        {"name": "neutral", "description": "Text with neutral emotions or indifference"}
    ],
    num_samples_per_prompt=3,
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

from datafast.llms import OpenAIProvider, AnthropicProvider, GeminiProvider, OllamaProvider

providers = [
    OpenAIProvider(model_id="gpt-4.1-mini-2025-04-14"),
    # AnthropicProvider(model_id="claude-3-5-haiku-latest"),
    # GeminiProvider(model_id="gemini-2.0-flash"),
    # OllamaProvider(model_id="gemma3:12b")
]

# Generate dataset
dataset = ClassificationDataset(config)
num_expected_rows = dataset.get_num_expected_rows(providers)
print(f"Expected number of rows: {num_expected_rows}")
dataset.generate(providers)

# Optional: Push to Hugging Face Hub
USERNAME = "username"  # <--- Your hugging face username
DATASET_NAME = "dataset_name"  # <--- Your hugging face dataset name
dataset.push_to_hub(
    repo_id=f"{USERNAME}/{DATASET_NAME}",
    train_size=0.6
)
