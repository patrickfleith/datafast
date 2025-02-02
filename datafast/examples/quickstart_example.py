from cmath import polar
from datafast.datasets import TextClassificationDataset
from datafast.schema.config import ClassificationConfig, PromptExpansionConfig
from dotenv import load_dotenv
load_dotenv("secrets.env")

config = ClassificationConfig(
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

from datafast.llms import OpenAIProvider, AnthropicProvider, GoogleProvider

providers = [
    OpenAIProvider(model_id="gpt-4o-mini"),
    AnthropicProvider(model_id="claude-3-5-haiku-latest"),
    GoogleProvider(model_id="gemini-1.5-flash")
]

# Generate dataset
dataset = TextClassificationDataset(config)
dataset.generate(providers)

# Optional: Push to Hugging Face Hub
dataset.push_to_hub(
    repo_id="YOUR_USERNAME/sentiment-dataset-quickstart",
    train_size=0.6
)

