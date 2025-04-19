from datafast.datasets import ClassificationDataset
from datafast.schema.config import ClassificationDatasetConfig, PromptExpansionConfig
from datafast.llms import OpenAIProvider, AnthropicProvider
from dotenv import load_dotenv

# Load API keys
load_dotenv("secrets.env")

# Configure dataset
config = ClassificationDatasetConfig(
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
    num_samples_per_prompt=10,
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
                "forest walk",
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
    OpenAIProvider(model_id="gpt-4.1-mini-2025-04-14"),
    AnthropicProvider(model_id="claude-3-5-haiku-latest")
]

# Generate dataset
dataset = ClassificationDataset(config)
dataset.generate(providers)

# Optional: Push to Hugging Face Hub
USERNAME = "YOUR_USERNAME"  # <--- Your hugging face username
DATASET_NAME = "YOUR_DATASET_NAME"  # <--- Your hugging face dataset name
dataset.push_to_hub(
    repo_id=f"{USERNAME}/{DATASET_NAME}",
    train_size=0.8,
    seed=42,
    shuffle=True
)