from datafast.datasets import TextClassificationDataset
from datafast.schema.config import ClassificationConfig, PromptExpansionConfig
from datafast.llms import create_provider, OpenAIProvider, AnthropicProvider, GoogleProvider
from datafast.expanders import expand_prompts

def main():
    # 1. Configure dataset generation with prompt expansion
    config = ClassificationConfig(
        classes=[
            {
                "name": "positive",
                "description": "Text expressing positive emotions, approval, or favorable opinions"
            },
            {
                "name": "negative",
                "description": "Text expressing negative emotions, criticism, or unfavorable opinions"
            },
            {
                "name": "neutral",
                "description": "Text expressing factual, objective, or balanced statements without strong emotions"
            }
        ],
        num_samples_per_prompt=5,  # Generate 5 samples per expanded prompt
        output_file="sentiment_dataset.jsonl",
        languages={'en': 'English'},  # English only
        prompts=[
            """Generate {num_samples} reviews in {language_name} which are diverse \
            and representative of a '{label_name}' sentiment class. {label_description}. \
            The reviews should be {{style}} and in the context of {{context}}."""
        ],
        expansion=PromptExpansionConfig(
            placeholders={
                "context": [
                    "product review",
                    "movie review",
                    "restaurant experience",
                    "customer service",
                    "travel experience"
                ],
                "style": [
                    "brief",
                    "detailed"
                ]
            },
            combinatorial=True  # Generate all combinations
        )
    )

    # 2. Create LLM providers with recommended models
    providers = [
        OpenAIProvider(model_id='gpt-4o-mini'),
    ]

    # 3. Generate the dataset
    dataset = TextClassificationDataset(config)

    dataset.generate(providers)

    # 4. Push to HF hub (optional)
    # USERNAME = "your-username"
    # url = dataset.push_to_hub(
    #     repo_id=f"{USERNAME}/sentiment-classification-demo",
    #     train_size=0.8,
    #     seed=42,
    #     shuffle=True,
    # )
    # print(f"\nDataset pushed to Hugging Face Hub: {url}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv('secrets.env')
    main()
