from datafast.datasets import TextClassificationDataset
from datafast.schema.config import ClassificationConfig, PromptExpansionConfig
from datafast.llms import create_provider

def main():
    # 1. Configure the dataset generation
    config = ClassificationConfig(
        classes=[
            {
                "name": "positive",
                "description": "A positive sentiment expressing approval, happiness, or satisfaction"
            },
            {
                "name": "negative", 
                "description": "A negative sentiment expressing disapproval, unhappiness, or dissatisfaction"
            }
        ],
        num_samples_per_prompt=3,  # Get 3 examples per prompt
        output_file="sentiment_dataset.jsonl",
    )

    # 2. Create LLM providers (will use default models)
    providers = [
        create_provider("anthropic"),  # Uses claude-3-5-haiku-latest
        create_provider("google"),     # Uses gemini-1.5-flash
        create_provider("openai")      # Uses gpt-4o-mini
    ]

    # 3. Generate the dataset
    dataset = TextClassificationDataset(config)
    dataset.generate(providers)

    print(f"Generated dataset saved to {config.output_file}")
    
    # Print first few examples
    for i, row in enumerate(dataset.data_rows[:5]):
        print(f"\nExample {i+1}:")
        print(f"Text: {row.text}")
        print(f"Label: {row.label}")
        print(f"Model: {row.model_id}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv('secrets.env')
    main()
