from datafast.datasets import TextDataset
from datafast.schema.config import TextDatasetConfig, PromptExpansionConfig
from datafast.llms import OpenAIProvider, AnthropicProvider, GoogleProvider


def main():
    # 1. Configure the dataset generation
    config = TextDatasetConfig(
        document_types=["tech journalism blog", "personal blog"],
        topics=["artificial intelligence", "cybersecurity"],
        num_samples_per_prompt=3,
        output_file="tech_posts.jsonl",
        languages={"en": "English", "fr": "French"},
        prompts=[
            (
                "Generate {num_samples} {document_type} entries in {language_name} about {topic}. "
                "The emphasis should be a perspective from {{country}}"
            )
        ],
        expansion=PromptExpansionConfig(
            placeholders={
                "country": ["United States", "Europe", "India"]
            },
            combinatorial=True,
        )
    )

    # 2. Create LLM providers with specific models
    providers = [
        OpenAIProvider(model_id="gpt-4o-mini"),
        # AnthropicProvider(model_id="claude-3-5-haiku-latest"),
    ]

    # 3. Generate the dataset
    dataset = TextDataset(config)
    dataset.generate(providers)

    # 4. Push to HF hub (optional)
    USERNAME = "your_huggingface_username"
    DATASET_NAME = "your_dataset_name"
    url = dataset.push_to_hub(
        repo_id=f"{USERNAME}/{DATASET_NAME}",
        train_size=0.7,  # for a 80/20 train/test split, otherwise omit
        seed=20250304,
        shuffle=True,
    )
    print(f"\nDataset pushed to Hugging Face Hub: {url}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv("secrets.env")
    main()